# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Planner-only Triton kernels for MoonEP-style expert replicas.

This module contains the route compaction, replica placement, and virtual expert
mapping kernels from NVIDIA/Megatron-LM PR #6892.  The weight transport kernels
from that PR intentionally stay out of this module; the common MoEScheduler
interface still hands expert materialization to ``ExpertDispatch``.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAVE_TRITON = False

MAX_REPLICA_EP_RANKS = 64
_MAX_PLANNER_PROGRAMS = 128


def planner_route_partition_count(num_routes: int) -> int:
    """Return the shared route-ranking and route-mapping grid width."""
    if num_routes <= 0:
        raise ValueError(f"num_routes must be positive, got {num_routes}.")
    return min(_MAX_PLANNER_PROGRAMS, num_routes)


def _require_triton(operation: str) -> None:
    if not HAVE_TRITON:
        raise RuntimeError(f"{operation} requires Triton.")


def _require_cuda_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


if HAVE_TRITON:
    _GRID_SYNC_TAG = tl.constexpr(0x40000000)

    @triton.jit
    def _emit_on_every_thread(ASM: tl.constexpr, THREADS: tl.constexpr):
        """Run one side-effecting PTX instruction on every thread of the block."""
        tl.inline_asm_elementwise(
            ASM,
            "=r,r",
            [tl.zeros([THREADS], tl.int32)],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def _grid_sync(grid_barrier, TAG: tl.constexpr, NUM_SMS: tl.constexpr):
        """Self-resetting cooperative-grid barrier."""
        tl.debug_barrier()
        increment = tl.where(tl.program_id(0) == 0, TAG - (NUM_SMS - 1), 1)
        previous = tl.atomic_add(grid_barrier, increment, sem="release", scope="gpu")
        complete = False
        while not complete:
            current = tl.atomic_add(grid_barrier, 0, sem="acquire", scope="gpu")
            complete = ((current ^ previous) & TAG) != 0
        tl.debug_barrier()

    @triton.jit(do_not_specialize=["source_rank"])
    def _plan_replica_placement_kernel(
        gathered_tokens_per_expert,
        rank_load_balance,
        expert_rank_allocations,
        destination_boundaries,
        experts_to_copy,
        expert_replica_slots,
        grid_sync,
        source_rank,
        RANK_ROUTE_CAPACITY: tl.constexpr,
        EP_SIZE: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        NUM_EXPERTS_PER_GPU: tl.constexpr,
        BLOCK_EP_SIZE: tl.constexpr,
        BLOCK_NUM_EXPERTS_PER_GPU: tl.constexpr,
        BLOCK_NUM_EXPERTS: tl.constexpr,
    ):
        """Compute deterministic replica placement in one cooperative launch."""
        rank = tl.program_id(0)
        ranks = tl.arange(0, BLOCK_EP_SIZE)
        valid_ranks = ranks < EP_SIZE
        local_experts = tl.arange(0, BLOCK_NUM_EXPERTS_PER_GPU)
        valid_local_experts = local_experts < NUM_EXPERTS_PER_GPU
        native_experts = rank * NUM_EXPERTS_PER_GPU + local_experts

        source_counts = tl.load(
            gathered_tokens_per_expert
            + ranks[:, None] * NUM_EXPERTS
            + native_experts[None, :],
            mask=valid_ranks[:, None] & valid_local_experts[None, :],
            other=0,
        )
        native_totals = tl.sum(source_counts, axis=0).to(tl.int32)
        routes_before_source = tl.sum(
            tl.where(ranks[:, None] < source_rank, source_counts, 0), axis=0
        ).to(tl.int32)
        tl.store(
            rank_load_balance + rank,
            tl.sum(native_totals, axis=0).to(tl.int32) - RANK_ROUTE_CAPACITY,
        )

        _grid_sync(grid_sync, _GRID_SYNC_TAG, EP_SIZE)

        balances = tl.load(rank_load_balance + ranks, mask=valid_ranks, other=0)
        quotas = tl.zeros((BLOCK_EP_SIZE,), dtype=tl.int32)
        for _ in tl.range(0, EP_SIZE, 1, loop_unroll_factor=1):
            maximum = tl.max(tl.where(valid_ranks, balances, -2147483648), axis=0)
            minimum = tl.min(tl.where(valid_ranks, balances, 2147483647), axis=0)
            overloaded = tl.min(
                tl.where(valid_ranks & (balances == maximum), ranks, BLOCK_EP_SIZE),
                axis=0,
            )
            receiver = tl.min(
                tl.where(valid_ranks & (balances == minimum), ranks, BLOCK_EP_SIZE),
                axis=0,
            )
            active = maximum > 0
            moved = tl.where(active, -minimum, 0).to(tl.int32)
            quotas = tl.where(
                active & (overloaded == rank) & (ranks == receiver), moved, quotas
            )
            balances = tl.where(active & (ranks == overloaded), balances - moved, balances)
            balances = tl.where(active & (ranks == receiver), 0, balances)

        remaining = native_totals
        allocations = tl.where(ranks[None, :] == rank, native_totals[:, None], 0)
        for _ in tl.range(0, EP_SIZE + NUM_EXPERTS_PER_GPU, 1, loop_unroll_factor=1):
            max_quota = tl.max(tl.where(valid_ranks, quotas, -1), axis=0)
            destination = tl.min(
                tl.where(valid_ranks & (quotas == max_quota), ranks, BLOCK_EP_SIZE),
                axis=0,
            )
            max_remaining = tl.max(tl.where(valid_local_experts, remaining, -1), axis=0)
            local_expert = tl.min(
                tl.where(
                    valid_local_experts & (remaining == max_remaining),
                    local_experts,
                    BLOCK_NUM_EXPERTS_PER_GPU,
                ),
                axis=0,
            )
            active = max_quota > 0
            moved = tl.where(active, tl.minimum(max_quota, max_remaining), 0).to(
                tl.int32
            )
            transfer = tl.where(
                ranks[None, :] == destination,
                moved,
                tl.where(ranks[None, :] == rank, -moved, 0),
            )
            allocations += tl.where(
                (local_experts[:, None] == local_expert) & active, transfer, 0
            )
            remaining = tl.where(
                active & (local_experts == local_expert), remaining - moved, remaining
            )
            quotas = tl.where(active & (ranks == destination), quotas - moved, quotas)

        tl.store(
            expert_rank_allocations + native_experts[:, None] * EP_SIZE + ranks[None, :],
            allocations,
            mask=valid_local_experts[:, None] & valid_ranks[None, :],
        )
        tl.store(
            destination_boundaries + native_experts[:, None] * BLOCK_EP_SIZE + ranks[None, :],
            tl.cumsum(allocations, axis=1) - routes_before_source[:, None],
            mask=valid_local_experts[:, None],
        )

        _grid_sync(grid_sync, _GRID_SYNC_TAG, EP_SIZE)

        experts = tl.arange(0, BLOCK_NUM_EXPERTS)
        owner = experts // NUM_EXPERTS_PER_GPU
        valid_remote = (experts < NUM_EXPERTS) & (owner != rank)
        counts = tl.load(
            expert_rank_allocations + experts * EP_SIZE + rank,
            mask=valid_remote,
            other=-1,
        )
        for slot in tl.range(0, NUM_EXPERTS_PER_GPU, 1, loop_unroll_factor=1):
            maximum = tl.max(tl.where(valid_remote, counts, -1), axis=0)
            expert = tl.max(tl.where(valid_remote & (counts == maximum), experts, -1), axis=0)
            selected = tl.where(maximum > 0, expert, -1).to(tl.int32)
            tl.store(experts_to_copy + rank * NUM_EXPERTS_PER_GPU + slot, selected)
            tl.store(expert_replica_slots + selected * EP_SIZE + rank, slot, mask=selected >= 0)
            counts = tl.where(experts == expert, -1, counts)
        if tl.max(tl.where(valid_remote, counts, -1), axis=0) > 0:
            tl.device_print("replica placement needs more replica slots than experts on rank", rank)
            _emit_on_every_thread("trap; mov.u32 $0, 0;", THREADS=1)

    @triton.jit
    def _rank_routes_within_experts_kernel(
        flat_topk_indices,
        route_metadata,
        partition_counts,
        grid_sync,
        NUM_ROUTES: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        BLOCK_NUM_EXPERTS: tl.constexpr,
        BLOCK_NUM_ROUTES: tl.constexpr,
        BLOCK_SCAN_PARTITIONS: tl.constexpr,
        NUM_SCAN_EXPERTS: tl.constexpr,
    ):
        """Give each route its stable ordinal within its expert's local stream."""
        partition = tl.program_id(0)
        num_partitions = tl.num_programs(0)
        expert_offsets = tl.arange(0, BLOCK_NUM_EXPERTS)
        valid_experts = expert_offsets < NUM_EXPERTS
        routes_per_partition = tl.cdiv(NUM_ROUTES, num_partitions)
        partition_start = partition * routes_per_partition
        partition_end = tl.minimum(partition_start + routes_per_partition, NUM_ROUTES)
        partition_histogram = tl.zeros((BLOCK_NUM_EXPERTS,), dtype=tl.int32)
        tile_offsets = tl.arange(0, BLOCK_NUM_ROUTES)

        for route_start in tl.range(
            partition_start, partition_end, BLOCK_NUM_ROUTES, loop_unroll_factor=1
        ):
            route_positions = route_start + tile_offsets
            valid_routes = route_positions < partition_end
            route_experts = tl.load(
                flat_topk_indices + route_positions,
                mask=valid_routes,
                other=NUM_EXPERTS + tile_offsets,
            ).to(tl.int32)
            ranks_in_tile = tl.inline_asm_elementwise(
                asm="""
                {
                    .reg .b32 matching_lanes;
                    .reg .b32 lower_lanes;
                    match.sync.any.b32 matching_lanes, $1, 0xffffffff;
                    mov.u32 lower_lanes, %lanemask_lt;
                    and.b32 matching_lanes, matching_lanes, lower_lanes;
                    popc.b32 $0, matching_lanes;
                }
                """,
                constraints="=r,r",
                args=[route_experts],
                dtype=tl.int32,
                is_pure=True,
                pack=1,
            )
            safe_route_experts = tl.where(valid_routes, route_experts, 0)
            first_warp_counts = tl.histogram(
                route_experts, BLOCK_NUM_EXPERTS, mask=valid_routes & (tile_offsets < 32)
            )
            second_warp_counts = tl.histogram(
                route_experts,
                BLOCK_NUM_EXPERTS,
                mask=valid_routes & (tile_offsets >= 32) & (tile_offsets < 64),
            )
            preceding_warp_counts = tl.gather(first_warp_counts, safe_route_experts, axis=0)
            ranks_in_tile += tl.where(tile_offsets >= 32, preceding_warp_counts, 0)
            ordinals_before_tile = tl.gather(partition_histogram, safe_route_experts, axis=0)
            local_ordinals = ordinals_before_tile + ranks_in_tile
            tl.store(
                route_metadata + route_positions,
                local_ordinals * BLOCK_NUM_EXPERTS + route_experts,
                mask=valid_routes,
            )
            partition_histogram += first_warp_counts + second_warp_counts

        tl.store(
            partition_counts + partition * NUM_EXPERTS + expert_offsets,
            partition_histogram,
            mask=valid_experts,
        )

        _grid_sync(grid_sync, _GRID_SYNC_TAG, num_partitions)

        partition_offsets = tl.arange(0, BLOCK_SCAN_PARTITIONS)
        valid_partitions = partition_offsets < num_partitions
        for scan_expert_offset in tl.static_range(0, NUM_SCAN_EXPERTS):
            scan_expert = partition + scan_expert_offset * num_partitions
            valid_scan = valid_partitions & (scan_expert < NUM_EXPERTS)
            counts_for_expert = tl.load(
                partition_counts + partition_offsets * NUM_EXPERTS + scan_expert,
                mask=valid_scan,
                other=0,
            )
            tl.store(
                partition_counts + partition_offsets * NUM_EXPERTS + scan_expert,
                tl.cumsum(counts_for_expert, axis=0) - counts_for_expert,
                mask=valid_scan,
            )

    @triton.jit
    def _map_virtual_experts_kernel(
        route_metadata,
        partition_counts,
        destination_boundaries,
        expert_replica_slots,
        virtual_experts,
        NUM_ROUTES: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        NUM_EXPERTS_PER_GPU: tl.constexpr,
        EP_SIZE: tl.constexpr,
        BLOCK_NUM_EXPERTS: tl.constexpr,
        BLOCK_NUM_ROUTES: tl.constexpr,
        BLOCK_EP_SIZE: tl.constexpr,
        LOG2_BLOCK_EP_SIZE: tl.constexpr,
    ):
        """Map local route ordinals to rank-major native or replica expert ids."""
        partition = tl.program_id(0)
        num_partitions = tl.num_programs(0)
        routes_per_partition = tl.cdiv(NUM_ROUTES, num_partitions)
        partition_start = partition * routes_per_partition
        partition_end = tl.minimum(partition_start + routes_per_partition, NUM_ROUTES)
        expert_offsets = tl.arange(0, BLOCK_NUM_EXPERTS)
        routes_before_partition = tl.load(
            partition_counts + partition * NUM_EXPERTS + expert_offsets,
            mask=expert_offsets < NUM_EXPERTS,
            other=0,
        )
        tile_offsets = tl.arange(0, BLOCK_NUM_ROUTES)

        for route_start in tl.range(
            partition_start, partition_end, BLOCK_NUM_ROUTES, loop_unroll_factor=1
        ):
            route_positions = route_start + tile_offsets
            valid_routes = route_positions < partition_end
            packed_metadata = tl.load(
                route_metadata + route_positions, mask=valid_routes, other=0
            ).to(tl.int32)
            experts = packed_metadata % BLOCK_NUM_EXPERTS
            ordinals_in_partition = packed_metadata // BLOCK_NUM_EXPERTS
            safe_experts = tl.where(valid_routes, experts, 0)
            local_ordinal = (
                tl.gather(routes_before_partition, safe_experts, axis=0)
                + ordinals_in_partition
            )

            boundary_base = destination_boundaries + safe_experts * BLOCK_EP_SIZE
            destination = tl.zeros((BLOCK_NUM_ROUTES,), dtype=tl.int32)
            for step in tl.static_range(0, LOG2_BLOCK_EP_SIZE):
                candidate = destination + (BLOCK_EP_SIZE >> (step + 1))
                boundary = tl.load(boundary_base + candidate - 1)
                destination = tl.where(local_ordinal >= boundary, candidate, destination)

            owner = experts // NUM_EXPERTS_PER_GPU
            owned_local = experts % NUM_EXPERTS_PER_GPU
            replica_slot = tl.load(
                expert_replica_slots + safe_experts * EP_SIZE + destination,
                mask=valid_routes & (destination != owner),
                other=-1,
            )
            runtime_local = tl.where(
                destination == owner, owned_local, NUM_EXPERTS_PER_GPU + replica_slot
            )
            virtual = destination.to(tl.int64) * (2 * NUM_EXPERTS_PER_GPU) + runtime_local
            tl.store(virtual_experts + route_positions, virtual, mask=valid_routes)

    @triton.jit
    def _compact_routing_map_kernel(
        routing_map,
        token_indices,
        tokens_per_expert,
        num_tokens,
        ROUTER_TOPK: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
        BLOCK_NUM_EXPERTS: tl.constexpr,
    ):
        """Compact a dense routing map and accumulate its expert histogram."""
        program = tl.program_id(0)
        experts = tl.arange(0, BLOCK_NUM_EXPERTS)
        valid_experts = experts < NUM_EXPERTS
        token_offsets = tl.arange(0, BLOCK_TOKENS)
        histogram = tl.zeros((BLOCK_NUM_EXPERTS,), dtype=tl.int32)
        tokens_per_program = tl.cdiv(num_tokens, tl.num_programs(0))
        program_start = program * tokens_per_program
        program_end = tl.minimum(program_start + tokens_per_program, num_tokens)

        for token_start in tl.range(program_start, program_end, BLOCK_TOKENS, loop_unroll_factor=1):
            tokens = token_start + token_offsets
            valid = (tokens[:, None] < program_end) & valid_experts[None, :]
            selected = tl.load(
                routing_map + tokens[:, None] * NUM_EXPERTS + experts[None, :],
                mask=valid,
                other=0,
            ).to(tl.int32)
            slots = tl.cumsum(selected, axis=1) - selected
            tl.store(
                token_indices + tokens[:, None] * ROUTER_TOPK + slots,
                tl.broadcast_to(experts[None, :], (BLOCK_TOKENS, BLOCK_NUM_EXPERTS)),
                mask=(selected != 0) & (slots < ROUTER_TOPK),
            )
            histogram += tl.sum(selected, axis=0)

        tl.atomic_add(tokens_per_expert + experts, histogram, mask=valid_experts)


def launch_replica_route_ranking(
    flat_topk_indices: torch.Tensor,
    route_metadata: torch.Tensor,
    partition_counts: torch.Tensor,
    grid_sync: torch.Tensor,
    *,
    num_experts: int,
    num_routes: int,
) -> None:
    """Launch one-kernel stable per-expert route ranking."""
    _require_triton("MoonEP route ranking")
    _require_cuda_contiguous("flat_topk_indices", flat_topk_indices)
    _require_cuda_contiguous("route_metadata", route_metadata)
    _require_cuda_contiguous("partition_counts", partition_counts)
    _require_cuda_contiguous("grid_sync", grid_sync)
    if flat_topk_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"flat_topk_indices must be int32 or int64, got {flat_topk_indices.dtype}.")
    if route_metadata.dtype != torch.int32 or partition_counts.dtype != torch.int32:
        raise ValueError("route_metadata and partition_counts must be int32 tensors.")
    if grid_sync.dtype != torch.int32:
        raise ValueError("grid_sync must be an int32 tensor.")

    num_programs = planner_route_partition_count(num_routes)
    _rank_routes_within_experts_kernel[(num_programs,)](
        flat_topk_indices,
        route_metadata,
        partition_counts,
        grid_sync,
        NUM_ROUTES=num_routes,
        NUM_EXPERTS=num_experts,
        BLOCK_NUM_EXPERTS=triton.next_power_of_2(num_experts),
        BLOCK_NUM_ROUTES=64,
        BLOCK_SCAN_PARTITIONS=triton.next_power_of_2(num_programs),
        NUM_SCAN_EXPERTS=triton.cdiv(num_experts, num_programs),
        launch_cooperative_grid=True,
        num_warps=2,
    )


def launch_replica_placement(
    gathered_counts: torch.Tensor,
    balance: torch.Tensor,
    allocation: torch.Tensor,
    destination_boundaries: torch.Tensor,
    experts_to_copy: torch.Tensor,
    expert_replica_slots: torch.Tensor,
    grid_sync: torch.Tensor,
    *,
    rank_route_capacity: int,
    source_rank: int,
    ep_size: int,
    num_experts: int,
    num_local_experts: int,
) -> None:
    """Launch deterministic single-kernel replica placement."""
    _require_triton("MoonEP replica placement")
    if ep_size > MAX_REPLICA_EP_RANKS:
        raise ValueError(
            f"MoonEP replica planner supports at most {MAX_REPLICA_EP_RANKS} EP ranks, "
            f"got {ep_size}."
        )
    for name, tensor in (
        ("gathered_counts", gathered_counts),
        ("balance", balance),
        ("allocation", allocation),
        ("destination_boundaries", destination_boundaries),
        ("experts_to_copy", experts_to_copy),
        ("expert_replica_slots", expert_replica_slots),
        ("grid_sync", grid_sync),
    ):
        _require_cuda_contiguous(name, tensor)
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must be an int32 tensor, got {tensor.dtype}.")

    _plan_replica_placement_kernel[(ep_size,)](
        gathered_counts,
        balance,
        allocation,
        destination_boundaries,
        experts_to_copy,
        expert_replica_slots,
        grid_sync,
        source_rank,
        RANK_ROUTE_CAPACITY=rank_route_capacity,
        EP_SIZE=ep_size,
        NUM_EXPERTS=num_experts,
        NUM_EXPERTS_PER_GPU=num_local_experts,
        BLOCK_EP_SIZE=triton.next_power_of_2(ep_size),
        BLOCK_NUM_EXPERTS_PER_GPU=triton.next_power_of_2(num_local_experts),
        BLOCK_NUM_EXPERTS=triton.next_power_of_2(num_experts),
        launch_cooperative_grid=True,
        num_warps=1,
    )


def launch_replica_route_mapping(
    route_metadata: torch.Tensor,
    partition_counts: torch.Tensor,
    destination_boundaries: torch.Tensor,
    expert_replica_slots: torch.Tensor,
    virtual_experts: torch.Tensor,
    *,
    ep_size: int,
    num_experts: int,
    num_local_experts: int,
    num_routes: int,
) -> None:
    """Map ranked routes to native-or-replica ids."""
    _require_triton("MoonEP route mapping")
    for name, tensor in (
        ("route_metadata", route_metadata),
        ("partition_counts", partition_counts),
        ("destination_boundaries", destination_boundaries),
        ("expert_replica_slots", expert_replica_slots),
        ("virtual_experts", virtual_experts),
    ):
        _require_cuda_contiguous(name, tensor)
    if virtual_experts.dtype != torch.int64:
        raise ValueError(f"virtual_experts must be int64, got {virtual_experts.dtype}.")

    block_ep_size = triton.next_power_of_2(ep_size)
    _map_virtual_experts_kernel[(planner_route_partition_count(num_routes),)](
        route_metadata,
        partition_counts,
        destination_boundaries,
        expert_replica_slots,
        virtual_experts,
        NUM_ROUTES=num_routes,
        NUM_EXPERTS=num_experts,
        NUM_EXPERTS_PER_GPU=num_local_experts,
        EP_SIZE=ep_size,
        BLOCK_NUM_EXPERTS=triton.next_power_of_2(num_experts),
        BLOCK_NUM_ROUTES=256,
        BLOCK_EP_SIZE=block_ep_size,
        LOG2_BLOCK_EP_SIZE=block_ep_size.bit_length() - 1,
        num_warps=8,
    )


def launch_compact_routing_map(
    routing_map: torch.Tensor,
    token_indices: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    *,
    num_tokens: int,
    router_topk: int,
    num_experts: int,
) -> None:
    """Compact selected semantic experts and accumulate their histogram."""
    _require_triton("MoonEP routing-map compaction")
    _require_cuda_contiguous("routing_map", routing_map)
    _require_cuda_contiguous("token_indices", token_indices)
    _require_cuda_contiguous("tokens_per_expert", tokens_per_expert)
    if routing_map.dtype != torch.bool:
        raise ValueError(f"routing_map must be bool, got {routing_map.dtype}.")
    if token_indices.dtype != torch.int32:
        raise ValueError(f"token_indices must be int32, got {token_indices.dtype}.")
    if tokens_per_expert.dtype != torch.int32:
        raise ValueError(f"tokens_per_expert must be int32, got {tokens_per_expert.dtype}.")
    if num_tokens == 0:
        tokens_per_expert.zero_()
        return

    block_num_experts = triton.next_power_of_2(num_experts)
    block_tokens = min(32, max(1, 16384 // block_num_experts))
    num_programs = min(_MAX_PLANNER_PROGRAMS, triton.cdiv(num_tokens, block_tokens))
    _compact_routing_map_kernel[(num_programs,)](
        routing_map,
        token_indices,
        tokens_per_expert,
        num_tokens,
        ROUTER_TOPK=router_topk,
        NUM_EXPERTS=num_experts,
        BLOCK_TOKENS=block_tokens,
        BLOCK_NUM_EXPERTS=block_num_experts,
        num_warps=8,
    )
