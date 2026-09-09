# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fused MoonEP virtual-expert planner from Megatron-LM PR #6892.

The cooperative kernel histograms compact router routes, exchanges the
histograms through an NCCL symmetric-memory window, computes deterministic
placement, and writes HybridEP's dense runtime routes in one launch.
"""

from __future__ import annotations

import functools

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
PLANNER_PROGRAMS = 128


def _require_triton() -> None:
    if not HAVE_TRITON:
        raise RuntimeError("MoonEP virtual-expert planning requires Triton.")


if HAVE_TRITON:
    _GRID_SYNC_TAG = tl.constexpr(0x40000000)
    _SIGNAL_STRIDE = tl.constexpr(4)
    _BARRIER_TIMEOUT_NS = tl.constexpr(100_000_000_000)
    _PLANNER_PROGRAMS = tl.constexpr(PLANNER_PROGRAMS)
    _FLAG_STRIDE = tl.constexpr(32)

    def _emit_on_every_thread(ASM: tl.constexpr, THREADS: tl.constexpr):
        """Run one proxy fence on every thread of the block; Triton has no primitive for them."""
        tl.inline_asm_elementwise(
            ASM, "=r,r", [tl.zeros([THREADS], tl.int32)], dtype=tl.int32, is_pure=False, pack=1
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


    @triton.jit
    def _argmax_lowest(values, ids, valid, SENTINEL: tl.constexpr):
        """``(max, id)`` over the valid entries; ties take the lowest id (``SENTINEL`` if none)."""
        best = tl.max(tl.where(valid, values, -2147483648), axis=0)
        return best, tl.min(tl.where(valid & (values == best), ids, SENTINEL), axis=0)


    @triton.jit
    def _argmax_highest(values, ids, valid):
        """``(max, id)`` over the valid entries; ties take the highest id (``-1`` if none)."""
        best = tl.max(tl.where(valid, values, -1), axis=0)
        return best, tl.max(tl.where(valid & (values == best), ids, -1), axis=0)


    @triton.jit
    def _argmin_lowest(values, ids, valid, SENTINEL: tl.constexpr):
        """``(min, id)`` over the valid entries; ties take the lowest id (``SENTINEL`` if none)."""
        best = tl.min(tl.where(valid, values, 2147483647), axis=0)
        return best, tl.min(tl.where(valid & (values == best), ids, SENTINEL), axis=0)


    @triton.jit
    def _planner_fields(scratch, EP_SIZE: tl.constexpr, NUM_EXPERTS: tl.constexpr):
        """Split the planner's int32 scratch arena into its fields (mirrors ``_scratch_layout`` on
        the host). Three flag words lead, each on its own 128-byte line so the programs spinning on
        the grid barrier do not contend with the placing programs' barrier and data."""
        BLOCK_EP_SIZE: tl.constexpr = 1 << (EP_SIZE - 1).bit_length()
        balance = scratch + 3 * _FLAG_STRIDE
        allocation = balance + EP_SIZE
        boundaries = allocation + NUM_EXPERTS * EP_SIZE
        slots = boundaries + NUM_EXPERTS * BLOCK_EP_SIZE
        histogram = slots + NUM_EXPERTS * EP_SIZE
        running = histogram + _PLANNER_PROGRAMS * NUM_EXPERTS
        totals = running + _PLANNER_PROGRAMS * NUM_EXPERTS
        return balance, allocation, boundaries, slots, histogram, running, totals


    @triton.jit
    def _place_virtual_experts(
        scratch,
        window,
        experts_to_copy,
        source_rank,
        num_routes,
        peer_bases,
        signal_bases,
        EP_SIZE: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        EXCHANGE: tl.constexpr,
    ):
        """Deterministic virtual-expert placement, run by the planner's first ``EP_SIZE`` programs.

        Program ``rank`` sums rank ``rank``'s native-expert columns of the histogram rows into this
        rank's histogram, publishes it into peer ``rank``'s symmetric window (row ``source_rank``)
        and waits for that peer's row, so ``window`` completes without a collective. The flags carry
        a per-launch sequence (every rank plans every layer; HybridEP's dispatch orders one layer's
        exchange after every peer read the previous one); a missing peer trips the device assert
        after the transport timeout. ``EXCHANGE=False`` is the process-local test seam.

        Then each program replays the quota greedy for its rank, assigns quotas across its experts
        and fills its virtual-expert slots, all in registers. Rank ties take the lowest rank,
        expert ties the lowest expert, slot ties the highest expert.
        """
        NUM_EXPERTS_PER_GPU: tl.constexpr = NUM_EXPERTS // EP_SIZE
        BLOCK_EP_SIZE: tl.constexpr = 1 << (EP_SIZE - 1).bit_length()
        BLOCK_NUM_EXPERTS_PER_GPU: tl.constexpr = 1 << (NUM_EXPERTS_PER_GPU - 1).bit_length()
        BLOCK_NUM_EXPERTS: tl.constexpr = 1 << (NUM_EXPERTS - 1).bit_length()
        if 8192 // BLOCK_NUM_EXPERTS > 16:
            HISTOGRAM_TILE: tl.constexpr = 16
        else:
            HISTOGRAM_TILE: tl.constexpr = 8192 // BLOCK_NUM_EXPERTS
        placement_sync = scratch
        sequence = scratch + 2 * _FLAG_STRIDE
        balance, allocation, boundaries, slots, histogram_rows, _, totals = _planner_fields(
            scratch, EP_SIZE, NUM_EXPERTS
        )
        rank = tl.program_id(0)
        ranks = tl.arange(0, BLOCK_EP_SIZE)
        valid_ranks = ranks < EP_SIZE
        local_experts = tl.arange(0, BLOCK_NUM_EXPERTS_PER_GPU)
        valid_local_experts = local_experts < NUM_EXPERTS_PER_GPU
        native_experts = rank * NUM_EXPERTS_PER_GPU + local_experts

        rows_tile = tl.arange(0, HISTOGRAM_TILE)
        local_totals = tl.zeros((BLOCK_NUM_EXPERTS_PER_GPU,), dtype=tl.int32)
        for row_start in tl.range(0, _PLANNER_PROGRAMS, HISTOGRAM_TILE):
            rows = row_start + rows_tile
            local_totals += tl.sum(
                tl.load(
                    histogram_rows + rows[:, None] * NUM_EXPERTS + native_experts[None, :],
                    mask=(rows[:, None] < _PLANNER_PROGRAMS) & valid_local_experts[None, :],
                    other=0,
                ),
                axis=0,
            )
        tl.store(totals + native_experts, local_totals, mask=valid_local_experts)
        _grid_sync(placement_sync, _GRID_SYNC_TAG, EP_SIZE)
        if EXCHANGE:
            sequence_number = tl.load(sequence) + 1
            experts = tl.arange(0, BLOCK_NUM_EXPERTS)
            valid_experts = experts < NUM_EXPERTS
            histogram = tl.load(totals + experts, mask=valid_experts, other=0)
            peer_window = tl.load(peer_bases.to(tl.pointer_type(tl.int64)) + rank)
            tl.store(
                peer_window.to(tl.pointer_type(tl.int32)) + source_rank * NUM_EXPERTS + experts,
                histogram,
                mask=valid_experts,
            )
            # Publish the stores through the alias proxy before the system-scope release.
            _emit_on_every_thread("fence.proxy.alias; mov.u32 $0, 0;", THREADS=32)
            signals = signal_bases.to(tl.pointer_type(tl.int64))
            peer_flag = (
                tl.load(signals + rank).to(tl.pointer_type(tl.int32)) + source_rank * _SIGNAL_STRIDE
            )
            tl.atomic_xchg(peer_flag, sequence_number, sem="release", scope="sys")
            own_flag = (
                tl.load(signals + source_rank).to(tl.pointer_type(tl.int32)) + rank * _SIGNAL_STRIDE
            )
            # Spin on the timer only; the assert's call site stays out of the loop, where it would
            # cost the placement a good part of its runtime (the kernel is debug=True).
            start = tl.extra.cuda.globaltimer()
            arrived = tl.atomic_add(own_flag, 0, sem="acquire", scope="sys") >= sequence_number
            while not arrived and tl.extra.cuda.globaltimer() - start < _BARRIER_TIMEOUT_NS:
                arrived = tl.atomic_add(own_flag, 0, sem="acquire", scope="sys") >= sequence_number
            tl.device_assert(arrived, "virtual-expert planner: histogram exchange stalled")
            # Every program has acquired its peer's row; the barrier hands them all to everyone.
            _grid_sync(placement_sync, _GRID_SYNC_TAG, EP_SIZE)
            if rank == 0:
                tl.store(sequence, sequence_number)

        source_counts = tl.load(
            window + ranks[:, None] * NUM_EXPERTS + native_experts[None, :],
            mask=valid_ranks[:, None] & valid_local_experts[None, :],
            other=0,
        )
        native_totals = tl.sum(source_counts, axis=0).to(tl.int32)
        routes_before_source = tl.sum(
            tl.where(ranks[:, None] < source_rank, source_counts, 0), axis=0
        ).to(tl.int32)
        # Every rank must contribute exactly num_routes routes: the
        # capacity math and the compaction's slot count both assume router_topk
        # selections per token. This kernel launches eagerly, so the assert is compiled in.
        source_total = tl.sum(
            tl.load(
                window + rank * NUM_EXPERTS + tl.arange(0, BLOCK_NUM_EXPERTS),
                mask=tl.arange(0, BLOCK_NUM_EXPERTS) < NUM_EXPERTS,
                other=0,
            ),
            axis=0,
        )
        tl.device_assert(
            source_total == num_routes,
            "virtual-expert planner: a rank's route count differs from tokens * topk",
        )
        tl.store(balance + rank, tl.sum(native_totals, axis=0).to(tl.int32) - num_routes)

        _grid_sync(placement_sync, _GRID_SYNC_TAG, EP_SIZE)

        # Pair the most overloaded rank with the emptiest one and move the
        # receiver's whole deficit from that single sender. This can send more than
        # the sender's excess, but it gives every receiver exactly one sender, and
        # a sender owns NUM_EXPERTS_PER_GPU experts, so a receiver never needs more
        # virtual-expert slots than it has. Moving only min(excess, deficit) would cut
        # traffic but let a receiver draw on several senders and overflow the slots.
        balances = tl.load(balance + ranks, mask=valid_ranks, other=0)
        quotas = tl.zeros((BLOCK_EP_SIZE,), dtype=tl.int32)
        for _ in tl.range(0, EP_SIZE, 1, loop_unroll_factor=1):
            maximum, overloaded = _argmax_lowest(balances, ranks, valid_ranks, BLOCK_EP_SIZE)
            minimum, receiver = _argmin_lowest(balances, ranks, valid_ranks, BLOCK_EP_SIZE)
            active = maximum > 0
            moved = tl.where(active, -minimum, 0).to(tl.int32)
            quotas = tl.where(active & (overloaded == rank) & (ranks == receiver), moved, quotas)
            balances = tl.where(active & (ranks == overloaded), balances - moved, balances)
            balances = tl.where(active & (ranks == receiver), 0, balances)
        remaining = native_totals
        allocations = tl.where(ranks[None, :] == rank, native_totals[:, None], 0)
        for _ in tl.range(0, EP_SIZE + NUM_EXPERTS_PER_GPU, 1, loop_unroll_factor=1):
            max_quota, destination = _argmax_lowest(quotas, ranks, valid_ranks, BLOCK_EP_SIZE)
            max_remaining, local_expert = _argmax_lowest(
                remaining, local_experts, valid_local_experts, BLOCK_NUM_EXPERTS_PER_GPU
            )
            active = max_quota > 0
            moved = tl.where(active, tl.minimum(max_quota, max_remaining), 0).to(tl.int32)
            transfer = tl.where(
                ranks[None, :] == destination, moved, tl.where(ranks[None, :] == rank, -moved, 0)
            )
            allocations += tl.where((local_experts[:, None] == local_expert) & active, transfer, 0)
            remaining = tl.where(active & (local_experts == local_expert), remaining - moved, remaining)
            quotas = tl.where(active & (ranks == destination), quotas - moved, quotas)
        tl.store(
            allocation + native_experts[:, None] * EP_SIZE + ranks[None, :],
            allocations,
            mask=valid_local_experts[:, None] & valid_ranks[None, :],
        )
        tl.store(
            boundaries + native_experts[:, None] * BLOCK_EP_SIZE + ranks[None, :],
            tl.cumsum(allocations, axis=1) - routes_before_source[:, None],
            mask=valid_local_experts[:, None],
        )

        _grid_sync(placement_sync, _GRID_SYNC_TAG, EP_SIZE)

        experts = tl.arange(0, BLOCK_NUM_EXPERTS)
        owner = experts // NUM_EXPERTS_PER_GPU
        valid_remote = (experts < NUM_EXPERTS) & (owner != rank)
        counts = tl.load(allocation + experts * EP_SIZE + rank, mask=valid_remote, other=-1)
        for slot in tl.range(0, NUM_EXPERTS_PER_GPU, 1, loop_unroll_factor=1):
            maximum, expert = _argmax_highest(counts, experts, valid_remote)
            selected = tl.where(maximum > 0, expert, -1).to(tl.int32)
            tl.store(experts_to_copy + rank * NUM_EXPERTS_PER_GPU + slot, selected)
            tl.store(slots + selected * EP_SIZE + rank, slot, mask=selected >= 0)
            counts = tl.where(experts == expert, -1, counts)
        # Allocations name the destination of every route, so an expert allocated
        # here without a slot would map its routes to a stale slot id. The
        # single-sender rule above makes this unreachable; keep it loud anyway.
        tl.device_assert(
            tl.max(tl.where(valid_remote, counts, -1), axis=0) <= 0,
            "virtual-expert placement needs more virtual-expert slots than experts",
        )


    # ``debug=True`` keeps the device asserts alive: the planner launches eagerly, so a failed
    # route-count, exchange or slot check traps the run instead of misrouting tokens.
    @triton.jit(debug=True, do_not_specialize=["source_rank", "num_tokens"])
    def _plan_virtual_expert_routes_kernel(
        top_indices,
        probs,
        virtual_experts,
        runtime_probs,
        experts_to_copy,
        scratch,
        window,
        source_rank,
        num_tokens,
        peer_bases,
        signal_bases,
        ROUTER_TOPK: tl.constexpr,
        EP_SIZE: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        EXCHANGE: tl.constexpr,
    ):
        """Plan one layer's virtual-expert routes in one cooperative launch.

        Phase 1: every program histograms its token range of the router's ``[num_tokens, topk]``
        ids into its row. Phase 2: the first ``EP_SIZE`` programs run :func:`_place_virtual_experts`
        while the rest wait at the grid barrier. Phase 3: every program maps its routes: the stable
        ordinal among this rank's routes to the same expert (earlier rows + running count + rank in
        the tile) against the placement's segment ends picks the destination, remote destinations
        take the slot the placement assigned, and the pass writes the int16 runtime ids and the dense
        ``[num_tokens, 2 * num_experts]`` runtime probabilities HybridEP consumes.
        """
        NUM_EXPERTS_PER_GPU: tl.constexpr = NUM_EXPERTS // EP_SIZE
        NUM_RUNTIME_EXPERTS: tl.constexpr = 2 * NUM_EXPERTS
        BLOCK_EP_SIZE: tl.constexpr = 1 << (EP_SIZE - 1).bit_length()
        BLOCK_NUM_EXPERTS: tl.constexpr = 1 << (NUM_EXPERTS - 1).bit_length()
        BLOCK_TOPK: tl.constexpr = 1 << (ROUTER_TOPK - 1).bit_length()
        BLOCK_TOKENS: tl.constexpr = 128 // BLOCK_TOPK
        if 2 * BLOCK_NUM_EXPERTS > 256:
            BLOCK_RUNTIME_EXPERTS: tl.constexpr = 256
        else:
            BLOCK_RUNTIME_EXPERTS: tl.constexpr = 2 * BLOCK_NUM_EXPERTS
        if 8192 // BLOCK_NUM_EXPERTS > 16:
            HISTOGRAM_TILE: tl.constexpr = 16
        else:
            HISTOGRAM_TILE: tl.constexpr = 8192 // BLOCK_NUM_EXPERTS
        grid_sync = scratch + _FLAG_STRIDE
        _, _, boundaries, slots, histogram_rows, running_counts, totals = _planner_fields(
            scratch, EP_SIZE, NUM_EXPERTS
        )
        program = tl.program_id(0)
        experts = tl.arange(0, BLOCK_NUM_EXPERTS)
        valid_experts = experts < NUM_EXPERTS
        tokens_per_program = tl.cdiv(num_tokens, _PLANNER_PROGRAMS)
        program_start = program * tokens_per_program
        program_end = tl.minimum(program_start + tokens_per_program, num_tokens)
        # One tile holds BLOCK_TOKENS tokens' routes in token-major order, flat index
        # local_token * BLOCK_TOPK + k, so ordering by flat index is the routes' order.
        flat = tl.arange(0, BLOCK_TOKENS * BLOCK_TOPK)
        tile_tokens = flat // BLOCK_TOPK
        tile_slots = flat % BLOCK_TOPK

        # Phase 1: this program's histogram row.
        row = tl.zeros((BLOCK_NUM_EXPERTS,), dtype=tl.int32)
        for token_start in tl.range(program_start, program_end, BLOCK_TOKENS, loop_unroll_factor=1):
            tokens = token_start + tile_tokens
            valid = (tokens < program_end) & (tile_slots < ROUTER_TOPK)
            ids = tl.load(top_indices + tokens * ROUTER_TOPK + tile_slots, mask=valid, other=0)
            row += tl.histogram(ids.to(tl.int32), BLOCK_NUM_EXPERTS, mask=valid)
        tl.store(histogram_rows + program * NUM_EXPERTS + experts, row, mask=valid_experts)
        _grid_sync(grid_sync, _GRID_SYNC_TAG, _PLANNER_PROGRAMS)

        # Phase 2: histogram exchange and placement, one program per EP rank.
        if program < EP_SIZE:
            _place_virtual_experts(
                scratch,
                window,
                experts_to_copy,
                source_rank,
                num_tokens * ROUTER_TOPK,
                peer_bases,
                signal_bases,
                EP_SIZE=EP_SIZE,
                NUM_EXPERTS=NUM_EXPERTS,
                EXCHANGE=EXCHANGE,
            )
        _grid_sync(grid_sync, _GRID_SYNC_TAG, _PLANNER_PROGRAMS)

        # Phase 3: map this program's routes. Routes of the same expert issued by earlier programs
        # come first in the ordinal space.
        running = tl.zeros((BLOCK_NUM_EXPERTS,), dtype=tl.int32)
        rows_tile = tl.arange(0, HISTOGRAM_TILE)
        for row_start in tl.range(0, program, HISTOGRAM_TILE):
            rows = row_start + rows_tile
            running += tl.sum(
                tl.load(
                    histogram_rows + rows[:, None] * NUM_EXPERTS + experts[None, :],
                    mask=(rows[:, None] < program) & valid_experts[None, :],
                    other=0,
                ),
                axis=0,
            )
        ranks = tl.arange(0, BLOCK_EP_SIZE)
        valid_ranks = ranks < EP_SIZE
        tile_rows = tl.arange(0, BLOCK_TOKENS)
        runtime_columns = tl.arange(0, BLOCK_RUNTIME_EXPERTS)
        for token_start in tl.range(program_start, program_end, BLOCK_TOKENS, loop_unroll_factor=1):
            # The running counts go through memory so every route can gather its expert's count.
            tl.store(running_counts + program * NUM_EXPERTS + experts, running, mask=valid_experts)
            tl.debug_barrier()
            tokens = token_start + tile_tokens
            valid = (tokens < program_end) & (tile_slots < ROUTER_TOPK)
            route_offsets = tokens * ROUTER_TOPK + tile_slots
            ids = tl.load(top_indices + route_offsets, mask=valid, other=0).to(tl.int32)
            earlier = tl.sum(
                ((ids[None, :] == ids[:, None]) & (flat[None, :] < flat[:, None]) & valid[None, :]).to(
                    tl.int32
                ),
                axis=1,
            )
            ordinal = tl.load(running_counts + program * NUM_EXPERTS + ids, mask=valid, other=0)
            ordinal += earlier
            # Destination = number of segment ends at or below the ordinal. Segment ends are the
            # placement's cumulative allocations in this rank's ordinal space, clipped to the
            # routes this rank actually holds.
            local_routes = tl.load(totals + ids, mask=valid, other=0)
            segment_ends = tl.load(
                boundaries + ids[:, None] * BLOCK_EP_SIZE + ranks[None, :],
                mask=valid[:, None] & valid_ranks[None, :],
                other=0,
            )
            segment_ends = tl.minimum(tl.maximum(segment_ends, 0), local_routes[:, None])
            destination = tl.sum(
                ((segment_ends <= ordinal[:, None]) & valid_ranks[None, :]).to(tl.int32), axis=1
            )
            remote = valid & (destination != ids // NUM_EXPERTS_PER_GPU)
            slot = tl.load(slots + ids * EP_SIZE + destination, mask=remote, other=0)
            runtime = destination * (2 * NUM_EXPERTS_PER_GPU) + tl.where(
                remote, NUM_EXPERTS_PER_GPU + slot, ids % NUM_EXPERTS_PER_GPU
            )
            tl.store(virtual_experts + route_offsets, runtime.to(tl.int16), mask=valid)
            # Dense runtime probabilities: clear the tile's rows, then scatter the routes into them.
            rows = token_start + tile_rows
            valid_rows = rows < program_end
            for column_start in tl.range(0, NUM_RUNTIME_EXPERTS, BLOCK_RUNTIME_EXPERTS):
                columns = column_start + runtime_columns
                tl.store(
                    runtime_probs + rows[:, None] * NUM_RUNTIME_EXPERTS + columns[None, :],
                    tl.zeros((BLOCK_TOKENS, BLOCK_RUNTIME_EXPERTS), dtype=tl.float32),
                    mask=valid_rows[:, None] & (columns[None, :] < NUM_RUNTIME_EXPERTS),
                )
            tl.debug_barrier()
            prob = tl.load(probs + route_offsets, mask=valid, other=0.0)
            tl.store(
                runtime_probs + tokens * NUM_RUNTIME_EXPERTS + runtime, prob.to(tl.float32), mask=valid
            )
            running += tl.histogram(ids, BLOCK_NUM_EXPERTS, mask=valid)



def launch_virtual_expert_planner(
    top_indices: torch.Tensor, probs: torch.Tensor, workspace, *, exchange: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plan one layer's routes in one cooperative launch.

    ``top_indices`` / ``probs`` are the router's ``[num_tokens, topk]`` expert ids and
    probabilities, ``workspace`` the planner scratch (``VirtualExpertPlannerWorkspace``) whose
    ``gathered_counts`` is this rank's symmetric window. Returns the int16 ``[num_tokens, topk]``
    runtime ids, the float32 ``[num_tokens, 2 * num_experts]`` runtime probabilities and the
    int32 ``[ep_size, num_local_experts]`` slot table. Without ``exchange`` (process-local
    tests) the window must already hold every rank's histogram.
    """
    _require_triton()
    num_tokens, router_topk = top_indices.shape
    ep_size, num_experts = workspace.ep_size, workspace.num_experts
    if ep_size > PLANNER_PROGRAMS or num_experts > 8192:
        raise ValueError(
            f"Virtual-expert planner supports at most {PLANNER_PROGRAMS} EP ranks and 8192 experts."
        )
    empty = functools.partial(torch.empty, device=top_indices.device)
    virtual_experts = empty((num_tokens, router_topk), dtype=torch.int16)
    runtime_probs = empty((num_tokens, 2 * num_experts), dtype=torch.float32)
    experts_to_copy = empty((ep_size, num_experts // ep_size), dtype=torch.int32)
    handle = workspace.histogram_handle
    _plan_virtual_expert_routes_kernel[(PLANNER_PROGRAMS,)](
        top_indices,
        probs,
        virtual_experts,
        runtime_probs,
        experts_to_copy,
        workspace.scratch,
        workspace.gathered_counts,
        workspace.rank,
        num_tokens,
        int(handle.buffer_ptrs_dev) if exchange else 0,
        int(handle.signal_pad_ptrs_dev) if exchange else 0,
        ROUTER_TOPK=router_topk,
        EP_SIZE=ep_size,
        NUM_EXPERTS=num_experts,
        EXCHANGE=exchange,
        launch_cooperative_grid=True,
        num_warps=4,
    )
    return virtual_experts, runtime_probs, experts_to_copy
