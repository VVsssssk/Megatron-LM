# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pure Python MoonEP load planner for MoEScheduler.

This module mirrors MoonEP's planning algorithm at the interface boundary.  It
does not depend on MoonEP's CuTe/CUDA package; instead it emits the common
``MoEPlannerOutput`` with a physical-to-logical expert layout and dense token
reroute tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from megatron.core import tensor_parallel
from megatron.core.transformer.moe.moe_scheduler import (
    MoELoadPlanner,
    MoEPlannerOutput,
    SchedulerContext,
)


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class PythonMoonEPCommPlan:
    """Python equivalent of the MoonEP communication plan fields.

    ``dst`` is flattened ``[num_tokens * topk]`` and uses MoonEP's duplicate
    destination-rank encoding: the first entry for a destination rank stays
    non-negative; later entries for the same token/rank are encoded as
    ``-raw_dst - 1``.
    """

    dst: torch.Tensor
    cu_seqlens: torch.Tensor
    experts_to_copy: torch.Tensor
    zero_fill_ranges: torch.Tensor
    remote_stats: torch.Tensor
    dup_groups: torch.Tensor
    dup_loffs: torch.Tensor
    dup_counts: torch.Tensor
    N: int
    R: int
    E: int
    B: int
    NvS: int
    K: int


def _dense_to_topk(
    routing_map: torch.Tensor, probs: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_ids = torch.empty(routing_map.size(0), topk, dtype=torch.long, device=routing_map.device)
    topk_probs = torch.empty(routing_map.size(0), topk, dtype=probs.dtype, device=probs.device)
    for token_idx in range(routing_map.size(0)):
        expert_ids = torch.nonzero(routing_map[token_idx], as_tuple=False).flatten()
        if expert_ids.numel() != topk:
            raise ValueError(
                f"Expected every token to route to {topk} experts, "
                f"got {expert_ids.numel()} for token {token_idx}."
            )
        topk_ids[token_idx] = expert_ids
        topk_probs[token_idx] = probs[token_idx, expert_ids]
    return topk_ids, topk_probs


def _global_home_physical_ids(
    logical_expert_ids: torch.Tensor, experts_per_rank: int, redundant_experts_per_rank: int
) -> torch.Tensor:
    local_physical_experts = experts_per_rank + redundant_experts_per_rank
    rank_ids = torch.div(logical_expert_ids, experts_per_rank, rounding_mode="floor")
    local_ids = logical_expert_ids.remainder(experts_per_rank)
    return rank_ids * local_physical_experts + local_ids


def _global_idle_physical_id(
    dest_rank: int, redundant_slot: int, experts_per_rank: int, redundant_experts_per_rank: int
) -> int:
    local_physical_experts = experts_per_rank + redundant_experts_per_rank
    return dest_rank * local_physical_experts + experts_per_rank + redundant_slot


class MoonEPLoadPlanner(MoELoadPlanner):
    """MoonEP-style L2 planner implemented in Python/Torch.

    The planner balances every destination rank to ``num_tokens * topk`` routed
    entries, chooses up to ``B`` remote experts per rank for MoonEP VM prefetch
    slots, and returns a dense physical routing map that can feed Megatron's
    existing token dispatchers.  MoonEP-native ``dst`` offsets are retained in
    metadata for debugging and future adapters.
    """

    planner_name = "moon_ep"

    def __init__(
        self,
        num_redundant_experts: Optional[int] = None,
        *,
        token_padding: int = 1,
    ) -> None:
        super().__init__()
        if num_redundant_experts is not None and num_redundant_experts < 0:
            raise ValueError("num_redundant_experts must be non-negative when provided.")
        if token_padding <= 0:
            raise ValueError("token_padding must be positive.")
        self.num_redundant_experts = num_redundant_experts
        self.token_padding = token_padding

    def _validate_context(self, context: SchedulerContext) -> None:
        if context.num_logical_experts % context.ep_size != 0:
            raise ValueError(
                "MoonEPLoadPlanner requires num_logical_experts divisible by ep_size."
            )

    def should_plan(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: Optional[torch.Tensor] = None,
    ) -> bool:
        del probs, routing_map, tokens_per_expert
        num_redundant_experts = self._resolve_num_redundant_experts(context)
        if num_redundant_experts == 0:
            return False
        self._validate_context(context)
        return True

    def _resolve_num_redundant_experts(self, context: SchedulerContext) -> int:
        if self.num_redundant_experts is not None:
            return self.num_redundant_experts
        return context.num_logical_experts // context.ep_size

    @staticmethod
    def _topk_from_dense(
        probs: torch.Tensor, routing_map: torch.Tensor, context: SchedulerContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _dense_to_topk(routing_map, probs, context.router_topk)

    @staticmethod
    def _count_topk_assignments(topk_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
        flat_topk_ids = topk_ids.reshape(-1)
        if flat_topk_ids.numel() == 0:
            return torch.zeros(num_experts, dtype=torch.int64, device=topk_ids.device)
        if bool((flat_topk_ids < 0).any()) or bool((flat_topk_ids >= num_experts).any()):
            raise ValueError("MoonEPLoadPlanner received topk_ids outside the expert range.")
        return torch.bincount(flat_topk_ids, minlength=num_experts).to(dtype=torch.int64)

    def _get_count_matrix(
        self,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        topk_ids: torch.Tensor,
        *,
        tokens_per_expert: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        local_counts = self._count_topk_assignments(topk_ids, context.num_logical_experts)
        if tokens_per_expert is not None:
            dense_counts = tokens_per_expert.to(
                dtype=torch.int64, device=local_counts.device
            )
            if dense_counts.numel() != context.num_logical_experts:
                raise ValueError(
                    "Expected local token counts to match the logical expert dimension, "
                    f"got {dense_counts.numel()} and {context.num_logical_experts}."
                )
            if not torch.equal(dense_counts, local_counts):
                raise ValueError(
                    "MoonEPLoadPlanner requires tokens_per_expert to match topk_ids counts."
                )

        if context.ep_size == 1:
            return local_counts.unsqueeze(0)

        ep_group = getattr(context.pg_collection, "ep", None)
        if ep_group is None:
            raise ValueError(
                "MoonEPLoadPlanner requires SchedulerContext.pg_collection.ep when ep_size > 1."
            )
        counts = tensor_parallel.gather_from_sequence_parallel_region(
            local_counts, group=ep_group
        ).reshape(context.ep_size, context.num_logical_experts)
        counts = counts.to(dtype=torch.int64)
        if not torch.equal(counts[context.ep_rank], local_counts):
            raise ValueError(
                "Expected gathered token counts for the current EP rank to match local topk_ids."
            )
        return counts

    def _build_allocation(
        self,
        counts_from_ep_rank: torch.Tensor,
        *,
        capacity_per_rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        counts = counts_from_ep_rank.to(device="cpu", dtype=torch.int64)
        num_ep_ranks, num_experts = counts.shape
        if num_experts % num_ep_ranks != 0:
            raise ValueError("MoonEPLoadPlanner requires num_experts to be divisible by ep_size.")
        if int(counts.sum().item()) != capacity_per_rank * num_ep_ranks:
            raise ValueError(
                "MoonEPLoadPlanner requires each planning window to have a fixed "
                "num_tokens * topk capacity per EP rank."
            )

        experts_per_rank = num_experts // num_ep_ranks
        expert_count = counts.sum(dim=0)
        group_tokens = torch.zeros(num_ep_ranks, dtype=torch.int64)
        for home_rank in range(num_ep_ranks):
            start = home_rank * experts_per_rank
            group_tokens[home_rank] = expert_count[start : start + experts_per_rank].sum()

        balance = group_tokens - capacity_per_rank
        initial_balance = balance.clone()
        alloc = torch.zeros(num_experts, num_ep_ranks, dtype=torch.int64)
        for expert_id in range(num_experts):
            alloc[expert_id, expert_id // experts_per_rank] = expert_count[expert_id]

        remote_quotas = torch.zeros(num_ep_ranks, num_ep_ranks, dtype=torch.int64)
        while True:
            overload_rank = int(torch.argmax(balance).item())
            underload_rank = int(torch.argmin(balance).item())
            if int(balance[overload_rank].item()) <= 0:
                break
            move = int(-balance[underload_rank].item())
            if move <= 0:
                break
            remote_quotas[overload_rank, underload_rank] += move
            balance[overload_rank] -= move
            balance[underload_rank] = 0

        for home_rank in range(num_ep_ranks):
            expert_start = home_rank * experts_per_rank
            remaining = expert_count[expert_start : expert_start + experts_per_rank].clone()
            quotas = remote_quotas[home_rank].clone()
            while True:
                dest_rank = int(torch.argmax(quotas).item())
                quota = int(quotas[dest_rank].item())
                if quota <= 0:
                    break
                local_expert = int(torch.argmax(remaining).item())
                expert_id = expert_start + local_expert
                take = min(int(remaining[local_expert].item()), quota)
                if take <= 0:
                    break
                alloc[expert_id, dest_rank] += take
                alloc[expert_id, home_rank] -= take
                remaining[local_expert] -= take
                quotas[dest_rank] -= take

        if not torch.equal(alloc.sum(dim=1), expert_count):
            raise AssertionError("MoonEPLoadPlanner failed per-expert token conservation.")
        if bool((alloc.sum(dim=0) > capacity_per_rank).any().item()):
            raise AssertionError("MoonEPLoadPlanner produced a rank allocation over capacity.")
        return alloc, group_tokens, initial_balance, remote_quotas

    @staticmethod
    def _cap_remote_experts_for_global_slots(
        alloc: torch.Tensor, num_redundant_experts: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep only copied remote experts that can be represented as global idle slots."""
        capped_alloc = alloc.clone()
        num_experts, num_ep_ranks = capped_alloc.shape
        experts_per_rank = num_experts // num_ep_ranks
        dropped_remote_tokens = torch.zeros(num_ep_ranks, dtype=torch.int64)

        for dest_rank in range(num_ep_ranks):
            local_start = dest_rank * experts_per_rank
            local_end = local_start + experts_per_rank
            remote_experts = [
                expert_id
                for expert_id in range(num_experts)
                if int(capped_alloc[expert_id, dest_rank].item()) > 0
                and not (local_start <= expert_id < local_end)
            ]
            remote_experts.sort(
                key=lambda expert_id: (int(capped_alloc[expert_id, dest_rank]), expert_id),
                reverse=True,
            )
            keep = set(remote_experts[:num_redundant_experts])
            for expert_id in remote_experts:
                if expert_id in keep:
                    continue
                moved_tokens = int(capped_alloc[expert_id, dest_rank].item())
                if moved_tokens == 0:
                    continue
                source_rank = expert_id // experts_per_rank
                capped_alloc[expert_id, dest_rank] = 0
                capped_alloc[expert_id, source_rank] += moved_tokens
                dropped_remote_tokens[dest_rank] += moved_tokens
        return capped_alloc, dropped_remote_tokens

    def _build_layout(
        self,
        alloc: torch.Tensor,
        *,
        num_redundant_experts: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        num_experts, num_ep_ranks = alloc.shape
        experts_per_rank = num_experts // num_ep_ranks
        num_physical_groups = num_experts + num_redundant_experts
        experts_to_copy = torch.full(
            (num_ep_ranks, num_redundant_experts), -1, dtype=torch.int32
        )
        remote_stats_all = torch.zeros(num_ep_ranks, 2, dtype=torch.int32)
        cu_seqlens_all = torch.zeros(num_ep_ranks, num_physical_groups, dtype=torch.int32)
        zero_fill_ranges_all = torch.zeros(
            num_ep_ranks, num_physical_groups, 2, dtype=torch.int32
        )
        expert_offsets = torch.zeros(num_ep_ranks, num_experts, dtype=torch.int32)
        expert_group_ids = torch.full((num_ep_ranks, num_experts), -1, dtype=torch.int32)

        for dest_rank in range(num_ep_ranks):
            local_start = dest_rank * experts_per_rank
            local_end = local_start + experts_per_rank
            remote_experts = [
                expert_id
                for expert_id in range(num_experts)
                if int(alloc[expert_id, dest_rank].item()) > 0
                and not (local_start <= expert_id < local_end)
            ]
            remote_experts.sort(
                key=lambda expert_id: (int(alloc[expert_id, dest_rank]), expert_id),
                reverse=True,
            )
            remote_stats_all[dest_rank, 0] = len(remote_experts)
            for redundant_slot, expert_id in enumerate(remote_experts[:num_redundant_experts]):
                experts_to_copy[dest_rank, redundant_slot] = expert_id
                remote_stats_all[expert_id // experts_per_rank, 1] += 1

            prefetched = set(remote_experts[:num_redundant_experts])
            start_offset = 0
            for group_id in range(num_physical_groups):
                count = 0
                expert_id = -1
                if group_id < num_experts:
                    if group_id not in prefetched:
                        expert_id = group_id
                        count = int(alloc[expert_id, dest_rank].item())
                else:
                    redundant_slot = group_id - num_experts
                    copied_expert = int(experts_to_copy[dest_rank, redundant_slot].item())
                    if copied_expert >= 0:
                        expert_id = copied_expert
                        count = int(alloc[expert_id, dest_rank].item())

                end_offset = start_offset + count
                padded = _align_up(count, self.token_padding) if count > 0 else 0
                aligned_end = start_offset + padded
                cu_seqlens_all[dest_rank, group_id] = aligned_end
                if count > 0:
                    expert_offsets[dest_rank, expert_id] = start_offset
                    expert_group_ids[dest_rank, expert_id] = group_id
                    pad_count = padded - count
                    if pad_count > 0:
                        zero_fill_ranges_all[dest_rank, group_id, 0] = end_offset
                        zero_fill_ranges_all[dest_rank, group_id, 1] = pad_count
                start_offset = aligned_end

        num_virtual_tokens_per_rank = max(1, int(cu_seqlens_all.max().item()))
        return (
            experts_to_copy,
            remote_stats_all,
            cu_seqlens_all,
            zero_fill_ranges_all,
            expert_offsets,
            expert_group_ids,
            num_virtual_tokens_per_rank,
        )

    @staticmethod
    def _build_raw_dst(
        topk_ids: torch.Tensor,
        counts_from_ep_rank: torch.Tensor,
        alloc: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_group_ids: torch.Tensor,
        *,
        rank: int,
        num_virtual_tokens_per_rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_topk_ids = topk_ids.reshape(-1).to(device="cpu", dtype=torch.long)
        counts = counts_from_ep_rank.to(device="cpu", dtype=torch.int64)
        alloc = alloc.to(device="cpu", dtype=torch.int64)
        alloc_cumsum = alloc.cumsum(dim=1)
        counts_cumsum = counts.cumsum(dim=0)
        num_experts, num_ep_ranks = alloc.shape

        dst = torch.zeros(flat_topk_ids.numel(), dtype=torch.int32)
        physical_group_ids = torch.zeros(flat_topk_ids.numel(), dtype=torch.long)
        local_counts_by_expert = torch.zeros(num_experts, dtype=torch.int64)
        for entry_idx, expert_id_tensor in enumerate(flat_topk_ids):
            expert_id = int(expert_id_tensor.item())
            local_count = int(local_counts_by_expert[expert_id].item())
            local_counts_by_expert[expert_id] += 1
            previous_source_count = 0 if rank == 0 else int(counts_cumsum[rank - 1, expert_id])
            global_expert_index = previous_source_count + local_count

            dest_rank = -1
            previous_alloc = 0
            for candidate_rank in range(num_ep_ranks):
                if int(alloc_cumsum[expert_id, candidate_rank].item()) > global_expert_index:
                    dest_rank = candidate_rank
                    if candidate_rank > 0:
                        previous_alloc = int(alloc_cumsum[expert_id, candidate_rank - 1].item())
                    break
            if dest_rank < 0:
                raise AssertionError("MoonEPLoadPlanner could not find a destination rank.")

            segment_pos = global_expert_index - previous_alloc
            base_offset = int(expert_offsets[dest_rank, expert_id].item())
            dst[entry_idx] = dest_rank * num_virtual_tokens_per_rank + base_offset + segment_pos
            group_id = int(expert_group_ids[dest_rank, expert_id].item())
            if group_id < 0:
                raise AssertionError("MoonEPLoadPlanner could not find an expert group id.")
            physical_group_ids[entry_idx] = group_id
        return dst, physical_group_ids

    @staticmethod
    def _canonicalize_duplicates(
        dst: torch.Tensor,
        *,
        num_tokens: int,
        topk: int,
        num_ep_ranks: int,
        rank: int,
        num_virtual_tokens_per_rank: int,
        group: Optional[Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        dst_for_encoding = dst.clone()
        gathered_rows: list[tuple[int, torch.Tensor]] = [(rank, dst.clone())]
        gathered_all_ranks = False
        if (
            num_ep_ranks > 1
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            gathered = [
                torch.zeros_like(dst_for_encoding, device=device) for _ in range(num_ep_ranks)
            ]
            torch.distributed.all_gather(
                gathered, dst_for_encoding.to(device=device), group=group
            )
            gathered_rows = [
                (src_rank, gathered[src_rank].to(device="cpu", dtype=torch.int32))
                for src_rank in range(num_ep_ranks)
            ]
            gathered_all_ranks = True

        dup_groups = torch.zeros(num_virtual_tokens_per_rank, 3, dtype=torch.int32)
        dup_loffs = torch.zeros(num_virtual_tokens_per_rank, dtype=torch.int32)
        dup_counts = torch.zeros(2, dtype=torch.int32)

        for src_rank, src_dst in gathered_rows:
            for token_idx in range(num_tokens):
                base_idx = token_idx * topk
                dst_vals = src_dst[base_idx : base_idx + topk]
                entries_by_dest: dict[int, list[tuple[int, int]]] = {}
                for topk_idx, encoded_dst in enumerate(dst_vals):
                    raw_dst = int(encoded_dst.item())
                    dest_rank = raw_dst // num_virtual_tokens_per_rank
                    local_offset = raw_dst % num_virtual_tokens_per_rank
                    entries_by_dest.setdefault(dest_rank, []).append((topk_idx, local_offset))

                for dest_rank, entries in entries_by_dest.items():
                    if len(entries) <= 1:
                        continue
                    if dest_rank == rank:
                        group_idx = int(dup_counts[0].item())
                        dup_start = int(dup_counts[1].item())
                        duplicate_offsets = [local_offset for _, local_offset in entries[1:]]
                        if group_idx >= dup_groups.size(0) or (
                            dup_start + len(duplicate_offsets) > dup_loffs.size(0)
                        ):
                            raise AssertionError("MoonEPLoadPlanner duplicate metadata overflow.")
                        dup_groups[group_idx, 0] = entries[0][1]
                        dup_groups[group_idx, 1] = dup_start
                        dup_groups[group_idx, 2] = len(duplicate_offsets)
                        dup_loffs[dup_start : dup_start + len(duplicate_offsets)] = (
                            dup_loffs.new_tensor(duplicate_offsets)
                        )
                        dup_counts[0] += 1
                        dup_counts[1] += len(duplicate_offsets)
                    if src_rank == rank:
                        for topk_idx, _ in entries[1:]:
                            local_index = base_idx + topk_idx
                            dst_for_encoding[local_index] = -dst_for_encoding[local_index] - 1

        return (
            dst_for_encoding.to(device=device),
            dup_groups.to(device=device),
            dup_loffs.to(device=device),
            dup_counts.to(device=device),
            gathered_all_ranks,
        )

    def _build_physical_to_logical_map(
        self,
        *,
        comm_plan: PythonMoonEPCommPlan,
        context: SchedulerContext,
        device: torch.device,
    ) -> torch.Tensor:
        return self._build_global_physical_to_logical_map(
            comm_plan=comm_plan,
            context=context,
            device=device,
        )

    def _build_global_physical_to_logical_map(
        self,
        *,
        comm_plan: PythonMoonEPCommPlan,
        context: SchedulerContext,
        device: torch.device,
    ) -> torch.Tensor:
        num_experts = context.num_logical_experts
        num_ep_ranks = context.ep_size
        experts_per_rank = num_experts // num_ep_ranks
        redundant_experts_per_rank = comm_plan.B
        local_physical_experts = experts_per_rank + redundant_experts_per_rank
        num_physical_experts = num_ep_ranks * local_physical_experts

        physical_to_logical = torch.full(
            (num_physical_experts,), -1, dtype=torch.long, device=device
        )

        logical_ids = torch.arange(num_experts, dtype=torch.long, device=device)
        home_physical_ids = _global_home_physical_ids(
            logical_ids, experts_per_rank, redundant_experts_per_rank
        )
        physical_to_logical[home_physical_ids] = logical_ids

        for dest_rank in range(num_ep_ranks):
            for redundant_slot in range(redundant_experts_per_rank):
                copied_expert = int(comm_plan.experts_to_copy[dest_rank, redundant_slot].item())
                if copied_expert < 0:
                    continue
                physical_id = _global_idle_physical_id(
                    dest_rank,
                    redundant_slot,
                    experts_per_rank,
                    redundant_experts_per_rank,
                )
                physical_to_logical[physical_id] = copied_expert

        return physical_to_logical

    @staticmethod
    def _build_token_reroute(
        *,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        topk_ids: torch.Tensor,
        topk_probs: torch.Tensor,
        dst: torch.Tensor,
        physical_group_ids: torch.Tensor,
        comm_plan: PythonMoonEPCommPlan,
        dedup_complete: bool,
    ) -> dict[str, Any]:
        del dedup_complete
        device = routing_map.device
        num_experts = context.num_logical_experts
        experts_per_rank = num_experts // context.ep_size
        redundant_experts_per_rank = comm_plan.B
        local_physical_experts = experts_per_rank + redundant_experts_per_rank
        num_physical_experts = context.ep_size * local_physical_experts

        flat_groups = physical_group_ids.reshape(-1).to(device="cpu", dtype=torch.long)
        flat_dst = dst.reshape(-1).to(device="cpu", dtype=torch.int64)
        raw_dst = torch.where(flat_dst < 0, -flat_dst - 1, flat_dst)
        flat_physical_ids = torch.empty(flat_groups.numel(), dtype=torch.long)

        for entry_idx, group_id_tensor in enumerate(flat_groups):
            group_id = int(group_id_tensor.item())
            dest_rank = int(raw_dst[entry_idx].item()) // comm_plan.NvS
            if group_id < num_experts:
                physical_id = int(
                    _global_home_physical_ids(
                        torch.tensor([group_id], dtype=torch.long),
                        experts_per_rank,
                        redundant_experts_per_rank,
                    ).item()
                )
            else:
                physical_id = _global_idle_physical_id(
                    dest_rank,
                    group_id - num_experts,
                    experts_per_rank,
                    redundant_experts_per_rank,
                )
            flat_physical_ids[entry_idx] = physical_id

        physical_ids = flat_physical_ids.reshape_as(topk_ids).to(device=device)
        rerouted_routing_map = torch.zeros(
            routing_map.size(0), num_physical_experts, dtype=torch.bool, device=device
        )
        rerouted_probs = torch.zeros(
            routing_map.size(0), num_physical_experts, dtype=probs.dtype, device=probs.device
        )
        for token_idx in range(routing_map.size(0)):
            for topk_idx in range(context.router_topk):
                physical_id = int(physical_ids[token_idx, topk_idx].item())
                if bool(rerouted_routing_map[token_idx, physical_id].item()):
                    raise ValueError(
                        "MoonEPLoadPlanner global mode produced duplicate physical experts "
                        f"for token {token_idx}."
                    )
                rerouted_routing_map[token_idx, physical_id] = True
                rerouted_probs[token_idx, physical_id] = topk_probs[token_idx, topk_idx]

        return {
            "routing_map": rerouted_routing_map,
            "probs": rerouted_probs,
        }

    def _build_output_from_counts(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        topk_ids: torch.Tensor,
        topk_probs: torch.Tensor,
        counts_from_ep_rank: torch.Tensor,
    ) -> MoEPlannerOutput:
        self._validate_context(context)
        num_redundant_experts = self._resolve_num_redundant_experts(context)
        if num_redundant_experts <= 0:
            raise ValueError("MoonEPLoadPlanner requires at least one redundant expert slot.")

        capacity_per_rank = routing_map.size(0) * context.router_topk
        alloc, _, _, _ = self._build_allocation(
            counts_from_ep_rank, capacity_per_rank=capacity_per_rank
        )
        alloc, _ = self._cap_remote_experts_for_global_slots(
            alloc, num_redundant_experts
        )
        (
            experts_to_copy,
            remote_stats_all,
            cu_seqlens_all,
            zero_fill_ranges_all,
            expert_offsets,
            expert_group_ids,
            num_virtual_tokens_per_rank,
        ) = self._build_layout(alloc, num_redundant_experts=num_redundant_experts)
        num_virtual_tokens_per_rank = max(num_virtual_tokens_per_rank, capacity_per_rank)
        raw_dst, physical_group_ids = self._build_raw_dst(
            topk_ids,
            counts_from_ep_rank,
            alloc,
            expert_offsets,
            expert_group_ids,
            rank=context.ep_rank,
            num_virtual_tokens_per_rank=num_virtual_tokens_per_rank,
        )
        ep_group = getattr(context.pg_collection, "ep", None)
        dst, dup_groups, dup_loffs, dup_counts, gathered_all_ranks = (
            self._canonicalize_duplicates(
                raw_dst,
                num_tokens=routing_map.size(0),
                topk=context.router_topk,
                num_ep_ranks=context.ep_size,
                rank=context.ep_rank,
                num_virtual_tokens_per_rank=num_virtual_tokens_per_rank,
                group=ep_group,
                device=routing_map.device,
            )
        )
        comm_plan = PythonMoonEPCommPlan(
            dst=dst.contiguous(),
            cu_seqlens=cu_seqlens_all[context.ep_rank].to(
                device=routing_map.device
            ).contiguous(),
            experts_to_copy=experts_to_copy.to(device=routing_map.device).contiguous(),
            zero_fill_ranges=zero_fill_ranges_all[context.ep_rank].to(
                device=routing_map.device
            ).contiguous(),
            remote_stats=remote_stats_all[context.ep_rank].to(
                device=routing_map.device
            ).contiguous(),
            dup_groups=dup_groups.contiguous(),
            dup_loffs=dup_loffs.contiguous(),
            dup_counts=dup_counts.contiguous(),
            N=routing_map.size(0) * context.router_topk,
            R=context.ep_size,
            E=context.num_logical_experts,
            B=num_redundant_experts,
            NvS=num_virtual_tokens_per_rank,
            K=context.router_topk,
        )
        physical_to_logical_map = self._build_physical_to_logical_map(
            comm_plan=comm_plan,
            context=context,
            device=routing_map.device,
        )
        token_reroute = self._build_token_reroute(
            probs=probs,
            routing_map=routing_map,
            context=context,
            topk_ids=topk_ids,
            topk_probs=topk_probs,
            dst=dst,
            physical_group_ids=physical_group_ids,
            comm_plan=comm_plan,
            dedup_complete=gathered_all_ranks or context.ep_size == 1,
        )
        return MoEPlannerOutput(
            physical_to_logical_map=physical_to_logical_map,
            routing_map=token_reroute["routing_map"],
            probs=token_reroute["probs"],
        )

    def plan(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: Optional[torch.Tensor] = None,
    ) -> MoEPlannerOutput:
        if routing_map.size(1) != context.num_logical_experts:
            raise ValueError(
                "routing_map logical expert dimension does not match SchedulerContext, "
                f"got {routing_map.size(1)} and {context.num_logical_experts}."
            )
        topk_ids, topk_probs = self._topk_from_dense(probs, routing_map, context)
        counts_from_ep_rank = self._get_count_matrix(
            routing_map, context, topk_ids, tokens_per_expert=tokens_per_expert
        )
        return self._build_output_from_counts(
            probs, routing_map, context, topk_ids, topk_probs, counts_from_ep_rank
        )

    def plan_with_count_matrix(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        counts_from_ep_rank: torch.Tensor,
    ) -> MoEPlannerOutput:
        """Plan with a caller-provided ``[ep_size, num_experts]`` count matrix.

        This is useful when another component has already gathered route
        statistics.  The local row is still checked against ``routing_map`` so
        token offsets remain well defined.
        """
        if routing_map.size(1) != context.num_logical_experts:
            raise ValueError(
                "routing_map logical expert dimension does not match SchedulerContext, "
                f"got {routing_map.size(1)} and {context.num_logical_experts}."
            )
        if counts_from_ep_rank.shape != (context.ep_size, context.num_logical_experts):
            raise ValueError(
                "Expected counts_from_ep_rank to have shape "
                f"({context.ep_size}, {context.num_logical_experts}), "
                f"got {tuple(counts_from_ep_rank.shape)}."
            )
        topk_ids, topk_probs = self._topk_from_dense(probs, routing_map, context)
        local_counts = self._count_topk_assignments(topk_ids, context.num_logical_experts)
        provided_local = counts_from_ep_rank[context.ep_rank].to(
            dtype=torch.int64, device=local_counts.device
        )
        if not torch.equal(provided_local, local_counts):
            raise ValueError(
                "Expected counts_from_ep_rank for the current rank to match local topk_ids."
            )
        return self._build_output_from_counts(
            probs, routing_map, context, topk_ids, topk_probs, counts_from_ep_rank
        )
