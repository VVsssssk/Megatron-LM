# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoonEP-style replica planner for MoEScheduler.

This module adapts the planner side of NVIDIA/Megatron-LM PR #6892 to the
MoEScheduler abstraction.  The public output stays backend-neutral:
``physical_to_logical_map`` plus dense physical ``routing_map``/``probs``.
Expert weight movement is still owned by ``ExpertDispatch``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from megatron.core.transformer.moe.moe_scheduler import (
    MoELoadPlanner,
    MoEPlannerOutput,
    SchedulerContext,
)
from megatron.core.transformer.moe.moonep_replica_triton import (
    MAX_REPLICA_EP_RANKS,
    launch_compact_routing_map,
    launch_replica_placement,
    launch_replica_route_mapping,
    launch_replica_route_ranking,
    planner_route_partition_count,
)


@dataclass(frozen=True, slots=True)
class ReplicaPlan:
    """Transport-facing planner result from PR #6892.

    ``virtual_experts`` has shape ``[num_tokens, router_topk]`` and contains
    rank-major physical expert ids.  ``experts_to_copy`` has shape
    ``[ep_size, num_experts_per_rank]`` and records which semantic expert each
    destination rank should materialize in its replica slots; unused slots are
    ``-1``.
    """

    virtual_experts: torch.Tensor
    experts_to_copy: torch.Tensor


@dataclass(slots=True)
class ReplicaPlannerWorkspace:
    """Fixed-address scratch/output buffers for one planner shape."""

    num_tokens: int
    router_topk: int
    num_experts: int
    ep_size: int
    num_local_experts: int
    gathered_counts: torch.Tensor
    balance: torch.Tensor
    allocation: torch.Tensor
    placement_grid_sync: torch.Tensor
    destination_boundaries: torch.Tensor
    expert_replica_slots: torch.Tensor
    sort_route_metadata: torch.Tensor
    sort_partition_counts: torch.Tensor
    sort_grid_sync: torch.Tensor
    sort_stream: torch.cuda.Stream
    virtual_experts: torch.Tensor
    experts_to_copy: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        num_tokens: int,
        router_topk: int,
        num_experts: int,
        ep_size: int,
        device: torch.device,
    ) -> "ReplicaPlannerWorkspace":
        """Allocate reusable planner state for a fixed route shape."""
        if device.type != "cuda":
            raise RuntimeError("ReplicaPlannerWorkspace requires a CUDA device.")
        if min(num_tokens, router_topk, num_experts, ep_size) <= 0:
            raise ValueError(
                "Replica planner dimensions must be positive, got "
                f"num_tokens={num_tokens}, router_topk={router_topk}, "
                f"num_experts={num_experts}, ep_size={ep_size}."
            )
        if ep_size > MAX_REPLICA_EP_RANKS:
            raise ValueError(
                f"MoonEP replica planner supports at most {MAX_REPLICA_EP_RANKS} EP ranks, "
                f"got {ep_size}."
            )
        if num_experts % ep_size:
            raise ValueError(
                "Replica planner requires equal experts per rank, got "
                f"num_experts={num_experts}, ep_size={ep_size}."
            )

        num_routes = num_tokens * router_topk
        int32 = dict(dtype=torch.int32, device=device)
        return cls(
            num_tokens=num_tokens,
            router_topk=router_topk,
            num_experts=num_experts,
            ep_size=ep_size,
            num_local_experts=num_experts // ep_size,
            gathered_counts=torch.empty((ep_size, num_experts), **int32),
            balance=torch.empty(ep_size, **int32),
            allocation=torch.empty((num_experts, ep_size), **int32),
            placement_grid_sync=torch.zeros(1, **int32),
            destination_boundaries=torch.empty(
                (num_experts, 1 << (ep_size - 1).bit_length()), **int32
            ),
            expert_replica_slots=torch.empty((num_experts, ep_size), **int32),
            sort_route_metadata=torch.empty(num_routes, **int32),
            sort_partition_counts=torch.empty(
                (planner_route_partition_count(num_routes), num_experts), **int32
            ),
            sort_grid_sync=torch.zeros(1, **int32),
            sort_stream=torch.cuda.Stream(device=device),
            virtual_experts=torch.empty(
                (num_tokens, router_topk), dtype=torch.int64, device=device
            ),
            experts_to_copy=torch.empty((ep_size, num_experts // ep_size), **int32),
        )


def extract_semantic_routes(
    routing_map: torch.Tensor, probs: torch.Tensor, router_topk: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover compact semantic routes from dense router output.

    ``routing_map`` is authoritative, so selected zero-probability routes remain
    selected.  The route order is ascending semantic expert id, matching the
    PR #6892 planner.
    """
    if routing_map.dim() != 2:
        raise ValueError(f"routing_map must be 2D, got shape {tuple(routing_map.shape)}.")
    if probs.shape != routing_map.shape:
        raise ValueError(
            f"probs and routing_map must have the same shape, got "
            f"{tuple(probs.shape)} and {tuple(routing_map.shape)}."
        )
    if routing_map.dtype != torch.bool:
        raise ValueError(f"routing_map must be bool, got {routing_map.dtype}.")
    if router_topk <= 0 or router_topk > routing_map.size(1):
        raise ValueError(
            f"router_topk must be in [1, {routing_map.size(1)}], got {router_topk}."
        )
    if not routing_map.is_cuda or not probs.is_cuda:
        raise RuntimeError("MoonEP replica planner requires CUDA routing tensors.")
    if routing_map.device != probs.device:
        raise ValueError(
            f"probs and routing_map must be on the same device, got "
            f"{probs.device} and {routing_map.device}."
        )
    if not routing_map.is_contiguous() or not probs.is_contiguous():
        raise ValueError("MoonEP replica planner requires contiguous routing tensors.")

    num_tokens, num_experts = (int(size) for size in routing_map.shape)
    tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32, device=routing_map.device)
    token_indices = torch.zeros(
        (num_tokens, router_topk), dtype=torch.int32, device=routing_map.device
    )
    launch_compact_routing_map(
        routing_map,
        token_indices,
        tokens_per_expert,
        num_tokens=num_tokens,
        router_topk=router_topk,
        num_experts=num_experts,
    )
    return torch.gather(probs, 1, token_indices.long()), token_indices, tokens_per_expert


def plan_replica_routes(
    topk_indices: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    ep_group: dist.ProcessGroup,
    workspace: ReplicaPlannerWorkspace,
    *,
    on_placement_ready: Optional[Callable[[ReplicaPlan], None]] = None,
) -> ReplicaPlan:
    """Plan deterministic replica placements for HybridEP-compatible layouts."""
    ep_size = dist.get_world_size(group=ep_group)
    expected = (
        workspace.num_tokens,
        workspace.router_topk,
        workspace.num_experts,
        workspace.ep_size,
    )
    if (
        topk_indices.dtype not in (torch.int32, torch.int64)
        or tokens_per_expert.dtype != torch.int32
        or not topk_indices.is_contiguous()
        or not tokens_per_expert.is_contiguous()
        or topk_indices.device != workspace.gathered_counts.device
        or tokens_per_expert.device != workspace.gathered_counts.device
        or (*topk_indices.shape, tokens_per_expert.numel(), ep_size) != expected
    ):
        raise ValueError(
            "Replica planner expects contiguous int32/int64 routes and an int32 histogram "
            f"on {workspace.gathered_counts.device} matching "
            f"(num_tokens, router_topk, num_experts, ep_size)={expected}; got "
            f"{tuple(topk_indices.shape)} {topk_indices.dtype} routes on {topk_indices.device} "
            f"and {tuple(tokens_per_expert.shape)} {tokens_per_expert.dtype} counts on "
            f"{tokens_per_expert.device} for ep_size={ep_size}."
        )

    num_tokens, router_topk, num_experts, _ = expected
    num_local_experts = num_experts // ep_size
    num_routes = num_tokens * router_topk

    current_stream = torch.cuda.current_stream(topk_indices.device)
    workspace.sort_stream.wait_stream(current_stream)
    with torch.cuda.stream(workspace.sort_stream):
        launch_replica_route_ranking(
            topk_indices.reshape(-1),
            workspace.sort_route_metadata,
            workspace.sort_partition_counts,
            workspace.sort_grid_sync,
            num_experts=num_experts,
            num_routes=num_routes,
        )

    dist.all_gather_into_tensor(
        workspace.gathered_counts.view(-1), tokens_per_expert, group=ep_group
    )

    launch_replica_placement(
        workspace.gathered_counts,
        workspace.balance,
        workspace.allocation,
        workspace.destination_boundaries,
        workspace.experts_to_copy,
        workspace.expert_replica_slots,
        workspace.placement_grid_sync,
        rank_route_capacity=num_routes,
        source_rank=dist.get_rank(group=ep_group),
        ep_size=ep_size,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
    )

    plan = ReplicaPlan(
        virtual_experts=workspace.virtual_experts,
        experts_to_copy=workspace.experts_to_copy,
    )
    workspace.sort_stream.wait_stream(current_stream)
    with torch.cuda.stream(workspace.sort_stream):
        launch_replica_route_mapping(
            workspace.sort_route_metadata,
            workspace.sort_partition_counts,
            workspace.destination_boundaries,
            workspace.expert_replica_slots,
            workspace.virtual_experts,
            num_routes=num_routes,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            ep_size=ep_size,
        )
    if on_placement_ready is not None:
        on_placement_ready(plan)
    current_stream.wait_stream(workspace.sort_stream)
    return plan


def map_replica_plan_to_hybridep(
    plan: ReplicaPlan, topk_probs: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter compact virtual routes into dense HybridEP token-dispatch inputs."""
    if plan.virtual_experts.shape != topk_probs.shape:
        raise ValueError(
            "Replica virtual experts and top-k probabilities must have the same shape, got "
            f"{tuple(plan.virtual_experts.shape)} and {tuple(topk_probs.shape)}."
        )
    dense_shape = (int(plan.virtual_experts.shape[0]), num_experts)
    routing_map = torch.zeros(dense_shape, dtype=torch.bool, device=plan.virtual_experts.device)
    dense_probs = torch.zeros(dense_shape, dtype=torch.float32, device=topk_probs.device)
    routing_map.scatter_(1, plan.virtual_experts, True)
    dense_probs.scatter_(1, plan.virtual_experts, topk_probs.to(torch.float32))
    return routing_map, dense_probs


def _physical_to_logical_map_from_experts_to_copy(
    experts_to_copy: torch.Tensor, context: SchedulerContext
) -> torch.Tensor:
    """Build rank-major physical layout from PR #6892 ``experts_to_copy``."""
    if experts_to_copy.shape != (
        context.ep_size,
        context.num_logical_experts // context.ep_size,
    ):
        raise ValueError(
            "experts_to_copy shape does not match SchedulerContext, got "
            f"{tuple(experts_to_copy.shape)}."
        )

    device = experts_to_copy.device
    num_local_home_experts = context.num_logical_experts // context.ep_size
    num_local_physical_experts = 2 * num_local_home_experts
    num_physical_experts = context.ep_size * num_local_physical_experts
    physical_to_logical = torch.full(
        (num_physical_experts,), -1, dtype=torch.long, device=device
    )

    logical_ids = torch.arange(context.num_logical_experts, dtype=torch.long, device=device)
    owner_ranks = torch.div(logical_ids, num_local_home_experts, rounding_mode="floor")
    owner_slots = logical_ids.remainder(num_local_home_experts)
    home_physical_ids = owner_ranks * num_local_physical_experts + owner_slots
    physical_to_logical[home_physical_ids] = logical_ids

    ranks = torch.arange(context.ep_size, dtype=torch.long, device=device)
    replica_slots = torch.arange(num_local_home_experts, dtype=torch.long, device=device)
    replica_physical_ids = (
        ranks[:, None] * num_local_physical_experts
        + num_local_home_experts
        + replica_slots[None, :]
    )
    replica_logical_ids = experts_to_copy.to(dtype=torch.long)
    valid = (replica_logical_ids >= 0) & (replica_logical_ids < context.num_logical_experts)
    physical_to_logical[replica_physical_ids[valid]] = replica_logical_ids[valid]
    return physical_to_logical


class MoonEPLoadPlanner(MoELoadPlanner):
    """PR #6892 MoonEP-style L2 planner.

    ``num_redundant_experts`` is the number of replica slots per EP rank.  The
    PR planner requires this to equal the number of native experts per rank,
    giving each rank ``[home experts][replica experts]`` physical slots.
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
        if token_padding != 1:
            raise ValueError("PR #6892 MoonEPLoadPlanner does not support token_padding.")
        self.num_redundant_experts = num_redundant_experts
        self.token_padding = token_padding
        self._workspaces: dict[tuple[int, int, int, int, int], ReplicaPlannerWorkspace] = {}

    def _resolve_num_redundant_experts(self, context: SchedulerContext) -> int:
        if self.num_redundant_experts is not None:
            return self.num_redundant_experts
        return context.num_logical_experts // context.ep_size

    def _validate_inputs(
        self, probs: torch.Tensor, routing_map: torch.Tensor, context: SchedulerContext
    ) -> None:
        if routing_map.dim() != 2:
            raise ValueError(f"routing_map must be 2D, got shape {tuple(routing_map.shape)}.")
        if probs.shape != routing_map.shape:
            raise ValueError(
                f"probs and routing_map must have the same shape, got "
                f"{tuple(probs.shape)} and {tuple(routing_map.shape)}."
            )
        if routing_map.size(1) != context.num_logical_experts:
            raise ValueError(
                "routing_map logical expert dimension does not match SchedulerContext, "
                f"got {routing_map.size(1)} and {context.num_logical_experts}."
            )
        if routing_map.dtype != torch.bool:
            raise ValueError(f"routing_map must be bool, got {routing_map.dtype}.")

    def _validate_context(self, context: SchedulerContext) -> None:
        if context.num_logical_experts % context.ep_size != 0:
            raise ValueError(
                "MoonEPLoadPlanner requires num_logical_experts divisible by ep_size."
            )
        if context.ep_size > MAX_REPLICA_EP_RANKS:
            raise ValueError(
                f"MoonEPLoadPlanner supports at most {MAX_REPLICA_EP_RANKS} EP ranks, "
                f"got {context.ep_size}."
            )
        num_local_home_experts = context.num_logical_experts // context.ep_size
        num_redundant_experts = self._resolve_num_redundant_experts(context)
        if num_redundant_experts not in (0, num_local_home_experts):
            raise ValueError(
                "PR #6892 MoonEPLoadPlanner requires one replica slot per local home expert; "
                f"got num_redundant_experts={num_redundant_experts}, "
                f"num_local_home_experts={num_local_home_experts}."
            )

    def should_plan(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: Optional[torch.Tensor] = None,
    ) -> bool:
        del tokens_per_expert
        self._validate_inputs(probs, routing_map, context)
        self._validate_context(context)
        return self._resolve_num_redundant_experts(context) != 0

    @staticmethod
    def _identity_output(
        probs: torch.Tensor, routing_map: torch.Tensor, context: SchedulerContext
    ) -> MoEPlannerOutput:
        physical_to_logical_map = torch.arange(
            context.num_logical_experts, dtype=torch.long, device=routing_map.device
        )
        return MoEPlannerOutput(
            physical_to_logical_map=physical_to_logical_map,
            routing_map=routing_map,
            probs=probs,
        )

    def _single_rank_output(
        self, probs: torch.Tensor, routing_map: torch.Tensor, context: SchedulerContext
    ) -> MoEPlannerOutput:
        num_redundant_experts = self._resolve_num_redundant_experts(context)
        num_physical_experts = context.num_logical_experts + num_redundant_experts
        physical_to_logical_map = torch.full(
            (num_physical_experts,), -1, dtype=torch.long, device=routing_map.device
        )
        physical_to_logical_map[: context.num_logical_experts] = torch.arange(
            context.num_logical_experts, dtype=torch.long, device=routing_map.device
        )
        physical_routing_map = torch.zeros(
            routing_map.size(0),
            num_physical_experts,
            dtype=torch.bool,
            device=routing_map.device,
        )
        physical_probs = torch.zeros(
            probs.size(0), num_physical_experts, dtype=probs.dtype, device=probs.device
        )
        physical_routing_map[:, : context.num_logical_experts] = routing_map
        physical_probs[:, : context.num_logical_experts] = probs
        return MoEPlannerOutput(
            physical_to_logical_map=physical_to_logical_map,
            routing_map=physical_routing_map,
            probs=physical_probs,
        )

    def _get_ep_group(self, context: SchedulerContext) -> dist.ProcessGroup:
        ep_group = getattr(context.pg_collection, "ep", None)
        if ep_group is None:
            raise ValueError(
                "MoonEPLoadPlanner requires SchedulerContext.pg_collection.ep when ep_size > 1."
            )
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("MoonEPLoadPlanner requires initialized torch.distributed.")
        if dist.get_world_size(group=ep_group) != context.ep_size:
            raise ValueError(
                "SchedulerContext.ep_size does not match the EP process group size, got "
                f"{context.ep_size} and {dist.get_world_size(group=ep_group)}."
            )
        if dist.get_rank(group=ep_group) != context.ep_rank:
            raise ValueError(
                "SchedulerContext.ep_rank does not match the EP process group rank, got "
                f"{context.ep_rank} and {dist.get_rank(group=ep_group)}."
            )
        return ep_group

    def _workspace_key(
        self,
        *,
        device: torch.device,
        num_tokens: int,
        router_topk: int,
        num_experts: int,
        ep_size: int,
    ) -> tuple[int, int, int, int, int]:
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        return (device_index, num_tokens, router_topk, num_experts, ep_size)

    def _get_workspace(
        self,
        *,
        device: torch.device,
        num_tokens: int,
        router_topk: int,
        num_experts: int,
        ep_size: int,
    ) -> ReplicaPlannerWorkspace:
        key = self._workspace_key(
            device=device,
            num_tokens=num_tokens,
            router_topk=router_topk,
            num_experts=num_experts,
            ep_size=ep_size,
        )
        workspace = self._workspaces.get(key)
        if workspace is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "MoonEPLoadPlanner workspace must be allocated before CUDA graph capture."
                )
            workspace = ReplicaPlannerWorkspace.allocate(
                num_tokens=num_tokens,
                router_topk=router_topk,
                num_experts=num_experts,
                ep_size=ep_size,
                device=device,
            )
            self._workspaces[key] = workspace
        return workspace

    def plan(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: Optional[torch.Tensor] = None,
    ) -> MoEPlannerOutput:
        """Return PR #6892 physical layout and dense rerouted token tensors."""
        del tokens_per_expert
        self._validate_inputs(probs, routing_map, context)
        self._validate_context(context)
        if self._resolve_num_redundant_experts(context) == 0:
            return self._identity_output(probs, routing_map, context)
        if context.ep_size == 1:
            return self._single_rank_output(probs, routing_map, context)

        ep_group = self._get_ep_group(context)
        if not routing_map.is_cuda or not probs.is_cuda:
            raise RuntimeError("MoonEPLoadPlanner requires CUDA tensors when ep_size > 1.")
        topk_probs, topk_indices, local_tokens_per_expert = extract_semantic_routes(
            routing_map, probs, context.router_topk
        )
        workspace = self._get_workspace(
            device=routing_map.device,
            num_tokens=routing_map.size(0),
            router_topk=context.router_topk,
            num_experts=context.num_logical_experts,
            ep_size=context.ep_size,
        )
        replica_plan = plan_replica_routes(
            topk_indices,
            local_tokens_per_expert,
            ep_group,
            workspace,
        )
        physical_routing_map, physical_probs = map_replica_plan_to_hybridep(
            replica_plan,
            topk_probs,
            context.num_logical_experts * 2,
        )
        physical_to_logical_map = _physical_to_logical_map_from_experts_to_copy(
            replica_plan.experts_to_copy,
            context,
        )
        return MoEPlannerOutput(
            physical_to_logical_map=physical_to_logical_map,
            routing_map=physical_routing_map,
            probs=physical_probs,
        )

    def plan_with_count_matrix(self, *args, **kwargs) -> MoEPlannerOutput:
        """Reject the removed Python count-matrix adapter."""
        del args, kwargs
        raise NotImplementedError(
            "PR #6892 MoonEPLoadPlanner plans from routing_map/probs and the EP process group; "
            "plan_with_count_matrix is not supported."
        )
