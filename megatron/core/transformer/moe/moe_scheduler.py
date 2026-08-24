# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoE scheduler interfaces for expert placement and route-map rewrites.

The scheduler runs after routing and before the token dispatcher.  It is split
into a load planner, which decides how logical experts should map to physical
expert instances and whether routing should be rewritten, and an expert
dispatcher, which materializes the selected placement before token dispatch.
Concrete backends such as UltraEP or MoonEP should implement these interfaces
without changing ``MoELayer``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional

import torch

from megatron.core import utils
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass(frozen=True)
class RouteInfo:
    """Router output consumed by MoE load planners."""

    probs: torch.Tensor
    routing_map: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.probs.dim() != 2:
            raise ValueError(
                f"Expected probs to be 2D, got shape {tuple(self.probs.shape)}"
            )
        if self.routing_map.dim() != 2:
            raise ValueError(
                f"Expected routing_map to be 2D, got shape {tuple(self.routing_map.shape)}"
            )
        if self.routing_map.dtype != torch.bool:
            raise ValueError(f"Expected bool routing_map, got {self.routing_map.dtype}")
        if self.probs.shape != self.routing_map.shape:
            raise ValueError(
                "Expected probs and routing_map to have the same shape, "
                f"got {tuple(self.probs.shape)} and {tuple(self.routing_map.shape)}"
            )

    @property
    def num_tokens(self) -> int:
        """Number of local token rows represented by this route map."""
        return self.routing_map.size(0)

    @property
    def num_logical_experts(self) -> int:
        """Number of global logical experts represented by this route map."""
        return self.routing_map.size(1)

    def with_routing(
        self, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> "RouteInfo":
        """Return a copy with rewritten routing tensors."""
        return replace(self, probs=probs, routing_map=routing_map)


@dataclass(frozen=True)
class SchedulerContext:
    """Static and per-forward context shared by planners and dispatchers."""

    config: TransformerConfig
    layer_number: Optional[int]
    num_local_experts: int
    local_expert_indices: tuple[int, ...]
    ep_size: int
    ep_rank: int
    training: bool
    pg_collection: Optional[ProcessGroupCollection] = None

    @property
    def num_global_experts(self) -> int:
        """Total number of logical experts in the MoE layer."""
        assert self.config.num_moe_experts is not None
        return self.config.num_moe_experts

    @property
    def router_topk(self) -> int:
        """Number of experts selected for each token."""
        return self.config.moe_router_topk


@dataclass(frozen=True)
class ExpertPlacementPlan:
    """Common placement IR passed from planners to expert dispatchers.

    The tensor fields intentionally match the placement state required by
    physical-replica dispatchers such as UltraEP. Backends that need additional
    native metadata should attach it through ``backend_metadata`` instead of
    teaching ``MoELayer`` about backend-specific objects.
    """

    backend: str = "none"
    physical_to_logical_map: Optional[torch.Tensor] = None
    logical_to_physical_map: Optional[torch.Tensor] = None
    logical_replica_counts: Optional[torch.Tensor] = None
    logical_instance_quota: Optional[torch.Tensor] = None
    logical_instance_quota_prefix: Optional[torch.Tensor] = None
    rank_quota_prefix: Optional[torch.Tensor] = None
    backend_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_identity(self) -> bool:
        """Whether this plan leaves the default logical expert layout unchanged."""
        return self.backend == "none" and self.physical_to_logical_map is None


@dataclass(frozen=True)
class PlannerResult:
    """Planner output consumed by ``MoEScheduler``."""

    probs: torch.Tensor
    routing_map: torch.Tensor
    placement_plan: ExpertPlacementPlan = field(default_factory=ExpertPlacementPlan)
    backend_metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def identity(cls, route_info: RouteInfo) -> "PlannerResult":
        """Build an identity result that preserves router outputs."""
        return cls(probs=route_info.probs, routing_map=route_info.routing_map)

    def to_route_info(self, base: RouteInfo) -> RouteInfo:
        """Convert the result back to route info for a subsequent planner."""
        return base.with_routing(self.probs, self.routing_map)


@dataclass
class DispatchHandle:
    """Handle returned by an expert dispatcher after materializing a plan."""

    placement_plan: ExpertPlacementPlan
    event: Any = None
    backend_metadata: Mapping[str, Any] = field(default_factory=dict)

    def wait(self, stream: Any = None) -> None:
        """Wait for asynchronous expert dispatch work, if the backend returned an event."""
        if self.event is None:
            return
        wait = getattr(self.event, "wait", None)
        if wait is None:
            return
        if stream is None:
            wait()
        else:
            wait(stream)


class MoELoadPlanner(torch.nn.Module, ABC):
    """Base class for route-map aware MoE load planners."""

    def should_plan(self, route_info: RouteInfo, context: SchedulerContext) -> bool:
        """Return whether this planner should run for the current forward."""
        del route_info, context
        return True

    @abstractmethod
    def plan(
        self,
        route_info: RouteInfo,
        context: SchedulerContext,
        base_plan: Optional[ExpertPlacementPlan] = None,
    ) -> PlannerResult:
        """Return the expert placement and any rewritten routing tensors."""


class L1LoadPlanner(MoELoadPlanner):
    """Base class for low-frequency history-based planners."""


class L2LoadPlanner(MoELoadPlanner):
    """Base class for per-forward route-map based planners."""


class NoOpLoadPlanner(MoELoadPlanner):
    """Identity planner used by default and in tests."""

    def should_plan(self, route_info: RouteInfo, context: SchedulerContext) -> bool:
        del route_info, context
        return False

    def plan(
        self,
        route_info: RouteInfo,
        context: SchedulerContext,
        base_plan: Optional[ExpertPlacementPlan] = None,
    ) -> PlannerResult:
        del context
        return PlannerResult(
            probs=route_info.probs,
            routing_map=route_info.routing_map,
            placement_plan=base_plan or ExpertPlacementPlan(),
        )


class AutoLoadPlanner(MoELoadPlanner):
    """Hierarchical planner shell that composes optional L1 and L2 planners."""

    def __init__(
        self,
        l1_planner: Optional[MoELoadPlanner] = None,
        l2_planner: Optional[MoELoadPlanner] = None,
    ) -> None:
        super().__init__()
        self.l1_planner = l1_planner or NoOpLoadPlanner()
        self.l2_planner = l2_planner or NoOpLoadPlanner()

    def plan(
        self,
        route_info: RouteInfo,
        context: SchedulerContext,
        base_plan: Optional[ExpertPlacementPlan] = None,
    ) -> PlannerResult:
        result = PlannerResult(
            probs=route_info.probs,
            routing_map=route_info.routing_map,
            placement_plan=base_plan or ExpertPlacementPlan(),
        )
        current_route_info = route_info
        current_plan = result.placement_plan

        if self.l1_planner.should_plan(current_route_info, context):
            result = self.l1_planner.plan(
                current_route_info, context, base_plan=current_plan
            )
            current_route_info = result.to_route_info(current_route_info)
            current_plan = result.placement_plan

        if self.l2_planner.should_plan(current_route_info, context):
            result = self.l2_planner.plan(
                current_route_info, context, base_plan=current_plan
            )
        elif result.placement_plan is not current_plan:
            result = replace(result, placement_plan=current_plan)

        return result


class ExpertDispatch(torch.nn.Module, ABC):
    """Base class for materializing planner-selected expert placement."""

    @abstractmethod
    def prepare(
        self,
        experts: torch.nn.Module,
        placement_plan: ExpertPlacementPlan,
        context: SchedulerContext,
    ) -> DispatchHandle:
        """Prepare experts for the selected placement before token dispatch."""

    def finalize(self, handle: DispatchHandle, context: SchedulerContext) -> None:
        """Release any transient placement resources after the MoE forward finishes."""
        del handle, context


class NoOpExpertDispatch(ExpertDispatch):
    """Default expert dispatcher that leaves experts in their original layout."""

    def prepare(
        self,
        experts: torch.nn.Module,
        placement_plan: ExpertPlacementPlan,
        context: SchedulerContext,
    ) -> DispatchHandle:
        del experts, context
        return DispatchHandle(placement_plan=placement_plan)


class MoEScheduler(torch.nn.Module):
    """Coordinates MoE load planning and expert dispatch for one MoE layer."""

    def __init__(
        self,
        config: TransformerConfig,
        num_local_experts: int,
        local_expert_indices: tuple[int, ...],
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_local_experts = num_local_experts
        self.local_expert_indices = local_expert_indices
        self.pg_collection = pg_collection
        self.enabled = bool(config.moe_enable_scheduler)
        self.load_planner = (
            self._build_load_planner(config) if self.enabled else NoOpLoadPlanner()
        )
        self.expert_dispatch = (
            self._build_expert_dispatch(config)
            if self.enabled
            else NoOpExpertDispatch()
        )
        self._cached_plan: Optional[ExpertPlacementPlan] = None
        self._active_dispatch_handle: Optional[DispatchHandle] = None
        self._active_context: Optional[SchedulerContext] = None

    @staticmethod
    def _build_load_planner(config: TransformerConfig) -> MoELoadPlanner:
        if config.moe_load_planner_type == "none":
            return NoOpLoadPlanner()
        if config.moe_load_planner_type == "auto":
            return AutoLoadPlanner()
        raise ValueError(
            f"Unsupported MoE load planner type: {config.moe_load_planner_type}"
        )

    @staticmethod
    def _build_expert_dispatch(config: TransformerConfig) -> ExpertDispatch:
        if config.moe_expert_dispatcher_type == "none":
            return NoOpExpertDispatch()
        raise ValueError(
            f"Unsupported MoE expert dispatcher type: {config.moe_expert_dispatcher_type}"
        )

    @property
    def active_dispatch_handle(self) -> Optional[DispatchHandle]:
        """The active expert-dispatch handle for the current forward, if any."""
        return self._active_dispatch_handle

    @property
    def cached_plan(self) -> Optional[ExpertPlacementPlan]:
        """Most recent placement plan retained for planners that need history."""
        return self._cached_plan

    def _build_context(
        self, layer_number: Optional[int], training: bool
    ) -> SchedulerContext:
        if self.pg_collection is None:
            ep_size = 1
            ep_rank = 0
        else:
            ep_size = utils.get_pg_size(self.pg_collection.ep)
            ep_rank = utils.get_pg_rank(self.pg_collection.ep)
        return SchedulerContext(
            config=self.config,
            layer_number=layer_number,
            num_local_experts=self.num_local_experts,
            local_expert_indices=self.local_expert_indices,
            ep_size=ep_size,
            ep_rank=ep_rank,
            training=training,
            pg_collection=self.pg_collection,
        )

    def schedule(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        *,
        hidden_states: Optional[torch.Tensor],
        experts: torch.nn.Module,
        layer_number: Optional[int],
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Plan expert placement and return routing tensors for token dispatch."""
        if not self.enabled:
            return probs, routing_map

        route_info = RouteInfo(
            probs=probs, routing_map=routing_map, hidden_states=hidden_states
        )
        context = self._build_context(layer_number=layer_number, training=training)
        planner_result = self.load_planner.plan(
            route_info, context, base_plan=self._cached_plan
        )
        result_route_info = route_info.with_routing(
            planner_result.probs, planner_result.routing_map
        )
        placement_plan = planner_result.placement_plan
        dispatch_handle = self.expert_dispatch.prepare(experts, placement_plan, context)
        if self.config.moe_scheduler_wait_for_dispatch:
            dispatch_handle.wait()
        self._cached_plan = placement_plan
        self._active_dispatch_handle = dispatch_handle
        self._active_context = context
        return result_route_info.probs, result_route_info.routing_map

    def finish_forward(self) -> None:
        """Finalize transient expert-dispatch state for the current forward."""
        if self._active_dispatch_handle is None:
            return
        assert self._active_context is not None
        self.expert_dispatch.finalize(
            self._active_dispatch_handle, self._active_context
        )
        self._active_dispatch_handle = None
        self._active_context = None
