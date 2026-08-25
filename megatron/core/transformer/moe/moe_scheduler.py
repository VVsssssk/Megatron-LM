# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Backend-neutral interfaces for MoE expert and token rerouting.

MoEScheduler is intended to run after router output is available and before the
normal MoE token dispatcher starts.  This module only defines the shared
contracts:

* planners translate a route map into an expert reroute plan plus a token
  reroute plan;
* expert dispatchers materialize the expert reroute plan before token dispatch;
* backend-specific objects can be attached as native metadata without changing
  the common interface.

Concrete Echo, UltraEP, and MoonEP planners should produce ``MoEPlannerOutput``.
Concrete Echo and UltraEP expert dispatchers should consume ``ExpertReroutePlan``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Optional

import torch

IDENTITY_BACKEND = "identity"
ECHO_BACKEND = "echo"
ULTRA_EP_BACKEND = "ultra_ep"
MOON_EP_BACKEND = "moon_ep"


def _validate_1d_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dim() != 1:
        raise ValueError(f"Expected {name} to be 1D, got shape {tuple(tensor.shape)}")


def _validate_2d_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dim() != 2:
        raise ValueError(f"Expected {name} to be 2D, got shape {tuple(tensor.shape)}")


@dataclass(frozen=True)
class RouteInfo:
    """Router output consumed by MoE load planners.

    ``probs`` and ``routing_map`` are the dense logical-expert tensors used by
    existing Megatron token dispatchers.  ``topk_ids`` and ``topk_probs`` are
    optional sparse views for planners whose native APIs use top-k assignments.
    """

    probs: torch.Tensor
    routing_map: torch.Tensor
    topk_ids: Optional[torch.Tensor] = None
    topk_probs: Optional[torch.Tensor] = None
    tokens_per_expert: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        _validate_2d_tensor("probs", self.probs)
        _validate_2d_tensor("routing_map", self.routing_map)
        if self.routing_map.dtype != torch.bool:
            raise ValueError(f"Expected bool routing_map, got {self.routing_map.dtype}")
        if self.probs.shape != self.routing_map.shape:
            raise ValueError(
                "Expected probs and routing_map to have the same shape, "
                f"got {tuple(self.probs.shape)} and {tuple(self.routing_map.shape)}"
            )
        if self.topk_ids is not None:
            _validate_2d_tensor("topk_ids", self.topk_ids)
            if self.topk_ids.size(0) != self.num_tokens:
                raise ValueError(
                    "Expected topk_ids to have the same token dimension as routing_map, "
                    f"got {self.topk_ids.size(0)} and {self.num_tokens}"
                )
        if self.topk_probs is not None:
            if self.topk_ids is None:
                raise ValueError("topk_probs requires topk_ids to be provided.")
            _validate_2d_tensor("topk_probs", self.topk_probs)
            if self.topk_probs.shape != self.topk_ids.shape:
                raise ValueError(
                    "Expected topk_probs and topk_ids to have the same shape, "
                    f"got {tuple(self.topk_probs.shape)} and {tuple(self.topk_ids.shape)}"
                )
        if self.tokens_per_expert is not None:
            _validate_1d_tensor("tokens_per_expert", self.tokens_per_expert)
            if self.tokens_per_expert.numel() != self.num_logical_experts:
                raise ValueError(
                    "Expected tokens_per_expert to match the logical expert dimension, "
                    f"got {self.tokens_per_expert.numel()} and {self.num_logical_experts}"
                )

    @property
    def num_tokens(self) -> int:
        """Number of local token rows represented by the route map."""
        return self.routing_map.size(0)

    @property
    def num_logical_experts(self) -> int:
        """Number of global logical experts represented by the route map."""
        return self.routing_map.size(1)


@dataclass(frozen=True)
class SchedulerContext:
    """Static and per-forward context shared by planners and dispatchers."""

    layer_number: Optional[int]
    num_logical_experts: int
    num_local_experts: int
    local_expert_indices: tuple[int, ...]
    ep_size: int
    ep_rank: int
    router_topk: int
    training: bool
    config: Any = None
    pg_collection: Any = None

    def __post_init__(self) -> None:
        if self.num_logical_experts <= 0:
            raise ValueError("num_logical_experts must be positive.")
        if self.num_local_experts <= 0:
            raise ValueError("num_local_experts must be positive.")
        if self.ep_size <= 0:
            raise ValueError("ep_size must be positive.")
        if self.ep_rank < 0 or self.ep_rank >= self.ep_size:
            raise ValueError(f"ep_rank must be in [0, {self.ep_size}), got {self.ep_rank}.")
        if self.router_topk <= 0:
            raise ValueError("router_topk must be positive.")
        if len(self.local_expert_indices) != self.num_local_experts:
            raise ValueError(
                "Expected local_expert_indices to match num_local_experts, "
                f"got {len(self.local_expert_indices)} and {self.num_local_experts}"
            )


@dataclass(frozen=True)
class ExpertTransferPlan:
    """Canonical expert-copy tasks needed to materialize a reroute plan.

    Each row describes copying one logical expert into one destination physical
    expert slot.  Backends with richer native state can attach it through
    ``metadata`` while still exposing the common task list when possible.
    """

    source_logical_expert_ids: Optional[torch.Tensor] = None
    dest_physical_expert_ids: Optional[torch.Tensor] = None
    dest_ranks: Optional[torch.Tensor] = None
    dest_local_slots: Optional[torch.Tensor] = None
    source_ranks: Optional[torch.Tensor] = None
    source_local_slots: Optional[torch.Tensor] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        core_tensors = (
            self.source_logical_expert_ids,
            self.dest_physical_expert_ids,
            self.dest_ranks,
            self.dest_local_slots,
        )
        if not any(tensor is not None for tensor in core_tensors):
            return
        if not all(tensor is not None for tensor in core_tensors):
            raise ValueError(
                "ExpertTransferPlan requires source_logical_expert_ids, "
                "dest_physical_expert_ids, dest_ranks, and dest_local_slots together."
            )

        assert self.source_logical_expert_ids is not None
        num_transfers = self.source_logical_expert_ids.numel()
        for name, tensor in (
            ("source_logical_expert_ids", self.source_logical_expert_ids),
            ("dest_physical_expert_ids", self.dest_physical_expert_ids),
            ("dest_ranks", self.dest_ranks),
            ("dest_local_slots", self.dest_local_slots),
            ("source_ranks", self.source_ranks),
            ("source_local_slots", self.source_local_slots),
        ):
            if tensor is None:
                continue
            _validate_1d_tensor(name, tensor)
            if tensor.numel() != num_transfers:
                raise ValueError(
                    f"Expected {name} to have {num_transfers} entries, got {tensor.numel()}."
                )

    @property
    def num_transfers(self) -> int:
        """Number of expert-copy tasks in this plan."""
        if self.source_logical_expert_ids is None:
            return 0
        return int(self.source_logical_expert_ids.numel())

    @property
    def is_empty(self) -> bool:
        """Whether the plan contains no expert-copy task."""
        return self.num_transfers == 0


@dataclass(frozen=True)
class ExpertReroutePlan:
    """Planner output describing the physical expert layout.

    Physical expert ids represent runtime slots, not logical expert ids.  Echo
    and UltraEP can use the tensor maps directly.  MoonEP can preserve native
    VM-layout state in ``native_plan`` while also filling these common maps when
    a direct physical-slot interpretation is available.
    """

    backend: str
    num_physical_experts: Optional[int] = None
    physical_to_logical_map: Optional[torch.Tensor] = None
    physical_to_rank_map: Optional[torch.Tensor] = None
    physical_to_local_slot_map: Optional[torch.Tensor] = None
    logical_to_physical_map: Optional[torch.Tensor] = None
    logical_replica_counts: Optional[torch.Tensor] = None
    transfer_plan: ExpertTransferPlan = field(default_factory=ExpertTransferPlan)
    native_plan: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_physical_experts is not None and self.num_physical_experts < 0:
            raise ValueError("num_physical_experts must be non-negative when provided.")

        for name, tensor in (
            ("physical_to_logical_map", self.physical_to_logical_map),
            ("physical_to_rank_map", self.physical_to_rank_map),
            ("physical_to_local_slot_map", self.physical_to_local_slot_map),
        ):
            if tensor is None:
                continue
            _validate_1d_tensor(name, tensor)
            if (
                self.num_physical_experts is not None
                and tensor.numel() != self.num_physical_experts
            ):
                raise ValueError(
                    f"Expected {name} to have {self.num_physical_experts} entries, "
                    f"got {tensor.numel()}."
                )

        if self.logical_to_physical_map is not None:
            _validate_2d_tensor("logical_to_physical_map", self.logical_to_physical_map)
        if self.logical_replica_counts is not None:
            _validate_1d_tensor("logical_replica_counts", self.logical_replica_counts)
            if (
                self.logical_to_physical_map is not None
                and self.logical_replica_counts.numel() != self.logical_to_physical_map.size(0)
            ):
                raise ValueError(
                    "Expected logical_replica_counts to match logical_to_physical_map rows, "
                    f"got {self.logical_replica_counts.numel()} and "
                    f"{self.logical_to_physical_map.size(0)}."
                )

    @classmethod
    def identity(cls, num_logical_experts: int) -> "ExpertReroutePlan":
        """Build a plan that leaves the default logical expert layout unchanged."""
        return cls(backend=IDENTITY_BACKEND, num_physical_experts=num_logical_experts)

    @property
    def resolved_num_physical_experts(self) -> Optional[int]:
        """Return the physical expert count if it is explicit or inferable."""
        if self.num_physical_experts is not None:
            return self.num_physical_experts
        for tensor in (
            self.physical_to_logical_map,
            self.physical_to_rank_map,
            self.physical_to_local_slot_map,
        ):
            if tensor is not None:
                return int(tensor.numel())
        return None

    @property
    def is_identity(self) -> bool:
        """Whether this plan keeps the original logical expert layout."""
        return (
            self.backend == IDENTITY_BACKEND
            and self.transfer_plan.is_empty
            and self.native_plan is None
        )


@dataclass(frozen=True)
class TokenReroutePlan:
    """Planner output describing token assignment after rerouting.

    Dense ``routing_map``/``probs`` keeps compatibility with Megatron token
    dispatchers.  Sparse ``logical_expert_ids``/``physical_expert_ids`` supports
    Echo and UltraEP-style physical slot assignment.  ``native_plan`` is reserved
    for backend-specific dispatch plans such as MoonEP communication plans.
    """

    backend: str
    routing_map: Optional[torch.Tensor] = None
    probs: Optional[torch.Tensor] = None
    logical_expert_ids: Optional[torch.Tensor] = None
    physical_expert_ids: Optional[torch.Tensor] = None
    assignment_probs: Optional[torch.Tensor] = None
    dispatch_indices: Optional[torch.Tensor] = None
    combine_indices: Optional[torch.Tensor] = None
    native_plan: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.routing_map is not None:
            _validate_2d_tensor("routing_map", self.routing_map)
            if self.routing_map.dtype != torch.bool:
                raise ValueError(f"Expected bool routing_map, got {self.routing_map.dtype}")
        if self.probs is not None:
            _validate_2d_tensor("probs", self.probs)
            if self.routing_map is not None and self.probs.shape != self.routing_map.shape:
                raise ValueError(
                    "Expected probs and routing_map to have the same shape, "
                    f"got {tuple(self.probs.shape)} and {tuple(self.routing_map.shape)}"
                )

        sparse_shape = None
        for name, tensor in (
            ("logical_expert_ids", self.logical_expert_ids),
            ("physical_expert_ids", self.physical_expert_ids),
            ("assignment_probs", self.assignment_probs),
        ):
            if tensor is None:
                continue
            _validate_2d_tensor(name, tensor)
            if sparse_shape is None:
                sparse_shape = tensor.shape
            elif tensor.shape != sparse_shape:
                raise ValueError(
                    f"Expected {name} to have sparse shape {tuple(sparse_shape)}, "
                    f"got {tuple(tensor.shape)}."
                )

        token_dims = []
        if self.routing_map is not None:
            token_dims.append(("routing_map", self.routing_map.size(0)))
        if self.logical_expert_ids is not None:
            token_dims.append(("logical_expert_ids", self.logical_expert_ids.size(0)))
        if self.physical_expert_ids is not None:
            token_dims.append(("physical_expert_ids", self.physical_expert_ids.size(0)))
        if self.dispatch_indices is not None:
            token_dims.append(("dispatch_indices", self.dispatch_indices.size(0)))
        if self.combine_indices is not None:
            token_dims.append(("combine_indices", self.combine_indices.size(0)))
        if token_dims and any(dim != token_dims[0][1] for _, dim in token_dims):
            details = ", ".join(f"{name}={dim}" for name, dim in token_dims)
            raise ValueError(f"Expected token reroute tensors to share token dimension: {details}.")

    @classmethod
    def identity(cls, route_info: RouteInfo) -> "TokenReroutePlan":
        """Build a plan that preserves the router's logical assignments."""
        return cls(
            backend=IDENTITY_BACKEND,
            routing_map=route_info.routing_map,
            probs=route_info.probs,
            logical_expert_ids=route_info.topk_ids,
            physical_expert_ids=route_info.topk_ids,
            assignment_probs=route_info.topk_probs,
        )

    @property
    def num_tokens(self) -> Optional[int]:
        """Return the token count if represented by any common tensor."""
        for tensor in (
            self.routing_map,
            self.logical_expert_ids,
            self.physical_expert_ids,
            self.dispatch_indices,
            self.combine_indices,
        ):
            if tensor is not None:
                return int(tensor.size(0))
        return None


@dataclass(frozen=True)
class MoEPlannerOutput:
    """Unified output that every MoE load planner must produce."""

    expert_plan: ExpertReroutePlan
    token_plan: TokenReroutePlan
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ExpertDispatchOutput:
    """Result returned after an expert dispatcher materializes an expert plan."""

    expert_plan: ExpertReroutePlan
    materialized_experts: Any = None
    event: Any = None
    handle: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

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
    """Base class for Echo, UltraEP, and MoonEP MoE load planners."""

    planner_name: ClassVar[str] = "abstract"

    def supports(self, context: SchedulerContext) -> bool:
        """Return whether the planner can run with the current context."""
        del context
        return True

    @abstractmethod
    def plan(self, route_info: RouteInfo, context: SchedulerContext) -> MoEPlannerOutput:
        """Return expert and token reroute plans for the current router output."""


class ExpertDispatch(torch.nn.Module, ABC):
    """Base class for Echo and UltraEP expert placement materialization."""

    dispatcher_name: ClassVar[str] = "abstract"
    supported_backends: ClassVar[frozenset[str]] = frozenset()

    def supports(self, expert_plan: ExpertReroutePlan, context: SchedulerContext) -> bool:
        """Return whether this dispatcher can materialize the given expert plan."""
        del context
        return not self.supported_backends or expert_plan.backend in self.supported_backends

    @abstractmethod
    def dispatch(
        self,
        experts: torch.nn.Module,
        expert_plan: ExpertReroutePlan,
        context: SchedulerContext,
    ) -> ExpertDispatchOutput:
        """Materialize the planned expert layout before token dispatch begins."""

    def finalize(self, output: ExpertDispatchOutput, context: SchedulerContext) -> None:
        """Release transient dispatch state after the MoE forward finishes."""
        del output, context


@dataclass(frozen=True)
class MoESchedulerOutput:
    """Combined result returned by ``MoEScheduler.schedule``."""

    planner_output: MoEPlannerOutput
    expert_dispatch_output: ExpertDispatchOutput

    @property
    def expert_plan(self) -> ExpertReroutePlan:
        """Expert reroute plan produced by the planner."""
        return self.planner_output.expert_plan

    @property
    def token_plan(self) -> TokenReroutePlan:
        """Token reroute plan produced by the planner."""
        return self.planner_output.token_plan

    def wait(self, stream: Any = None) -> None:
        """Wait for the expert-dispatch portion of the schedule."""
        self.expert_dispatch_output.wait(stream=stream)


class MoEScheduler(torch.nn.Module):
    """Small orchestrator that connects a planner to an expert dispatcher."""

    def __init__(self, planner: MoELoadPlanner, expert_dispatch: ExpertDispatch) -> None:
        super().__init__()
        self.planner = planner
        self.expert_dispatch = expert_dispatch

    def schedule(
        self,
        route_info: RouteInfo,
        experts: torch.nn.Module,
        context: SchedulerContext,
    ) -> MoESchedulerOutput:
        """Plan token/expert rerouting and materialize the expert side."""
        if not self.planner.supports(context):
            raise ValueError(
                f"Planner {self.planner.planner_name!r} does not support the current context."
            )

        planner_output = self.planner.plan(route_info, context)
        expert_plan = planner_output.expert_plan
        if not self.expert_dispatch.supports(expert_plan, context):
            raise ValueError(
                f"Expert dispatcher {self.expert_dispatch.dispatcher_name!r} "
                f"does not support planner backend {expert_plan.backend!r}."
            )

        dispatch_output = self.expert_dispatch.dispatch(experts, expert_plan, context)
        return MoESchedulerOutput(
            planner_output=planner_output, expert_dispatch_output=dispatch_output
        )

    def finalize(self, output: MoESchedulerOutput, context: SchedulerContext) -> None:
        """Finalize the expert-dispatch portion of a scheduled forward."""
        self.expert_dispatch.finalize(output.expert_dispatch_output, context)
