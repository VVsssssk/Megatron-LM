# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Backend-neutral contracts for planning and materializing MoE expert placements.

This module has no dependency on a particular transport or MoE execution path.
Concrete planners and dispatchers can therefore be introduced independently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Literal

import torch

PlannerType = Literal["echo", "eplb", "moon_ep", "ultra_ep"]
TransportType = Literal["replica_nccl", "replica_peer_tma"]
ReplicaStorageType = Literal["shared", "layerwise"]
MXFP8StorageType = Literal["reuse", "separate"]


@dataclass(frozen=True)
class MoESchedulerConfig:
    """Backend selection and placement size, independent of TransformerConfig.

    The training configuration will construct this value when a concrete
    scheduler is wired into MoELayer. Defining it here does not enable a new
    execution path or change the existing MoonEP load balancer.
    """

    planner: PlannerType
    transport: TransportType
    num_logical_experts: int
    num_idle_experts: int
    ep_size: int
    replica_storage: ReplicaStorageType = "shared"
    replica_mxfp8_storage: MXFP8StorageType = "reuse"
    replica_exchange_buffer_mb: int = 256
    grad_reduce_in_bf16: bool = False
    num_sms: int | None = None

    def __post_init__(self) -> None:
        if self.planner not in ("echo", "eplb", "moon_ep", "ultra_ep"):
            raise ValueError(f"Unsupported MoE scheduler planner: {self.planner}")
        if self.transport not in ("replica_nccl", "replica_peer_tma"):
            raise ValueError(f"Unsupported MoE scheduler transport: {self.transport}")
        if self.ep_size <= 1:
            raise ValueError("MoE scheduler requires expert parallel size greater than one.")
        if self.num_logical_experts <= 0 or self.num_logical_experts % self.ep_size:
            raise ValueError("Logical expert count must be positive and divisible by EP size.")
        if self.num_idle_experts < 0 or self.num_idle_experts % self.ep_size:
            raise ValueError("Replica slot count must be nonnegative and divisible by EP size.")
        if self.planner == "moon_ep" and self.num_idle_experts != self.num_logical_experts:
            raise ValueError("MoonEP requires one replica slot per logical expert.")
        if self.replica_storage not in ("shared", "layerwise"):
            raise ValueError("Replica storage must be shared or layerwise.")
        if self.replica_mxfp8_storage not in ("reuse", "separate"):
            raise ValueError("MXFP8 replica storage must be reuse or separate.")
        if self.replica_exchange_buffer_mb <= 0:
            raise ValueError("Replica exchange buffer size must be positive.")
        if self.num_sms is not None and self.num_sms <= 0:
            raise ValueError("Replica transport SM count must be positive.")


@dataclass(frozen=True)
class SchedulerContext:
    """Layer and EP topology supplied to a planner for one forward."""

    layer_number: int | None
    num_logical_experts: int
    num_local_experts: int
    local_expert_indices: tuple[int, ...]
    ep_size: int
    ep_rank: int
    router_topk: int
    training: bool

    def __post_init__(self) -> None:
        if self.ep_size <= 0 or not 0 <= self.ep_rank < self.ep_size:
            raise ValueError("Invalid expert parallel rank or size.")
        if self.num_logical_experts <= 0 or self.num_local_experts <= 0:
            raise ValueError("Expert counts must be positive.")
        if len(self.local_expert_indices) != self.num_local_experts:
            raise ValueError("Local expert indices must match the local expert count.")
        if self.router_topk <= 0 or self.router_topk > self.num_logical_experts:
            raise ValueError("Router top-k must be within the logical expert count.")


@dataclass(frozen=True)
class MoEPlacementResult:
    """Planner-private state plus optional upper bounds for token dispatch.

    A subclass may carry tensors needed for rerouting. Capacity bounds are
    Python integers so dispatchers can allocate without synchronizing a GPU.
    """

    max_dispatch_routes_per_rank: int | None = None
    max_dispatch_padded_tokens_per_rank: int | None = None

    def __post_init__(self) -> None:
        for bound in (self.max_dispatch_routes_per_rank, self.max_dispatch_padded_tokens_per_rank):
            if bound is not None and bound < 0:
                raise ValueError("Dispatch capacity bounds must be nonnegative.")


class MoELoadPlanner(torch.nn.Module, ABC):
    """Choose physical slots and reroute tokens without owning expert weights."""

    planner_name: ClassVar[str] = "abstract"

    @abstractmethod
    def update_placement(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, MoEPlacementResult]:
        """Return [EP, local replica slots] source slots and reroute state."""

    @abstractmethod
    def reroute(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        placement_result: MoEPlacementResult,
        context: SchedulerContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return physical routing map and probabilities, in that order."""


class ExpertDispatch(torch.nn.Module, ABC):
    """Materialize planned expert replicas before normal token dispatch."""

    dispatcher_name: ClassVar[str] = "abstract"

    def supports(self, placement: torch.Tensor, context: SchedulerContext) -> bool:
        """Whether this dispatcher can materialize the planner's placement."""
        del placement, context
        return True

    @abstractmethod
    def dispatch(
        self, experts: torch.nn.Module, placement: torch.Tensor, context: SchedulerContext
    ) -> None:
        """Start materializing physical slots for the current forward."""

    def finalize(self, context: SchedulerContext) -> None:
        """Release transient per-forward state."""
        del context

    def bind_experts(self, experts: torch.nn.Module) -> None:
        """Bind persistent expert weights before the first scheduled forward."""
        del experts

    def assert_idle(self) -> None:
        """Reject state changes while a forward or backward still owns a plan."""

    def wrap_layer_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Attach the outermost gradient and plan-lifetime boundary."""
        return hidden_states

    def before_token_dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Attach work whose backward runs after token dispatch backward."""
        return hidden_states

    def after_token_dispatch(self, dispatched_hidden: torch.Tensor) -> torch.Tensor:
        """Attach work between token dispatch and expert compute."""
        return dispatched_hidden

    def before_token_combine(self, expert_output: torch.Tensor) -> torch.Tensor:
        """Attach work whose backward runs before expert backward."""
        return expert_output

    def after_token_combine(self, combined_hidden: torch.Tensor) -> torch.Tensor:
        """Finish the current forward plan scope after token combine."""
        return combined_hidden


class MoEScheduler(torch.nn.Module):
    """Sequence a planner and an expert dispatcher around existing token dispatch."""

    def __init__(self, planner: MoELoadPlanner, expert_dispatch: ExpertDispatch) -> None:
        super().__init__()
        self.planner = planner
        self.expert_dispatch = expert_dispatch
        self.max_dispatch_routes_per_rank: int | None = None
        self.max_dispatch_padded_tokens_per_rank: int | None = None

    def schedule(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        experts: torch.nn.Module,
        context: SchedulerContext,
        *,
        tokens_per_expert: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize placement, then return physical probabilities and routes."""
        if probs.ndim != 2 or routing_map.ndim != 2 or probs.shape != routing_map.shape:
            raise ValueError("Scheduler inputs must have matching 2D shapes.")
        self.max_dispatch_routes_per_rank = None
        self.max_dispatch_padded_tokens_per_rank = None
        placement, result = self.planner.update_placement(
            probs, routing_map, context, tokens_per_expert=tokens_per_expert
        )
        num_physical_experts = context.num_logical_experts
        if placement is not None:
            if (
                placement.ndim != 2
                or placement.shape[0] != context.ep_size
                or placement.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("Expected integer replica placement [EP, local slots].")
            if not self.expert_dispatch.supports(placement, context):
                raise ValueError("Expert dispatcher does not support planner output.")
            self.expert_dispatch.dispatch(experts, placement, context)
            num_physical_experts += placement.numel()

        physical_routes, physical_probs = self.planner.reroute(probs, routing_map, result, context)
        if (
            physical_routes.ndim != 2
            or physical_probs.ndim != 2
            or physical_routes.shape != physical_probs.shape
            or physical_routes.shape[0] != routing_map.shape[0]
        ):
            raise ValueError("Physical probabilities and routing map must have the same shape.")
        if physical_routes.dtype == torch.bool:
            if physical_routes.shape[1] != num_physical_experts:
                raise ValueError("Dense routing map width must equal physical expert count.")
        elif physical_routes.dtype not in (torch.int16, torch.int32, torch.int64):
            raise ValueError("Physical routing map must contain bool or integer expert IDs.")
        self.max_dispatch_routes_per_rank = result.max_dispatch_routes_per_rank
        self.max_dispatch_padded_tokens_per_rank = result.max_dispatch_padded_tokens_per_rank
        return physical_probs, physical_routes

    def finalize(self, context: SchedulerContext) -> None:
        """Finish dispatcher work for the current forward."""
        self.expert_dispatch.finalize(context)

    def bind_experts(self, experts: torch.nn.Module) -> None:
        """Bind expert parameters to the selected dispatcher."""
        self.expert_dispatch.bind_experts(experts)

    def wrap_layer_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.expert_dispatch.wrap_layer_input(hidden_states)

    def before_token_dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.expert_dispatch.before_token_dispatch(hidden_states)

    def after_token_dispatch(self, dispatched_hidden: torch.Tensor) -> torch.Tensor:
        return self.expert_dispatch.after_token_dispatch(dispatched_hidden)

    def before_token_combine(self, expert_output: torch.Tensor) -> torch.Tensor:
        return self.expert_dispatch.before_token_combine(expert_output)

    def after_token_combine(self, combined_hidden: torch.Tensor) -> torch.Tensor:
        return self.expert_dispatch.after_token_combine(combined_hidden)
