# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Backend-neutral interfaces for MoE expert and token rerouting.

MoEScheduler is intended to run after router output is available and before the
normal MoE token dispatcher starts.  This module only defines the shared
contracts:

* planners translate dense router output into final dense token reroute tensors
  plus an expert placement;
* expert dispatchers materialize the expert placement before token dispatch;
* backend-specific objects can be attached as native metadata without changing
  the common interface.

Concrete Echo, UltraEP, and MoonEP planners should produce ``MoEPlannerOutput``.
Concrete Echo and UltraEP expert dispatchers should consume ``ExpertPlacementPlan``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Optional, Sequence

import torch

IDENTITY_BACKEND = "identity"
ECHO_BACKEND = "echo"
ULTRA_EP_BACKEND = "ultra_ep"
MOON_EP_BACKEND = "moon_ep"


def _rank0_info(message: str) -> None:
    try:
        is_rank0 = (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
    except RuntimeError:
        is_rank0 = True
    if is_rank0:
        print(f"INFO:MoEScheduler: {message}", flush=True)


def _tensor_shape(tensor: Optional[torch.Tensor]) -> Optional[tuple[int, ...]]:
    return None if tensor is None else tuple(tensor.shape)


def _validate_1d_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dim() != 1:
        raise ValueError(f"Expected {name} to be 1D, got shape {tuple(tensor.shape)}")


def _validate_2d_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dim() != 2:
        raise ValueError(f"Expected {name} to be 2D, got shape {tuple(tensor.shape)}")


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
class ExpertPlacementPlan:
    """Physical expert placement to materialize before token dispatch.

    ``physical_to_logical_map`` describes the post-placement physical slots.
    Copy tasks are inlined as parallel 1D tensors; each row copies one logical
    expert into one destination physical slot.  Planner-specific details can be
    attached through ``metadata`` without becoming part of the common contract.
    """

    backend: str
    num_physical_experts: Optional[int] = None
    physical_to_logical_map: Optional[torch.Tensor] = None
    source_logical_expert_ids: Optional[torch.Tensor] = None
    dest_physical_expert_ids: Optional[torch.Tensor] = None
    dest_ranks: Optional[torch.Tensor] = None
    dest_local_slots: Optional[torch.Tensor] = None
    source_ranks: Optional[torch.Tensor] = None
    source_local_slots: Optional[torch.Tensor] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_physical_experts is not None and self.num_physical_experts < 0:
            raise ValueError("num_physical_experts must be non-negative when provided.")
        if self.physical_to_logical_map is not None:
            _validate_1d_tensor("physical_to_logical_map", self.physical_to_logical_map)
            if (
                self.num_physical_experts is not None
                and self.physical_to_logical_map.numel() != self.num_physical_experts
            ):
                raise ValueError(
                    "Expected physical_to_logical_map to have "
                    f"{self.num_physical_experts} entries, "
                    f"got {self.physical_to_logical_map.numel()}."
                )

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
                "ExpertPlacementPlan requires source_logical_expert_ids, "
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

    @classmethod
    def identity(
        cls, num_logical_experts: int, *, device: Optional[torch.device] = None
    ) -> "ExpertPlacementPlan":
        """Build a placement that leaves the default logical expert layout unchanged."""
        return cls(
            backend=IDENTITY_BACKEND,
            num_physical_experts=num_logical_experts,
            physical_to_logical_map=torch.arange(
                num_logical_experts, dtype=torch.long, device=device
            ),
        )

    @property
    def resolved_num_physical_experts(self) -> Optional[int]:
        """Return the physical expert count if it is explicit or inferable."""
        if self.num_physical_experts is not None:
            return self.num_physical_experts
        if self.physical_to_logical_map is not None:
            return int(self.physical_to_logical_map.numel())
        return None

    @property
    def num_transfers(self) -> int:
        """Number of expert-copy tasks in this placement."""
        if self.source_logical_expert_ids is None:
            return 0
        return int(self.source_logical_expert_ids.numel())

    @property
    def is_identity(self) -> bool:
        """Whether this placement keeps the original logical expert layout."""
        return self.backend == IDENTITY_BACKEND and self.num_transfers == 0


@dataclass(frozen=True)
class MoEPlannerOutput:
    """Unified planner output consumed by MoELayer and ExpertDispatch."""

    expert_placement: ExpertPlacementPlan
    routing_map: torch.Tensor
    probs: torch.Tensor

    def __post_init__(self) -> None:
        _validate_2d_tensor("routing_map", self.routing_map)
        _validate_2d_tensor("probs", self.probs)
        if self.routing_map.dtype != torch.bool:
            raise ValueError(f"Expected bool routing_map, got {self.routing_map.dtype}")
        if self.probs.shape != self.routing_map.shape:
            raise ValueError(
                "Expected probs and routing_map to have the same shape, "
                f"got {tuple(self.probs.shape)} and {tuple(self.routing_map.shape)}"
            )


class MoELoadPlanner(torch.nn.Module, ABC):
    """Base class for Echo, UltraEP, and MoonEP MoE load planners."""

    planner_name: ClassVar[str] = "abstract"

    def should_plan(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: Optional[torch.Tensor] = None,
    ) -> bool:
        """Return whether the planner should run for this router output."""
        del probs, routing_map, context, tokens_per_expert
        return True

    @abstractmethod
    def plan(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: Optional[torch.Tensor] = None,
    ) -> MoEPlannerOutput:
        """Return expert placement and token reroute tensors for the current router output."""


class ExpertDispatch(torch.nn.Module, ABC):
    """Base class for Echo and UltraEP expert placement materialization."""

    dispatcher_name: ClassVar[str] = "abstract"
    supported_backends: ClassVar[frozenset[str]] = frozenset()

    def supports(
        self, expert_placement: ExpertPlacementPlan, context: SchedulerContext
    ) -> bool:
        """Return whether this dispatcher can materialize the given expert placement."""
        del context
        return (
            not self.supported_backends
            or expert_placement.backend in self.supported_backends
        )

    @abstractmethod
    def dispatch(
        self,
        experts: torch.nn.Module,
        expert_placement: ExpertPlacementPlan,
        context: SchedulerContext,
    ) -> None:
        """Materialize the planned expert placement before token dispatch begins."""

    def finalize(self, context: SchedulerContext) -> None:
        """Release transient dispatch state after the MoE forward finishes."""
        del context


class MoEScheduler(torch.nn.Module):
    """Small orchestrator that connects a planner to an expert dispatcher."""

    _logged_config_signatures: ClassVar[set[tuple[str, str, int, str]]] = set()
    _logged_runtime_summary: ClassVar[bool] = False

    def __init__(self, planner: MoELoadPlanner, expert_dispatch: ExpertDispatch) -> None:
        super().__init__()
        self.planner = planner
        self.expert_dispatch = expert_dispatch

    @classmethod
    def from_config(
        cls,
        config: Any,
        pg_collection: Any,
        *,
        home_expert_indices: Sequence[int],
        idle_expert_indices: Sequence[int],
    ) -> "MoEScheduler":
        """Build the configured MoEScheduler backend stack."""
        planner_type = getattr(config, "moe_scheduler_planner_type", None)
        expert_dispatcher_type = getattr(config, "moe_scheduler_expert_dispatcher_type", None)
        if planner_type not in ("echo", "moon_ep"):
            raise ValueError(f"Unsupported MoEScheduler planner: {planner_type}")
        if expert_dispatcher_type != "hybridep":
            raise ValueError(
                f"Unsupported MoEScheduler expert dispatcher: {expert_dispatcher_type}"
            )

        num_idle_experts = getattr(config, "moe_scheduler_num_idle_experts", None)
        if num_idle_experts is None:
            raise ValueError(
                "moe_scheduler_num_idle_experts must be set when MoEScheduler is enabled."
            )
        assignment_algorithm = getattr(
            config, "moe_scheduler_assignment_algorithm", "approx_bin_packing"
        )

        from megatron.core.transformer.moe.echo_moe_scheduler import (
            EchoExpertDispatch,
            EchoLoadPlanner,
            HybridEPEchoExpertDispatchBackend,
        )

        if planner_type == "echo":
            planner = EchoLoadPlanner(
                num_idle_experts,
                assignment_algorithm=assignment_algorithm,
            )
        else:
            from megatron.core.transformer.moe.moonep_moe_scheduler import MoonEPLoadPlanner

            ep_size = getattr(config, "expert_model_parallel_size", 1)
            planner = MoonEPLoadPlanner(
                num_redundant_experts=num_idle_experts // ep_size,
            )
        hidden_size = (
            config.hidden_size
            if getattr(config, "moe_latent_size", None) is None
            else config.moe_latent_size
        )
        materializer = HybridEPEchoExpertDispatchBackend(
            config=config,
            pg_collection=pg_collection,
            num_idle_experts=num_idle_experts,
            hidden_size=hidden_size,
        )
        expert_dispatch = EchoExpertDispatch(
            materializer=materializer,
            home_expert_indices=home_expert_indices,
            idle_expert_indices=idle_expert_indices,
        )
        config_signature = (
            str(planner_type),
            str(expert_dispatcher_type),
            int(num_idle_experts),
            str(assignment_algorithm),
        )
        if config_signature not in cls._logged_config_signatures:
            cls._logged_config_signatures.add(config_signature)
            _rank0_info(
                "configured "
                f"planner={planner_type} "
                f"expert_dispatcher={expert_dispatcher_type} "
                f"num_idle_experts={num_idle_experts} "
                f"assignment_algorithm={assignment_algorithm}"
            )
        return cls(planner=planner, expert_dispatch=expert_dispatch)

    def _log_first_schedule(
        self,
        input_routing_map: torch.Tensor,
        planner_output: Optional[MoEPlannerOutput],
        context: SchedulerContext,
        *,
        planning_skipped: bool,
        dispatch_materialized: bool,
    ) -> None:
        if MoEScheduler._logged_runtime_summary:
            return
        MoEScheduler._logged_runtime_summary = True
        if planner_output is None:
            output_routing_map = input_routing_map
            expert_backend = IDENTITY_BACKEND
            num_physical_experts = context.num_logical_experts
            num_transfers = 0
            assignment_backend = None
            reroute_backend = "identity"
        else:
            expert_placement = planner_output.expert_placement
            output_routing_map = planner_output.routing_map
            expert_backend = expert_placement.backend
            num_physical_experts = expert_placement.resolved_num_physical_experts
            num_transfers = expert_placement.num_transfers
            assignment_backend = expert_placement.metadata.get('assignment_backend')
            reroute_backend = expert_placement.metadata.get('reroute_backend')
        materializer = getattr(self.expert_dispatch, "materializer", None)
        materializer_name = type(materializer).__name__ if materializer is not None else None
        _rank0_info(
            "first schedule completed "
            f"layer={context.layer_number} "
            f"training={context.training} "
            f"planner={self.planner.planner_name} "
            f"dispatcher={self.expert_dispatch.dispatcher_name} "
            f"materializer={materializer_name} "
            f"input_routing_map_shape={_tensor_shape(input_routing_map)} "
            f"output_routing_map_shape={_tensor_shape(output_routing_map)} "
            f"expert_backend={expert_backend} "
            f"num_physical_experts={num_physical_experts} "
            f"num_transfers={num_transfers} "
            f"assignment_backend={assignment_backend} "
            f"reroute_backend={reroute_backend} "
            f"planning_skipped={planning_skipped} "
            f"dispatch_materialized={dispatch_materialized}"
        )

    def schedule(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        experts: torch.nn.Module,
        context: SchedulerContext,
        *,
        tokens_per_expert: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Plan token/expert rerouting and materialize the expert side."""
        if not self.planner.should_plan(
            probs, routing_map, context, tokens_per_expert=tokens_per_expert
        ):
            self._log_first_schedule(
                routing_map,
                None,
                context,
                planning_skipped=True,
                dispatch_materialized=False,
            )
            return probs, routing_map

        planner_output = self.planner.plan(
            probs, routing_map, context, tokens_per_expert=tokens_per_expert
        )
        expert_placement = planner_output.expert_placement
        if not self.expert_dispatch.supports(expert_placement, context):
            raise ValueError(
                f"Expert dispatcher {self.expert_dispatch.dispatcher_name!r} "
                f"does not support planner output backend={expert_placement.backend!r}."
            )

        self.expert_dispatch.dispatch(experts, expert_placement, context)
        self._log_first_schedule(
            routing_map,
            planner_output,
            context,
            planning_skipped=False,
            dispatch_materialized=True,
        )
        return planner_output.probs, planner_output.routing_map

    def finalize(self, context: SchedulerContext) -> None:
        """Finalize the expert-dispatch portion of a scheduled forward."""
        self.expert_dispatch.finalize(context)
