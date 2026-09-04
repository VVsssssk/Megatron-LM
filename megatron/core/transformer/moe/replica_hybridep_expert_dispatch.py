# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""ReplicaWeightBridge adapter for the backend-neutral MoEScheduler contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from megatron.core.transformer.moe.moe_scheduler import ExpertDispatch, SchedulerContext
from megatron.core.transformer.moe.replica_planner import (
    ReplicaPlan,
    ReplicaWeightBridge,
    start_replica_grad_reduce_after_expert_backward,
    start_replica_weight_prefetch_before_combine_backward,
    wait_replica_grad_reduce_after_dispatch_backward,
    wait_replica_weight_prefetch_before_expert_backward,
)


@dataclass
class _ReplicaPlanSlot:
    """Stable placement storage retained through one forward/backward lifetime."""

    experts_to_copy: torch.Tensor
    plan: Optional[ReplicaPlan] = None
    in_use: bool = False
    lifetime_tracked: bool = False


class _ReplicaPlanLifetime(torch.autograd.Function):
    """Release a placement slot after dispatch backward has consumed the plan."""

    @staticmethod
    def forward(ctx, hidden_states, *args):
        ctx.dispatcher, ctx.slot = args[-2:]
        ctx.num_source_parameters = len(args) - 2
        return hidden_states

    @staticmethod
    def backward(ctx, grad_hidden_states):
        ctx.dispatcher._release_plan_slot(ctx.slot)
        return grad_hidden_states, *([None] * (ctx.num_source_parameters + 2))


class ReplicaHybridEPExpertDispatch(ExpertDispatch):
    """Materialize an ``E + R`` layout with PR #6892's ReplicaWeightBridge.

    The public placement is rank-major and contains each rank's native slots
    followed by its replica slots. This adapter lowers only the replica suffix
    to the bridge's ``experts_to_copy[rank, slot]`` input. Weight
    push and replica-gradient reduction remain in the original bridge.
    """

    dispatcher_name = "replica_hybridep"

    def __init__(self, *, config, pg_collection) -> None:
        super().__init__()
        self.config = config
        self.group = pg_collection.ep
        self.num_experts = int(config.num_moe_experts)
        self.ep_size = int(config.expert_model_parallel_size)
        self.num_replica_slots = int(config.moe_scheduler_num_idle_experts)
        self.num_local_home_experts = self.num_experts // self.ep_size
        self.num_local_replica_slots = self.num_replica_slots // self.ep_size
        self.num_local_runtime_experts = self.num_local_home_experts + self.num_local_replica_slots
        self.bridge: Optional[ReplicaWeightBridge] = None
        self._plan_slots: list[_ReplicaPlanSlot] = []
        self._active_plan_slot: Optional[_ReplicaPlanSlot] = None
        self._active_plan: Optional[ReplicaPlan] = None

    def bind_experts(self, experts: torch.nn.Module) -> None:
        """Bind native expert parameters before the first scheduled forward."""
        if self.bridge is not None:
            raise RuntimeError("Replica-HybridEP experts were already bound.")
        self.bridge = ReplicaWeightBridge(
            experts=experts,
            group=self.group,
            num_experts=self.num_experts,
            num_local_home_experts=self.num_local_home_experts,
            num_local_replica_slots=self.num_local_replica_slots,
            grad_dtype=torch.bfloat16 if self.config.grad_reduce_in_bf16 else torch.float32,
            num_sms=self.config.moe_flex_dispatcher_num_sms,
        )
        experts.set_replica_weight_bridge(self.bridge)

    def supports(self, physical_to_logical_map: torch.Tensor, context: SchedulerContext) -> bool:
        return (
            physical_to_logical_map.dim() == 1
            and context.num_logical_experts == self.num_experts
            and context.ep_size == self.ep_size
            and physical_to_logical_map.numel() == self.num_experts + self.num_replica_slots
        )

    def _acquire_plan_slot(self, device: torch.device) -> _ReplicaPlanSlot:
        for slot in self._plan_slots:
            if not slot.in_use:
                slot.in_use = True
                slot.lifetime_tracked = False
                return slot
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Replica-HybridEP needs another in-flight placement slot during CUDA graph "
                "capture. Warm up the same number of outstanding forwards before capture."
            )
        slot = _ReplicaPlanSlot(
            experts_to_copy=torch.empty(
                (self.ep_size, self.num_local_replica_slots), dtype=torch.int32, device=device
            ),
            in_use=True,
        )
        self._plan_slots.append(slot)
        return slot

    def _release_plan_slot(self, slot: _ReplicaPlanSlot) -> None:
        if not slot.in_use:
            raise RuntimeError("Replica-HybridEP placement slot was released twice.")
        slot.plan = None
        slot.in_use = False
        slot.lifetime_tracked = False

    def dispatch(
        self,
        experts: torch.nn.Module,
        physical_to_logical_map: torch.Tensor,
        context: SchedulerContext,
    ) -> None:
        """Start asynchronous weight prefetch for the common physical layout."""
        del experts
        if self.bridge is None:
            raise RuntimeError("Replica-HybridEP experts must be bound before dispatch.")
        if self._active_plan is not None or self._active_plan_slot is not None:
            raise RuntimeError(
                "Replica-HybridEP requires the previous token combine to finish before dispatch."
            )
        if not self.supports(physical_to_logical_map, context):
            raise ValueError(
                "ReplicaWeightBridge requires a rank-major E+R placement matching its "
                "configured replica slots."
            )

        slot = self._acquire_plan_slot(physical_to_logical_map.device)
        rank_layout = physical_to_logical_map.reshape(self.ep_size, self.num_local_runtime_experts)
        slot.experts_to_copy.copy_(rank_layout[:, self.num_local_home_experts :])
        plan = ReplicaPlan(
            virtual_experts=physical_to_logical_map, experts_to_copy=slot.experts_to_copy
        )
        slot.plan = plan
        self._active_plan_slot = slot
        self._active_plan = plan
        try:
            self.bridge.last_plan = plan
            self.bridge.start_prefetch(plan)
        except Exception:
            self._active_plan_slot = None
            self._active_plan = None
            self._release_plan_slot(slot)
            raise

    def before_token_dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Install the dispatch-backward gradient-reduction boundary."""
        if self._active_plan is None or not torch.is_grad_enabled():
            return hidden_states
        slot = self._active_plan_slot
        if slot is None or slot.plan is not self._active_plan:
            raise RuntimeError("Replica-HybridEP lost its active placement slot.")
        hidden_states = _ReplicaPlanLifetime.apply(
            hidden_states, *self.bridge.source_parameters, self, slot
        )
        slot.lifetime_tracked = hidden_states.requires_grad
        return wait_replica_grad_reduce_after_dispatch_backward(
            hidden_states, self.bridge, self._active_plan
        )

    def after_token_dispatch(self, dispatched_hidden: torch.Tensor) -> torch.Tensor:
        """Start replica-gradient reduction after expert backward."""
        if self._active_plan is None:
            return dispatched_hidden
        return start_replica_grad_reduce_after_expert_backward(
            dispatched_hidden, self.bridge, self._active_plan
        )

    def before_token_combine(self, expert_output: torch.Tensor) -> torch.Tensor:
        """Wait for backward-direction weights immediately before expert backward."""
        if self._active_plan is None:
            return expert_output
        return wait_replica_weight_prefetch_before_expert_backward(
            expert_output, self.bridge, self._active_plan
        )

    def after_token_combine(self, combined_hidden: torch.Tensor) -> torch.Tensor:
        """Start backward weight prefetch and finish the forward plan scope."""
        plan = self._active_plan
        slot = self._active_plan_slot
        if plan is None or slot is None or slot.plan is not plan:
            return combined_hidden
        if torch.is_grad_enabled() and combined_hidden.requires_grad:
            combined_hidden = start_replica_weight_prefetch_before_combine_backward(
                combined_hidden, self.bridge, plan
            )
        self._active_plan = None
        self._active_plan_slot = None
        if not slot.lifetime_tracked:
            self._release_plan_slot(slot)
        return combined_hidden
