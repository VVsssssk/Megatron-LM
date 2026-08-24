# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe.moe_scheduler import (
    AutoLoadPlanner,
    DispatchHandle,
    ExpertDispatch,
    ExpertPlacementPlan,
    L1LoadPlanner,
    L2LoadPlanner,
    MoEScheduler,
    PlannerResult,
    RouteInfo,
    SchedulerContext,
)
from megatron.core.transformer.transformer_config import TransformerConfig


def _moe_config(**overrides):
    defaults = dict(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_moe_experts=4,
        moe_ffn_hidden_size=16,
        add_bias_linear=False,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def test_disabled_scheduler_returns_original_route_tensors():
    config = _moe_config()
    scheduler = MoEScheduler(
        config=config,
        num_local_experts=4,
        local_expert_indices=(0, 1, 2, 3),
        pg_collection=None,
    )
    probs = torch.randn(3, 4)
    routing_map = torch.zeros(3, 4, dtype=torch.bool)
    routing_map[:, 0] = True

    out_probs, out_routing_map = scheduler.schedule(
        probs,
        routing_map,
        hidden_states=None,
        experts=torch.nn.Identity(),
        layer_number=1,
        training=True,
    )

    assert out_probs is probs
    assert out_routing_map is routing_map
    assert scheduler.active_dispatch_handle is None


class _L1Planner(L1LoadPlanner):
    def plan(
        self,
        route_info: RouteInfo,
        context: SchedulerContext,
        base_plan: ExpertPlacementPlan | None = None,
    ) -> PlannerResult:
        del context
        assert base_plan is None or base_plan.is_identity
        return PlannerResult(
            probs=route_info.probs,
            routing_map=route_info.routing_map,
            placement_plan=ExpertPlacementPlan(backend="l1-test"),
        )


class _L2Planner(L2LoadPlanner):
    def plan(
        self,
        route_info: RouteInfo,
        context: SchedulerContext,
        base_plan: ExpertPlacementPlan | None = None,
    ) -> PlannerResult:
        del context
        assert base_plan is not None
        assert base_plan.backend == "l1-test"
        probs = torch.zeros_like(route_info.probs)
        routing_map = torch.zeros_like(route_info.routing_map)
        probs[:, 1] = 1.0
        routing_map[:, 1] = True
        return PlannerResult(
            probs=probs,
            routing_map=routing_map,
            placement_plan=ExpertPlacementPlan(
                backend="l2-test", backend_metadata={"base_backend": base_plan.backend}
            ),
        )


class _RecordingDispatch(ExpertDispatch):
    def __init__(self):
        super().__init__()
        self.prepared_plan = None
        self.finalized_plan = None

    def prepare(
        self,
        experts: torch.nn.Module,
        placement_plan: ExpertPlacementPlan,
        context: SchedulerContext,
    ) -> DispatchHandle:
        del experts, context
        self.prepared_plan = placement_plan
        return DispatchHandle(placement_plan=placement_plan)

    def finalize(self, handle: DispatchHandle, context: SchedulerContext) -> None:
        del context
        self.finalized_plan = handle.placement_plan


def test_auto_scheduler_composes_l1_l2_and_dispatch():
    config = _moe_config(moe_enable_scheduler=True, moe_load_planner_type="auto")
    scheduler = MoEScheduler(
        config=config,
        num_local_experts=4,
        local_expert_indices=(0, 1, 2, 3),
        pg_collection=None,
    )
    dispatch = _RecordingDispatch()
    scheduler.load_planner = AutoLoadPlanner(
        l1_planner=_L1Planner(), l2_planner=_L2Planner()
    )
    scheduler.expert_dispatch = dispatch
    probs = torch.randn(3, 4)
    routing_map = torch.zeros(3, 4, dtype=torch.bool)
    routing_map[:, 0] = True

    out_probs, out_routing_map = scheduler.schedule(
        probs,
        routing_map,
        hidden_states=None,
        experts=torch.nn.Identity(),
        layer_number=1,
        training=True,
    )

    assert torch.equal(out_probs[:, 1], torch.ones(3))
    assert out_routing_map[:, 1].all()
    assert dispatch.prepared_plan is scheduler.cached_plan
    assert scheduler.cached_plan.backend == "l2-test"
    assert scheduler.active_dispatch_handle is not None

    scheduler.finish_forward()

    assert scheduler.active_dispatch_handle is None
    assert dispatch.finalized_plan is scheduler.cached_plan


def test_route_info_validates_shapes_and_dtype():
    probs = torch.randn(2, 4)
    routing_map = torch.zeros(2, 4, dtype=torch.int64)

    with pytest.raises(ValueError, match="Expected bool routing_map"):
        RouteInfo(probs=probs, routing_map=routing_map)


def test_scheduler_config_rejects_unregistered_dispatch_backend():
    with pytest.raises(ValueError, match="moe_expert_dispatcher_type must be 'none'"):
        _moe_config(moe_enable_scheduler=True, moe_expert_dispatcher_type="ultra_ep")
