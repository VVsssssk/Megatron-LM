# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.moe_scheduler import (
    ECHO_BACKEND,
    IDENTITY_BACKEND,
    MOON_EP_BACKEND,
    ULTRA_EP_BACKEND,
    ExpertDispatch,
    ExpertPlacementPlan,
    MoELoadPlanner,
    MoEPlannerOutput,
    MoEScheduler,
    SchedulerContext,
)


def _route_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.zeros(3, 4)
    routing_map = torch.zeros(3, 4, dtype=torch.bool)
    topk_ids = torch.tensor([[0, 1], [1, 2], [3, 0]])
    topk_probs = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]])
    routing_map.scatter_(1, topk_ids, True)
    probs.scatter_(1, topk_ids, topk_probs)
    return probs, routing_map, routing_map.sum(dim=0)


def _context() -> SchedulerContext:
    return SchedulerContext(
        layer_number=1,
        num_logical_experts=4,
        num_local_experts=2,
        local_expert_indices=(0, 1),
        ep_size=2,
        ep_rank=0,
        router_topk=2,
        training=True,
    )


def _physical_token_reroute(route_info: RouteInfo) -> dict:
    physical_ids = route_info.topk_ids.clone()
    physical_ids[0, 0] = 4
    physical_ids[2, 0] = 5
    routing_map = torch.zeros(route_info.num_tokens, 6, dtype=torch.bool)
    probs = torch.zeros(route_info.num_tokens, 6)
    routing_map.scatter_(1, physical_ids, True)
    probs.scatter_(1, physical_ids, route_info.topk_probs)
    return {
        "routing_map": routing_map,
        "probs": probs,
        "logical_expert_ids": route_info.topk_ids,
        "physical_expert_ids": physical_ids,
        "assignment_probs": route_info.topk_probs,
    }


def test_route_info_validates_dense_sparse_and_counts():
    route_info = _route_info()

    assert route_info.num_tokens == 3
    assert route_info.num_logical_experts == 4

    with pytest.raises(ValueError, match="Expected bool routing_map"):
        RouteInfo(probs=route_info.probs, routing_map=route_info.routing_map.long())

    with pytest.raises(ValueError, match="topk_probs requires topk_ids"):
        RouteInfo(
            probs=route_info.probs,
            routing_map=route_info.routing_map,
            topk_probs=route_info.topk_probs,
        )

    with pytest.raises(ValueError, match="tokens_per_expert"):
        RouteInfo(
            probs=route_info.probs,
            routing_map=route_info.routing_map,
            tokens_per_expert=torch.ones(3),
        )


def test_identity_plans_preserve_route_info():
    route_info = _route_info()
    expert_placement = ExpertPlacementPlan.identity(route_info.num_logical_experts)
    planner_output = MoEPlannerOutput(
        expert_placement=expert_placement,
        routing_map=route_info.routing_map,
        probs=route_info.probs,
        logical_expert_ids=route_info.topk_ids,
        physical_expert_ids=route_info.topk_ids,
        assignment_probs=route_info.topk_probs,
    )

    assert expert_placement.backend == IDENTITY_BACKEND
    assert expert_placement.resolved_num_physical_experts == route_info.num_logical_experts
    assert expert_placement.is_identity
    assert planner_output.routing_map is route_info.routing_map
    assert planner_output.probs is route_info.probs
    assert planner_output.physical_expert_ids is route_info.topk_ids


def test_unified_planner_output_can_hold_echo_ultraep_and_moonep_shapes():
    route_info = _route_info()
    physical_to_logical = torch.tensor([0, 1, 2, 3, 0, 3])
    source_logical_expert_ids = torch.tensor([0, 3])
    dest_physical_expert_ids = torch.tensor([4, 5])
    dest_ranks = torch.tensor([1, 1])
    dest_local_slots = torch.tensor([0, 1])

    planner_output = MoEPlannerOutput(
        expert_placement=ExpertPlacementPlan(
            backend=ECHO_BACKEND,
            num_physical_experts=6,
            physical_to_logical_map=physical_to_logical,
            source_logical_expert_ids=source_logical_expert_ids,
            dest_physical_expert_ids=dest_physical_expert_ids,
            dest_ranks=dest_ranks,
            dest_local_slots=dest_local_slots,
        ),
        **_physical_token_reroute(route_info),
    )

    ultraep_output = MoEPlannerOutput(
        expert_placement=ExpertPlacementPlan(
            backend=ULTRA_EP_BACKEND,
            num_physical_experts=6,
            physical_to_logical_map=physical_to_logical,
            source_logical_expert_ids=source_logical_expert_ids,
            dest_physical_expert_ids=dest_physical_expert_ids,
            dest_ranks=dest_ranks,
            dest_local_slots=dest_local_slots,
            metadata={"logical_instance_quota_prefix": torch.arange(7)},
        ),
        **_physical_token_reroute(route_info),
    )

    moonep_output = MoEPlannerOutput(
        expert_placement=ExpertPlacementPlan(
            backend=MOON_EP_BACKEND,
            metadata={"experts_to_copy": torch.tensor([[0, 3], [1, 2]])},
        ),
        **_physical_token_reroute(route_info),
        metadata={"moon_ep_dst": torch.arange(6).reshape(3, 2)},
    )

    assert planner_output.expert_placement.num_transfers == 2
    assert planner_output.routing_map.shape == (3, 6)
    assert ultraep_output.expert_placement.metadata["logical_instance_quota_prefix"].numel() == 7
    assert moonep_output.expert_placement.metadata["experts_to_copy"].shape == (2, 2)
    assert moonep_output.num_tokens == route_info.num_tokens
    assert moonep_output.metadata["moon_ep_dst"].shape == (3, 2)


class _EchoStylePlanner(MoELoadPlanner):
    planner_name = "echo-test"

    def plan(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: torch.Tensor | None = None,
    ) -> MoEPlannerOutput:
        del tokens_per_expert
        route_info = SimpleNamespace(
            probs=probs,
            routing_map=routing_map,
            topk_ids=torch.nonzero(routing_map, as_tuple=False)[:, 1].reshape(
                routing_map.size(0), -1
            ),
            num_tokens=routing_map.size(0),
            num_logical_experts=routing_map.size(1),
        )
        route_info.topk_probs = torch.gather(probs, 1, route_info.topk_ids)
        del context
        return MoEPlannerOutput(
            expert_placement=ExpertPlacementPlan(
                backend=ECHO_BACKEND,
                num_physical_experts=6,
                physical_to_logical_map=torch.tensor([0, 1, 2, 3, 0, 3]),
                source_logical_expert_ids=torch.tensor([0, 3]),
                dest_physical_expert_ids=torch.tensor([4, 5]),
                dest_ranks=torch.tensor([1, 1]),
                dest_local_slots=torch.tensor([0, 1]),
            ),
            **_physical_token_reroute(route_info),
        )


class _SkipPlanner(MoELoadPlanner):
    planner_name = "skip-test"

    def __init__(self) -> None:
        super().__init__()
        self.plan_called = False

    def should_plan(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: torch.Tensor | None = None,
    ) -> bool:
        del probs, routing_map, context, tokens_per_expert
        return False

    def plan(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: torch.Tensor | None = None,
    ) -> MoEPlannerOutput:
        del probs, routing_map, context, tokens_per_expert
        self.plan_called = True
        raise AssertionError("plan should not be called when should_plan returns False.")


class _RecordingDispatch(ExpertDispatch):
    dispatcher_name = "recording-test"
    supported_backends = frozenset({ECHO_BACKEND})

    def __init__(self) -> None:
        super().__init__()
        self.dispatched_placement = None
        self.materialized_experts = None
        self.finalized = False
        self.supports_called = False

    def dispatch(
        self,
        experts: torch.nn.Module,
        expert_placement: ExpertPlacementPlan,
        context: SchedulerContext,
    ) -> None:
        del context
        self.dispatched_placement = expert_placement
        self.materialized_experts = experts

    def supports(
        self, expert_placement: ExpertPlacementPlan, context: SchedulerContext
    ) -> bool:
        self.supports_called = True
        return super().supports(expert_placement, context)

    def finalize(self, context: SchedulerContext) -> None:
        del context
        self.finalized = True


class _UltraEPOnlyDispatch(ExpertDispatch):
    dispatcher_name = "ultraep-only-test"
    supported_backends = frozenset({ULTRA_EP_BACKEND})

    def dispatch(
        self,
        experts: torch.nn.Module,
        expert_placement: ExpertPlacementPlan,
        context: SchedulerContext,
    ) -> None:
        del experts, expert_placement, context


def test_scheduler_passes_expert_placement_to_matching_dispatcher():
    probs, routing_map, tokens_per_expert = _route_inputs()
    context = _context()
    dispatch = _RecordingDispatch()
    scheduler = MoEScheduler(planner=_EchoStylePlanner(), expert_dispatch=dispatch)
    experts = torch.nn.Identity()

    output_probs, output_routing_map = scheduler.schedule(
        probs, routing_map, experts, context, tokens_per_expert=tokens_per_expert
    )

    assert dispatch.dispatched_placement.backend == ECHO_BACKEND
    assert dispatch.materialized_experts is experts
    assert output_probs.shape == output_routing_map.shape

    scheduler.finalize(context)

    assert dispatch.finalized


def test_scheduler_skips_planner_and_dispatch_when_should_plan_is_false():
    probs, routing_map, tokens_per_expert = _route_inputs()
    context = _context()
    planner = _SkipPlanner()
    dispatch = _RecordingDispatch()
    scheduler = MoEScheduler(planner=planner, expert_dispatch=dispatch)
    experts = torch.nn.Identity()

    output_probs, output_routing_map = scheduler.schedule(
        probs, routing_map, experts, context, tokens_per_expert=tokens_per_expert
    )

    assert not planner.plan_called
    assert not dispatch.supports_called
    assert dispatch.dispatched_placement is None
    assert output_routing_map is routing_map
    assert output_probs is probs


def test_scheduler_rejects_dispatcher_that_does_not_support_planner_backend():
    scheduler = MoEScheduler(
        planner=_EchoStylePlanner(), expert_dispatch=_UltraEPOnlyDispatch()
    )

    with pytest.raises(ValueError, match="does not support planner output backend='echo'"):
        probs, routing_map, tokens_per_expert = _route_inputs()
        scheduler.schedule(
            probs,
            routing_map,
            torch.nn.Identity(),
            _context(),
            tokens_per_expert=tokens_per_expert,
        )
