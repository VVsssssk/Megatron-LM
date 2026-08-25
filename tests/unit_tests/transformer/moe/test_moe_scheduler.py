# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe.moe_scheduler import (
    ECHO_BACKEND,
    IDENTITY_BACKEND,
    MOON_EP_BACKEND,
    ULTRA_EP_BACKEND,
    ExpertDispatch,
    ExpertDispatchOutput,
    ExpertReroutePlan,
    ExpertTransferPlan,
    MoELoadPlanner,
    MoEPlannerOutput,
    MoEScheduler,
    RouteInfo,
    SchedulerContext,
    TokenReroutePlan,
)


def _route_info() -> RouteInfo:
    probs = torch.zeros(3, 4)
    routing_map = torch.zeros(3, 4, dtype=torch.bool)
    topk_ids = torch.tensor([[0, 1], [1, 2], [3, 0]])
    topk_probs = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]])
    routing_map.scatter_(1, topk_ids, True)
    probs.scatter_(1, topk_ids, topk_probs)
    return RouteInfo(
        probs=probs,
        routing_map=routing_map,
        topk_ids=topk_ids,
        topk_probs=topk_probs,
        tokens_per_expert=routing_map.sum(dim=0),
    )


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


def _physical_token_plan(route_info: RouteInfo, backend: str) -> TokenReroutePlan:
    physical_ids = route_info.topk_ids.clone()
    physical_ids[0, 0] = 4
    physical_ids[2, 0] = 5
    routing_map = torch.zeros(route_info.num_tokens, 6, dtype=torch.bool)
    probs = torch.zeros(route_info.num_tokens, 6)
    routing_map.scatter_(1, physical_ids, True)
    probs.scatter_(1, physical_ids, route_info.topk_probs)
    return TokenReroutePlan(
        backend=backend,
        routing_map=routing_map,
        probs=probs,
        logical_expert_ids=route_info.topk_ids,
        physical_expert_ids=physical_ids,
        assignment_probs=route_info.topk_probs,
    )


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
    expert_plan = ExpertReroutePlan.identity(route_info.num_logical_experts)
    token_plan = TokenReroutePlan.identity(route_info)

    assert expert_plan.backend == IDENTITY_BACKEND
    assert expert_plan.resolved_num_physical_experts == route_info.num_logical_experts
    assert expert_plan.is_identity
    assert token_plan.routing_map is route_info.routing_map
    assert token_plan.probs is route_info.probs
    assert token_plan.physical_expert_ids is route_info.topk_ids


def test_unified_planner_output_can_hold_echo_ultraep_and_moonep_shapes():
    route_info = _route_info()
    physical_to_logical = torch.tensor([0, 1, 2, 3, 0, 3])
    logical_to_physical = torch.tensor([[0, 4], [1, -1], [2, -1], [3, 5]])
    replica_counts = torch.tensor([2, 1, 1, 2])
    transfer_plan = ExpertTransferPlan(
        source_logical_expert_ids=torch.tensor([0, 3]),
        dest_physical_expert_ids=torch.tensor([4, 5]),
        dest_ranks=torch.tensor([1, 1]),
        dest_local_slots=torch.tensor([0, 1]),
    )

    echo_output = MoEPlannerOutput(
        expert_plan=ExpertReroutePlan(
            backend=ECHO_BACKEND,
            num_physical_experts=6,
            physical_to_logical_map=physical_to_logical,
            physical_to_rank_map=torch.tensor([0, 0, 1, 1, 1, 1]),
            physical_to_local_slot_map=torch.tensor([0, 1, 0, 1, 0, 1]),
            logical_to_physical_map=logical_to_physical,
            logical_replica_counts=replica_counts,
            transfer_plan=transfer_plan,
        ),
        token_plan=_physical_token_plan(route_info, ECHO_BACKEND),
    )

    ultraep_output = MoEPlannerOutput(
        expert_plan=ExpertReroutePlan(
            backend=ULTRA_EP_BACKEND,
            num_physical_experts=6,
            physical_to_logical_map=physical_to_logical,
            logical_to_physical_map=logical_to_physical,
            logical_replica_counts=replica_counts,
            transfer_plan=transfer_plan,
            native_plan={"logical_instance_quota_prefix": torch.arange(7)},
        ),
        token_plan=_physical_token_plan(route_info, ULTRA_EP_BACKEND),
    )

    moonep_output = MoEPlannerOutput(
        expert_plan=ExpertReroutePlan(
            backend=MOON_EP_BACKEND,
            native_plan={"experts_to_copy": torch.tensor([[0, 3], [1, 2]])},
        ),
        token_plan=TokenReroutePlan(
            backend=MOON_EP_BACKEND,
            logical_expert_ids=route_info.topk_ids,
            assignment_probs=route_info.topk_probs,
            dispatch_indices=torch.arange(6).reshape(3, 2),
            native_plan={"dst": torch.arange(6)},
        ),
    )

    assert echo_output.expert_plan.transfer_plan.num_transfers == 2
    assert echo_output.token_plan.routing_map.shape == (3, 6)
    assert ultraep_output.expert_plan.native_plan["logical_instance_quota_prefix"].numel() == 7
    assert moonep_output.expert_plan.native_plan["experts_to_copy"].shape == (2, 2)
    assert moonep_output.token_plan.num_tokens == route_info.num_tokens


class _EchoStylePlanner(MoELoadPlanner):
    planner_name = "echo-test"

    def plan(self, route_info: RouteInfo, context: SchedulerContext) -> MoEPlannerOutput:
        del context
        return MoEPlannerOutput(
            expert_plan=ExpertReroutePlan(
                backend=ECHO_BACKEND,
                num_physical_experts=6,
                physical_to_logical_map=torch.tensor([0, 1, 2, 3, 0, 3]),
                transfer_plan=ExpertTransferPlan(
                    source_logical_expert_ids=torch.tensor([0, 3]),
                    dest_physical_expert_ids=torch.tensor([4, 5]),
                    dest_ranks=torch.tensor([1, 1]),
                    dest_local_slots=torch.tensor([0, 1]),
                ),
            ),
            token_plan=_physical_token_plan(route_info, ECHO_BACKEND),
        )


class _RecordingDispatch(ExpertDispatch):
    dispatcher_name = "recording-test"
    supported_backends = frozenset({ECHO_BACKEND})

    def __init__(self) -> None:
        super().__init__()
        self.dispatched_plan = None
        self.finalized_output = None

    def dispatch(
        self,
        experts: torch.nn.Module,
        expert_plan: ExpertReroutePlan,
        context: SchedulerContext,
    ) -> ExpertDispatchOutput:
        del context
        self.dispatched_plan = expert_plan
        return ExpertDispatchOutput(
            expert_plan=expert_plan,
            materialized_experts=experts,
            metadata={"recorded": True},
        )

    def finalize(self, output: ExpertDispatchOutput, context: SchedulerContext) -> None:
        del context
        self.finalized_output = output


class _UltraEPOnlyDispatch(ExpertDispatch):
    dispatcher_name = "ultraep-only-test"
    supported_backends = frozenset({ULTRA_EP_BACKEND})

    def dispatch(
        self,
        experts: torch.nn.Module,
        expert_plan: ExpertReroutePlan,
        context: SchedulerContext,
    ) -> ExpertDispatchOutput:
        del experts, context
        return ExpertDispatchOutput(expert_plan=expert_plan)


def test_scheduler_passes_expert_plan_to_matching_dispatcher():
    route_info = _route_info()
    context = _context()
    dispatch = _RecordingDispatch()
    scheduler = MoEScheduler(planner=_EchoStylePlanner(), expert_dispatch=dispatch)
    experts = torch.nn.Identity()

    output = scheduler.schedule(route_info, experts, context)

    assert output.expert_plan.backend == ECHO_BACKEND
    assert output.token_plan.physical_expert_ids.shape == (3, 2)
    assert output.expert_dispatch_output.materialized_experts is experts
    assert output.expert_dispatch_output.metadata["recorded"]
    assert dispatch.dispatched_plan is output.expert_plan

    scheduler.finalize(output, context)

    assert dispatch.finalized_output is output.expert_dispatch_output


def test_scheduler_rejects_dispatcher_that_does_not_support_planner_backend():
    scheduler = MoEScheduler(
        planner=_EchoStylePlanner(), expert_dispatch=_UltraEPOnlyDispatch()
    )

    with pytest.raises(ValueError, match="does not support planner backend 'echo'"):
        scheduler.schedule(_route_info(), torch.nn.Identity(), _context())
