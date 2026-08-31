# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import pytest
import torch

from megatron.core.transformer.moe.moe_scheduler import (
    ExpertDispatch,
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


def _dense_to_topk(
    routing_map: torch.Tensor, probs: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_ids = torch.empty(routing_map.size(0), topk, dtype=torch.long)
    topk_probs = torch.empty(routing_map.size(0), topk, dtype=probs.dtype)
    for token_idx in range(routing_map.size(0)):
        expert_ids = torch.nonzero(routing_map[token_idx], as_tuple=False).flatten()
        topk_ids[token_idx] = expert_ids
        topk_probs[token_idx] = probs[token_idx, expert_ids]
    return topk_ids, topk_probs


def _physical_token_reroute(
    probs: torch.Tensor, routing_map: torch.Tensor, context: SchedulerContext
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_ids, topk_probs = _dense_to_topk(routing_map, probs, context.router_topk)
    physical_ids = topk_ids.clone()
    physical_ids[0, 0] = 2
    physical_ids[2, 0] = 5
    physical_routing_map = torch.zeros(routing_map.size(0), 6, dtype=torch.bool)
    physical_probs = torch.zeros(routing_map.size(0), 6, dtype=probs.dtype)
    physical_routing_map.scatter_(1, physical_ids, True)
    physical_probs.scatter_(1, physical_ids, topk_probs)
    return physical_probs, physical_routing_map


def test_planner_output_validates_physical_layout_and_dense_route_tensors():
    probs, routing_map, _ = _route_inputs()
    output = MoEPlannerOutput(
        physical_to_logical_map=torch.arange(4, dtype=torch.long),
        routing_map=routing_map,
        probs=probs,
    )

    assert output.physical_to_logical_map.tolist() == [0, 1, 2, 3]
    assert output.routing_map is routing_map
    assert output.probs is probs

    with pytest.raises(ValueError, match="physical_to_logical_map"):
        MoEPlannerOutput(
            physical_to_logical_map=torch.arange(4, dtype=torch.long).reshape(2, 2),
            routing_map=routing_map,
            probs=probs,
        )

    with pytest.raises(ValueError, match="same shape"):
        MoEPlannerOutput(
            physical_to_logical_map=torch.arange(4, dtype=torch.long),
            routing_map=routing_map,
            probs=torch.zeros(3, 5),
        )

    with pytest.raises(ValueError, match="expert dimension"):
        MoEPlannerOutput(
            physical_to_logical_map=torch.arange(5, dtype=torch.long),
            routing_map=routing_map,
            probs=probs,
        )


def test_unified_planner_output_carries_only_layout_and_rerouted_tensors():
    probs, routing_map, _ = _route_inputs()
    context = _context()
    physical_probs, physical_routing_map = _physical_token_reroute(
        probs, routing_map, context
    )
    output = MoEPlannerOutput(
        physical_to_logical_map=torch.tensor([0, 1, 0, 2, 3, 3]),
        routing_map=physical_routing_map,
        probs=physical_probs,
    )

    assert output.physical_to_logical_map.tolist() == [0, 1, 0, 2, 3, 3]
    assert output.routing_map.shape == (3, 6)
    assert output.probs.shape == (3, 6)


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
        physical_probs, physical_routing_map = _physical_token_reroute(
            probs, routing_map, context
        )
        return MoEPlannerOutput(
            physical_to_logical_map=torch.tensor([0, 1, 0, 2, 3, 3]),
            routing_map=physical_routing_map,
            probs=physical_probs,
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

    def __init__(self) -> None:
        super().__init__()
        self.dispatched_physical_to_logical_map = None
        self.materialized_experts = None
        self.finalized = False
        self.supports_called = False

    def dispatch(
        self,
        experts: torch.nn.Module,
        physical_to_logical_map: torch.Tensor,
        context: SchedulerContext,
    ) -> None:
        del context
        self.dispatched_physical_to_logical_map = physical_to_logical_map
        self.materialized_experts = experts

    def supports(
        self, physical_to_logical_map: torch.Tensor, context: SchedulerContext
    ) -> bool:
        self.supports_called = True
        return super().supports(physical_to_logical_map, context)

    def finalize(self, context: SchedulerContext) -> None:
        del context
        self.finalized = True


class _RejectingDispatch(ExpertDispatch):
    dispatcher_name = "rejecting-test"

    def supports(
        self, physical_to_logical_map: torch.Tensor, context: SchedulerContext
    ) -> bool:
        del physical_to_logical_map, context
        return False

    def dispatch(
        self,
        experts: torch.nn.Module,
        physical_to_logical_map: torch.Tensor,
        context: SchedulerContext,
    ) -> None:
        del experts, physical_to_logical_map, context


def test_scheduler_passes_physical_layout_to_matching_dispatcher():
    probs, routing_map, tokens_per_expert = _route_inputs()
    context = _context()
    dispatch = _RecordingDispatch()
    scheduler = MoEScheduler(planner=_EchoStylePlanner(), expert_dispatch=dispatch)
    experts = torch.nn.Identity()

    output_probs, output_routing_map = scheduler.schedule(
        probs, routing_map, experts, context, tokens_per_expert=tokens_per_expert
    )

    assert dispatch.dispatched_physical_to_logical_map.tolist() == [0, 1, 0, 2, 3, 3]
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
    assert dispatch.dispatched_physical_to_logical_map is None
    assert output_routing_map is routing_map
    assert output_probs is probs


def test_scheduler_rejects_dispatcher_that_does_not_support_planner_output():
    scheduler = MoEScheduler(planner=_EchoStylePlanner(), expert_dispatch=_RejectingDispatch())

    with pytest.raises(ValueError, match="does not support planner output"):
        probs, routing_map, tokens_per_expert = _route_inputs()
        scheduler.schedule(
            probs,
            routing_map,
            torch.nn.Identity(),
            _context(),
            tokens_per_expert=tokens_per_expert,
        )
