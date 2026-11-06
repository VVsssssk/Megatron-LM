# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU coverage for the scheduler boundary before GPU backends are introduced."""

import pytest
import torch

from megatron.core.transformer.moe.moe_scheduler import (
    ExpertDispatch,
    MoELoadPlanner,
    MoEPlacementResult,
    MoEScheduler,
    MoESchedulerConfig,
    SchedulerContext,
)


def _context():
    return SchedulerContext(1, 4, 2, (0, 1), 2, 0, 1, True)


class _Planner(MoELoadPlanner):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def update_placement(self, probs, routing_map, context, *, tokens_per_expert=None):
        self.events.append("plan")
        return torch.tensor([[0], [3]], dtype=torch.int32), MoEPlacementResult(8, 12)

    def reroute(self, probs, routing_map, placement_result, context):
        self.events.append("reroute")
        routes = torch.zeros(2, 6, dtype=torch.bool)
        routes[0, 4] = True
        routes[1, 3] = True
        physical_probs = torch.zeros(2, 6)
        physical_probs[0, 4] = probs[0, 0]
        physical_probs[1, 3] = probs[1, 3]
        return routes, physical_probs


class _Dispatcher(ExpertDispatch):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def dispatch(self, experts, placement, context):
        self.events.append("dispatch")
        assert placement.tolist() == [[0], [3]]

    def finalize(self, context):
        self.events.append("finalize")


def test_scheduler_orders_placement_before_rerouting_and_preserves_probabilities():
    events = []
    scheduler = MoEScheduler(_Planner(events), _Dispatcher(events))
    probs = torch.tensor([[0.7, 0, 0, 0], [0, 0, 0, 0.6]])
    routes = probs.bool()
    output_probs, output_routes = scheduler.schedule(probs, routes, torch.nn.Identity(), _context())

    assert events == ["plan", "dispatch", "reroute"]
    assert output_routes[0, 4] and output_routes[1, 3]
    assert output_probs[0, 4] == probs[0, 0]
    assert output_probs[1, 3] == probs[1, 3]
    assert scheduler.max_dispatch_routes_per_rank == 8
    assert scheduler.max_dispatch_padded_tokens_per_rank == 12
    scheduler.finalize(_context())
    assert events[-1] == "finalize"


def test_scheduler_rejects_bad_dispatcher_placement_before_rerouting():
    class BadPlanner(_Planner):
        def update_placement(self, probs, routing_map, context, *, tokens_per_expert=None):
            return torch.arange(4), MoEPlacementResult()

    events = []
    scheduler = MoEScheduler(BadPlanner(events), _Dispatcher(events))
    with pytest.raises(ValueError, match="replica placement"):
        scheduler.schedule(
            torch.ones(2, 4), torch.ones(2, 4, dtype=torch.bool), torch.nn.Identity(), _context()
        )
    assert events == []


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_logical_experts": 5}, "Logical expert count"),
        ({"num_idle_experts": 3}, "Replica slot count"),
        ({"planner": "moon_ep", "num_idle_experts": 2}, "MoonEP requires"),
        ({"transport": "replica_hybridep"}, "Unsupported MoE scheduler transport"),
    ],
)
def test_scheduler_config_rejects_invalid_placement(kwargs, message):
    options = dict(
        planner="echo",
        transport="replica_nccl",
        num_logical_experts=4,
        num_idle_experts=2,
        ep_size=2,
    )
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        MoESchedulerConfig(**options)
