# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.moe.echo_moe_scheduler import (
    EchoExpertDispatch,
    EchoLoadPlanner,
    HybridEPEchoExpertDispatchBackend,
)
from megatron.core.transformer.moe.moonep_moe_scheduler import MoonEPLoadPlanner
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_scheduler import (
    ECHO_BACKEND,
    ExpertPlacementPlan,
    MoEPlannerOutput,
    MoEScheduler,
    SchedulerContext,
)
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(autouse=True)
def _patch_echo_count_gather(monkeypatch):
    import megatron.core.transformer.moe.echo_moe_scheduler as echo_moe_scheduler

    def fake_gather_from_sequence_parallel_region(tokens_per_expert, group=None):
        assert group is not None
        return torch.tensor(
            [4, 0, 0, 0, 0, 0, 1, 1],
            dtype=tokens_per_expert.dtype,
            device=tokens_per_expert.device,
        )

    monkeypatch.setattr(
        echo_moe_scheduler.tensor_parallel,
        "gather_from_sequence_parallel_region",
        fake_gather_from_sequence_parallel_region,
    )


def _echo_context() -> SchedulerContext:
    return SchedulerContext(
        layer_number=1,
        num_logical_experts=4,
        num_local_experts=2,
        local_expert_indices=(0, 1),
        ep_size=2,
        ep_rank=0,
        router_topk=1,
        training=True,
        pg_collection=SimpleNamespace(ep=object()),
    )


def _hot_expert_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.zeros(4, 4)
    routing_map = torch.zeros(4, 4, dtype=torch.bool)
    topk_ids = torch.zeros(4, 1, dtype=torch.long)
    topk_probs = torch.ones(4, 1)
    routing_map[:, 0] = True
    probs[:, 0] = 1
    return probs, routing_map, routing_map.sum(dim=0)


def test_echo_planner_reroutes_hot_expert_tokens_to_echo_slot():
    probs, routing_map, tokens_per_expert = _hot_expert_inputs()
    context = _echo_context()
    output = EchoLoadPlanner(2).plan(
        probs, routing_map, context, tokens_per_expert=tokens_per_expert
    )

    expert_placement = output.expert_placement
    expert_offloading_map = output.expert_placement.metadata["expert_offloading_map"]

    assert expert_placement.backend == ECHO_BACKEND
    assert expert_placement.resolved_num_physical_experts == 6
    assert expert_placement.physical_to_logical_map.tolist() == [0, 1, -1, 2, 3, 0]
    assert expert_placement.source_logical_expert_ids.tolist() == [0]
    assert expert_placement.dest_physical_expert_ids.tolist() == [5]
    assert expert_placement.dest_ranks.tolist() == [1]
    assert expert_offloading_map.tolist() == [
        [False, True],
        [False, False],
        [False, False],
        [False, False],
    ]

    assert output.routing_map.shape == (4, 6)
    assert output.routing_map[:, 0].sum().item() == 3
    assert output.routing_map[:, 5].sum().item() == 1
    assert torch.equal(output.probs.sum(dim=1), probs.sum(dim=1))
    assert output.logical_expert_ids.tolist() == [[0], [0], [0], [0]]


def test_echo_planner_should_not_plan_without_idle_experts():
    probs, routing_map, tokens_per_expert = _hot_expert_inputs()
    assert (
        EchoLoadPlanner(0).should_plan(
            probs, routing_map, _echo_context(), tokens_per_expert=tokens_per_expert
        )
        is False
    )


def test_echo_planner_exposes_pr_style_allocation_metadata():
    probs, routing_map, tokens_per_expert = _hot_expert_inputs()
    context = _echo_context()
    output = EchoLoadPlanner(2).plan(
        probs, routing_map, context, tokens_per_expert=tokens_per_expert
    )

    assert output.expert_placement.metadata["assignment_algorithm"] == "approx_bin_packing"
    assert output.expert_placement.metadata["count_tokens_from_home_expert_to_echo"].tolist() == [
        [0, 1],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
    assert output.expert_placement.metadata[
        "count_tokens_offloaded_from_ep_rank_to_echo"
    ].tolist() == [[0, 1], [0, 0]]
    assert output.expert_placement.metadata[
        "count_tokens_offloaded_from_ep_rank_from_home_expert"
    ].tolist() == [[1, 0, 0, 0], [0, 0, 0, 0]]
    assert output.expert_placement.metadata["count_tokens_per_expert_after_offload"].tolist() == [
        [3, 0, 0, 0],
        [0, 0, 1, 1],
    ]


def test_echo_planner_requires_ep_group_for_multi_ep():
    probs, routing_map, tokens_per_expert = _hot_expert_inputs()
    context = SchedulerContext(
        layer_number=1,
        num_logical_experts=4,
        num_local_experts=2,
        local_expert_indices=(0, 1),
        ep_size=2,
        ep_rank=0,
        router_topk=1,
        training=True,
    )

    with pytest.raises(ValueError, match="pg_collection.ep"):
        EchoLoadPlanner(2).plan(
            probs, routing_map, context, tokens_per_expert=tokens_per_expert
        )


def test_echo_expert_dispatch_delegates_to_materializer():
    probs, routing_map, tokens_per_expert = _hot_expert_inputs()
    context = _echo_context()
    planner_output = EchoLoadPlanner(2).plan(route_info, context)
    experts = torch.nn.Identity()
    calls = []

    def materializer(experts_arg, expert_placement_arg, context_arg):
        calls.append((experts_arg, expert_placement_arg, context_arg))
        return ExpertDispatchOutput(
            expert_placement=expert_placement_arg,
            materialized_experts="materialized-echo-experts",
            metadata={"materialized": True},
        )

    dispatcher = EchoExpertDispatch(materializer=materializer)
    dispatch_output = dispatcher.dispatch(experts, planner_output.expert_placement, context)

    assert dispatch_output.materialized_experts == "materialized-echo-experts"
    assert dispatch_output.metadata["materialized"]
    assert calls == [(experts, planner_output.expert_placement, context)]


def test_echo_expert_dispatch_builds_pr_style_metadata_and_dispatches_weights():
    probs, routing_map, tokens_per_expert = _hot_expert_inputs()
    context = _echo_context()
    planner_output = EchoLoadPlanner(2).plan(route_info, context)

    class _Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = {
                "fc1": [torch.tensor([1.0]), torch.tensor([2.0])],
                "fc2": [torch.tensor([3.0]), torch.tensor([4.0])],
            }
            self.set_calls = []

        def get_expert_weights(self, module, expert_indices):
            return [self.weights[module][idx] for idx in expert_indices]

        def set_expert_weights(self, module, expert_weights, expert_indices):
            self.set_calls.append((module, expert_weights, expert_indices))

    class _PRStyleBackend:
        def __init__(self):
            self.preprocess_calls = []
            self.dispatch_calls = []

        def preprocess(self, expert_offloading_map):
            self.preprocess_calls.append(expert_offloading_map)
            return {"expert_offloading_map": expert_offloading_map}

        def expert_dispatch(self, metadata, *expert_weights):
            self.dispatch_calls.append((metadata, expert_weights))
            return [torch.tensor([99.0])]

    experts = _Experts()
    backend = _PRStyleBackend()
    dispatcher = EchoExpertDispatch(materializer=backend)

    dispatch_output = dispatcher.dispatch(experts, planner_output.expert_placement, context)
    dispatch_metadata = dispatch_output.metadata["echo_dispatch_metadata"]

    assert dispatch_output.metadata["materialized"]
    assert dispatch_metadata.input_splits == [0, 1]
    assert dispatch_metadata.output_splits == [0, 0]
    assert dispatch_metadata.has_experts_per_slot.tolist() == [0]
    assert len(backend.preprocess_calls) == 2
    assert [call[0] for call in experts.set_calls] == ["fc1", "fc2"]
    assert [call[2] for call in experts.set_calls] == [[2], [2]]
    assert len(backend.dispatch_calls) == 2


def test_hybridep_echo_backend_slices_local_routing_map_and_calls_kernel(monkeypatch):
    probs, routing_map, tokens_per_expert = _hot_expert_inputs()
    context = _echo_context()
    planner_output = EchoLoadPlanner(2).plan(route_info, context)

    class _Group:
        def size(self):
            return 2

        def rank(self):
            return 0

    calls = []
    dispatched_weight = torch.tensor([7.0])
    handle = object()

    def fake_hybrid_ep_expert_dispatch(*args):
        calls.append(args)
        return (dispatched_weight, handle)

    import megatron.core.transformer.moe.fused_a2a as fused_a2a

    monkeypatch.setattr(fused_a2a, "hybrid_ep_expert_dispatch", fake_hybrid_ep_expert_dispatch)

    backend = HybridEPEchoExpertDispatchBackend(
        ep_group=_Group(),
        hidden_size=4,
        num_idle_experts=2,
        num_sms_dispatch_api=7,
        num_sms_combine_api=8,
        weight_chunk_size=4,
    )
    metadata = backend.preprocess(planner_output.metadata["expert_offloading_map"])
    result = backend.expert_dispatch(metadata, torch.ones(4), torch.ones(4) * 2)

    assert metadata.routing_map.tolist() == [
        [False, True],
        [False, False],
    ]
    assert result == [dispatched_weight]
    assert metadata.handle is handle
    assert calls[0][0] is metadata.routing_map
    assert calls[0][1] is backend.ep_group
    assert calls[0][2] is None
    assert calls[0][3:8] == (1, 7, 8, 1, 4)


def test_echo_scheduler_runs_planner_and_dispatch_adapter():
    probs, routing_map, tokens_per_expert = _hot_expert_inputs()
    context = _echo_context()
    scheduler = MoEScheduler(
        planner=EchoLoadPlanner(2), expert_dispatch=EchoExpertDispatch()
    )

    output = scheduler.schedule(route_info, torch.nn.Identity(), context)

    assert output.expert_placement.backend == ECHO_BACKEND
    assert output.expert_dispatch_output.metadata["materialized"] is False
    assert output.routing_map[:, 5].sum().item() == 1


def test_moe_scheduler_builds_echo_stack_from_config():
    class _Group:
        def size(self):
            return 1

        def rank(self):
            return 0

    scheduler = MoEScheduler.from_config(
        _scheduler_config(moe_scheduler_num_idle_experts=2),
        SimpleNamespace(ep=_Group()),
        home_expert_indices=(0, 1, 2, 3),
        idle_expert_indices=(4, 5),
    )

    assert isinstance(scheduler.planner, EchoLoadPlanner)
    assert isinstance(scheduler.expert_dispatch, EchoExpertDispatch)
    assert scheduler.planner.num_echo_experts == 2


def test_moe_scheduler_builds_moonep_planner_with_echo_dispatch_from_config():
    class _Group:
        def size(self):
            return 1

        def rank(self):
            return 0

    scheduler = MoEScheduler.from_config(
        _scheduler_config(moe_scheduler_planner_type="moon_ep"),
        SimpleNamespace(ep=_Group()),
        home_expert_indices=(0, 1, 2, 3),
        idle_expert_indices=(4, 5),
    )

    assert isinstance(scheduler.planner, MoonEPLoadPlanner)
    assert scheduler.planner.num_redundant_experts == 2
    assert isinstance(scheduler.expert_dispatch, EchoExpertDispatch)


def _scheduler_config(**overrides) -> TransformerConfig:
    defaults = {
        "num_layers": 1,
        "hidden_size": 12,
        "num_attention_heads": 4,
        "num_moe_experts": 4,
        "use_cpu_initialization": True,
        "moe_router_topk": 1,
        "moe_router_pre_softmax": True,
        "moe_grouped_gemm": False,
        "add_bias_linear": False,
        "moe_enable_scheduler": True,
        "moe_scheduler_num_idle_experts": 2,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def test_transformer_config_validates_moe_scheduler_requirements():
    config = _scheduler_config()

    assert config.moe_enable_scheduler
    assert config.moe_scheduler_planner_type == "echo"
    assert config.moe_scheduler_expert_dispatcher_type == "hybridep"

    with pytest.raises(ValueError, match="moe_scheduler_num_idle_experts"):
        _scheduler_config(moe_scheduler_num_idle_experts=None)
    with pytest.raises(ValueError, match="dropless"):
        _scheduler_config(moe_expert_capacity_factor=1.0)
    with pytest.raises(ValueError, match="add_bias_linear"):
        _scheduler_config(add_bias_linear=True)


def test_moe_layer_scheduler_helper_uses_unified_scheduler_output():
    probs = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    routing_map = probs.bool()
    physical_probs = torch.zeros(3, 6)
    physical_routing_map = torch.zeros(3, 6, dtype=torch.bool)
    physical_probs[:, 5] = probs[:, 0]
    physical_routing_map[:, 5] = routing_map[:, 0]
    physical_probs[:, 1] = probs[:, 1]
    physical_routing_map[:, 1] = routing_map[:, 1]

    class _RecordingScheduler:
        def __init__(self):
            self.route_info = None
            self.context = None
            self.experts = None

        def schedule(self, route_info, experts, context):
            self.route_info = route_info
            self.context = context
            self.experts = experts
            expert_placement = ExpertPlacementPlan(
                backend=ECHO_BACKEND, num_physical_experts=6
            )
            return MoESchedulerOutput(
                planner_output=MoEPlannerOutput(
                    expert_placement=expert_placement,
                    routing_map=physical_routing_map,
                    probs=physical_probs,
                ),
                expert_dispatch_output=ExpertDispatchOutput(
                    expert_placement=expert_placement
                ),
            )

    layer = object.__new__(MoELayer)
    layer.moe_scheduler = _RecordingScheduler()
    layer.experts = object()
    layer.num_logical_experts = 4
    layer.num_local_home_experts = 4
    layer.local_home_expert_indices = [0, 1, 2, 3]
    layer.ep_group = None
    layer.config = type("_Config", (), {"moe_router_topk": 1})()
    layer.layer_number = 7
    layer.training = True
    layer.pg_collection = object()

    hidden_states = torch.randn(3, 12)
    new_probs, new_routing_map = MoELayer._maybe_schedule_moe(
        layer, hidden_states, probs, routing_map
    )

    assert new_probs is physical_probs
    assert new_routing_map is physical_routing_map
    assert layer.moe_scheduler.route_info.routing_map is routing_map
    assert layer.moe_scheduler.route_info.hidden_states is hidden_states
    assert layer.moe_scheduler.route_info.tokens_per_expert.tolist() == [2, 1, 0, 0]
    assert layer.moe_scheduler.experts is layer.experts
    assert layer.moe_scheduler.context.layer_number == 7
    assert layer.moe_scheduler.context.num_logical_experts == 4
    assert layer.moe_scheduler.context.num_local_experts == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_moe_layer_auto_instantiates_scheduler_from_config():
    Utils.initialize_model_parallel(1, 1)
    try:
        config = _scheduler_config()
        submodules = get_submodules(
            get_gpt_layer_local_submodules(
                num_experts=config.num_moe_experts, moe_grouped_gemm=False
            ).mlp
        )
        assert isinstance(submodules, MoESubmodules)

        layer = MoELayer(config, submodules)

        assert layer.num_logical_experts == 4
        assert layer.num_physical_experts == 6
        assert layer.num_local_home_experts == 4
        assert layer.num_local_experts == 6
        assert layer._token_dispatcher_config.num_moe_experts == 6
        assert layer.token_dispatcher.config.num_moe_experts == 6
        assert layer.config.num_moe_experts == 4
        assert layer.moe_scheduler is not None
        assert layer.home_expert_indices == [0, 1, 2, 3]
        assert layer.idle_expert_indices == [4, 5]
        assert not hasattr(layer.experts.local_experts[4].linear_fc1, "weight")
        assert not hasattr(layer.experts.local_experts[5].linear_fc2, "weight")
    finally:
        Utils.destroy_model_parallel()
