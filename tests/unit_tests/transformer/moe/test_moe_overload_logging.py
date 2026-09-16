# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.transformer.moe import moe_logging
from megatron.core.transformer.moe.moe_logging import MoEOverloadFactorTracker
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.launch_on_gb200


@pytest.fixture(scope="module", autouse=True)
def _distributed_device():
    Utils.initialize_distributed()


def _metrics(tracker, iteration=1):
    writer = Mock()
    tracker.report(iteration, wandb_writer=writer, per_layer_logging=True)
    return {key: value for call in writer.log.call_args_list for key, value in call.args[0].items()}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_overload_snapshots_each_event_and_clears_in_place(device):
    tracker = MoEOverloadFactorTracker(capacity=8)
    values = torch.tensor([24.0, 16.0], device=device)
    tracker.record_fwd(3, *values)
    tracker.record_bwd(*values)
    values[0] = 8
    tracker.record_fwd(3, *values)
    tracker.record_bwd(*values)
    pointers = (
        tracker._events.data_ptr(),
        tracker._event_layers.data_ptr(),
        tracker._cursor.data_ptr(),
    )
    metrics = _metrics(tracker)
    assert metrics["moe/avg_overload_factor"] == pytest.approx(1.0)
    assert metrics["moe/max_overload_factor"] == pytest.approx(1.5)
    assert metrics["moe/max_cum_overload_factor"] == pytest.approx(1.5)
    assert metrics["moe/avg_overload_factor_layer_2"] == pytest.approx(1.0)
    assert tracker.report(2) == ""
    assert pointers == (
        tracker._events.data_ptr(),
        tracker._event_layers.data_ptr(),
        tracker._cursor.data_ptr(),
    )
    tracker.record_fwd(3, *values)
    tracker.record_bwd(*values)
    assert _metrics(tracker, 3)["moe/avg_overload_factor"] == pytest.approx(0.5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("existing_events", [False, True])
def test_capture_warmup_preserves_training_interval(monkeypatch, device, existing_events):
    tracker = MoEOverloadFactorTracker(capacity=8)
    monkeypatch.setattr(moe_logging, "_MOE_OVERLOAD_FACTOR_TRACKER", tracker)
    values = torch.tensor([24.0, 16.0], device=device)
    if existing_events:
        tracker.record_fwd(1, *values)
        tracker.record_bwd(*values)
    # More warmup events than capacity are harmless only because this entire
    # construction interval is explicitly restored, not because overflow is ignored.
    with moe_logging.preserve_moe_overload_state():
        for _ in range(10):
            tracker.record_fwd(1, values[0] * 10, values[1])
            tracker.record_bwd(values[0] * 10, values[1])
    if existing_events:
        assert _metrics(tracker)["moe/avg_overload_factor"] == pytest.approx(1.5)
    else:
        assert tracker.report(1) == ""


def test_overload_capacity_errors_are_explicit():
    tracker = MoEOverloadFactorTracker(capacity=2)
    values = torch.tensor([16.0, 16.0])
    for _ in range(3):
        tracker.record_fwd(1, *values)
    with pytest.raises(RuntimeError, match="journal overflow"):
        tracker.report(1)
    with pytest.raises(RuntimeError, match="Clear/report"):
        tracker.reserve(4, "cpu")
    tracker.clear()
    tracker.reserve(4, "cpu")
    tracker._captured = True
    with pytest.raises(RuntimeError, match="referenced by CUDA Graphs"):
        tracker.reserve(8, "cpu")


def test_overload_retry_discards_failed_attempt(monkeypatch):
    from megatron.core.transformer.moe.paged_stash import PagedStashRunner

    tracker = MoEOverloadFactorTracker(capacity=8)
    monkeypatch.setattr(moe_logging, "_MOE_OVERLOAD_FACTOR_TRACKER", tracker)
    values = torch.tensor([32.0, 16.0])
    tracker.record_fwd(1, *values)
    tracker.record_bwd(*values)
    runner = object.__new__(PagedStashRunner)
    runner.moe_layers = []
    runner.stash_manager = SimpleNamespace(
        overflow=None, host_spill=None, release_stash_buffers=Mock()
    )
    runner._set_moe_paged_stash_all = Mock()
    runner._reset_qb_histograms = Mock()
    runner.model = []
    runner.optimizer = None
    runner.copy_main_params = False
    runner.forward_backward_func = Mock()
    runner.prepare_for_rerun(is_training=True)
    values[0] = 16
    tracker.record_fwd(1, *values)
    tracker.record_bwd(*values)
    metrics = _metrics(tracker)
    for name in ("avg_overload_factor", "max_overload_factor", "max_cum_overload_factor"):
        assert metrics[f"moe/{name}"] == pytest.approx(1.0)


@pytest.mark.parametrize("forward_first", [False, True])
def test_overload_graph_reuse_tracks_each_microbatch_and_actual_timeline(
    monkeypatch, forward_first
):
    tracker = MoEOverloadFactorTracker(capacity=16)
    monkeypatch.setattr(moe_logging, "_MOE_OVERLOAD_FACTOR_TRACKER", tracker)
    values = torch.tensor([1000.0, 16.0], device="cuda")
    tracker.reserve(16, values.device)
    fwd, bwd = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with moe_logging.preserve_moe_overload_state():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                tracker.record_fwd(1, *values)
                tracker.record_bwd(*values)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(fwd):
            tracker.record_fwd(1, *values)
        with torch.cuda.graph(bwd):
            tracker.record_bwd(*values)
    # Reuse the SAME graphs multiple times within a step, with different routes.
    # Also run another step with a different number of microbatches.
    for step, loads in enumerate(([24.0, 8.0], [16.0, 32.0, 24.0]), 1):
        timeline = []
        if forward_first:
            operations = [(fwd, x, 1) for x in loads] + [(bwd, x, -1) for x in reversed(loads)]
        else:
            operations = [(g, x, s) for x in loads for g, s in ((fwd, 1), (bwd, -1))]
        for graph, load, sign in operations:
            values[0] = load
            graph.replay()
            timeline.append((sign * load, sign * 16.0))
        reference = torch.tensor(timeline).cumsum(0).amax(0)
        metrics = _metrics(tracker, step)
        assert metrics["moe/avg_overload_factor"] == pytest.approx(sum(loads) / len(loads) / 16)
        assert metrics["moe/max_overload_factor"] == pytest.approx(max(loads) / 16)
        assert metrics["moe/max_cum_overload_factor"] == pytest.approx(
            (reference[0] / reference[1]).item()
        )
        assert int(tracker._cursor.item()) == 0
    assert tracker.report(3) == ""


def test_overload_graph_distributed_max_rank_over_average_rank(monkeypatch):
    Utils.initialize_model_parallel()
    try:
        group = torch.distributed.group.WORLD
        rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        tracker = MoEOverloadFactorTracker(capacity=16)
        tracker.set_process_groups(tp_ep_group=group)
        monkeypatch.setattr(moe_logging, "_MOE_OVERLOAD_FACTOR_TRACKER", tracker)
        values = torch.tensor([16.0, 16.0], device="cuda")
        tracker.reserve(16, values.device)
        graph = torch.cuda.CUDAGraph()
        with moe_logging.preserve_moe_overload_state():
            with torch.cuda.graph(graph):
                tracker.record_fwd(1, *values)
                tracker.record_bwd(*values)
        for step in (1, 2):
            # Global routed counts equal global balanced counts; source lengths
            # deliberately differ so the denominator must SUM and divide by world.
            for microbatch in (0, 1):
                values[0] = 8.0 * (1 + (rank + microbatch + step) % world)
                values[1] = 8.0 * (rank + 1)
                graph.replay()
            metrics = _metrics(tracker, step)
            expected = 2.0 * world / (world + 1)
            assert metrics["moe/avg_overload_factor"] == pytest.approx(expected)
            assert metrics["moe/max_overload_factor"] == pytest.approx(expected)
    finally:
        Utils.destroy_model_parallel()
