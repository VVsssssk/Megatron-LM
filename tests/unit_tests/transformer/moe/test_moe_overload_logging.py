# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest import mock

import pytest
import torch

from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.moe.moe_logging import MoEOverloadFactorTracker


def test_overload_report_supports_uneven_layer_entry_counts():
    """Recompute and CUDA-graph setup can dispatch different layer counts per step."""
    tracker = MoEOverloadFactorTracker()
    writer = mock.MagicMock()

    tracker.record_fwd(1, torch.tensor(10.0), torch.tensor(8.0))
    tracker.record_fwd(1, torch.tensor(14.0), torch.tensor(8.0))
    tracker.record_fwd(2, torch.tensor(6.0), torch.tensor(8.0))

    log = tracker.report(iteration=7, writer=writer, per_layer_logging=True)

    assert "avg overload factor: 1.250" in log
    assert "max overload factor: 1.750" in log
    assert "max cum overload factor: 1.250" in log

    scalars = {
        call.args[0]: call.args[1] for call in writer.add_scalar.call_args_list
    }
    assert scalars["moe/avg_overload_factor"] == pytest.approx(1.25)
    assert scalars["moe/max_overload_factor"] == pytest.approx(1.75)
    assert scalars["moe/max_cum_overload_factor"] == pytest.approx(1.25)
    assert scalars["moe/avg_overload_factor_layer_0"] == pytest.approx(1.5)
    assert scalars["moe/max_overload_factor_layer_0"] == pytest.approx(1.75)
    assert scalars["moe/avg_overload_factor_layer_1"] == pytest.approx(0.75)
    assert scalars["moe/max_overload_factor_layer_1"] == pytest.approx(0.75)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_dispatch_counts_are_cuda_graph_safe(monkeypatch):
    """Static CPU expert counts can be recorded while capturing a CUDA graph."""
    tracker = mock.MagicMock()
    monkeypatch.setattr(moe_utils, "get_moe_overload_factor_tracker", lambda: tracker)

    tensor = torch.ones(4, device="cuda", requires_grad=True)
    tokens_per_expert = torch.tensor([2, 3], device="cpu")
    balanced = torch.tensor(5.0, device="cuda")
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()

    with torch.cuda.graph(graph):
        output = moe_utils.record_dispatch_token_counts(tensor, tokens_per_expert, balanced, 1)

    graph.replay()
    torch.cuda.synchronize()
    assert output.data_ptr() == tensor.data_ptr()
    recorded_tokens = tracker.record_fwd.call_args.args[1]
    assert recorded_tokens.item() == pytest.approx(5.0)
