# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest import mock

import pytest
import torch

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
