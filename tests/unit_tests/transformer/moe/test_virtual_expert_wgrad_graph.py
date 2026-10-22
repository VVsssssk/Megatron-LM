# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU-only checks of the MoonEP/TE deferred-wgrad graph contract.

Also executable inside the training container before distributed training, to
validate both TE implementations against the exact installed dependency.
"""

import queue
from types import SimpleNamespace

import torch

from megatron.core.extensions.transformer_engine import TEGroupedLinear
from megatron.core.transformer.moe.experts import (
    _GroupedLinearWithWgradStore,
    _VirtualExpertFC2WgradStore,
)


def _linear(cls, store):
    # Only the graph contract is under test; constructing weights requires a GPU.
    linear = cls.__new__(cls)
    torch.nn.Module.__init__(linear)
    linear.wgrad_store = store
    return linear


def test_immediate_fc2_wgrad_needs_no_separate_graph():
    events = []

    class Manager:
        def start_fc2_grad_reduce(self):
            events.append("reduce")

    manager = Manager()
    store = _VirtualExpertFC2WgradStore(manager)
    linears = [_linear(cls, store) for cls in (TEGroupedLinear, _GroupedLinearWithWgradStore)]
    assert store.delay_wgrad_compute()  # GEMM interception remains enabled.
    assert all(not linear.need_backward_dw() for linear in linears)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dy = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
    dw = torch.zeros(2, 2)

    def gemm(x, dy, dw):
        dw.add_(dy.T @ x)
        events.append("gemm")

    store.put([x, dy, dw], gemm)
    assert events == ["gemm", "reduce"]
    torch.testing.assert_close(dw, dy.T @ x, rtol=0, atol=0)
    assert all(not linear.need_backward_dw() for linear in linears)


def test_real_deferred_wgrad_keeps_separate_graph():
    for enabled in (False, True):
        store = SimpleNamespace(
            context=queue.Queue() if enabled else None,
            delay_wgrad_compute=lambda enabled=enabled: enabled,
        )
        for cls in (TEGroupedLinear, _GroupedLinearWithWgradStore):
            linear = _linear(cls, store)
            assert linear.need_backward_dw() is enabled
            if enabled:
                # Capture must preserve future wgrad calls even with an empty queue.
                assert store.context.empty()
                store.context.put("pending wgrad")
                assert linear.need_backward_dw()
                store.context.get()


if __name__ == "__main__":
    test_immediate_fc2_wgrad_needs_no_separate_graph()
    test_real_deferred_wgrad_keeps_separate_graph()
    print("MoonEP wgrad graph contract: PASS (immediate/deferred, module/op)", flush=True)
