# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU checks for placement slot ownership across forward and backward."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.moe_scheduler import MoESchedulerConfig, SchedulerContext
from megatron.core.transformer.moe import replica_expert_dispatch as dispatch_module
from megatron.core.transformer.moe.replica_expert_dispatch import ReplicaExpertDispatch


def _dispatcher(events):
    config = MoESchedulerConfig("echo", "replica_nccl", 4, 2, 2)
    dispatcher = ReplicaExpertDispatch(config=config, pg_collection=SimpleNamespace(ep=None))

    class Runtime:
        source_parameters = ()
        last_plan = None

        def acquire_workspace(self):
            events.append("acquire")

        def release_workspace(self):
            events.append("release")

        def start_prefetch(self, plan):
            events.append("forward_prefetch")

        def start_backward_prefetch(self, plan):
            events.append("backward_prefetch")

        def wait_prefetch_for_backward(self, plan):
            events.append("wait_backward")

        def start_pending_grad_reduces(self, plan):
            events.append("start_grad_reduce")

        def wait_grad_reduce(self, plan):
            events.append("wait_grad_reduce")
            return ()

    dispatcher.runtime = Runtime()
    return dispatcher


def _context():
    return SchedulerContext(1, 4, 2, (0, 1), 2, 0, 1, True)


def _placement():
    return torch.tensor([[2], [0]], dtype=torch.int32)


def test_replica_plan_slot_survives_forward_until_backward_finishes():
    events = []
    dispatcher = _dispatcher(events)
    hidden = dispatcher.wrap_layer_input(torch.ones(2, requires_grad=True))
    dispatcher.dispatch(torch.nn.Identity(), _placement(), _context())
    output = dispatcher.before_token_dispatch(hidden) * 2
    output = dispatcher.before_token_combine(output)
    output = dispatcher.after_token_combine(output)

    with pytest.raises(RuntimeError, match="all replica forwards/backwards"):
        dispatcher.assert_idle()
    output.sum().backward()

    dispatcher.assert_idle()
    assert events.index("backward_prefetch") < events.index("wait_backward")
    assert events.index("start_grad_reduce") < events.index("wait_grad_reduce")


def test_replica_dispatch_requires_wrapped_input_without_leaking_a_slot():
    dispatcher = _dispatcher([])
    with pytest.raises(RuntimeError, match="layer input must be wrapped"):
        dispatcher.dispatch(torch.nn.Identity(), _placement(), _context())
    assert dispatcher._plan_slots == []
    dispatcher.assert_idle()


def test_replica_binding_failure_destroys_allocated_runtime(monkeypatch):
    destroyed = []

    class Runtime:
        def __init__(self, **kwargs):
            pass

        def destroy(self):
            destroyed.append(True)

    monkeypatch.setattr(dispatch_module, "ReplicaExpertRuntime", Runtime)
    dispatcher = ReplicaExpertDispatch(
        config=MoESchedulerConfig("echo", "replica_nccl", 4, 2, 2),
        pg_collection=SimpleNamespace(ep=None),
    )

    def reject_runtime(runtime):
        raise RuntimeError("attachment failed")

    experts = SimpleNamespace(_fused_ops=None, set_replica_expert_runtime=reject_runtime)
    with pytest.raises(RuntimeError, match="attachment failed"):
        dispatcher.bind_experts(experts)
    assert destroyed == [True]
    assert dispatcher.runtime is None
