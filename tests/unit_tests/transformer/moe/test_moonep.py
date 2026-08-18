# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import squared_relu
from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.core.transformer.moe import fused_a2a
from megatron.core.transformer.moe.fused_a2a import moonep_combine, moonep_dispatch
from megatron.core.transformer.transformer_config import TransformerConfig


class _FakeMoonEPBuffer:
    """CPU implementation of MoonEP's saved-plan dispatch/combine contract."""

    def dispatch(
        self,
        hidden,
        route_weights=None,
        topk_experts=None,
        tokens_per_expert=None,
        plan=None,
        *,
        zero_copy=False,
        zero_copy_weights=None,
        hidden_buffer=None,
    ):
        del zero_copy, zero_copy_weights, hidden_buffer
        if plan is None:
            num_tokens, topk = topk_experts.shape
            flat_experts = topk_experts.reshape(-1).long()
            order = torch.argsort(flat_experts, stable=True)
            source_tokens = torch.arange(num_tokens).repeat_interleave(topk)[order]
            source_routes = torch.arange(topk).repeat(num_tokens)[order]
            plan = SimpleNamespace(
                source_tokens=source_tokens,
                source_routes=source_routes,
                num_tokens=num_tokens,
                topk=topk,
                experts_to_copy=torch.full((1, tokens_per_expert.numel()), -1, dtype=torch.int32),
            )
            counts = torch.cat(
                [tokens_per_expert, tokens_per_expert.new_zeros(tokens_per_expert.numel())]
            )
            cu_seqlens = counts.cumsum(0)
        else:
            cu_seqlens = None

        dispatched_hidden = hidden[plan.source_tokens]
        dispatched_weights = (
            None if route_weights is None else route_weights[plan.source_tokens, plan.source_routes]
        )
        return dispatched_hidden, dispatched_weights, cu_seqlens, plan

    def combine(
        self, *, plan, hidden_nvsh, route_weights_nvs=None, zero_copy=False, hidden_buffer=None
    ):
        del zero_copy, hidden_buffer
        hidden = hidden_nvsh.new_zeros((plan.num_tokens, hidden_nvsh.shape[-1]))
        hidden.index_add_(0, plan.source_tokens, hidden_nvsh)
        weights = None
        if route_weights_nvs is not None:
            weights = route_weights_nvs.new_zeros((plan.num_tokens, plan.topk))
            weights[plan.source_tokens, plan.source_routes] = route_weights_nvs
        return hidden, weights, None


class _FakeWeightBridge:
    def __init__(self, device=None):
        self.parameters = (
            torch.nn.Parameter(torch.ones((), device=device)),
            torch.nn.Parameter(torch.ones((), device=device)),
        )
        self.last_plan = None
        self.reduced_plans = []
        self.prefetched_plans = []
        self.buffer = None

    @property
    def source_parameters(self):
        return self.parameters

    @property
    def dummy_grads(self):
        return tuple(torch.zeros_like(parameter) for parameter in self.parameters)

    def reduce_grads(self, plan):
        self.reduced_plans.append(plan)

    def prefetch(self, plan):
        self.prefetched_plans.append(plan)

    def attach_buffer(self, buffer):
        self.buffer = buffer


class _FakeDispatchBufferPool:
    def __init__(self):
        self.acquired = []
        self.released = []

    def acquire(self):
        pair = (object(), object())
        self.acquired.append(pair)
        return pair

    def release(self, pair):
        self.released.append(pair)


def _run_fake_moonep(hidden, probs, indices, bridge, buffer, dispatch_buffer_pool=None):
    tokens_per_expert = torch.bincount(indices.reshape(-1), minlength=int(indices.max()) + 1).to(
        torch.int32
    )
    dispatched, dispatched_probs, runtime_counts = moonep_dispatch(
        hidden, probs, indices, tokens_per_expert, buffer, bridge, dispatch_buffer_pool
    )
    expert_output = dispatched * dispatched_probs.unsqueeze(-1)
    output = moonep_combine(expert_output, buffer, bridge.last_plan, bridge)
    return output, runtime_counts


def test_moonep_autograd_wrappers_preserve_hidden_and_probability_gradients():
    buffer = _FakeMoonEPBuffer()
    bridge = _FakeWeightBridge()
    indices = torch.tensor([[0, 2], [1, 2], [0, 1]], dtype=torch.int32)
    hidden = torch.randn(3, 4, requires_grad=True)
    probs = torch.randn(3, 2, requires_grad=True)
    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_probs = probs.detach().clone().requires_grad_(True)

    output, runtime_counts = _run_fake_moonep(hidden, probs, indices, bridge, buffer)
    ref_output = (ref_hidden.unsqueeze(1) * ref_probs.unsqueeze(2)).sum(dim=1)
    grad = torch.randn_like(output)
    output.backward(grad)
    ref_output.backward(grad)

    torch.testing.assert_close(output, ref_output)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad)
    torch.testing.assert_close(probs.grad, ref_probs.grad)
    assert runtime_counts.numel() == 6  # E+B, with B=E for the one-rank fake.
    assert runtime_counts.sum() == indices.numel()
    assert list(map(id, bridge.prefetched_plans)) == list(map(id, bridge.reduced_plans))
    assert len(bridge.reduced_plans) == 1
    assert all(parameter.grad.item() == 0 for parameter in bridge.parameters)


def test_moonep_saved_plans_are_restored_for_multiple_outstanding_forwards():
    buffer = _FakeMoonEPBuffer()
    bridge = _FakeWeightBridge()
    dispatch_buffer_pool = _FakeDispatchBufferPool()
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
    hidden_1 = torch.randn(2, 4, requires_grad=True)
    hidden_2 = torch.randn(2, 4, requires_grad=True)
    probs_1 = torch.randn(2, 2, requires_grad=True)
    probs_2 = torch.randn(2, 2, requires_grad=True)

    output_1, _ = _run_fake_moonep(hidden_1, probs_1, indices, bridge, buffer, dispatch_buffer_pool)
    plan_1 = bridge.last_plan
    output_2, _ = _run_fake_moonep(
        hidden_2, probs_2, indices.flip(0), bridge, buffer, dispatch_buffer_pool
    )
    plan_2 = bridge.last_plan
    assert len(dispatch_buffer_pool.acquired) == 2
    assert dispatch_buffer_pool.released == []
    (output_1.sum() + output_2.sum()).backward()

    assert set(map(id, bridge.prefetched_plans)) == {id(plan_1), id(plan_2)}
    assert set(map(id, bridge.reduced_plans)) == {id(plan_1), id(plan_2)}
    assert set(map(id, dispatch_buffer_pool.released)) == set(
        map(id, dispatch_buffer_pool.acquired)
    )


def test_moonep_manager_preserves_static_dispatch_capacity():
    from megatron.core.transformer.moe.token_dispatcher import _MoonEPManager

    manager = _MoonEPManager.__new__(_MoonEPManager)
    manager.dispatched_probs = torch.randn(12)
    manager._zero_copy_token_buffers = object()
    hidden = torch.randn(12, 4)

    expert_hidden, expert_probs = manager.get_permuted_hidden_states_by_experts(hidden)

    assert expert_hidden.data_ptr() == hidden.data_ptr()
    assert expert_hidden.shape == (12, 4)
    assert expert_probs.data_ptr() == manager.dispatched_probs.data_ptr()


def test_moonep_manager_narrows_non_zero_copy_dispatch_and_restores_capacity():
    from megatron.core.transformer.moe.token_dispatcher import _MoonEPManager

    manager = _MoonEPManager.__new__(_MoonEPManager)
    manager._zero_copy_token_buffers = None
    manager.tokens_per_expert = torch.tensor([2, 1, 0], dtype=torch.int64)
    manager.dispatched_probs = torch.randn(6)
    manager._dispatch_capacity = 6
    hidden = torch.randn(6, 4)

    expert_hidden, expert_probs = manager.get_permuted_hidden_states_by_experts(hidden)
    restored = manager.get_restored_hidden_states_by_experts(expert_hidden)

    assert expert_hidden.shape == (3, 4)
    assert expert_probs.shape == (3,)
    torch.testing.assert_close(expert_hidden, hidden[:3])
    torch.testing.assert_close(expert_probs, manager.dispatched_probs[:3])
    assert restored.shape == hidden.shape
    torch.testing.assert_close(restored[:3], hidden[:3])
    torch.testing.assert_close(restored[3:], torch.zeros_like(restored[3:]))


def test_moonep_manager_exposes_shared_expert_zero_copy_buffers():
    from megatron.core.transformer.moe.token_dispatcher import _MoonEPManager

    manager = _MoonEPManager.__new__(_MoonEPManager)
    output_buffer = torch.empty(12, 4)
    dgrad_buffer = torch.empty(12, 4)
    manager._zero_copy_token_buffers = SimpleNamespace(
        forward=(object(), output_buffer), backward=(object(), dgrad_buffer)
    )

    actual_output, actual_dgrad = manager.get_expert_zero_copy_buffers()

    assert actual_output.data_ptr() == output_buffer.data_ptr()
    assert actual_dgrad.data_ptr() == dgrad_buffer.data_ptr()


def test_moonep_manager_dispatch_combine_accepts_missing_zero_copy_buffers(monkeypatch):
    from megatron.core.transformer.moe import token_dispatcher
    from megatron.core.transformer.moe.token_dispatcher import _MoonEPManager

    manager = _MoonEPManager.__new__(_MoonEPManager)
    manager._zero_copy_token_buffers = None
    manager._dispatch_hidden_buffer_pool = None
    manager._buffer = object()
    manager._bridge = SimpleNamespace(last_plan="plan")
    manager.token_probs = torch.randn(3, 2)
    manager.token_indices = torch.zeros(3, 2, dtype=torch.int32)
    manager.tokens_per_expert = torch.ones(2, dtype=torch.int32)
    hidden = torch.randn(3, 4)
    calls = {}

    def ensure_buffer(actual_hidden):
        calls["ensure_buffer"] = actual_hidden

    def fake_dispatch(
        hidden_states,
        topk_probs,
        topk_indices,
        tokens_per_expert,
        buffer,
        bridge,
        dispatch_buffer_pool=None,
        dgrad_hidden_buffer=None,
    ):
        calls["dispatch"] = {
            "hidden": hidden_states,
            "probs": topk_probs,
            "indices": topk_indices,
            "tokens_per_expert": tokens_per_expert,
            "buffer": buffer,
            "bridge": bridge,
            "dispatch_buffer_pool": dispatch_buffer_pool,
            "dgrad_hidden_buffer": dgrad_hidden_buffer,
        }
        return hidden_states + 1, topk_probs, torch.ones(2, dtype=torch.int64)

    def fake_combine(expert_output, buffer, plan, bridge, fwd_hidden_buffer=None):
        calls["combine"] = {
            "expert_output": expert_output,
            "buffer": buffer,
            "plan": plan,
            "bridge": bridge,
            "fwd_hidden_buffer": fwd_hidden_buffer,
        }
        return expert_output - 1

    manager._ensure_buffer = ensure_buffer
    monkeypatch.setattr(token_dispatcher, "moonep_dispatch", fake_dispatch)
    monkeypatch.setattr(token_dispatcher, "moonep_combine", fake_combine)

    dispatched = manager.dispatch(hidden)
    combined = manager.combine(dispatched)

    assert calls["ensure_buffer"] is hidden
    assert calls["dispatch"]["dgrad_hidden_buffer"] is None
    assert calls["dispatch"]["dispatch_buffer_pool"] is None
    assert calls["dispatch"]["buffer"] is manager._buffer
    assert calls["combine"]["fwd_hidden_buffer"] is None
    assert calls["combine"]["plan"] == "plan"
    torch.testing.assert_close(combined, hidden)
    assert manager.handle is None
    assert manager.dispatched_probs is None
    assert manager._dispatch_capacity is None


def test_moonep_metadata_uses_fixed_gpu_histogram(monkeypatch):
    from megatron.core.transformer.moe.token_dispatcher import _MoonEPManager

    manager = _MoonEPManager.__new__(_MoonEPManager)
    manager.num_experts = 4
    manager.router_topk = 2
    probs = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 4.0, 3.0, 2.0], [4.0, 1.0, 3.0, 2.0]])
    routing_map = torch.zeros_like(probs, dtype=torch.bool)

    def unexpected_bincount(*_args, **_kwargs):
        raise AssertionError("MoonEP metadata must not call torch.bincount")

    monkeypatch.setattr(torch, "bincount", unexpected_bincount)
    manager.setup_metadata(routing_map, probs)

    torch.testing.assert_close(
        manager.tokens_per_expert, torch.tensor([2, 2, 2, 0], dtype=torch.int32)
    )


def test_moonep_finalize_is_idempotent(monkeypatch):
    class _Resource:
        def __init__(self):
            self.destroy_calls = 0

        def destroy(self):
            self.destroy_calls += 1

    buffer = _Resource()
    bridge = _Resource()
    dispatch_pool = _Resource()
    token_pool = _Resource()
    token_buffers = {"test": token_pool}
    monkeypatch.setattr(fused_a2a, "_moonep_buffers", weakref.WeakSet([buffer]))
    monkeypatch.setattr(fused_a2a, "_moonep_bridges", weakref.WeakSet([bridge]))
    monkeypatch.setattr(
        fused_a2a, "_moonep_dispatch_buffer_pools", weakref.WeakSet([dispatch_pool])
    )
    monkeypatch.setattr(fused_a2a, "_moonep_token_buffer_pools", token_buffers)

    fused_a2a.moonep_finalize()
    fused_a2a.moonep_finalize()

    assert buffer.destroy_calls == 1
    assert bridge.destroy_calls == 1
    assert dispatch_pool.destroy_calls == 1
    assert token_pool.destroy_calls == 1
    assert token_buffers == {}


def test_moonep_map_chunks_supports_shareables_api(monkeypatch):
    calls = []

    def fake_map(*, chunk_shape, dtype, shareables, local_rank, world_size):
        calls.append(
            {
                "chunk_shape": chunk_shape,
                "dtype": dtype,
                "shareables": shareables,
                "local_rank": local_rank,
                "world_size": world_size,
            }
        )
        return "mapped"

    fake_map.__doc__ = (
        "nvl_dist_map(chunk_shape: Sequence[int], dtype: torch.dtype, "
        "shareables: torch.Tensor, local_rank: int, world_size: int)"
    )
    monkeypatch.setattr(fused_a2a, "_moonep_nvl_dist_map", fake_map, raising=False)

    mapped = fused_a2a._map_moonep_chunks(
        chunk_shape=[2, 3],
        dtype=torch.float32,
        fds=[torch.tensor(7), 8],
        local_rank=1,
        world_size=2,
    )

    assert mapped == "mapped"
    assert calls[0]["chunk_shape"] == [2, 3]
    assert calls[0]["dtype"] == torch.float32
    assert calls[0]["shareables"].dtype == torch.int64
    assert calls[0]["shareables"].tolist() == [7, 8]
    assert calls[0]["local_rank"] == 1
    assert calls[0]["world_size"] == 2


def test_moonep_map_chunks_supports_fds_api(monkeypatch):
    calls = []

    def fake_map(*, chunk_shape, dtype, fds, local_rank, world_size):
        calls.append(
            {
                "chunk_shape": chunk_shape,
                "dtype": dtype,
                "fds": fds,
                "local_rank": local_rank,
                "world_size": world_size,
            }
        )
        return "mapped"

    fake_map.__doc__ = (
        "nvl_dist_map(chunk_shape: Sequence[int], dtype: torch.dtype, "
        "fds: Sequence[int], local_rank: int, world_size: int)"
    )
    monkeypatch.setattr(fused_a2a, "_moonep_nvl_dist_map", fake_map, raising=False)

    mapped = fused_a2a._map_moonep_chunks(
        chunk_shape=[2, 3],
        dtype=torch.float32,
        fds=[torch.tensor(7), 8],
        local_rank=1,
        world_size=2,
    )

    assert mapped == "mapped"
    assert calls[0]["chunk_shape"] == [2, 3]
    assert calls[0]["dtype"] == torch.float32
    assert calls[0]["fds"] == [7, 8]
    assert calls[0]["local_rank"] == 1
    assert calls[0]["world_size"] == 2


def test_moonep_buffer_dispatch_omits_external_buffer_when_unsupported():
    calls = []

    class NewBuffer:
        def dispatch(
            self,
            hidden_sh,
            route_weights_sk=None,
            topk_experts_sk=None,
            tokens_per_expert=None,
            plan=None,
            *,
            inter_rank_sync=True,
            zero_copy=False,
            router_weights_zero_copy=False,
        ):
            del route_weights_sk, topk_experts_sk, tokens_per_expert, plan, inter_rank_sync
            calls.append(
                {
                    "hidden": hidden_sh,
                    "zero_copy": zero_copy,
                    "router_weights_zero_copy": router_weights_zero_copy,
                }
            )
            return "hidden", "weights", "cu_seqlens", "plan"

    hidden = torch.randn(2, 3)
    returned = fused_a2a._call_moonep_buffer_dispatch(
        NewBuffer(),
        hidden,
        torch.randn(2, 1),
        torch.zeros(2, 1, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
        zero_copy=True,
        hidden_buffer=(object(), torch.empty_like(hidden)),
    )

    assert returned == ("hidden", "weights", "cu_seqlens", "plan")
    assert calls[0]["hidden"] is hidden
    assert calls[0]["zero_copy"] is False
    assert calls[0]["router_weights_zero_copy"] is False


def test_moonep_buffer_combine_omits_external_buffer_when_unsupported():
    calls = []

    class NewBuffer:
        def combine(
            self,
            plan=None,
            hidden_nvsh=None,
            route_weights_nvs=None,
            async_finish=False,
            inter_rank_sync=True,
            *,
            zero_copy=False,
            router_weights_zero_copy=False,
        ):
            del async_finish, inter_rank_sync
            calls.append(
                {
                    "plan": plan,
                    "hidden": hidden_nvsh,
                    "route_weights": route_weights_nvs,
                    "zero_copy": zero_copy,
                    "router_weights_zero_copy": router_weights_zero_copy,
                }
            )
            return "combined", "weights", None

    hidden = torch.randn(2, 3)
    route_weights = torch.randn(2)
    returned = fused_a2a._call_moonep_buffer_combine(
        NewBuffer(),
        plan="plan",
        hidden_nvsh=hidden,
        route_weights_nvs=route_weights,
        zero_copy=True,
        hidden_buffer=(object(), torch.empty_like(hidden)),
    )

    assert returned == ("combined", "weights", None)
    assert calls[0]["plan"] == "plan"
    assert calls[0]["hidden"] is hidden
    assert calls[0]["route_weights"] is route_weights
    assert calls[0]["zero_copy"] is False
    assert calls[0]["router_weights_zero_copy"] is False


def _make_moonep_config(**overrides) -> TransformerConfig:
    kwargs = {
        "num_layers": 1,
        "hidden_size": 256,
        "ffn_hidden_size": 512,
        "num_attention_heads": 4,
        "num_moe_experts": 8,
        "expert_model_parallel_size": 8,
        "expert_tensor_parallel_size": 1,
        "moe_router_topk": 2,
        "moe_token_dispatcher_type": "flex",
        "moe_flex_dispatcher_backend": "moonep",
        "moe_router_dtype": "fp32",
        "moe_grouped_gemm": True,
        "moe_single_grouped_weight": True,
        "use_transformer_engine_op_fuser": True,
        "gradient_accumulation_fusion": True,
        "add_bias_linear": False,
        "bf16": True,
        "params_dtype": torch.bfloat16,
        "gated_linear_unit": True,
        "activation_func": F.silu,
    }
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


@pytest.fixture
def moonep_config(monkeypatch):
    """Avoid making config validation depend on the TE version in the unit-test environment."""
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_config.is_te_min_version", lambda _: True
    )
    return _make_moonep_config


@pytest.mark.parametrize(
    "activation_overrides",
    [
        {"activation_func": F.silu, "gated_linear_unit": True},
        {"activation_func": quick_gelu, "gated_linear_unit": True},
        {
            "activation_func": squared_relu,
            "gated_linear_unit": False,
            "use_fused_weighted_squared_relu": True,
        },
    ],
)
def test_moonep_accepts_supported_activations(moonep_config, activation_overrides):
    config = moonep_config(**activation_overrides)

    assert config.moe_flex_dispatcher_backend == "moonep"


def test_moonep_accepts_latent_moe(moonep_config):
    config = moonep_config(moe_latent_size=128)

    assert config.moe_latent_size == 128


def test_moonep_rejects_unaligned_latent_size(moonep_config):
    with pytest.raises(ValueError, match="moe_latent_size.*divisible by 128"):
        moonep_config(moe_latent_size=64)


@pytest.mark.parametrize(
    ("override", "requirement"),
    [
        ({"moe_token_dispatcher_type": "alltoall"}, "moe_token_dispatcher_type='flex'"),
        ({"bf16": False, "params_dtype": torch.float32}, "BF16 execution"),
        ({"add_bias_linear": True}, "add_bias_linear=False"),
        ({"moe_grouped_gemm": False}, "moe_grouped_gemm=True"),
        ({"moe_single_grouped_weight": False}, "moe_single_grouped_weight=True"),
        ({"use_transformer_engine_op_fuser": False}, "use_transformer_engine_op_fuser=True"),
        ({"gradient_accumulation_fusion": False}, "gradient_accumulation_fusion=True"),
        ({"moe_router_dtype": None}, "moe_router_dtype='fp32'"),
        ({"expert_tensor_parallel_size": 2}, "expert_tensor_parallel_size=1"),
        ({"moe_router_topk": 33}, "moe_router_topk<=32"),
    ],
)
def test_moonep_rejects_missing_required_flags(moonep_config, override, requirement):
    with pytest.raises(ValueError, match=requirement):
        moonep_config(**override)


@pytest.mark.parametrize(
    "override",
    [
        {"fp8": "e4m3", "fp8_recipe": "mxfp8"},
        {"cuda_graph_impl": "local"},
        {"delay_wgrad_compute": True},
        {"overlap_dispatch_backward_with_experts_wgrad": True},
        {"overlap_moe_expert_parallel_comm": True},
        {"moe_shared_expert_overlap": True},
        {"moe_expert_capacity_factor": 1.0},
        {"moe_pad_expert_input_to_capacity": True, "moe_expert_capacity_factor": 1.0},
        {"moe_router_padding_for_quantization": True},
        {"moe_apply_probs_on_input": True},
    ],
)
def test_moonep_rejects_unsupported_features(moonep_config, override):
    with pytest.raises(ValueError, match="MoonEP flex dispatcher configuration is unsupported"):
        moonep_config(**override)


def test_moonep_rejects_unsupported_activation(moonep_config):
    with pytest.raises(ValueError, match="weighted squared-ReLU activation"):
        moonep_config(activation_func=F.gelu, gated_linear_unit=True)


def test_moonep_manager_reports_missing_optional_package(moonep_config, monkeypatch):
    from megatron.core.transformer.moe.token_dispatcher import _MoonEPManager

    config = moonep_config()
    monkeypatch.setattr(
        "megatron.core.transformer.moe.token_dispatcher.is_moonep_available", lambda: False
    )

    with pytest.raises(ImportError, match="MoonEP is not installed"):
        _MoonEPManager(
            group=None,
            num_local_experts=1,
            router_topk=config.moe_router_topk,
            num_experts=config.num_moe_experts,
            config=config,
        )


@pytest.mark.skipif(not fused_a2a.HAVE_MOONEP, reason="MoonEP is not installed")
def test_moonep_availability_helper():
    assert fused_a2a.is_moonep_available()


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_MOONEP,
    reason="CUDA and MoonEP are required",
)
def test_moonep_four_rank_dispatch_probability_grad_and_redundant_counts():
    """Run with ``torch.distributed.run --nproc_per_node=4`` on an NVLink node."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("MoonEP distributed coverage requires a 4-rank torchrun launch")

    from megatron.core import parallel_state
    from megatron.core.transformer.moe.token_dispatcher import _MoonEPManager
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, expert_model_parallel_size=4, expert_tensor_parallel_size=1
    )
    group = parallel_state.get_expert_tensor_and_model_parallel_group()
    config = SimpleNamespace(hidden_size=128, moe_flex_dispatcher_num_sms=None)
    manager = _MoonEPManager(
        group=group, num_local_experts=1, router_topk=2, num_experts=4, config=config
    )
    manager._bridge = _FakeWeightBridge(device="cuda")

    try:
        num_tokens = 16
        hidden = torch.randn(num_tokens, 128, device="cuda", dtype=torch.bfloat16)
        hidden.requires_grad_(True)
        # Every rank strongly favors experts 0 and 1. Non-owner ranks must use
        # MoonEP's redundant B slot for at least one of those experts.
        logits = torch.full((num_tokens, 4), -8.0, device="cuda")
        logits[:, 0] = 8.0
        logits[:, 1] = 7.0
        logits.requires_grad_(True)
        dense_probs = torch.softmax(logits, dim=-1)
        _, indices = torch.topk(dense_probs, 2, dim=-1)
        routing_map = torch.zeros_like(dense_probs, dtype=torch.bool)
        routing_map.scatter_(1, indices, True)

        manager.setup_metadata(routing_map, dense_probs)
        dispatched = manager.dispatch(hidden)
        runtime_counts = manager.get_number_of_tokens_per_expert()
        valid_hidden, valid_probs = manager.get_permuted_hidden_states_by_experts(dispatched)
        expert_output = (valid_hidden * valid_probs.unsqueeze(-1)).to(hidden.dtype)
        expert_output = manager.get_restored_hidden_states_by_experts(expert_output)
        output = manager.combine(expert_output)

        expected = hidden * manager.token_probs.sum(dim=-1, keepdim=True).to(hidden.dtype)
        torch.testing.assert_close(output, expected)
        output.float().sum().backward()
        assert hidden.grad is not None
        assert logits.grad is not None and torch.count_nonzero(logits.grad) > 0
        assert runtime_counts.numel() == 5  # E+B with E=4 and B=1.
        slot_tokens = runtime_counts[4:].sum().to(torch.int64)
        torch.distributed.all_reduce(slot_tokens, group=group)
        assert slot_tokens.item() > 0
    finally:
        fused_a2a.moonep_finalize()
        Utils.destroy_model_parallel()


def _set_main_grad(parameter):
    rowwise_data = getattr(parameter, "rowwise_data", parameter)
    parameter.main_grad = torch.zeros_like(rowwise_data).view(parameter.shape)
    parameter.grad_added_to_main_grad = False
    parameter.overwrite_main_grad = True


def _set_main_grads(layer):
    for linear in (layer.experts.linear_fc1, layer.experts.linear_fc2):
        _set_main_grad(linear.get_parameter("weight"))
    if layer.config.moe_latent_size is not None:
        _set_main_grad(layer.fc1_latent_proj.weight)
        _set_main_grad(layer.fc2_latent_proj.weight)


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_MOONEP,
    reason="CUDA and MoonEP are required",
)
@pytest.mark.parametrize(
    (
        "activation_func",
        "gated_linear_unit",
        "weighted_squared_relu",
        "glu_interleave",
        "moe_latent_size",
    ),
    [
        (F.silu, True, False, None, None),
        (F.silu, True, False, 128, None),
        (quick_gelu, True, False, None, None),
        (squared_relu, False, True, None, None),
        (F.silu, True, False, None, 512),
    ],
)
def test_moonep_full_layer_parity_with_alltoall(
    monkeypatch,
    activation_func,
    gated_linear_unit,
    weighted_squared_relu,
    glu_interleave,
    moe_latent_size,
):
    """Compare full expert/router fwd+bwd and grouped main_grads on 4 NVLink GPUs."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("MoonEP distributed coverage requires a 4-rank torchrun launch")

    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.core.transformer.transformer_config import TransformerConfig
    from tests.unit_tests.test_utilities import Utils

    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, expert_model_parallel_size=4, expert_tensor_parallel_size=1
    )

    common = {
        "num_layers": 1,
        "hidden_size": 1024,
        "ffn_hidden_size": 1024,
        "moe_ffn_hidden_size": 1024,
        "num_attention_heads": 8,
        "num_moe_experts": 4,
        "expert_model_parallel_size": 4,
        "expert_tensor_parallel_size": 1,
        "moe_router_topk": 2,
        "moe_router_load_balancing_type": "none",
        "moe_router_dtype": "fp32",
        "moe_grouped_gemm": True,
        "moe_single_grouped_weight": True,
        "use_transformer_engine_op_fuser": True,
        "gradient_accumulation_fusion": True,
        "add_bias_linear": False,
        "bf16": True,
        "params_dtype": torch.bfloat16,
        "use_cpu_initialization": False,
        "activation_func": activation_func,
        "gated_linear_unit": gated_linear_unit,
        "use_fused_weighted_squared_relu": weighted_squared_relu,
        "moe_mlp_glu_interleave_size": glu_interleave,
        "moe_latent_size": moe_latent_size,
    }
    alltoall_config = TransformerConfig(**common, moe_token_dispatcher_type="alltoall")
    moonep_config = TransformerConfig(
        **common, moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="moonep"
    )
    mlp_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=4, moe_grouped_gemm=True
    ).submodules.mlp
    submodules = get_submodules(mlp_spec)

    try:
        ref_layer = MoELayer(alltoall_config, submodules).cuda()
        moonep_layer = MoELayer(moonep_config, submodules).cuda()
        moonep_layer.load_state_dict(ref_layer.state_dict())
        assert moonep_layer.state_dict().keys() == ref_layer.state_dict().keys()
        _set_main_grads(ref_layer)
        _set_main_grads(moonep_layer)

        torch.manual_seed(1234)
        test_input = torch.randn(2, 4, 1024, device="cuda", dtype=torch.bfloat16)

        def run(layer):
            hidden = test_input.detach().clone().requires_grad_(True)
            output, _ = layer(hidden)
            output.float().sum().backward()
            values = [
                output.detach(),
                hidden.grad.detach(),
                layer.router.weight.grad.detach().clone(),
                layer.experts.linear_fc1.weight.main_grad.detach().clone(),
                layer.experts.linear_fc2.weight.main_grad.detach().clone(),
            ]
            if layer.config.moe_latent_size is not None:
                values.extend(
                    [
                        layer.fc1_latent_proj.weight.main_grad.detach().clone(),
                        layer.fc2_latent_proj.weight.main_grad.detach().clone(),
                    ]
                )
            return values

        ref_values = run(ref_layer)
        moonep_values = run(moonep_layer)
        value_names = ["output", "input grad", "router grad", "FC1 main_grad", "FC2 main_grad"]
        if moe_latent_size is not None:
            value_names.extend(["latent FC1 main_grad", "latent FC2 main_grad"])
        for value_name, actual, expected in zip(value_names, moonep_values, ref_values):
            torch.testing.assert_close(
                actual, expected, rtol=2e-2, atol=2e-2, msg=lambda msg: f"{value_name}: {msg}"
            )
    finally:
        fused_a2a.moonep_finalize()
        Utils.destroy_model_parallel()
