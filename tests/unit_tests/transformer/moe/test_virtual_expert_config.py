# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import re
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from megatron.core.transformer import transformer_config as transformer_config_module
from megatron.core.transformer.cuda_graph_config import validate_moe_cuda_graph_support
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture(autouse=True)
def _assume_required_transformer_engine_version(monkeypatch):
    """These tests exercise config semantics without constructing Transformer Engine ops."""
    monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda *_args: True)


def _virtual_expert_hybridep_config(**overrides):
    """Build a minimal virtual-expert HybridEP config, then apply overrides."""
    kwargs = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        expert_model_parallel_size=2,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_virtual_expert_load_balance=True,
        moe_grouped_gemm=True,
        moe_router_dtype="fp32",
        use_transformer_engine_op_fuser=True,
        gradient_accumulation_fusion=True,
        add_bias_linear=False,
        activation_func=F.silu,
        gated_linear_unit=True,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_virtual_expert_hybridep_defaults_a_dropless_rank_capacity():
    """The backend is dropless by construction and allows the whole-layer MoE graph."""
    config = _virtual_expert_hybridep_config(
        cuda_graph_impl="local", cuda_graph_modules=["moe"]
    )

    assert config.moe_expert_rank_capacity_factor == 1.0
    assert config.moe_single_grouped_weight is False


@pytest.mark.parametrize("cuda_graph_impl", ["local", "transformer_engine"])
def test_virtual_expert_hybridep_allows_whole_moe_cuda_graph(cuda_graph_impl):
    """The runtime graph validator must preserve the config-time support decision."""
    config = _virtual_expert_hybridep_config(
        cuda_graph_impl=cuda_graph_impl, cuda_graph_modules=["moe"]
    )

    validate_moe_cuda_graph_support(config)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"moe_flex_dispatcher_backend": "deepep"},
            "--moe-token-dispatcher-type flex and --moe-flex-dispatcher-backend hybridep",
        ),
        (
            {"moe_token_dispatcher_type": "alltoall"},
            "--moe-token-dispatcher-type flex and --moe-flex-dispatcher-backend hybridep",
        ),
        ({"moe_single_grouped_weight": True}, "moe_single_grouped_weight=False"),
        ({"moe_grouped_gemm": False}, "moe_grouped_gemm=True"),
        ({"use_transformer_engine_op_fuser": False}, "use_transformer_engine_op_fuser=True"),
        ({"gradient_accumulation_fusion": False}, "gradient_accumulation_fusion=True"),
        ({"add_bias_linear": True}, "add_bias_linear=False"),
        ({"moe_router_dtype": "fp64"}, "moe_router_dtype='fp32'"),
        ({"delay_wgrad_compute": True}, "delay_wgrad_compute=False"),
        ({"moe_shared_expert_overlap": True}, "moe_shared_expert_overlap=False"),
        ({"moe_expert_capacity_factor": 1.0}, "moe_expert_capacity_factor=None"),
        ({"moe_hybridep_pad_variable_tokens": True}, "moe_hybridep_pad_variable_tokens=False"),
        ({"moe_router_topk": 33}, "moe_router_topk<=32"),
        ({"moe_ffn_hidden_size": 129}, "moe_ffn_hidden_size divisible by 128"),
        ({"hidden_size": 129, "kv_channels": 32}, "moe_latent_size (or hidden_size)"),
        ({"activation_func": F.gelu}, "fused SwiGLU"),
        ({"params_dtype": torch.float32}, "BF16 execution and BF16 parameters"),
        (
            {
                "fine_grained_activation_offloading": True,
                "offload_modules": ["expert_fc1"],
            },
            "no expert_fc1 or moe_act fine-grained activation offloading",
        ),
    ],
)
def test_virtual_expert_hybridep_rejects_unsupported_configurations(overrides, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        _virtual_expert_hybridep_config(**overrides)


def test_virtual_expert_hybridep_accepts_native_mxfp8_with_router_padding():
    """Native MXFP8 parameters are the only quantized storage the push understands."""
    config = _virtual_expert_hybridep_config(
        fp8="e4m3", fp8_recipe="mxfp8", fp8_param=True, moe_router_padding_for_quantization=True
    )

    assert (config.fp8, config.fp8_recipe, config.fp8_param) == ("e4m3", "mxfp8", True)
    assert config.moe_router_padding_for_quantization


@pytest.mark.parametrize(
    ("fp8", "fp8_recipe", "fp8_param"),
    [("e4m3", "mxfp8", False), ("e4m3", "tensorwise", True), ("hybrid", "mxfp8", True)],
)
def test_virtual_expert_hybridep_rejects_unsupported_fp8_parameter_storage(
    fp8, fp8_recipe, fp8_param
):
    with pytest.raises(ValueError, match="MXFP8 E4M3 with native FP8 parameters"):
        _virtual_expert_hybridep_config(fp8=fp8, fp8_recipe=fp8_recipe, fp8_param=fp8_param)


@pytest.mark.parametrize("scope", ["moe_router", "moe_preprocess"])
def test_virtual_expert_hybridep_rejects_partial_moe_cuda_graph_scopes(scope):
    """Only the whole-layer MoE scope preserves the planner's per-forward metadata."""
    with pytest.raises(AssertionError, match="moe CUDA graph scope only"):
        _virtual_expert_hybridep_config(cuda_graph_impl="local", cuda_graph_modules=[scope])


def test_virtual_expert_hash_routing_returns_compact_routes():
    """DSv4 hash layers feed compact top-k ids directly to the virtual-expert planner."""
    router = SimpleNamespace(
        score_function="softmax",
        topk=2,
        tid2eid=torch.tensor([[0, 1], [2, 3], [1, 3]], dtype=torch.int32),
        config=SimpleNamespace(
            moe_router_force_load_balancing=False,
            moe_router_force_biased=None,
            moe_router_topk_scaling_factor=1.0,
            moe_virtual_expert_load_balance=True,
        ),
    )
    logits = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [4.0, 3.0, 2.0, 1.0], [0.0, 2.0, 1.0, 3.0]]
    )
    input_ids = torch.tensor([[0], [1], [2]])

    probs, expert_ids = TopKRouter._hash_routing(router, logits, input_ids)

    expected_ids = router.tid2eid.long()
    expected_probs = torch.softmax(logits, dim=-1).gather(1, expected_ids)
    torch.testing.assert_close(probs, expected_probs)
    torch.testing.assert_close(expert_ids, expected_ids)
