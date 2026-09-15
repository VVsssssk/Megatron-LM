# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import re
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from megatron.core.transformer.transformer_config import TransformerConfig


def test_virtual_expert_hash_router_keeps_compact_semantic_ids():
    from megatron.core.transformer.moe.router import TopKRouter

    config = SimpleNamespace(
        moe_virtual_expert_load_balance=True,
        moe_router_force_load_balancing=False,
        moe_router_force_biased=None,
        moe_router_topk_scaling_factor=1.5,
    )
    router = SimpleNamespace(
        config=config, score_function="sigmoid", topk=2, tid2eid=torch.tensor([[0, 3], [2, 1]])
    )
    logits = torch.randn(2, 4, requires_grad=True)
    probs, indices = TopKRouter._hash_routing(router, logits, torch.tensor([0, 1]))
    assert indices.dtype == torch.int64
    torch.testing.assert_close(indices, router.tid2eid)
    assert probs.shape == indices.shape
    (probs * indices).sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_virtual_expert_rejects_gtp_weights_before_allocating_shared_arenas():
    from megatron.core.transformer.moe.virtual_expert_load_balancer import _VirtualExperts

    parameter = torch.nn.Parameter(torch.ones(1))
    parameter.is_gtp_weight_remat = True
    with pytest.raises(ValueError, match="does not support GTP"):
        _VirtualExperts(None, None, ((parameter,),))


def _virtual_expert_hybridep_config(**overrides):
    """Build a minimal virtual-expert HybridEP config, then apply one override."""
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
    """The backend is dropless by construction and allows the whole-layer moe graph."""
    config = _virtual_expert_hybridep_config(cuda_graph_impl="local", cuda_graph_modules=["moe"])

    assert config.moe_expert_rank_capacity_factor == 1.0
    assert config.moe_single_grouped_weight is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        # The feature augments HybridEP rather than introducing another dispatcher backend.
        (
            {"moe_flex_dispatcher_backend": "deepep"},
            "--moe-token-dispatcher-type flex and --moe-flex-dispatcher-backend hybridep",
        ),
        (
            {"moe_token_dispatcher_type": "alltoall"},
            "--moe-token-dispatcher-type flex and --moe-flex-dispatcher-backend hybridep",
        ),
        # Every runtime expert needs its own weight address for the owner push.
        ({"moe_single_grouped_weight": True}, "moe_single_grouped_weight=False"),
        ({"moe_grouped_gemm": False}, "moe_grouped_gemm=True"),
        ({"use_transformer_engine_op_fuser": False}, "use_transformer_engine_op_fuser=True"),
        # The bridge reads wgrads out of main_grad buffers.
        ({"gradient_accumulation_fusion": False}, "gradient_accumulation_fusion=True"),
        ({"add_bias_linear": True}, "add_bias_linear=False"),
        ({"moe_router_dtype": "fp64"}, "moe_router_dtype='fp32'"),
        # The planner owns dispatch scheduling, so these overlap paths conflict.
        ({"delay_wgrad_compute": True}, "delay_wgrad_compute=False"),
        ({"moe_shared_expert_overlap": True}, "moe_shared_expert_overlap=False"),
        ({"moe_expert_capacity_factor": 1.0}, "moe_expert_capacity_factor=None"),
        # Route ids are packed against these limits.
        ({"moe_router_topk": 33}, "moe_router_topk<=32"),
        # The transport tile assumes 128-aligned projections.
        ({"moe_ffn_hidden_size": 129}, "moe_ffn_hidden_size divisible by 128"),
        ({"hidden_size": 129, "kv_channels": 32}, "moe_latent_size (or hidden_size)"),
        # Only fused SwiGLU, quick-GeGLU and weighted squared-ReLU are supported.
        ({"activation_func": F.gelu}, "fused SwiGLU"),
        ({"params_dtype": torch.float32}, "BF16 execution and BF16 parameters"),
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
    """Only the whole-layer moe scope preserves the planner's per-forward metadata."""
    with pytest.raises(AssertionError, match="moe CUDA graph scope only"):
        _virtual_expert_hybridep_config(cuda_graph_impl="local", cuda_graph_modules=[scope])
