# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import re
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """Configuration and compact-routing tests need no downloaded assets."""


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
        _VirtualExperts(None, ((parameter,),))


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

    assert config.moe_expert_rank_capacity_factor is None
    assert config.moe_single_grouped_weight is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        # The feature augments HybridEP rather than introducing another dispatcher backend.
        ({"moe_flex_dispatcher_backend": "deepep"}, "moe_flex_dispatcher_backend='hybridep'"),
        ({"moe_token_dispatcher_type": "alltoall"}, "moe_token_dispatcher_type='flex'"),
        # Every runtime expert needs its own weight address for the owner push.
        ({"moe_single_grouped_weight": True}, "moe_single_grouped_weight=False"),
        ({"moe_grouped_gemm": False}, "moe_grouped_gemm=True"),
        ({"use_transformer_engine_op_fuser": False}, "moe_use_grouped_tensor=True"),
        # The bridge reads wgrads out of main_grad buffers.
        ({"gradient_accumulation_fusion": False}, "gradient_accumulation_fusion=True"),
        ({"add_bias_linear": True}, "add_bias_linear=False"),
        ({"moe_router_dtype": "fp64"}, "moe_router_dtype='fp32'"),
        # The planner owns dispatch scheduling, so these overlap paths conflict.
        ({"delay_wgrad_compute": True}, "delay_wgrad_compute=False"),
        ({"moe_shared_expert_overlap": True}, "moe_shared_expert_overlap=False"),
        ({"moe_expert_capacity_factor": 1.0}, "moe_expert_capacity_factor=None"),
        # Route ids are packed against these limits.
        ({"moe_router_topk": 33}, "moe_router_topk<=min(32, num_moe_experts)"),
        # The transport tile assumes 128-aligned projections.
        ({"moe_ffn_hidden_size": 129}, "moe_ffn_hidden_size divisible by 128"),
        ({"hidden_size": 129, "kv_channels": 32}, "moe_latent_size (or hidden_size)"),
        # Only fused SwiGLU, quick-GeGLU and weighted squared-ReLU are supported.
        ({"activation_func": F.gelu}, "fused SwiGLU"),
        ({"params_dtype": torch.float32}, "params_dtype=torch.bfloat16"),
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
    with pytest.raises(ValueError, match="fp8_param=True"):
        _virtual_expert_hybridep_config(fp8=fp8, fp8_recipe=fp8_recipe, fp8_param=fp8_param)


@pytest.mark.parametrize("scope", ["moe_router", "moe_preprocess"])
def test_virtual_expert_hybridep_rejects_partial_moe_cuda_graph_scopes(scope):
    """Only the whole-layer moe scope preserves the planner's per-forward metadata."""
    with pytest.raises(AssertionError, match="moe CUDA graph scope only"):
        _virtual_expert_hybridep_config(cuda_graph_impl="local", cuda_graph_modules=[scope])


@pytest.mark.parametrize("scope", [None, "full", "selective"])
def test_virtual_expert_accepts_grouped_tensor_without_op_fuser(scope):
    """Runtime binding supports both TE execution paths, including recomputation."""
    kwargs = dict(
        use_transformer_engine_op_fuser=False,
        moe_use_grouped_tensor=True,
        recompute_granularity=scope,
    )
    if scope == "full":
        kwargs.update(recompute_method="uniform", recompute_num_layers=1)
    elif scope == "selective":
        kwargs.update(recompute_modules=["moe"])
    config = _virtual_expert_hybridep_config(**kwargs)
    assert config.recompute_granularity == scope
    assert not config.use_transformer_engine_op_fuser


@pytest.mark.parametrize(
    "fp8_param_gather,reuse_grad_buf", [(False, False), (False, True), (True, False), (True, True)]
)
def test_virtual_expert_mxfp8_requires_persistent_ddp_parameter_gather(
    fp8_param_gather, reuse_grad_buf
):
    from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel

    model = SimpleNamespace(moe_virtual_expert_load_balance=True, fp8="e4m3")
    ddp = SimpleNamespace(
        fp8_param_gather=fp8_param_gather, reuse_grad_buf_for_mxfp8_param_ag=reuse_grad_buf
    )
    if fp8_param_gather and reuse_grad_buf:
        DistributedDataParallel._validate_config(model, ddp)
    else:
        with pytest.raises(ValueError, match="MoonEP with MXFP8 requires"):
            DistributedDataParallel._validate_config(model, ddp)
    # The additional gather contract applies only to native MXFP8 MoonEP.
    model.fp8 = None
    DistributedDataParallel._validate_config(model, ddp)
    model.fp8, model.moe_virtual_expert_load_balance = "e4m3", False
    DistributedDataParallel._validate_config(model, ddp)


def test_virtual_expert_accepts_legacy_bias_and_explicit_capacity():
    config = _virtual_expert_hybridep_config(
        moe_router_enable_expert_bias=True,
        moe_router_score_function="sigmoid",
        moe_expert_rank_capacity_factor=2.0,
    )
    assert config.moe_router_enable_expert_bias
    assert config.moe_expert_rank_capacity_factor == 2.0
    # Op-fuser normalizes the old CLI default to the required runtime format.
    assert config.moe_use_grouped_tensor


@pytest.mark.parametrize("factor", [0.0, -1.0, float("nan"), float("inf")])
def test_virtual_expert_rejects_invalid_explicit_capacity(factor):
    with pytest.raises(ValueError, match="finite positive"):
        _virtual_expert_hybridep_config(moe_expert_rank_capacity_factor=factor)


@pytest.mark.parametrize("moonep", [False, True])
def test_expert_bias_counts_semantic_routes_and_ignores_padding_and_no_grad(moonep):
    from megatron.core.transformer.moe.router import TopKRouter

    router = SimpleNamespace(
        enable_expert_bias=True,
        config=SimpleNamespace(moe_virtual_expert_load_balance=moonep, num_moe_experts=4),
        local_tokens_per_expert=torch.zeros(4, device="cuda"),
    )
    routes = torch.tensor([[0, 3], [0, 2], [2, 3]], device="cuda")
    padding = torch.tensor([False, True, False], device="cuda")
    expected = torch.tensor([1.0, 0.0, 1.0, 2.0], device="cuda")
    with torch.no_grad():
        TopKRouter._apply_expert_bias(router, routes, padding)
    torch.testing.assert_close(router.local_tokens_per_expert, torch.zeros_like(expected))
    TopKRouter._apply_expert_bias(router, routes, padding)
    torch.testing.assert_close(router.local_tokens_per_expert, expected)
