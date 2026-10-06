# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Non-GTP regressions adapted from PR 7450 (242c8dbc6) to dev compact routing."""

import gc
import os
from functools import partialmethod

import pytest
import torch
import torch.distributed as dist
from torch.nn import functional as F
from transformer_engine.pytorch.quantization import FP8GlobalStateManager

from megatron.core import parallel_state as ps
from megatron.core.activations import squared_relu
from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context, is_mxfp8tensor
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as offload,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.moe import fused_a2a, moe_utils
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_logging import destroy_moe_metrics_tracker
from megatron.core.transformer.moe.virtual_expert_load_balancer import (
    VirtualExpertLoadBalancer,
    plan_virtual_expert_routes,
)
from megatron.core.transformer.moe.virtual_expert_triton import VirtualExpertPlannerWorkspace
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

requires_hybridep = pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_HYBRIDEP,
    reason="requires CUDA and HybridEP",
)


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """These synthetic tests do not require downloaded datasets."""


def _assert_numerical_parity(actual, expected, tolerance, name, peak_tolerance=None):
    """Bound aggregate and peak error against each parameter's own signal scale."""
    actual, expected = (actual.float(), expected.float())
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    error = actual - expected
    error_norm, expected_norm = error.norm().item(), expected.norm().item()
    relative_l2 = (
        error_norm / expected_norm if expected_norm else (0 if error_norm == 0 else float('inf'))
    )
    assert (
        error_norm <= tolerance * expected_norm
    ), f'{name}: relative L2 error {relative_l2:.6%} exceeds {tolerance:.6%}'
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0,
        atol=(tolerance if peak_tolerance is None else peak_tolerance)
        * expected.abs().max().item(),
        msg=lambda msg: f'{name}: {msg}',
    )


def _assert_bitwise_parity(actual, expected, name):
    """Require identical finite values and storage bits, including signed zero."""
    assert actual.dtype == expected.dtype, name
    assert actual.shape == expected.shape, name
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    torch.testing.assert_close(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
        rtol=0,
        atol=0,
        msg=lambda msg: f'{name}: {msg}',
    )


@requires_hybridep
@pytest.mark.parametrize("use_op_fuser", [True, False], ids=["op-fuser", "grouped-tensor"])
def test_bf16_virtual_expert_routing_parity(monkeypatch, use_op_fuser):
    """Moving routes preserves BF16 outputs/dgrads; only wgrad summation may round differently."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.spec_utils import get_submodules

    monkeypatch.setattr(
        fused_a2a.HybridEPBuffer,
        '__init__',
        partialmethod(fused_a2a.HybridEPBuffer.__init__, load_cached_kernels=True),
    )
    monkeypatch.setenv('NVTE_GROUPED_LINEAR_SINGLE_PARAM', '0')
    Utils.initialize_model_parallel(expert_model_parallel_size=2)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    group = ps.get_model_parallel_group()
    finalize = VirtualExpertLoadBalancer.finalize
    resources_alive_at_finalize = []

    def observe_finalize():
        resources_alive_at_finalize.append(
            ps.get_model_parallel_group() is group
            and bool(VirtualExpertLoadBalancer.storages)
            and VirtualExpertLoadBalancer.planner is not None
        )
        finalize()

    monkeypatch.setattr(VirtualExpertLoadBalancer, 'finalize', observe_finalize)
    spec = get_submodules(
        get_gpt_layer_with_transformer_engine_spec(
            num_experts=4, moe_grouped_gemm=True
        ).submodules.mlp
    )

    def run(virtual):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        model_parallel_cuda_manual_seed(1234)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            ffn_hidden_size=256,
            moe_ffn_hidden_size=256,
            num_moe_experts=4,
            expert_model_parallel_size=2,
            moe_router_topk=2,
            moe_router_score_function='sigmoid',
            moe_router_topk_scaling_factor=3.16,
            moe_router_dtype='fp32',
            moe_router_load_balancing_type='none',
            moe_aux_loss_coeff=0,
            moe_token_dispatcher_type='flex',
            moe_flex_dispatcher_backend='hybridep',
            moe_virtual_expert_load_balance=virtual,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
            activation_func=squared_relu,
            gated_linear_unit=False,
            gradient_accumulation_fusion=True,
            moe_grouped_gemm=True,
            use_transformer_engine_op_fuser=use_op_fuser,
            use_fused_weighted_squared_relu=True,
            moe_use_grouped_tensor=True,
        )
        layer = None
        try:
            layer = MoELayer(config, spec, pg_collection=pg).cuda()
            assert config.fp8 is None
            for parameter in layer.parameters():
                parameter.main_grad = torch.zeros_like(parameter, dtype=torch.float32)
                parameter.grad_added_to_main_grad = False
            for parameter in layer.experts.parameters():
                assert not is_mxfp8tensor(parameter)
                assert parameter.dtype == torch.bfloat16
            if virtual:
                for linear in (layer.experts.linear_fc1, layer.experts.linear_fc2):
                    assert linear.num_gemms == 4
                    assert set(dict(linear.named_parameters(recurse=False))) == {
                        "weight0",
                        "weight1",
                    }
                    assert not any(
                        key.startswith(("weight2", "weight3")) for key in linear.state_dict()
                    )
            # Force traffic to opposite EP owners on successive uses of the same layer.
            # Distinct per-rank inputs expose dropped or duplicated contributions.
            with torch.no_grad():
                layer.router.weight.zero_()
                layer.router.weight[:, 0].copy_(torch.tensor([1, 0.5, -0.5, -1], device='cuda'))
                layer.router.weight[2, 1] = 2
            routes, inputs, outputs, upstreams, plans = [], [], [], [], []

            def capture_routes(module, args, output):
                probs, indices = output
                probs.retain_grad()
                if indices.dtype == torch.bool:
                    indices = indices.to(torch.int8).topk(config.moe_router_topk, dim=-1).indices
                else:
                    indices = indices.long()
                routes.append((probs, indices.clone()))

            layer.router.register_forward_hook(capture_routes)
            if virtual:
                manager = layer.token_dispatcher._comm_manager
                assert manager.moe_expert_rank_capacity_factor == 1.0
                assert manager.config.moe_expert_rank_capacity_factor is None
                dispatch = manager.plan_dispatch

                def record(*args):
                    dispatch(*args)
                    plans.append(manager._plan)

                monkeypatch.setattr(manager, 'plan_dispatch', record)
            weights = {
                name: p.detach().float().cpu().clone() for name, p in layer.named_parameters()
            }
            # Use multiple alignment blocks to retain native and virtual work for the hot expert.
            num_tokens = 512
            for use in range(2):
                rng = torch.Generator(device='cuda').manual_seed(8765 + 10 * use + pg.ep.rank())
                x = torch.randn(
                    num_tokens, 1, 128, device='cuda', dtype=torch.bfloat16, generator=rng
                )
                x[..., 0] = 1 if use == 0 else -1
                x[..., 1] = 0
                x[-num_tokens // 4 :, :, 1] = 1 if use == 0 else -1
                x.requires_grad_()
                y, bias = layer(x)
                assert bias is None
                if virtual:
                    assert manager._alignment == moe_utils.get_align_size_for_quantization(config)
                    assert manager.pad_multiple == manager._alignment
                    assert manager.tokens_per_expert.sum() <= manager.rank_capacity
                    assert (manager.tokens_per_expert % manager._alignment == 0).all()
                if virtual and not use_op_fuser:
                    for fc_layer, linear in enumerate(
                        (layer.experts.linear_fc1, layer.experts.linear_fc2)
                    ):
                        runtime_weights = manager.runtime_weights(fc_layer)
                        assert not hasattr(linear, "_runtime_weights")
                        assert all(
                            getattr(linear, f"weight{index}") is weight
                            for index, weight in enumerate(runtime_weights)
                        )
                        registered = {id(param) for param in linear.parameters(recurse=False)}
                        assert all(id(weight) not in registered for weight in runtime_weights)
                        assert linear._parameters["weight2"] is None
                        assert linear._parameters["weight3"] is None
                        assert not any(
                            key.startswith(("weight2", "weight3")) for key in linear.state_dict()
                        )
                inputs.append(x)
                outputs.append(y)
                upstreams.append(torch.randn(y.shape, device='cuda', dtype=y.dtype, generator=rng))
            if virtual:
                assert len(plans) == 2
                assert plans[0].experts_to_copy.data_ptr() != plans[1].experts_to_copy.data_ptr()
                for use, plan in enumerate(plans):
                    # The hot expert must run on both its owner and a virtual copy, so the
                    # comparison exercises split/reduced wgrads, not just whole-expert moves.
                    hot_routes = routes[use][1] == (0 if use == 0 else 3)
                    destinations = (plan.virtual_experts[hot_routes] // 4).long()
                    counts = torch.bincount(destinations, minlength=2)
                    torch.distributed.all_reduce(counts, group=pg.ep)
                    assert (counts > 0).all(), 'hot expert must retain native and virtual work'
            # Both forwards precede backward, as with a repeated MTP layer. Weights stay fixed;
            # optimizer drift must not weaken exact forward/input-gradient expectations.
            # Backpropagate in reverse use order, as the repeated MTP dependency does.
            for y, upstream in reversed(list(zip(outputs, upstreams))):
                y.backward(upstream)
            torch.cuda.synchronize()
            values = {}
            for use, (x, y, (probs, indices)) in enumerate(zip(inputs, outputs, routes)):
                # Both paths retain upstream full-width router probabilities. Compare
                # selected routes in expert order using gradients from the actual outputs.
                indices, order = indices.sort(dim=-1)
                route_probs = probs.gather(1, order)
                route_grads = probs.grad.gather(1, order)
                expected_routes = (
                    torch.tensor([0, 1] if use == 0 else [2, 3], device='cuda')
                    .expand(num_tokens, -1)
                    .clone()
                )
                expected_routes[-num_tokens // 4 :] = torch.tensor(
                    [0, 2] if use == 0 else [1, 3], device='cuda'
                )
                torch.testing.assert_close(indices.sort(dim=-1).values.long(), expected_routes)
                values[f'use {use}: routes'] = indices.cpu()
                values[f'use {use}: probabilities'] = route_probs.detach().cpu()
                values[f'use {use}: output'] = y.detach().cpu()
                values[f'use {use}: input gradient'] = x.grad.cpu()
                values[f'use {use}: probability gradient'] = route_grads.cpu()
            for name, parameter in layer.named_parameters():
                grad = parameter.main_grad if name.startswith('experts.') else parameter.grad
                values[f'wgrad {name}'] = grad.detach().cpu().clone()
            return weights, values
        finally:
            torch.cuda.synchronize()
            del layer
            destroy_moe_metrics_tracker()
            gc.collect()
            fused_a2a.reset_hybrid_ep_buffer()

    try:
        reference_weights, reference = run(False)
        actual_weights, actual = run(True)
        assert actual_weights.keys() == reference_weights.keys()
        for name in reference_weights:
            _assert_bitwise_parity(
                actual_weights[name], reference_weights[name], f'initial {name}'
            )
        assert actual.keys() == reference.keys()
        for name, expected in reference.items():
            if name.startswith('wgrad experts.'):
                assert expected.norm() > 0, name
                _assert_numerical_parity(actual[name], expected, 1e-5, f'BF16 {name}')
            else:
                _assert_bitwise_parity(actual[name], expected, f'BF16 {name}')
    finally:
        Utils.destroy_model_parallel()
    assert resources_alive_at_finalize == [True]
    assert not VirtualExpertLoadBalancer.storages
    assert VirtualExpertLoadBalancer.planner is None
    finalize()  # Cleanup stays safe after the model-parallel groups are gone.


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
        moe_use_grouped_tensor=True,
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


@requires_hybridep
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize(
    "recompute,offload_scope,use_op_fuser",
    [
        ("moe_act", None, False),
        ("moe", None, True),
        ("full", None, True),
        (None, "expert_fc1", False),
        (None, "moe_act", False),
        (None, "fused_group_mlp", True),
    ],
    ids=[
        "moe-act-unfused",
        "moe-layer",
        "full-layer",
        "offload-fc1",
        "offload-moe-act",
        "offload-fused-mlp",
    ],
)
def test_virtual_expert_recompute_offload_scopes(
    monkeypatch, precision, recompute, offload_scope, use_op_fuser
):
    """Exercise real MoonEP transfers through two forwards and reverse-order backward."""
    if offload_scope == "expert_fc1":
        pytest.skip("FC1 offload needs a separate HybridEP/TE capacity-buffer fix.")

    from transformer_engine.pytorch.ops.fused import grouped_mlp
    from transformer_engine.pytorch.ops.fuser import (
        OperationFuser,
        register_forward_backward_fusion,
    )

    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.tensor_parallel import checkpoint
    from megatron.core.transformer.spec_utils import get_submodules

    checks = (
        grouped_mlp._GroupedMLP_CuTeGEMMBase.is_supported,
        grouped_mlp.GroupedMLP_CuTeGEMMUnary.is_supported,
        grouped_mlp._grouped_gemm_dsrelu_backward_supported,
    )
    for check in checks:
        check.cache_clear()
    monkeypatch.setattr(
        OperationFuser,
        "forward_backward_fusion_functions",
        list(OperationFuser.forward_backward_fusion_functions),
    )
    for cls, fuse in (
        (grouped_mlp.GroupedMLP_CuTeGEMMGLU, grouped_mlp.fuse_ops),
        (grouped_mlp.GroupedMLP_CuTeGEMMUnary, grouped_mlp.fuse_srelu_ops),
    ):
        if cls.is_supported() and fuse not in OperationFuser.forward_backward_fusion_functions:
            register_forward_backward_fusion(fuse, prepend=True)
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "0")
    monkeypatch.setattr(
        fused_a2a.HybridEPBuffer,
        "__init__",
        partialmethod(fused_a2a.HybridEPBuffer.__init__, load_cached_kernels=True),
    )
    Utils.initialize_model_parallel(expert_model_parallel_size=2)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    spec = get_submodules(
        get_gpt_layer_with_transformer_engine_spec(
            num_experts=4, moe_grouped_gemm=True
        ).submodules.mlp
    )

    def run(enabled):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        overrides = dict(
            num_moe_experts=4,
            ffn_hidden_size=256,
            moe_ffn_hidden_size=256,
            moe_router_topk=2,
            moe_router_score_function="sigmoid",
            moe_router_topk_scaling_factor=3.16,
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0,
            gated_linear_unit=False,
            activation_func=squared_relu,
            use_fused_weighted_squared_relu=True,
            use_transformer_engine_op_fuser=use_op_fuser,
            moe_use_grouped_tensor=True,
            min_offloaded_tensor_size=0,
        )
        if precision == "mxfp8":
            overrides.update(
                fp8="e4m3",
                fp8_recipe=Fp8Recipe.mxfp8,
                fp8_param=True,
                moe_router_padding_for_quantization=True,
            )
        if enabled and recompute:
            overrides.update(
                recompute_granularity="full" if recompute == "full" else "selective",
                recompute_modules=[] if recompute == "full" else [recompute],
            )
            if recompute == "full":
                overrides.update(recompute_method="uniform", recompute_num_layers=1)
        if enabled and offload_scope:
            overrides.update(
                fine_grained_activation_offloading=True, offload_modules=[offload_scope]
            )
        config = _virtual_expert_hybridep_config(**overrides)
        layer = None
        try:
            with get_fp8_context(config, is_init=True):
                layer = MoELayer(config, spec, pg_collection=pg).cuda()
            weights = {
                name: parameter.detach().float().cpu().clone()
                for name, parameter in layer.named_parameters()
            }
            for parameter in layer.parameters():
                parameter.main_grad = torch.zeros_like(parameter, dtype=torch.float32)
                parameter.grad_added_to_main_grad = False
            with torch.no_grad():
                layer.router.weight.zero_()
                layer.router.weight[:, 0].copy_(torch.tensor([1, 0.5, -0.5, -1], device="cuda"))
                layer.router.weight[2, 1] = 2
            if enabled and offload_scope:
                offload.init_chunk_handler(
                    pp_rank=pg.pp.rank(),
                    vp_size=None,
                    vp_stage=None,
                    min_offloaded_tensor_size=0,
                    delta_offload_bytes_across_pp_ranks=0,
                    activation_offload_fraction=1.0,
                )
            inputs, outputs, upstreams = [], [], []
            for use in range(2):
                generator = torch.Generator(device="cuda").manual_seed(
                    8765 + 10 * use + pg.ep.rank()
                )
                x = torch.randn(
                    32, 1, 128, device="cuda", dtype=torch.bfloat16, generator=generator
                )
                x[..., 0] = 1 if use == 0 else -1
                x.requires_grad_()
                if enabled and recompute == "full":

                    def replay(input_tensor):
                        with get_fp8_context(config):
                            return layer(input_tensor)[0]

                    y = checkpoint(replay, False, x)
                else:
                    with get_fp8_context(config):
                        y, bias = layer(x)
                    assert bias is None
                inputs.append(x)
                outputs.append(y)
                upstreams.append(
                    torch.randn(y.shape, device="cuda", dtype=y.dtype, generator=generator)
                )
            for y, upstream in reversed(list(zip(outputs, upstreams))):
                y.backward(upstream)
            torch.cuda.synchronize()
            values = {
                **{f"output {use}": y.detach().cpu() for use, y in enumerate(outputs)},
                **{f"input gradient {use}": x.grad.cpu() for use, x in enumerate(inputs)},
                **{
                    f"expert gradient {name}": parameter.main_grad.cpu().clone()
                    for name, parameter in layer.experts.named_parameters()
                },
            }
            if enabled and offload_scope:
                offload.reset(process_group=pg.tp_dp_cp)
                assert (
                    PipelineOffloadManager.get_instance().offload_summary_bytes[offload_scope] > 0
                )
            return weights, values
        finally:
            torch.cuda.synchronize()
            del layer
            VirtualExpertLoadBalancer.finalize()
            FP8GlobalStateManager.reset()
            offload.reset_instance()
            destroy_moe_metrics_tracker()
            gc.collect()
            fused_a2a.reset_hybrid_ep_buffer()

    try:
        reference_weights, reference = run(False)
        actual_weights, actual = run(True)
        assert actual_weights.keys() == reference_weights.keys()
        for name, expected in reference_weights.items():
            _assert_bitwise_parity(actual_weights[name], expected, f"initial {name}")
        assert actual.keys() == reference.keys()
        for name, expected in reference.items():
            _assert_bitwise_parity(actual[name], expected, name)
    finally:
        Utils.destroy_model_parallel()
        for check in checks:
            check.cache_clear()
