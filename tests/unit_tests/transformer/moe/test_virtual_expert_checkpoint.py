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


from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.utils import init_method_normal

HIDDEN, FFN = 128, 256
_IS_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def bytes_equal(a, b):
    return torch.equal(
        a.detach().contiguous().view(torch.uint8), b.detach().contiguous().view(torch.uint8)
    )


def _config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=HIDDEN,
        ffn_hidden_size=FFN,
        num_attention_heads=16,
        num_query_groups=4,
        kv_channels=128,
        use_cpu_initialization=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        deterministic_mode=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


@pytest.mark.parametrize("parallel_mode", ["column", "row"])
@pytest.mark.parametrize("singleton_local_shards", [False, True])
@pytest.mark.parametrize(
    "quantized",
    [
        False,
        pytest.param(
            True,
            marks=[
                pytest.mark.internal,
                pytest.mark.launch_on_gb200,
                pytest.mark.skipif(not _IS_BLACKWELL, reason="MXFP8 parameters need Blackwell"),
            ],
        ),
    ],
    ids=["bf16", "mxfp8"],
)
def test_te_grouped_linear_virtual_expert_checkpoint(
    parallel_mode, singleton_local_shards, quantized
):
    """Runtime replicas preserve owned initialization and the upstream checkpoint layout."""
    Utils.initialize_distributed()
    if torch.distributed.get_world_size() % 2:
        pytest.skip("requires a world size divisible by EP=2")
    Utils.initialize_model_parallel(expert_model_parallel_size=2)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    config = _config(
        num_moe_experts=4,
        expert_model_parallel_size=2,
        moe_grouped_gemm=True,
        moe_single_grouped_weight=False,
        fp8="e4m3" if quantized else None,
        fp8_recipe=Fp8Recipe.mxfp8,
        fp8_param=quantized,
    )
    linear = (
        TEColumnParallelGroupedLinear if parallel_mode == "column" else TERowParallelGroupedLinear
    )

    def build(num_virtual_experts):
        with get_fp8_context(config, 0, is_init=True):
            return linear(
                2,
                128,
                256,
                num_virtual_experts=num_virtual_experts,
                config=config,
                init_method=init_method_normal(0.02),
                bias=False,
                skip_bias_add=False,
                is_expert=True,
                pg_collection=pg,
            ).cuda()

    def assert_weight_equal(actual, expected):
        if is_mxfp8tensor(actual):
            actual = actual.dequantize()
        if is_mxfp8tensor(expected):
            expected = expected.dequantize()
        assert bytes_equal(actual, expected)

    try:
        model_parallel_cuda_manual_seed(123)
        owned, following_owned = build(0), build(0)
        model_parallel_cuda_manual_seed(123)
        virtual, following_virtual = build(2), build(0)
        for module in (owned, following_owned, virtual, following_virtual):
            assert is_mxfp8tensor(module.weight0) is quantized
        assert virtual.num_gemms == 4
        assert set(dict(virtual.named_parameters())) == {"weight0", "weight1"}
        assert virtual.state_dict().keys() == owned.state_dict().keys()
        for index in range(2):
            assert_weight_equal(
                getattr(virtual, f"weight{index}"), getattr(owned, f"weight{index}")
            )
            assert_weight_equal(
                getattr(following_virtual, f"weight{index}"),
                getattr(following_owned, f"weight{index}"),
            )

        metadata = {"dp_cp_group": pg.dp_cp, "singleton_local_shards": singleton_local_shards}
        expected = owned.sharded_state_dict(prefix="experts.", metadata=metadata)
        actual = virtual.sharded_state_dict(prefix="experts.", metadata=metadata)
        assert actual.keys() == expected.keys()
        for index in range(2):
            global_expert = torch.distributed.get_rank(pg.ep) * 2 + index
            shard = actual[f"experts.weight{index}"]
            reference = expected[f"experts.weight{index}"]
            assert shard.key == reference.key
            assert shard.global_shape == reference.global_shape
            assert shard.global_offset == reference.global_offset
            assert_weight_equal(shard.data, reference.data)
            if singleton_local_shards:
                assert shard.key == f"{global_expert}.experts.weight"
                assert shard.global_shape == (256, 128)
                assert shard.global_offset == (0, 0)
            else:
                assert shard.global_shape == (4, 256, 128)
                assert shard.global_offset == (global_expert, 0, 0)

        # Distributed checkpoint loading presents one extra-state object per owned expert.
        restored = build(2)
        restored.load_state_dict(
            {name.removeprefix("experts."): shard.data for name, shard in actual.items()},
            strict=True,
        )
        for index in range(2):
            assert_weight_equal(
                getattr(restored, f"weight{index}"), getattr(virtual, f"weight{index}")
            )
        if quantized:
            assert restored.fp8_meta["num_gemms"] == restored.num_gemms

        # Loading in either direction must work regardless of the runtime slot count.
        with torch.no_grad():
            for index, parameter in enumerate(virtual.parameters()):
                parameter.fill_(torch.distributed.get_rank(pg.ep) * 2 + index + 1)
        owned.load_state_dict(virtual.state_dict(), strict=True)
        for index in range(2):
            assert_weight_equal(
                getattr(owned, f"weight{index}"), getattr(virtual, f"weight{index}")
            )
        with torch.no_grad():
            for parameter in virtual.parameters():
                parameter.zero_()
        virtual.load_state_dict(owned.state_dict(), strict=True)
        for index in range(2):
            assert_weight_equal(
                getattr(virtual, f"weight{index}"), getattr(owned, f"weight{index}")
            )
    finally:
        Utils.destroy_model_parallel()
