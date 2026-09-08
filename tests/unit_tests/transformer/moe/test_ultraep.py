# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.distributed.param_and_grad_buffer import group_params_for_buffers
from megatron.core.optimizer import _get_param_groups
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer.moe import ultraep_manager
from megatron.core.transformer.moe.experts import TEGroupedMLP, _parse_te_expert_idx
from megatron.core.transformer.moe.ultraep_manager import UltraEPManager
from megatron.core.transformer.transformer_config import TransformerConfig


def _ultraep_config_kwargs(**overrides):
    kwargs = dict(
        num_layers=43,
        hidden_size=64,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_ffn_hidden_size=128,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        pipeline_model_parallel_size=1,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type='alltoall',
        gradient_accumulation_fusion=True,
        params_dtype=torch.bfloat16,
        gated_linear_unit=True,
        add_bias_linear=False,
        mtp_num_layers=1,
        moe_enable_ultraep=True,
        moe_num_redundant_experts_per_rank=1,
    )
    kwargs.update(overrides)
    return kwargs


def test_ultraep_transformer_config_accepts_bf16_mtp_configuration():
    config = TransformerConfig(**_ultraep_config_kwargs())

    assert config.moe_enable_ultraep
    assert config.mtp_num_layers == 1


def test_ultraep_transformer_config_accepts_hybridep_dispatch():
    config = TransformerConfig(
        **_ultraep_config_kwargs(
            moe_token_dispatcher_type='flex', moe_flex_dispatcher_backend='hybridep'
        )
    )

    assert config.moe_flex_dispatcher_backend == 'hybridep'


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"moe_num_redundant_experts_per_rank": 0}, "greater than zero"),
        ({"moe_token_dispatcher_type": "allgather"}, "token_dispatcher_type"),
        ({"moe_grouped_gemm": False}, "moe_grouped_gemm"),
        ({"add_bias_linear": True}, "add_bias_linear"),
        (
            {"pipeline_model_parallel_size": 2, "pipeline_dtype": torch.bfloat16},
            "pipeline parallel size 1",
        ),
        ({"use_transformer_engine_op_fuser": True}, "operation fuser"),
        ({"fp8": "mxfp8"}, "BF16"),
        ({"moe_paged_stash": True}, "paged stash"),
        ({"moe_expert_rank_capacity_factor": 1.5}, "dropless"),
    ],
)
def test_ultraep_transformer_config_rejects_unsupported_configuration(overrides, match):
    with pytest.raises((ValueError, AssertionError), match=match):
        TransformerConfig(**_ultraep_config_kwargs(**overrides))


def test_ultraep_runtime_capacity_and_mtp_layer_id(monkeypatch):
    runtime = SimpleNamespace(
        local_replica_fc1_weight_buffer=object(),
        local_replica_fc2_weight_buffer=object(),
        local_replica_fc1_grad_buffer=object(),
        local_replica_fc2_grad_buffer=object(),
    )
    manager_constructor = Mock(return_value=runtime)
    monkeypatch.setattr(ultraep_manager, "HAVE_ULTRAEP", True)
    monkeypatch.setattr(
        ultraep_manager, "ultra_ep", SimpleNamespace(Manager=manager_constructor)
    )
    ep_group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)

    manager = UltraEPManager(TransformerConfig(**_ultraep_config_kwargs()), ep_group)

    assert manager.num_layers == 44
    assert manager.layer_id(43) == 43
    assert manager.layer_id(1, is_mtp_layer=True) == 44
    assert manager_constructor.call_args.kwargs["num_layers"] == 44


def test_ultraep_manager_rejects_overlapping_or_out_of_range_layer_ids():
    manager = UltraEPManager.__new__(UltraEPManager)
    manager.num_decoder_layers = 43
    manager.num_mtp_layers = 1
    manager.num_layers = 44

    with pytest.raises(ValueError, match="MTP layer_number"):
        manager.layer_id(2, is_mtp_layer=True)
    with pytest.raises(ValueError, match="real layer ID"):
        manager.allocate_microbatch_slot(45)


def test_ultraep_replicas_are_excluded_from_ddp_layout():
    master = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    replica = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    replica._ultraep_is_replica = True

    grouped = group_params_for_buffers([master, replica], grad_reduce_in_fp32=True)

    grouped_params = [param for params, _ in grouped.values() for param in params]
    assert grouped_params == [master]


def test_ultraep_replicas_are_excluded_from_optimizer(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    replica = model[1].weight
    replica._ultraep_is_replica = True
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, value, group=None: output.__setitem__(0, value),
    )

    groups = _get_param_groups([model], OptimizerConfig(optimizer='adam', lr=0.01), {})

    optimizer_params = {param for group in groups for param in group["params"]}
    assert replica not in optimizer_params
    assert optimizer_params == set(model.parameters()) - {replica}


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("linear_fc1.weight0", 0),
        ("linear_fc1.bias12", 12),
        ("linear_fc1._extra_state", 0),
        ("linear_fc1._extra_state3", 3),
        ("linear_fc1._extra_statebad", None),
        ("linear_fc2.weight1", None),
    ],
)
def test_parse_te_expert_idx(key, expected):
    assert _parse_te_expert_idx(key, "linear_fc1") == expected


def test_ultraep_checkpoint_filter_drops_replicas_and_restores_logical_metadata():
    experts = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(experts)
    experts.ep_group = SimpleNamespace(size=lambda: 2, rank=lambda: 1)
    master = ShardedTensor.from_rank_offsets(
        "linear_fc2.weight0", torch.zeros(2, 3), (0, 2, 6), prepend_axis_num=1
    )
    replica = ShardedTensor.from_rank_offsets(
        "linear_fc2.weight1", torch.zeros(2, 3), (0, 3, 6), prepend_axis_num=1
    )
    extra_state = ShardedObject("linear_fc2._extra_state", object(), (6,), (2,))

    filtered = experts._ultraep_filter_replica_checkpoint_entries(
        {
            "linear_fc2.weight0": master,
            "linear_fc2.weight1": replica,
            "linear_fc2._extra_state": extra_state,
        },
        "linear_fc2",
        ep_axis=0,
        num_local_master_experts=1,
        fix_metadata=True,
    )

    assert set(filtered) == {"linear_fc2.weight0", "linear_fc2._extra_state"}
    assert filtered["linear_fc2.weight0"].global_shape == (2, 2, 3)
    assert filtered["linear_fc2.weight0"].global_offset == (1, 0, 0)
    assert filtered["linear_fc2._extra_state"].global_shape == (2,)
    assert filtered["linear_fc2._extra_state"].global_offset == (1,)


def test_destroy_ultraep_managers_closes_and_clears_registry():
    manager_a = SimpleNamespace(close=Mock())
    manager_b = SimpleNamespace(close=Mock())
    ultraep_manager._ULTRAEP_MANAGER_REGISTRY.update({1: manager_a, 2: manager_b})

    ultraep_manager.destroy_ultraep_managers()

    manager_a.close.assert_called_once_with()
    manager_b.close.assert_called_once_with()
    assert ultraep_manager._ULTRAEP_MANAGER_REGISTRY == {}


def test_ultraep_manager_close_is_idempotent():
    manager = UltraEPManager.__new__(UltraEPManager)
    manager.runtime = SimpleNamespace(destroy=Mock())
    manager._closed = False

    manager.close()
    manager.close()

    manager.runtime.destroy.assert_called_once_with()
