# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe.mock_router import (
    ImbalanceRouteMockGenerator,
    compute_mock_route_stats,
)
from megatron.core.transformer.transformer_config import TransformerConfig


def _config(**overrides):
    kwargs = dict(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=64,
        expert_model_parallel_size=8,
        moe_router_topk=8,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.0,
        use_cpu_initialization=True,
    )
    kwargs.update(overrides)
    config = TransformerConfig(**kwargs)
    config.moe_router_mock_base_seed = 1234
    return config


def _generator(config):
    return ImbalanceRouteMockGenerator(
        config=config,
        num_experts=config.num_moe_experts,
        num_ep_rank=config.expert_model_parallel_size,
        topk=config.moe_router_topk,
        ep_group=None,
    )


def _overlap(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs_set = set(lhs.cpu().tolist())
    rhs_set = set(rhs.cpu().tolist())
    return len(lhs_set & rhs_set) / float(max(1, len(lhs_set)))


def test_rank_maxvio_definition():
    balanced = torch.tensor([[0, 2, 4, 6]])
    stats = compute_mock_route_stats(balanced, num_experts=8, num_ep_rank=4)
    assert stats.actual_rank_maxvio == pytest.approx(1.0)

    rank0_twice_avg = torch.tensor([[0, 0], [0, 0], [2, 2], [4, 6]])
    stats = compute_mock_route_stats(rank0_twice_avg, num_experts=8, num_ep_rank=4)
    assert stats.tokens_per_ep_rank.tolist() == [4, 2, 1, 1]
    assert stats.actual_rank_maxvio == pytest.approx(2.0)


def test_imbalance_generator_basic_target_maxvio():
    config = _config(
        moe_router_mock_imbalance=True,
        moe_router_mock_maxvio=2.0,
        moe_router_mock_concentration=0.8,
        moe_router_mock_consistency=0.8,
    )
    route = _generator(config).generate(
        num_tokens=4096,
        global_step=3,
        microbatch_id=1,
        layer_idx=7,
        device=torch.device("cpu"),
    )

    assert route.topk_indices.shape == (4096, 8)
    assert route.topk_probs.shape == (4096, 8)
    assert route.routing_probs.shape == (4096, 64)
    assert route.routing_map.shape == (4096, 64)
    assert int(route.routing_map.sum().item()) == 4096 * 8
    assert route.stats.actual_rank_maxvio == pytest.approx(2.0, rel=0.02)
    assert 0.0 <= route.stats.actual_concentration <= 1.0
    assert route.stats.duplicate_topk_rate <= 0.01


def test_imbalance_generator_consistency_and_jitter():
    high_consistency_config = _config(
        moe_router_mock_imbalance=True,
        moe_router_mock_maxvio=2.0,
        moe_router_mock_concentration=0.8,
        moe_router_mock_consistency=0.8,
        moe_router_mock_maxvio_jitter=0.1,
    )
    low_consistency_config = _config(
        moe_router_mock_imbalance=True,
        moe_router_mock_maxvio=2.0,
        moe_router_mock_concentration=0.8,
        moe_router_mock_consistency=0.0,
    )

    high_routes = [
        _generator(high_consistency_config).generate(
            num_tokens=2048,
            global_step=step,
            microbatch_id=0,
            layer_idx=1,
            device=torch.device("cpu"),
        )
        for step in range(20)
    ]
    low_routes = [
        _generator(low_consistency_config).generate(
            num_tokens=2048,
            global_step=step,
            microbatch_id=0,
            layer_idx=1,
            device=torch.device("cpu"),
        )
        for step in range(20)
    ]

    high_overlap = sum(
        _overlap(high_routes[idx].hot_ep_ranks, high_routes[idx + 1].hot_ep_ranks)
        for idx in range(19)
    ) / 19.0
    low_overlap = sum(
        _overlap(low_routes[idx].hot_ep_ranks, low_routes[idx + 1].hot_ep_ranks)
        for idx in range(19)
    ) / 19.0
    assert high_overlap > low_overlap
    assert high_overlap >= 0.7

    effective_maxvios = [route.effective_maxvio for route in high_routes]
    assert min(effective_maxvios) >= 1.9
    assert max(effective_maxvios) <= 2.1
    assert len({round(value, 4) for value in effective_maxvios}) > 1


def test_mock_route_rng_isolated_and_deterministic():
    config = _config(
        moe_router_mock_imbalance=True,
        moe_router_mock_maxvio=2.0,
        moe_router_mock_concentration=0.7,
        moe_router_mock_consistency=0.5,
    )
    generator = _generator(config)

    torch.manual_seed(999)
    expected = torch.rand(8)
    torch.manual_seed(999)
    route = generator.generate(
        num_tokens=512,
        global_step=5,
        microbatch_id=2,
        layer_idx=3,
        device=torch.device("cpu"),
    )
    actual = torch.rand(8)
    torch.testing.assert_close(actual, expected)

    repeated = generator.generate(
        num_tokens=512,
        global_step=5,
        microbatch_id=2,
        layer_idx=3,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(route.topk_indices, repeated.topk_indices)
    torch.testing.assert_close(route.topk_probs, repeated.topk_probs)

    changed = generator.generate(
        num_tokens=512,
        global_step=6,
        microbatch_id=2,
        layer_idx=3,
        device=torch.device("cpu"),
    )
    assert not torch.equal(route.topk_indices, changed.topk_indices)


def test_mock_route_config_validation_and_force_balance_regression():
    _config(moe_router_force_load_balancing=True)
    _config(moe_router_mock_force_balance=True)

    with pytest.raises(ValueError, match="mutually exclusive"):
        _config(moe_router_force_load_balancing=True, moe_router_mock_imbalance=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _config(moe_router_mock_force_balance=True, moe_router_mock_imbalance=True)
    with pytest.raises(ValueError, match="maxvio"):
        _config(moe_router_mock_maxvio=0.99)
    with pytest.raises(ValueError, match="concentration"):
        _config(moe_router_mock_concentration=1.1)
    with pytest.raises(ValueError, match="consistency"):
        _config(moe_router_mock_consistency=-0.1)

