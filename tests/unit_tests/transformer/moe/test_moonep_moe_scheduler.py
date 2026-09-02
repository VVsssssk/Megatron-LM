# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import pytest
import torch

from megatron.core.transformer.moe.echo_moe_scheduler import EchoExpertDispatch
from megatron.core.transformer.moe.moonep_moe_scheduler import (
    MoonEPLoadPlanner,
    _physical_to_logical_map_from_experts_to_copy,
)
from megatron.core.transformer.moe.moe_scheduler import SchedulerContext


def _route_inputs(
    topk_ids: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_probs = torch.full(topk_ids.shape, 1.0 / topk_ids.size(1), dtype=torch.float32)
    probs = torch.zeros(topk_ids.size(0), num_experts)
    routing_map = torch.zeros(topk_ids.size(0), num_experts, dtype=torch.bool)
    routing_map.scatter_(1, topk_ids, True)
    probs.scatter_(1, topk_ids, topk_probs)
    tokens_per_expert = torch.bincount(topk_ids.reshape(-1), minlength=num_experts)
    return probs, routing_map, tokens_per_expert


def _context(ep_size: int, ep_rank: int, topk: int = 1, num_experts: int = 4) -> SchedulerContext:
    experts_per_rank = num_experts // ep_size
    local_start = ep_rank * experts_per_rank
    return SchedulerContext(
        layer_number=1,
        num_logical_experts=num_experts,
        num_local_experts=experts_per_rank,
        local_expert_indices=tuple(range(local_start, local_start + experts_per_rank)),
        ep_size=ep_size,
        ep_rank=ep_rank,
        router_topk=topk,
        training=True,
    )


def test_moonep_planner_emits_2x_identity_layout_for_single_ep():
    topk_ids = torch.tensor([[0], [1], [2], [3]])
    probs, routing_map, tokens_per_expert = _route_inputs(topk_ids, num_experts=4)
    context = _context(ep_size=1, ep_rank=0)
    output = MoonEPLoadPlanner(num_redundant_experts=4).plan(
        probs, routing_map, context, tokens_per_expert=tokens_per_expert
    )

    assert output.physical_to_logical_map.tolist() == [0, 1, 2, 3, -1, -1, -1, -1]
    assert output.routing_map.shape == (4, 8)
    assert output.routing_map[:, :4].sum().item() == 4
    assert output.routing_map[:, 4:].sum().item() == 0
    assert torch.equal(output.probs.sum(dim=1), probs.sum(dim=1))


def test_moonep_planner_should_not_plan_without_redundant_experts():
    topk_ids = torch.tensor([[0], [1], [2], [3]])
    probs, routing_map, tokens_per_expert = _route_inputs(topk_ids, num_experts=4)

    assert (
        MoonEPLoadPlanner(num_redundant_experts=0).should_plan(
            probs,
            routing_map,
            _context(ep_size=1, ep_rank=0),
            tokens_per_expert=tokens_per_expert,
        )
        is False
    )


def test_moonep_planner_rejects_partial_replica_slots():
    topk_ids = torch.tensor([[0], [0], [1], [1]])
    probs, routing_map, tokens_per_expert = _route_inputs(topk_ids, num_experts=4)

    with pytest.raises(ValueError, match="one replica slot per local home expert"):
        MoonEPLoadPlanner(num_redundant_experts=1).should_plan(
            probs,
            routing_map,
            _context(ep_size=2, ep_rank=0),
            tokens_per_expert=tokens_per_expert,
        )


def test_moonep_planner_rejects_token_padding():
    with pytest.raises(ValueError, match="token_padding"):
        MoonEPLoadPlanner(num_redundant_experts=4, token_padding=2)


def test_moonep_planner_requires_ep_group_for_multi_ep():
    topk_ids = torch.tensor([[0], [0], [1], [1]])
    probs, routing_map, tokens_per_expert = _route_inputs(topk_ids, num_experts=4)

    with pytest.raises(ValueError, match="pg_collection.ep"):
        MoonEPLoadPlanner(num_redundant_experts=2).plan(
            probs,
            routing_map,
            _context(ep_size=2, ep_rank=0),
            tokens_per_expert=tokens_per_expert,
        )


def test_moonep_single_ep_layout_can_feed_echo_expert_dispatch_metadata():
    topk_ids = torch.tensor([[0], [1], [2], [3]])
    probs, routing_map, tokens_per_expert = _route_inputs(topk_ids, num_experts=4)
    context = _context(ep_size=1, ep_rank=0)
    output = MoonEPLoadPlanner(num_redundant_experts=4).plan(
        probs, routing_map, context, tokens_per_expert=tokens_per_expert
    )

    metadata = EchoExpertDispatch().build_metadata(output.physical_to_logical_map, context)

    assert metadata.expert_offloading_map.shape == (4, 4)
    assert not bool(metadata.expert_offloading_map.any().item())
    assert metadata.input_splits == [0]
    assert metadata.output_splits == [0]
    assert metadata.has_experts_per_slot.tolist() == [0, 0, 0, 0]


def test_moonep_experts_to_copy_builds_rank_major_physical_layout():
    context = _context(ep_size=2, ep_rank=1)
    experts_to_copy = torch.tensor(
        [
            [-1, -1],
            [0, 1],
        ],
        dtype=torch.int32,
    )

    physical_to_logical_map = _physical_to_logical_map_from_experts_to_copy(
        experts_to_copy, context
    )

    assert physical_to_logical_map.tolist() == [0, 1, -1, -1, 2, 3, 0, 1]


def test_moonep_planner_rejects_removed_count_matrix_adapter():
    with pytest.raises(NotImplementedError, match="plan_with_count_matrix"):
        MoonEPLoadPlanner(num_redundant_experts=2).plan_with_count_matrix()
