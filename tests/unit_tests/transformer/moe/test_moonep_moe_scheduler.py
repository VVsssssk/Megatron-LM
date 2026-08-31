# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import torch

from megatron.core.transformer.moe.echo_moe_scheduler import EchoExpertDispatch
from megatron.core.transformer.moe.moonep_moe_scheduler import MoonEPLoadPlanner
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


def test_moonep_planner_emits_dense_reroute_for_single_ep():
    topk_ids = torch.tensor([[0], [1], [2], [3]])
    probs, routing_map, tokens_per_expert = _route_inputs(topk_ids, num_experts=4)
    context = _context(ep_size=1, ep_rank=0)
    output = MoonEPLoadPlanner(num_redundant_experts=1, token_padding=2).plan(
        probs, routing_map, context, tokens_per_expert=tokens_per_expert
    )

    assert output.physical_to_logical_map.tolist() == [0, 1, 2, 3, -1]
    assert output.routing_map.shape == (4, 5)
    assert output.routing_map[:, :4].sum().item() == 4
    assert output.routing_map[:, 4].sum().item() == 0
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


def test_moonep_planner_balances_overloaded_home_group_with_remote_copy():
    counts_from_ep_rank = torch.tensor(
        [
            [4, 0, 0, 0],
            [4, 0, 0, 0],
        ],
        dtype=torch.int64,
    )
    topk_ids = torch.tensor([[0], [0], [0], [0]])
    probs, routing_map, _ = _route_inputs(topk_ids, num_experts=4)
    context = _context(ep_size=2, ep_rank=1)
    output = MoonEPLoadPlanner(num_redundant_experts=1).plan_with_count_matrix(
        probs, routing_map, context, counts_from_ep_rank
    )

    assert output.physical_to_logical_map.tolist() == [0, 1, -1, 2, 3, 0]
    assert output.routing_map.shape == (4, 6)
    assert output.routing_map[:, 5].sum().item() == 4
    assert torch.equal(output.probs.sum(dim=1), probs.sum(dim=1))


def test_moonep_global_mode_can_feed_echo_expert_dispatch_metadata():
    counts_from_ep_rank = torch.tensor(
        [
            [4, 0, 0, 0],
            [4, 0, 0, 0],
        ],
        dtype=torch.int64,
    )
    topk_ids = torch.tensor([[0], [0], [0], [0]])
    probs, routing_map, _ = _route_inputs(topk_ids, num_experts=4)
    context = _context(ep_size=2, ep_rank=1)
    output = MoonEPLoadPlanner(num_redundant_experts=1).plan_with_count_matrix(
        probs, routing_map, context, counts_from_ep_rank
    )

    metadata = EchoExpertDispatch().build_metadata(output.physical_to_logical_map, context)

    assert output.physical_to_logical_map.tolist() == [0, 1, -1, 2, 3, 0]
    assert metadata.expert_offloading_map.tolist() == [
        [False, True],
        [False, False],
        [False, False],
        [False, False],
    ]
    assert metadata.output_splits == [1, 0]


def test_moonep_global_mode_falls_back_when_idle_slots_cannot_cover_remote_experts():
    counts_from_ep_rank = torch.tensor(
        [
            [2, 2, 2, 0, 0, 0],
            [2, 2, 2, 0, 0, 0],
        ],
        dtype=torch.int64,
    )
    topk_ids = torch.tensor([[0], [0], [1], [1], [2], [2]])
    probs, routing_map, _ = _route_inputs(topk_ids, num_experts=6)
    context = _context(ep_size=2, ep_rank=1, num_experts=6)
    output = MoonEPLoadPlanner(num_redundant_experts=1).plan_with_count_matrix(
        probs, routing_map, context, counts_from_ep_rank
    )

    assert output.physical_to_logical_map.tolist() == [0, 1, 2, -1, 3, 4, 5, 0]
    assert output.routing_map[:, 7].sum().item() == 2
    assert output.routing_map[:, 1].sum().item() == 2
    assert output.routing_map[:, 2].sum().item() == 2
    assert torch.equal(output.probs.sum(dim=1), probs.sum(dim=1))


def test_moonep_planner_handles_duplicate_destination_ranks():
    topk_ids = torch.tensor([[0, 1], [0, 1]])
    probs, routing_map, tokens_per_expert = _route_inputs(topk_ids, num_experts=4)
    context = _context(ep_size=1, ep_rank=0, topk=2)
    output = MoonEPLoadPlanner(num_redundant_experts=1).plan(
        probs, routing_map, context, tokens_per_expert=tokens_per_expert
    )

    assert output.physical_to_logical_map.tolist() == [0, 1, 2, 3, -1]
    assert output.routing_map.shape == (2, 5)
    assert torch.equal(output.probs.sum(dim=1), probs.sum(dim=1))
