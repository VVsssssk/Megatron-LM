# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Non-GTP regressions adapted from PR 7450 (242c8dbc6) to dev compact routing."""

import gc
import os
from functools import partialmethod
from types import SimpleNamespace

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


@pytest.fixture
def virtual_expert_planner_group(ep_size):
    """Run the planner independently of model parallelism and HybridEP transport."""
    Utils.initialize_distributed()
    world_size = dist.get_world_size()
    if world_size % ep_size:
        pytest.skip(f"EP={ep_size} requires a world size divisible by {ep_size}")
    groups = (
        [
            dist.new_group(list(range(start, start + ep_size)))
            for start in range(0, world_size, ep_size)
        ]
        if ep_size != world_size
        else []
    )
    group = groups[dist.get_rank() // ep_size] if groups else dist.group.WORLD
    try:
        yield group
    finally:
        torch.cuda.synchronize()
        dist.barrier(device_ids=[torch.cuda.current_device()])
        if groups:
            dist.destroy_process_group(group)


def _check_equal(actual, expected, label, errors):
    """Keep comparisons collective-safe while checking every element, including canaries."""
    try:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    except AssertionError as exc:
        errors.append(f"{label}: {exc}")


def _report(errors, group):
    """Fail on every rank if any rank saw a mismatch."""
    gathered = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(gathered, errors, group=group)
    combined = [error for rank_errors in gathered for error in rank_errors]
    assert not combined, "\n".join(combined)


def _reference_plan(routes, num_experts, alignment=1):
    """CPU greedy placement and stable route assignment, using only semantic input routes."""
    ep_size, num_tokens, topk = routes.shape
    local_experts = num_experts // ep_size
    counts = torch.stack([torch.bincount(row.flatten(), minlength=num_experts) for row in routes])
    totals = counts.sum(0).tolist()
    blocks = [(count + alignment - 1) // alignment for count in totals]
    total_blocks = sum(blocks)
    targets = [total_blocks // ep_size + (rank < total_blocks % ep_size) for rank in range(ep_size)]
    loads = [
        sum(blocks[r * local_experts : (r + 1) * local_experts]) - targets[r]
        for r in range(ep_size)
    ]
    quotas = [[0] * ep_size for _ in range(ep_size)]
    while max(loads) > 0:
        sender = max(range(ep_size), key=lambda r: (loads[r], -r))
        receiver = min(range(ep_size), key=lambda r: (loads[r], r))
        # A receiver takes its entire deficit from one sender, even beyond that sender's
        # excess. This bounds the number of virtual slots by the sender's native experts.
        moved = -loads[receiver]
        quotas[sender][receiver] += moved
        loads[sender] -= moved
        loads[receiver] = 0
    allocation = [[0] * ep_size for _ in range(num_experts)]
    for expert, count in enumerate(blocks):
        allocation[expert][expert // local_experts] = count
    for sender, pending in enumerate(quotas):
        experts = range(sender * local_experts, (sender + 1) * local_experts)
        while any(pending):
            destination = max(range(ep_size), key=lambda r: (pending[r], -r))
            expert = max(experts, key=lambda e: (allocation[e][sender], -e))
            moved = min(pending[destination], allocation[expert][sender])
            assert moved > 0
            allocation[expert][sender] -= moved
            allocation[expert][destination] += moved
            pending[destination] -= moved
    allocation = [[count * alignment for count in row] for row in allocation]
    copies = []
    for destination in range(ep_size):
        remote = sorted(
            (
                e
                for e in range(num_experts)
                if e // local_experts != destination and allocation[e][destination]
            ),
            key=lambda e: (allocation[e][destination], e),
            reverse=True,
        )
        assert len(remote) <= local_experts
        copies.append(remote + [-1] * (local_experts - len(remote)))
    # Stable global order is source rank, token, top-k lane. Partition each expert's
    # positions directly by the CPU allocation, without reading kernel boundaries or slots.
    order = routes.flatten().argsort(stable=True)
    mapped = torch.empty(routes.numel(), dtype=torch.int16)
    cursor = 0
    for expert, destinations in enumerate(allocation):
        remaining = totals[expert]
        for destination, count in enumerate(destinations):
            count = min(count, remaining)
            remaining -= count
            if count:
                local = (
                    expert % local_experts
                    if destination == expert // local_experts
                    else (local_experts + copies[destination].index(expert))
                )
                mapped[order[cursor : cursor + count]] = destination * (2 * local_experts) + local
                cursor += count
    assert cursor == routes.numel()
    return (
        counts.int(),
        torch.tensor(allocation, dtype=torch.int32),
        torch.tensor(copies, dtype=torch.int32),
        mapped.view_as(routes),
    )


def _routes_for_skew(ep_size, num_experts, num_tokens, topk, skew):
    """Generate distinct top-k choices with exact balance, ties or controlled load skew."""
    if skew == "boundary":
        assert topk == 1
        counts = (31, 32, 33, 127, 128, 129, 255, 256, 257)
        flat = torch.cat([torch.full((count,), i % num_experts) for i, count in enumerate(counts)])
        size = ep_size * num_tokens
        return flat.repeat((size + flat.numel() - 1) // flat.numel())[:size].reshape(
            ep_size, num_tokens, 1
        )
    if skew == "ties":
        return torch.arange(9).repeat_interleave(4).reshape(4, 9, 1)
    rows = torch.arange(num_tokens)[:, None] + torch.arange(topk)
    local_experts = num_experts // ep_size
    if skew == "rank_ties":
        return (rows % (num_experts // 2)).repeat(ep_size, 1, 1)
    if skew in ("balanced", "local", "remote"):
        return torch.stack(
            [
                (
                    (rows + rank * local_experts) % num_experts
                    if skew == "balanced"
                    else rows % local_experts
                    + ((rank + (skew == "remote")) % ep_size) * local_experts
                )
                for rank in range(ep_size)
            ]
        )
    if skew == "hot_expert" and topk == 1:
        return torch.zeros((ep_size, num_tokens, topk), dtype=torch.int64)
    weights = torch.ones(num_experts)
    if skew == "hot_expert":
        weights[0] = 20
    elif skew == "hot_rank":
        weights[:local_experts] = 12
    elif skew == "concentrated":
        weights[topk:] = 0
    else:
        assert skew == "random"
    return torch.multinomial(
        weights.expand(ep_size * num_tokens, -1),
        topk,
        replacement=False,
        generator=torch.Generator().manual_seed(1234),
    ).reshape(ep_size, num_tokens, topk)


@pytest.mark.internal
def test_virtual_expert_planner_reference_tie_breaks():
    """Hand-computed over-migration exercises rank/expert ties and reversed slot ties."""
    # Two equally overloaded senders and two equally empty receivers pair in rank order.
    _, _, copies, mapped = _reference_plan(_routes_for_skew(4, 8, 4, 1, "rank_ties"), 8)
    assert copies.tolist() == [[-1, -1], [-1, -1], [0, -1], [2, -1]]
    assert mapped.flatten().tolist() == [10, 1, 14, 5] * 4
    routes = _routes_for_skew(4, 12, 9, 1, "ties")
    _, allocation, copies, mapped = _reference_plan(routes, 12)
    assert allocation.tolist() == [
        [0, 0, 0, 4],
        [0, 0, 0, 4],
        [3, 0, 0, 1],
        [4, 0, 0, 0],
        [2, 2, 0, 0],
        [0, 4, 0, 0],
        [0, 3, 1, 0],
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert copies.tolist() == [[3, 4, -1], [6, -1, -1], [-1, -1, -1], [1, 0, 2]]
    expected = [22] * 4 + [21] * 4 + [2] * 3 + [23] + [3] * 4 + [4] * 2 + [7] * 2
    expected += [8] * 4 + [9] * 3 + [12] + [13] * 4 + [14] * 4
    assert mapped.flatten().tolist() == expected


def _check_planner_reference(group, num_experts, num_tokens, topk, skews, alignments=(1,)):
    ep_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    device = torch.device("cuda", torch.cuda.current_device())
    local_experts = num_experts // ep_size
    workspace = VirtualExpertPlannerWorkspace(num_experts=num_experts, device=device, group=group)
    errors = []
    try:
        cases = [(skew, alignment) for alignment in alignments for skew in skews.split()]
        for iteration, (skew, alignment) in enumerate(cases):
            routes = _routes_for_skew(ep_size, num_experts, num_tokens, topk, skew)
            counts, allocation, copies, mapped = _reference_plan(routes, num_experts, alignment)
            dtype = torch.int32 if iteration % 2 else torch.int64
            own = routes[rank].to(device=device, dtype=dtype)
            if iteration % 3 and num_tokens > 1:
                # A nonzero offset and poisoned gaps catch a wrapper that forgets contiguous().
                storage = torch.full(
                    (num_tokens, 2 * topk + 1), num_experts + 7, device=device, dtype=dtype
                )
                storage[:, 1::2] = own
                own = storage[:, 1::2]
                assert not own.is_contiguous()
            elif num_tokens:
                # A contiguous view need not have a 16-byte-aligned base. Launcher reuse must
                # not borrow the compiler's stronger pointer alignment from another input.
                storage = torch.empty(own.numel() + 1, device=device, dtype=dtype)
                storage[1:].copy_(own.flatten())
                own = storage[1:].view_as(own)
                assert own.is_contiguous() and own.data_ptr() % 16 != 0
            plan = plan_virtual_expert_routes(own, workspace, alignment)
            actual = plan.virtual_experts.cpu()
            actual_copies = plan.experts_to_copy.cpu()
            for label, value, expected in (
                ("histogram", workspace.gathered_counts.cpu(), counts),
                ("allocation", workspace.field("allocation").cpu(), allocation),
                ("copies", actual_copies, copies),
                ("routes", actual, mapped[rank]),
            ):
                _check_equal(value, expected, f"{skew} {label}", errors)
            # Independently decode the public outputs and count real routes, rather than
            # accepting a self-consistent allocation/slot table with the wrong expert owners.
            try:
                assert actual.shape == (num_tokens, topk) and actual.dtype == torch.int16
                assert ((actual >= 0) & (actual < 2 * num_experts)).all()
                destination, local = actual.long() // (2 * local_experts), actual.long() % (
                    2 * local_experts
                )
                semantic = destination * local_experts + local
                remote = local >= local_experts
                semantic[remote] = actual_copies[
                    destination[remote], local[remote] - local_experts
                ].long()
                torch.testing.assert_close(semantic, routes[rank], rtol=0, atol=0)
                if alignment == 1 and skew in ("balanced", "local", "remote"):
                    assert (actual_copies == -1).all()
                if alignment == 1 and skew in ("local", "remote"):
                    assert (
                        (destination == rank) if skew == "local" else (destination != rank)
                    ).all()
                observed = torch.bincount(
                    (routes[rank] * ep_size + destination).flatten(),
                    minlength=num_experts * ep_size,
                ).to(device=device, dtype=torch.int32)
            except (AssertionError, IndexError) as exc:
                errors.append(f"{skew} semantic routes: {exc}")
                observed = torch.zeros(num_experts * ep_size, device=device, dtype=torch.int32)
            dist.all_reduce(observed, group=group)
            real_counts = observed.cpu().reshape(num_experts, ep_size)
            padded_counts = ((real_counts + alignment - 1) // alignment) * alignment
            _check_equal(padded_counts, allocation, f"{skew} route counts", errors)
            _check_equal(
                padded_counts.sum(0),
                torch.tensor(
                    [
                        (
                            int(allocation.sum()) // alignment // ep_size
                            + (r < int(allocation.sum()) // alignment % ep_size)
                        )
                        * alignment
                        for r in range(ep_size)
                    ]
                ),
                f"{skew} balanced load",
                errors,
            )
            assert int(real_counts.sum()) == routes.numel()
            capacity_owner = object.__new__(VirtualExpertLoadBalancer)
            capacity_owner.config = SimpleNamespace(moe_expert_rank_capacity_factor=None)
            capacity_owner.ep_size = ep_size
            capacity_owner.num_owned_experts = local_experts
            capacity_owner.router_topk = topk
            capacity_owner._alignment = alignment
            assert int(padded_counts.sum(0).max()) <= capacity_owner._compute_rank_capacity(
                num_tokens
            )
            # Every nonempty replica on a receiver belongs to a single owner.
            for row in copies:
                assert len(set((row[row >= 0] // local_experts).tolist())) <= 1
            # The real dispatcher provides this cross-rank ordering between planner launches.
            dist.barrier(group=group, device_ids=[device.index])
        _report(errors, dist.group.WORLD)
    finally:
        workspace.destroy()


@pytest.mark.internal
@pytest.mark.parametrize(
    "ep_size,num_experts,num_tokens,topk,skews",
    [
        (1, 8, 17, 3, "balanced concentrated balanced"),
        (2, 8, 257, 3, "random concentrated balanced"),
        (2, 8, 0, 1, "balanced"),
        (4, 8, 17, 3, "balanced concentrated balanced"),
        (4, 8, 1, 1, "hot_expert random"),
        (4, 8, 0, 1, "balanced"),
        (4, 8, 257, 1, "hot_expert random balanced boundary"),
        (4, 12, 9, 1, "ties local remote"),
        (4, 512, 33, 10, "random concentrated balanced"),
        (8, 16, 17, 3, "rank_ties random balanced"),
    ],
)
def test_virtual_expert_planner_matches_reference(
    virtual_expert_planner_group, ep_size, num_experts, num_tokens, topk, skews
):
    """Aligned remapping preserves identity and balances loads, including reused workspace."""
    _check_planner_reference(
        virtual_expert_planner_group,
        num_experts,
        num_tokens,
        topk,
        skews,
        alignments=(1, 32, 128, 256, 1, 256),
    )


@pytest.mark.internal
@pytest.mark.parametrize("ep_size", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_virtual_expert_planner_graph_replay(virtual_expert_planner_group, ep_size, dtype):
    """Changing routes and cached shapes/alignments must agree with the CPU oracle."""
    group = virtual_expert_planner_group
    rank = dist.get_rank(group)
    device = torch.device("cuda", torch.cuda.current_device())
    workspace = VirtualExpertPlannerWorkspace(num_experts=8, device=device, group=group)
    errors = []
    try:
        for num_tokens, topk, alignment in (
            (17, 1, 1),
            (257, 3, 32),
            (257, 3, 128),
            (33, 1, 256),
            (257, 3, 32),
            (17, 1, 1),
        ):
            routes = _routes_for_skew(ep_size, 8, num_tokens, topk, "balanced")
            own = routes[rank].to(device=device, dtype=dtype)
            # Compile before capture, including a cached launcher on the same workspace.
            plan_virtual_expert_routes(own, workspace, alignment)
            dist.barrier(group=group)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = plan_virtual_expert_routes(own, workspace, alignment)
            dist.barrier(group=group)
            for skew in ("concentrated", "balanced", "random", "balanced"):
                routes = _routes_for_skew(ep_size, 8, num_tokens, topk, skew)
                counts, allocation, copies, mapped = _reference_plan(routes, 8, alignment)
                own.copy_(routes[rank])
                for replay in range(3):
                    graph.replay()
                    dist.barrier(group=group)
                    for label, value, expected in (
                        ("histogram", workspace.gathered_counts.cpu(), counts),
                        ("allocation", workspace.field("allocation").cpu(), allocation),
                        ("copies", captured.experts_to_copy.cpu(), copies),
                        ("routes", captured.virtual_experts.cpu(), mapped[rank]),
                    ):
                        _check_equal(
                            value,
                            expected,
                            f"graph {num_tokens=} {topk=} {alignment=} {skew=} {replay=} {label}",
                            errors,
                        )
        _report(errors, dist.group.WORLD)
    finally:
        workspace.destroy()
