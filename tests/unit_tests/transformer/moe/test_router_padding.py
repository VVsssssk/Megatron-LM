# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Padding-aware expert-bias counts must be fixed-shape for CUDA Graph replay."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.router import TopKRouter


def _router(device):
    return SimpleNamespace(
        enable_expert_bias=True,
        config=SimpleNamespace(moe_virtual_expert_load_balance=False, num_moe_experts=8),
        local_tokens_per_expert=torch.zeros(8, device=device),
    )


def _expected_counts(indices, mask):
    # Deliberately use the original filtering semantics outside graph capture.
    valid = indices if mask is None else indices[~mask.reshape(-1)]
    return torch.bincount(valid.reshape(-1).long(), minlength=8).float()


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("padding", [None, [False] * 4, [False, True, False, True], [True] * 4])
def test_index_route_padding_counts(deterministic, padding):
    router = _router("cuda")
    indices = torch.tensor([[0, 3], [1, 4], [0, 7], [2, 5]], device="cuda", dtype=torch.int16)
    mask = None if padding is None else torch.tensor(padding, device="cuda").reshape(2, 2)
    if mask is not None:
        # Padded rows used to be discarded, including otherwise invalid indices.
        indices.masked_fill_(mask.reshape(-1, 1), -1)
    original = indices.clone()
    expected = _expected_counts(indices, mask)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic)
        with torch.enable_grad():
            TopKRouter._apply_expert_bias(router, indices, padding_mask=mask)
            TopKRouter._apply_expert_bias(router, indices, padding_mask=mask)
        torch.testing.assert_close(router.local_tokens_per_expert, 2 * expected, rtol=0, atol=0)
        torch.testing.assert_close(indices, original, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("deterministic", [False, True])
def test_index_route_padding_cuda_graph_replay(deterministic):
    router = _router("cuda")
    indices = torch.tensor([[0, 3], [1, 4], [0, 7], [2, 5]], device="cuda", dtype=torch.int16)
    mask = torch.zeros((2, 2), device="cuda", dtype=torch.bool)
    original = indices.clone()
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic)
        with torch.enable_grad():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    TopKRouter._apply_expert_bias(router, indices, padding_mask=mask)
            torch.cuda.current_stream().wait_stream(stream)
            router.local_tokens_per_expert.zero_()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                TopKRouter._apply_expert_bias(router, indices, padding_mask=mask)
            # Replay with different padding contents without changing any addresses.
            for padding in ([False, True, False, True], [True] * 4, [False] * 4):
                mask.copy_(torch.tensor(padding, device="cuda").reshape_as(mask))
                indices.copy_(original)
                indices.masked_fill_(mask.reshape(-1, 1), -1)
                expected = _expected_counts(indices, mask)
                router.local_tokens_per_expert.zero_()
                graph.replay()
                graph.replay()
                torch.testing.assert_close(
                    router.local_tokens_per_expert, 2 * expected, rtol=0, atol=0
                )
    finally:
        torch.use_deterministic_algorithms(previous)
