# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.fused_a2a import HYBRIDEP_HANDLE_OVERFLOW_FLAG
from megatron.core.transformer.moe.token_dispatcher import _HybridEPManager
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.launch_on_gb200


@pytest.fixture(scope="module", autouse=True)
def _distributed_device():
    Utils.initialize_distributed()


@pytest.mark.parametrize("dense_topk", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("routes", [[], [[0, 2], [1, 3]], [[0, -1], [-1, 2]], [[-1, -1], [-1, -1]]])
def test_compact_routes_preserve_padding_and_probability_gradients(dense_topk, device, routes):
    manager = SimpleNamespace(num_experts=4, _dense_topk_routing=dense_topk)
    ids = torch.tensor(routes, device=device, dtype=torch.int64).reshape(-1, 2)
    saved_ids = ids.clone()
    # Nonzero padding probabilities must also be ignored. In particular an
    # invalid route must not overwrite a real route to expert zero in its row.
    probs = torch.arange(ids.numel(), device=device, dtype=torch.float32).reshape_as(ids) + 1
    probs.requires_grad_()
    routing_map, topk_idx, dense_probs = _HybridEPManager._expand_compact_routes(
        manager, ids, probs
    )
    expected = torch.zeros((ids.shape[0], 4), device=device)
    expected_map = torch.zeros_like(expected, dtype=torch.bool)
    for row, selections in enumerate(routes):
        for col, expert in enumerate(selections):
            if expert >= 0:
                expected[row, expert] = probs.detach()[row, col]
                expected_map[row, expert] = True
    torch.testing.assert_close(dense_probs, expected)
    torch.testing.assert_close(ids, saved_ids)
    assert dense_probs.is_contiguous()
    if dense_topk:
        assert routing_map is None
        torch.testing.assert_close(topk_idx, ids.to(torch.int16))
    else:
        assert topk_idx is None
        torch.testing.assert_close(routing_map, expected_map)
        assert routing_map.is_contiguous()
    coefficients = torch.arange(1, 5, device=device, dtype=torch.float32)
    (dense_probs * coefficients).sum().backward()
    expected_grad = torch.where(ids >= 0, (ids + 1).float(), 0.0)
    torch.testing.assert_close(probs.grad, expected_grad)


@pytest.mark.parametrize("dense_topk", [False, True])
@pytest.mark.parametrize("invalid_expert", [-2, 4])
def test_compact_routes_do_not_hide_invalid_expert_ids(dense_topk, invalid_expert):
    manager = SimpleNamespace(num_experts=4, _dense_topk_routing=dense_topk)
    # Run invalid-index checks on CPU so a device assertion cannot poison the
    # CUDA context used by the remaining tests.
    with pytest.raises(RuntimeError, match="out of bounds"):
        _HybridEPManager._expand_compact_routes(
            manager, torch.tensor([[0, invalid_expert]]), torch.ones(1, 2)
        )


@pytest.mark.parametrize("dense_topk", [False, True])
def test_compact_padding_routes_cuda_graph_replay(dense_topk):
    manager = SimpleNamespace(num_experts=4, _dense_topk_routing=dense_topk)
    ids = torch.tensor([[0, -1], [-1, 2]], device="cuda")
    probs = torch.ones((2, 2), device="cuda", requires_grad=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _, _, output = _HybridEPManager._expand_compact_routes(manager, ids, probs)
            output.sum().backward()
            probs.grad.zero_()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        routing_map, topk_idx, output = _HybridEPManager._expand_compact_routes(manager, ids, probs)
        (output * torch.arange(1, 5, device="cuda")).sum().backward()
    for selections in ([[0, -1], [-1, 2]], [[-1, -1], [-1, -1]], [[3, 1], [0, -1]]):
        ids.copy_(torch.tensor(selections, device="cuda"))
        probs.grad.zero_()
        graph.replay()
        _, _, expected = _HybridEPManager._expand_compact_routes(manager, ids, probs.detach())
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(probs.grad, torch.where(ids >= 0, (ids + 1).float(), 0.0))
        if dense_topk:
            torch.testing.assert_close(topk_idx, ids.to(torch.int16))
        else:
            torch.testing.assert_close(routing_map, expected.bool())


@pytest.mark.parametrize("ragged_handle", [False, True])
@pytest.mark.parametrize("overflow", [0, 1])
def test_overflow_flag_is_not_valid_token_count(ragged_handle, overflow):
    handle = (None,) * 10
    if ragged_handle:
        handle += (torch.tensor([4096], dtype=torch.int32),)
    handle += (torch.tensor([overflow], dtype=torch.int32),)
    assert (handle[HYBRIDEP_HANDLE_OVERFLOW_FLAG] != 0).item() == bool(overflow)


@pytest.mark.parametrize(
    ("expert_rank_capacity_factor", "expected_num_permuted_tokens"), [(None, None), (1.5, 128)]
)
def test_combine_only_releases_dynamic_token_count(
    monkeypatch, expert_rank_capacity_factor, expected_num_permuted_tokens
):
    manager = object.__new__(_HybridEPManager)
    manager.config = SimpleNamespace(moe_permute_fusion_into_hybridep=False)
    manager.handle = object()
    manager.num_permuted_tokens = 128
    manager.pad_multiple = None
    manager.drop_and_pad = False
    manager.moe_expert_rank_capacity_factor = expert_rank_capacity_factor
    manager._original_num_tokens = None
    manager._padded_num_tokens = None

    def fake_hybrid_ep_combine(**kwargs):
        assert kwargs["handle"] is manager.handle
        assert kwargs["num_permuted_tokens"] == 128
        return kwargs["x"]

    monkeypatch.setattr(
        "megatron.core.transformer.moe.token_dispatcher.hybrid_ep_combine", fake_hybrid_ep_combine
    )

    hidden_states = torch.empty(4, 8)
    assert manager.combine(hidden_states) is hidden_states
    assert manager.handle is None
    assert manager.num_permuted_tokens == expected_num_permuted_tokens
