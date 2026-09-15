# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe import token_dispatcher
from megatron.core.transformer.moe.token_dispatcher import _HybridEPManager


def test_drop_and_pad_preallocates_pinned_static_token_counts(monkeypatch):
    """Captured grouped experts need stable pinned host metadata from HybridEP."""
    real_empty = torch.empty
    real_zeros = torch.zeros
    requested = {}

    def record_empty(*args, **kwargs):
        requested["pin_memory"] = kwargs.get("pin_memory", False)
        kwargs["pin_memory"] = False
        return real_empty(*args, **kwargs)

    def redirect_cuda_zeros(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(token_dispatcher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(token_dispatcher.torch, "empty", record_empty)
    monkeypatch.setattr(token_dispatcher.torch, "zeros", redirect_cuda_zeros)
    monkeypatch.setattr(token_dispatcher, "hybrid_ep_dispatch", object())
    monkeypatch.setattr(token_dispatcher, "hybrid_ep_dense_topk_routing", lambda *_: False)
    monkeypatch.setattr(token_dispatcher, "uses_compact_routes", lambda _: False)

    config = SimpleNamespace(
        moe_router_topk=8,
        moe_permute_fusion=True,
        moe_expert_capacity_factor=4.0,
        moe_pad_expert_input_to_capacity=True,
        moe_virtual_expert_load_balance=False,
        moe_expert_rank_capacity_factor=None,
    )
    manager = _HybridEPManager(
        group=object(), num_local_experts=16, num_experts=128, config=config
    )

    assert requested["pin_memory"] is True
    assert manager._static_tokens_per_expert.shape == (16,)


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
