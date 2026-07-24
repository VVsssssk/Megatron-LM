# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Qwen3.5-VL decoder-only CUDA graph static inputs."""

import pytest
import torch

from examples.multimodal_dev.models.qwen35_vl.decoder_cuda_graph import _StaticLanguageInputStore
from megatron.core.packed_seq_params import PackedSeqParams


def _packed_seq_params(seq_len: int) -> PackedSeqParams:
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens.clone(),
        cu_seqlens_kv=cu_seqlens.clone(),
        cu_seqlens_q_padded=cu_seqlens.clone(),
        cu_seqlens_kv_padded=cu_seqlens.clone(),
        max_seqlen_q=seq_len,
        max_seqlen_kv=seq_len,
        total_tokens=seq_len,
    )


def _language_inputs(seq_len: int = 4, fill_value: int = 1):
    return {
        "input_ids": torch.full((1, seq_len), fill_value, dtype=torch.long),
        "position_ids": torch.full((3, 1, seq_len), fill_value, dtype=torch.long),
        "attention_mask": None,
        "decoder_input": torch.full(
            (seq_len, 1, 8), float(fill_value), dtype=torch.float32, requires_grad=True
        ),
        "labels": torch.full((1, seq_len), fill_value, dtype=torch.long),
        "loss_mask": torch.ones((1, seq_len), dtype=torch.float32),
        "padding_mask": torch.zeros((1, seq_len), dtype=torch.bool),
        "packed_seq_params": _packed_seq_params(seq_len),
    }


def test_static_language_input_store_reuses_and_updates_buffers():
    store = _StaticLanguageInputStore()

    first = store.copy_microbatch(0, _language_inputs(fill_value=1))
    decoder_input_id = id(first["decoder_input"])
    cu_seqlens_id = id(first["packed_seq_params"].cu_seqlens_q)

    second = store.copy_microbatch(0, _language_inputs(fill_value=3))

    assert id(second["decoder_input"]) == decoder_input_id
    assert id(second["packed_seq_params"].cu_seqlens_q) == cu_seqlens_id
    assert second["decoder_input"].requires_grad
    assert torch.equal(second["input_ids"], torch.full((1, 4), 3, dtype=torch.long))
    assert torch.equal(second["decoder_input"], torch.full((4, 1, 8), 3.0))


def test_static_language_input_store_rejects_shape_changes():
    store = _StaticLanguageInputStore()
    store.copy_microbatch(0, _language_inputs(seq_len=4))

    with pytest.raises(AssertionError, match="input shape changed"):
        store.copy_microbatch(0, _language_inputs(seq_len=5))
