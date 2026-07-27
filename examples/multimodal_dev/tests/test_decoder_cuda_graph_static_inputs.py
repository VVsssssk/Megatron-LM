# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Qwen3.5-VL decoder-only CUDA graph static inputs."""

import pytest
import torch

from examples.multimodal_dev.models.qwen35_vl.decoder_cuda_graph import (
    Qwen35VLDecoderFullCudaGraphWrapper,
    _StaticLanguageInputStore,
    _iter_megatron_fsdp_modules,
)
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


class _FakeAllGatherPipeline:
    def __init__(self):
        self.ag_stream = object()
        self.outer_fsdp_group_param_gather_stream = object()
        self.reset_calls = []

    def reset(self, preserve_non_fsdp_units=True):
        self.reset_calls.append(preserve_non_fsdp_units)


class _FakeMegatronFSDP(torch.nn.Module):
    def __init__(self, pipeline):
        super().__init__()
        self.all_gather_pipeline = pipeline

    def all_gather_and_wait_parameters_ready(self):
        pass


class _FakeSynchronizingMegatronFSDP(_FakeMegatronFSDP):
    def __init__(self, pipeline):
        super().__init__(pipeline)
        self.synchronize_param_gather_calls = 0

    def synchronize_param_gather(self):
        self.synchronize_param_gather_calls += 1


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


def test_iter_megatron_fsdp_modules_deduplicates_shared_pipeline():
    pipeline = _FakeAllGatherPipeline()
    root = torch.nn.Module()
    root.first = _FakeMegatronFSDP(pipeline)
    root.second = _FakeMegatronFSDP(pipeline)

    assert list(_iter_megatron_fsdp_modules(root)) == [root.first]


def test_fsdp_pipeline_capture_helpers_drain_and_restore_streams(monkeypatch):
    pipeline = _FakeAllGatherPipeline()
    fsdp = _FakeMegatronFSDP(pipeline)
    wrapper = Qwen35VLDecoderFullCudaGraphWrapper(lambda **kwargs: None)
    synchronize_calls = []
    original_ag_stream = pipeline.ag_stream
    original_outer_stream = pipeline.outer_fsdp_group_param_gather_stream
    capture_stream = object()

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synchronize_calls.append(True))

    wrapper._synchronize_fsdp_all_gather_pipelines([fsdp])

    assert pipeline.reset_calls == [True]
    assert synchronize_calls == [True]

    with wrapper._fsdp_all_gather_on_capture_stream([fsdp], capture_stream):
        assert pipeline.ag_stream is None
        assert pipeline.outer_fsdp_group_param_gather_stream is capture_stream

    assert pipeline.ag_stream is original_ag_stream
    assert pipeline.outer_fsdp_group_param_gather_stream is original_outer_stream


def test_fsdp_pipeline_drain_prefers_megatron_fsdp_synchronize_param_gather(monkeypatch):
    pipeline = _FakeAllGatherPipeline()
    fsdp = _FakeSynchronizingMegatronFSDP(pipeline)
    wrapper = Qwen35VLDecoderFullCudaGraphWrapper(lambda **kwargs: None)

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    wrapper._synchronize_fsdp_all_gather_pipelines([fsdp])

    assert fsdp.synchronize_param_gather_calls == 1
    assert pipeline.reset_calls == []
