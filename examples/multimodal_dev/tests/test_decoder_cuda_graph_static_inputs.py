# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Qwen3.5-VL decoder-only CUDA graph static inputs."""

from enum import Enum

import pytest
import torch

from examples.multimodal_dev.models.qwen35_vl.decoder_cuda_graph import (
    Qwen35VLDecoderFullCudaGraphWrapper,
    _StaticLanguageInputStore,
    _iter_megatron_fsdp_modules,
)
from megatron.core.packed_seq_params import PackedSeqParams


class _FakeBucketStatus(Enum):
    EMPTY = 1
    PRESERVED = 2
    COMMUNICATING = 3
    READY_TO_USE = 4


class _FakeParameterGroup:
    def __init__(self, fsdp_unit_id):
        self.fsdp_unit_id = fsdp_unit_id


class _FakeParamAndGradBuffer:
    def __init__(self, fsdp_unit_ids):
        self.parameter_groups = [_FakeParameterGroup(item) for item in fsdp_unit_ids]
        self.num_buckets = len(self.parameter_groups)


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
    def __init__(self, fsdp_unit_ids=(0, None)):
        self.ag_stream = object()
        self.outer_fsdp_group_param_gather_stream = object()
        self.buffer = _FakeParamAndGradBuffer(fsdp_unit_ids)
        self.param_gather_event_map = {}
        self.bucket_status = {}
        self.bucket_can_be_released = {}
        self.wait_bucket_ready_calls = []
        for bucket_id in range(self.num_buckets):
            for bwd in (False, True):
                bucket_key = self.get_bucket_key(bucket_id, bwd)
                self.bucket_status[bucket_key] = _FakeBucketStatus.EMPTY
                self.bucket_can_be_released[bucket_key] = False

    @property
    def num_buckets(self):
        return self.buffer.num_buckets

    def get_bucket_key(self, bucket_id, bwd):
        return (bucket_id, bwd)

    def mark_communicating(self, bucket_id, bwd):
        bucket_key = self.get_bucket_key(bucket_id, bwd)
        self.bucket_status[bucket_key] = _FakeBucketStatus.COMMUNICATING
        self.param_gather_event_map[bucket_key] = object()

    def wait_bucket_ready(self, bucket_id, bwd):
        bucket_key = self.get_bucket_key(bucket_id, bwd)
        self.wait_bucket_ready_calls.append(bucket_key)
        self.param_gather_event_map.pop(bucket_key)
        self.bucket_status[bucket_key] = _FakeBucketStatus.READY_TO_USE


class _FakeMegatronFSDP(torch.nn.Module):
    def __init__(self, pipeline):
        super().__init__()
        self.all_gather_pipeline = pipeline
        self.replace_param_with_distributed_calls = 0
        self.synchronize_param_gather_calls = 0

    def all_gather_and_wait_parameters_ready(self):
        pass

    def synchronize_param_gather(self):
        self.synchronize_param_gather_calls += 1

    def _replace_param_with_distributed_if_needed(self):
        self.replace_param_with_distributed_calls += 1


def test_static_language_input_store_reuses_and_updates_buffers():
    store = _StaticLanguageInputStore()

    first = store.copy_microbatch(0, _language_inputs(fill_value=1))
    decoder_input_id = id(first["decoder_input"])
    cu_seqlens_id = id(first["packed_seq_params"].cu_seqlens_q)

    second = store.copy_microbatch(0, _language_inputs(fill_value=3))

    assert id(second["decoder_input"]) == decoder_input_id
    assert id(second["packed_seq_params"].cu_seqlens_q) == cu_seqlens_id
    assert second["packed_seq_params"].cu_seqlens_q_cpu.device.type == "cpu"
    assert second["packed_seq_params"].cu_seqlens_q_cpu.dtype == torch.long
    assert torch.equal(second["packed_seq_params"].cu_seqlens_q_cpu, torch.tensor([0, 4]))
    assert second["decoder_input"].requires_grad
    assert torch.equal(second["input_ids"], torch.full((1, 4), 3, dtype=torch.long))
    assert torch.equal(second["decoder_input"], torch.full((4, 1, 8), 3.0))


def test_static_language_input_store_rejects_shape_changes():
    store = _StaticLanguageInputStore()
    store.copy_microbatch(0, _language_inputs(seq_len=4))

    with pytest.raises(AssertionError, match="input shape changed"):
        store.copy_microbatch(0, _language_inputs(seq_len=5))


def test_static_language_input_store_rejects_cu_seqlens_changes():
    store = _StaticLanguageInputStore()
    store.copy_microbatch(0, _language_inputs(seq_len=4))

    changed = _language_inputs(seq_len=4)
    changed_cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
    params = changed["packed_seq_params"]
    params.cu_seqlens_q = changed_cu_seqlens.clone()
    params.cu_seqlens_kv = changed_cu_seqlens.clone()
    params.cu_seqlens_q_padded = changed_cu_seqlens.clone()
    params.cu_seqlens_kv_padded = changed_cu_seqlens.clone()

    with pytest.raises(AssertionError, match="static packed sequence boundaries"):
        store.copy_microbatch(0, changed)


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

    pipeline.mark_communicating(0, False)
    pipeline.bucket_status[pipeline.get_bucket_key(1, False)] = _FakeBucketStatus.READY_TO_USE
    pipeline.bucket_can_be_released[pipeline.get_bucket_key(0, True)] = True

    wrapper._synchronize_fsdp_all_gather_pipelines([fsdp])

    assert pipeline.wait_bucket_ready_calls == [(0, False)]
    assert pipeline.param_gather_event_map == {}
    assert pipeline.bucket_status[pipeline.get_bucket_key(0, False)] == _FakeBucketStatus.EMPTY
    assert pipeline.bucket_status[pipeline.get_bucket_key(0, True)] == _FakeBucketStatus.EMPTY
    assert (
        pipeline.bucket_status[pipeline.get_bucket_key(1, False)]
        == _FakeBucketStatus.PRESERVED
    )
    assert (
        pipeline.bucket_status[pipeline.get_bucket_key(1, True)]
        == _FakeBucketStatus.PRESERVED
    )
    assert not any(pipeline.bucket_can_be_released.values())
    assert fsdp.replace_param_with_distributed_calls == 0
    assert synchronize_calls == [True]

    with wrapper._fsdp_all_gather_on_capture_stream([fsdp], capture_stream):
        assert pipeline.ag_stream is None
        assert pipeline.outer_fsdp_group_param_gather_stream is capture_stream

    assert pipeline.ag_stream is original_ag_stream
    assert pipeline.outer_fsdp_group_param_gather_stream is original_outer_stream


def test_fsdp_pipeline_capture_cleanup_discards_work_without_waiting():
    pipeline = _FakeAllGatherPipeline()
    fsdp = _FakeMegatronFSDP(pipeline)
    wrapper = Qwen35VLDecoderFullCudaGraphWrapper(lambda **kwargs: None)

    pipeline.mark_communicating(0, False)

    wrapper._discard_captured_fsdp_all_gather_work([fsdp])

    assert pipeline.wait_bucket_ready_calls == []
    assert pipeline.param_gather_event_map == {}
    assert pipeline.bucket_status[pipeline.get_bucket_key(0, False)] == _FakeBucketStatus.EMPTY
    assert fsdp.replace_param_with_distributed_calls == 0


def test_fsdp_finalize_sync_override_preserves_buckets_and_restores_method():
    pipeline = _FakeAllGatherPipeline()
    fsdp = _FakeMegatronFSDP(pipeline)
    wrapper = Qwen35VLDecoderFullCudaGraphWrapper(lambda **kwargs: None)

    pipeline.mark_communicating(0, False)
    with wrapper._fsdp_param_gather_sync_without_releasing_buckets([fsdp]):
        fsdp.synchronize_param_gather()

    assert fsdp.synchronize_param_gather_calls == 0
    assert pipeline.wait_bucket_ready_calls == [(0, False)]
    assert pipeline.param_gather_event_map == {}
    assert pipeline.bucket_status[pipeline.get_bucket_key(0, False)] == _FakeBucketStatus.EMPTY
    assert fsdp.replace_param_with_distributed_calls == 0

    fsdp.synchronize_param_gather()
    assert fsdp.synchronize_param_gather_calls == 1
