# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Decoder-only full-iteration CUDA graph support for Qwen3.5-VL training."""

import contextlib
import gc
import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional

import torch

from examples.multimodal_dev.forward_step import get_batch, loss_func
from megatron.core import parallel_state
from megatron.core.full_cuda_graph import get_graph_pool, get_shared_capture_stream
from megatron.core.packed_seq_params import (
    PackedSeqParams,
    get_thd_padding_kwargs,
    pad_sequence_for_thd,
    resolve_thd_tail_padding_policy,
)
from megatron.core.tensor_parallel.random import get_all_rng_states
from megatron.core.transformer.cuda_graphs import set_current_microbatch
from megatron.core.utils import get_attr_wrapped_model, get_model_config
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    to_local_if_dtensor,
)


_STAGE_TRAINING = "training"
logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").lower() in ("1", "true", "yes", "on")


def _env_flag_default(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


@dataclass
class _PreparedMicrobatch:
    """Static decoder inputs plus the batch needed to recompute bridge gradients."""

    static_inputs: Dict[str, Any]
    static_loss_mask: torch.Tensor
    bridge_batch: Optional[Dict[str, Any]]
    static_decoder_input: Optional[torch.Tensor]


def _clone_tensor(src: torch.Tensor, *, requires_grad: bool = False) -> torch.Tensor:
    cloned = src.detach().clone()
    if requires_grad:
        cloned.requires_grad_(True)
    return cloned


def _copy_tensor(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    assert dst.shape == src.shape, (
        f"Qwen3.5-VL decoder CUDA graph input shape changed from {tuple(dst.shape)} "
        f"to {tuple(src.shape)}."
    )
    assert dst.dtype == src.dtype, (
        f"Qwen3.5-VL decoder CUDA graph input dtype changed from {dst.dtype} to {src.dtype}."
    )
    with torch.no_grad():
        dst.copy_(src, non_blocking=True)
    return dst


def _assert_tensor_contents_unchanged(field: str, dst: torch.Tensor, src: torch.Tensor) -> None:
    if not torch.equal(dst, src):
        raise AssertionError(
            "Qwen3.5-VL decoder CUDA graph THD PackedSeqParams field "
            f"{field} changed after capture. THD FLA kernels derive Python-side "
            "chunk metadata from cu_seqlens during capture, so full-iteration "
            "CUDA graph replay currently requires static packed sequence boundaries."
        )


def _is_megatron_fsdp_module(module: Any) -> bool:
    return hasattr(module, "all_gather_pipeline") and hasattr(
        module, "all_gather_and_wait_parameters_ready"
    )


def _is_megatron_fsdp_grad_reduce_module(module: Any) -> bool:
    return hasattr(module, "grad_reduce_pipeline") and hasattr(module, "param_and_grad_buffer")


def _rank0() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _tensor_nbytes(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None or not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
        return 0
    return tensor.numel() * tensor.element_size()


def _mb(num_bytes: int) -> float:
    return num_bytes / (1024**2)


def _enable_stale_capture_stream_override() -> None:
    if hasattr(torch.autograd.graph, 'set_override_stale_capture_stream'):
        torch.autograd.graph.set_override_stale_capture_stream(True)
    else:
        logger.warning(
            'torch.autograd.graph.set_override_stale_capture_stream is not '
            'available in this PyTorch version; CUDA graph capture may fail '
            'if autograd nodes hold stale references to non-capturing streams. '
            'Upgrade to a PyTorch build that includes pytorch/pytorch#180090.'
        )


def _iter_megatron_fsdp_modules(model: Any):
    roots = model if isinstance(model, (list, tuple)) else [model]
    seen_modules = set()
    seen_pipelines = set()

    for root in roots:
        modules = root.modules() if isinstance(root, torch.nn.Module) else [root]
        for module in modules:
            if id(module) in seen_modules or not _is_megatron_fsdp_module(module):
                continue
            pipeline = getattr(module, "all_gather_pipeline", None)
            if pipeline is None or id(pipeline) in seen_pipelines:
                continue
            seen_modules.add(id(module))
            seen_pipelines.add(id(pipeline))
            yield module


def _iter_megatron_fsdp_grad_reduce_modules(model: Any):
    roots = model if isinstance(model, (list, tuple)) else [model]
    seen_modules = set()
    seen_pipelines = set()

    for root in roots:
        modules = root.modules() if isinstance(root, torch.nn.Module) else [root]
        for module in modules:
            if id(module) in seen_modules or not _is_megatron_fsdp_grad_reduce_module(module):
                continue
            pipeline = getattr(module, "grad_reduce_pipeline", None)
            if pipeline is None or id(pipeline) in seen_pipelines:
                continue
            seen_modules.add(id(module))
            seen_pipelines.add(id(pipeline))
            yield module


def _copy_packed_seq_params(src: PackedSeqParams) -> PackedSeqParams:
    copied = PackedSeqParams(
        qkv_format=src.qkv_format,
        cu_seqlens_q=_clone_tensor(src.cu_seqlens_q) if src.cu_seqlens_q is not None else None,
        cu_seqlens_kv=_clone_tensor(src.cu_seqlens_kv) if src.cu_seqlens_kv is not None else None,
        cu_seqlens_q_padded=(
            _clone_tensor(src.cu_seqlens_q_padded)
            if src.cu_seqlens_q_padded is not None
            else None
        ),
        cu_seqlens_kv_padded=(
            _clone_tensor(src.cu_seqlens_kv_padded)
            if src.cu_seqlens_kv_padded is not None
            else None
        ),
        max_seqlen_q=src.max_seqlen_q,
        max_seqlen_kv=src.max_seqlen_kv,
        local_cp_size=src.local_cp_size,
        cp_group=src.cp_group,
        total_tokens=src.total_tokens,
        seq_idx=_clone_tensor(src.seq_idx) if src.seq_idx is not None else None,
        pad_between_seqs=src.pad_between_seqs,
        cp_partition_mode=src.cp_partition_mode,
    )
    copied.refresh_cpu_cu_seqlens_cache()
    return copied


def _update_packed_seq_params(dst: PackedSeqParams, src: PackedSeqParams) -> PackedSeqParams:
    static_fields = (
        "qkv_format",
        "max_seqlen_q",
        "max_seqlen_kv",
        "local_cp_size",
        "cp_group",
        "total_tokens",
        "pad_between_seqs",
        "cp_partition_mode",
    )
    for field in static_fields:
        assert getattr(dst, field) == getattr(src, field), (
            f"Qwen3.5-VL decoder CUDA graph PackedSeqParams field {field} changed "
            f"from {getattr(dst, field)!r} to {getattr(src, field)!r}."
        )

    tensor_fields = (
        "cu_seqlens_q",
        "cu_seqlens_kv",
        "cu_seqlens_q_padded",
        "cu_seqlens_kv_padded",
        "seq_idx",
    )
    for field in tensor_fields:
        src_value = getattr(src, field)
        dst_value = getattr(dst, field)
        if src_value is None:
            if field == "seq_idx":
                continue
            assert dst_value is None, (
                f"Qwen3.5-VL decoder CUDA graph PackedSeqParams field {field} changed "
                "from tensor to None."
            )
            continue
        if dst_value is None:
            setattr(dst, field, _clone_tensor(src_value))
        else:
            if field.startswith("cu_seqlens"):
                _assert_tensor_contents_unchanged(field, dst_value, src_value)
            _copy_tensor(dst_value, src_value)
    dst.refresh_cpu_cu_seqlens_cache()
    return dst


class _StaticLanguageInputStore:
    """Per-microbatch static CUDA buffers for prepared language-model inputs."""

    def __init__(self):
        self.inputs = []

    def copy_microbatch(
        self,
        microbatch: int,
        inputs: Dict[str, Any],
        *,
        decoder_input_requires_grad: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if microbatch == len(self.inputs):
            static_inputs = self._copy_inputs(
                inputs, decoder_input_requires_grad=decoder_input_requires_grad
            )
            self.inputs.append(static_inputs)
            return static_inputs

        assert microbatch < len(self.inputs)
        return self._update_inputs(self.inputs[microbatch], inputs)

    def _copy_inputs(
        self, inputs: Dict[str, Any], *, decoder_input_requires_grad: Optional[bool]
    ) -> Dict[str, Any]:
        static_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                requires_grad = value.requires_grad
                if key == "decoder_input" and decoder_input_requires_grad is not None:
                    requires_grad = decoder_input_requires_grad
                static_inputs[key] = _clone_tensor(
                    value, requires_grad=key == "decoder_input" and requires_grad
                )
            elif isinstance(value, PackedSeqParams):
                static_inputs[key] = _copy_packed_seq_params(value)
            else:
                static_inputs[key] = value
        return static_inputs

    def _update_inputs(self, static_inputs: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        assert static_inputs.keys() == inputs.keys(), (
            "Qwen3.5-VL decoder CUDA graph input keys changed from "
            f"{sorted(static_inputs.keys())} to {sorted(inputs.keys())}."
        )
        for key, value in inputs.items():
            static_value = static_inputs[key]
            if isinstance(value, torch.Tensor):
                _copy_tensor(static_value, value)
            elif isinstance(value, PackedSeqParams):
                _update_packed_seq_params(static_value, value)
            else:
                assert static_value == value, (
                    f"Qwen3.5-VL decoder CUDA graph non-tensor input {key} changed "
                    f"from {static_value!r} to {value!r}."
                )
        return static_inputs


class Qwen35VLDecoderFullCudaGraphWrapper:
    """Capture Qwen3.5-VL language decoder forward/backward while vision stays eager.

    The wrapper preserves the current full-iteration optimizer boundary: optimizer.step()
    stays outside this graph. Gradient finalization is also run outside the graph so the
    eager vision/text prelude receives bridge gradients before DP/FSDP synchronization.
    """

    def __init__(self, forward_backward_func, cuda_graph_warmup_steps=1, use_single_mempool=False):
        self.forward_backward_func = forward_backward_func
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps
        self.use_single_mempool = use_single_mempool
        self.curr_iteration = {_STAGE_TRAINING: 0}
        self.cuda_graph = {_STAGE_TRAINING: None}
        self.result = {_STAGE_TRAINING: None}
        self.static_store = {_STAGE_TRAINING: _StaticLanguageInputStore()}
        self.freeze_non_decoder = _env_flag("QWEN35_VL_DECODER_CG_FREEZE_NON_DECODER")
        self.skip_decoder_input_grad_bridge = _env_flag(
            "QWEN35_VL_DECODER_CG_SKIP_GRAD_BRIDGE"
        )
        self.pre_capture_cleanup = _env_flag("QWEN35_VL_DECODER_CG_PRE_CAPTURE_CLEANUP")
        self.memory_probe = _env_flag_default("QWEN35_VL_DECODER_CG_MEMORY_PROBE", True)
        self.eager_fsdp_all_gather = _env_flag_default(
            "QWEN35_VL_DECODER_CG_EAGER_FSDP_ALL_GATHER", True
        )
        self._freeze_applied = False

    def __call__(self, *args, **kwargs):
        assert len(args) == 0, 'forward_backward_func does not accept positional args'
        assert all(
            key in kwargs
            for key in ['model', 'data_iterator', 'num_microbatches', 'seq_length', 'forward_only']
        )

        if kwargs['forward_only']:
            return self.forward_backward_func(*args, **kwargs)

        self._validate_supported_schedule(kwargs)
        self._maybe_freeze_non_decoder(kwargs['model'])

        stage = _STAGE_TRAINING
        iteration = self.curr_iteration[stage]
        self._log_memory_probe(f"iter={iteration}:entry", kwargs['model'])
        if iteration < self.cuda_graph_warmup_steps:
            _enable_stale_capture_stream_override()
            warmup_stream = get_shared_capture_stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                self._log_memory_probe(f"iter={iteration}:warmup_before_forward_backward")
                self.result[stage] = self.forward_backward_func(*args, **kwargs)
            torch.cuda.current_stream().wait_stream(warmup_stream)
            self._log_memory_probe(f"iter={iteration}:warmup_after_forward_backward", kwargs['model'])
            self.curr_iteration[stage] += 1
            return self.result[stage]

        self._log_memory_probe(f"iter={iteration}:before_prepare_microbatches", kwargs['model'])
        records, total_num_tokens = self._prepare_microbatches(kwargs)
        self._log_memory_probe(
            f"iter={iteration}:after_prepare_microbatches static={self._static_input_summary()}",
            kwargs['model'],
        )
        self._zero_static_decoder_input_grads(records)
        graph_kwargs = self._graph_kwargs(kwargs, records)
        config = get_model_config(
            kwargs['model'][0] if isinstance(kwargs['model'], list) else kwargs['model']
        )
        finalize_model_grads_func = config.finalize_model_grads_func

        if self.eager_fsdp_all_gather:
            self._refresh_eager_fsdp_all_gather_buckets(kwargs['model'])
            self._log_memory_probe(f"iter={iteration}:after_eager_fsdp_all_gather", kwargs['model'])
        else:
            self._synchronize_fsdp_all_gather_pipelines(kwargs['model'])
            self._log_memory_probe(f"iter={iteration}:after_sync_fsdp_all_gather", kwargs['model'])

        release_context = (
            self._preserve_eager_fsdp_all_gather_buckets(kwargs['model'])
            if self.eager_fsdp_all_gather
            else contextlib.nullcontext()
        )
        with release_context:
            if self.cuda_graph[stage] is None:
                _enable_stale_capture_stream_override()
                torch.distributed.barrier()
                self.cuda_graph[stage] = torch.cuda.CUDAGraph()
                for _, state in get_all_rng_states().items():
                    self.cuda_graph[stage].register_generator_state(state)
                torch.cuda.synchronize()
                capture_stream = get_shared_capture_stream()
                with torch.cuda.stream(capture_stream):
                    self._refresh_static_decoder_input_leaves(records)
                self._log_memory_probe(f"iter={iteration}:after_refresh_decoder_leaves", kwargs['model'])
                self._cleanup_cuda_allocator_before_capture()
                self._log_memory_probe(f"iter={iteration}:after_pre_capture_cleanup", kwargs['model'])
                torch.distributed.barrier()
                torch.cuda.synchronize()
                self._log_memory_probe(f"iter={iteration}:before_cuda_graph_capture", kwargs['model'])
                fsdp_stream_context = (
                    contextlib.nullcontext()
                    if self.eager_fsdp_all_gather
                    else self._fsdp_all_gather_on_capture_stream(kwargs['model'], capture_stream)
                )
                with fsdp_stream_context:
                    with torch.cuda.graph(
                        self.cuda_graph[stage],
                        stream=capture_stream,
                        pool=get_graph_pool(self.use_single_mempool),
                        capture_error_mode="thread_local",
                    ):
                        self.result[stage] = self._run_without_grad_finalization(config, graph_kwargs)
                torch.cuda.synchronize()
                self._log_memory_probe(f"iter={iteration}:after_cuda_graph_capture", kwargs['model'])
                if self.eager_fsdp_all_gather:
                    self._assert_no_captured_fsdp_all_gather_work(kwargs['model'])
                else:
                    self._discard_captured_fsdp_all_gather_work(kwargs['model'])
                self._log_memory_probe(
                    f"iter={iteration}:after_discard_captured_fsdp_work", kwargs['model']
                )
                torch.distributed.barrier()
            else:
                self.cuda_graph[stage].replay()
                self._log_memory_probe(f"iter={iteration}:after_cuda_graph_replay", kwargs['model'])

            self._bridge_decoder_input_grads(records, config, kwargs['model'])
            self._log_memory_probe(f"iter={iteration}:after_bridge_decoder_input_grads", kwargs['model'])
            self._complete_fsdp_gradient_reductions_after_graph(kwargs['model'])
            self._log_memory_probe(f"iter={iteration}:after_complete_fsdp_grad_reduce", kwargs['model'])
            with self._fsdp_param_gather_sync_without_releasing_buckets(kwargs['model']):
                self._finalize_model_grads(kwargs, finalize_model_grads_func, total_num_tokens)
            self._log_memory_probe(f"iter={iteration}:after_finalize_model_grads", kwargs['model'])
        self.curr_iteration[stage] += 1
        return self.result[stage]

    def _prepare_microbatches(self, kwargs):
        model = kwargs['model']
        model_for_call = model[0] if isinstance(model, list) else model
        data_iterator = kwargs['data_iterator']
        iterator = data_iterator[0] if isinstance(data_iterator, list) else data_iterator
        num_microbatches = kwargs['num_microbatches']
        config = get_model_config(model_for_call)
        records = []
        total_num_tokens = torch.zeros([], dtype=torch.int, device='cuda')

        for microbatch in range(num_microbatches):
            batch = get_batch(iterator)
            assert batch is not None, "Qwen3.5-VL decoder CUDA graph received an empty batch."
            batch = self._pad_batch_for_decoder_graph(batch, config)
            prepared_inputs = self._prepare_language_model_inputs(
                model_for_call, batch, config, microbatch, track_grad=False
            )
            static_inputs = self.static_store[_STAGE_TRAINING].copy_microbatch(
                microbatch,
                prepared_inputs,
                decoder_input_requires_grad=True,
            )
            static_loss_mask = static_inputs.get("loss_mask")
            if static_loss_mask is None:
                static_loss_mask = torch.ones_like(static_inputs["input_ids"], dtype=torch.float)
            total_num_tokens += static_loss_mask.contiguous().view(-1).float().sum().to(torch.int)
            records.append(
                _PreparedMicrobatch(
                    static_inputs=static_inputs,
                    static_loss_mask=static_loss_mask,
                    bridge_batch=None if self.skip_decoder_input_grad_bridge else batch,
                    static_decoder_input=static_inputs.get("decoder_input"),
                )
            )
        return records, total_num_tokens

    def _prepare_language_model_inputs(
        self, model_for_call, batch, config, microbatch: int, *, track_grad: bool
    ):
        pixel_values = batch.get("pixel_values", None)
        if (
            pixel_values is not None
            and pixel_values.is_floating_point()
            and pixel_values.dtype == torch.float32
        ):
            pixel_values = pixel_values.bfloat16()

        if microbatch == 0 and hasattr(model_for_call, "set_is_first_microbatch"):
            model_for_call.set_is_first_microbatch()
        set_current_microbatch(model_for_call, microbatch)
        set_input_tensor = get_attr_wrapped_model(model_for_call, "set_input_tensor")
        set_input_tensor([None])

        context_manager = (
            torch.autocast("cuda", dtype=config.autocast_dtype)
            if config.enable_autocast
            else contextlib.nullcontext()
        )
        grad_context = contextlib.nullcontext() if track_grad else torch.no_grad()
        with grad_context, context_manager:
            return model_for_call(
                input_ids=batch["input_ids"],
                position_ids=batch.get("position_ids"),
                attention_mask=batch.get("attention_mask", None),
                labels=batch.get("labels", None),
                loss_mask=batch.get("loss_mask", None),
                padding_mask=batch.get("padding_mask", None),
                pixel_values=pixel_values,
                image_grid_thw=batch.get("image_grid_thw", None),
                packed_seq_params=batch.get("packed_seq_params", None),
                return_language_model_inputs=True,
            )

    def _maybe_freeze_non_decoder(self, model):
        if not self.freeze_non_decoder or self._freeze_applied:
            return

        model_for_call = model[0] if isinstance(model, list) else model
        root_model = get_attr_wrapped_model(
            model_for_call, "language_model", allow_none=False, return_model_obj=True
        )
        language_model = root_model.language_model
        decoder = getattr(language_model, "decoder", None)
        if decoder is None:
            raise RuntimeError(
                "QWEN35_VL_DECODER_CG_FREEZE_NON_DECODER requires "
                "language_model.decoder to exist."
            )

        for param in root_model.parameters():
            param.requires_grad_(False)
        for param in decoder.parameters():
            param.requires_grad_(True)
        self._freeze_applied = True

    def _validate_supported_schedule(self, kwargs):
        model = kwargs['model']
        data_iterator = kwargs['data_iterator']
        assert not isinstance(model, list) or len(model) == 1, (
            "Qwen3.5-VL decoder full-iteration CUDA graph currently supports only "
            "the non-pipeline single-model schedule."
        )
        assert not isinstance(data_iterator, list) or len(data_iterator) == 1, (
            "Qwen3.5-VL decoder full-iteration CUDA graph currently supports only "
            "one data iterator."
        )
        assert parallel_state.get_tensor_model_parallel_world_size() == 1, (
            "Qwen3.5-VL decoder full-iteration CUDA graph currently supports TP=1. "
            "The multimodal THD packer applies TP padding before the model CP/decoder split."
        )
        assert parallel_state.get_pipeline_model_parallel_world_size() == 1, (
            "Qwen3.5-VL decoder full-iteration CUDA graph currently supports PP=1."
        )
        assert parallel_state.get_context_parallel_world_size() == 1, (
            "Qwen3.5-VL decoder full-iteration CUDA graph currently supports CP=1."
        )
        assert parallel_state.get_virtual_pipeline_model_parallel_world_size() is None, (
            "Qwen3.5-VL decoder full-iteration CUDA graph currently does not support VPP."
        )

    def _pad_batch_for_decoder_graph(self, batch, config):
        packed_seq_params = batch.get("packed_seq_params", None)
        if packed_seq_params is None:
            return batch

        assert config.pad_packed_seq_alignment is not None, (
            "Qwen3.5-VL decoder full-iteration CUDA graph with THD inputs requires "
            "--pad-packed-seq-alignment=max or --pad-packed-seq-alignment equal to "
            "--max-seqlen-per-dp-cp-rank."
        )
        assert config.max_seqlen_per_dp_cp_rank is not None, (
            "Qwen3.5-VL decoder full-iteration CUDA graph with THD inputs requires "
            "--max-seqlen-per-dp-cp-rank."
        )

        alignment, target_len, max_num_seqs = get_thd_padding_kwargs(
            config.pad_packed_seq_alignment,
            config.max_seqlen_per_dp_cp_rank,
            config.thd_max_packed_sequences,
            cuda_graph_static=True,
        )
        (
            input_ids,
            labels,
            loss_mask,
            position_ids,
            packed_seq_params,
            padding_mask,
        ) = pad_sequence_for_thd(
            batch.get("input_ids", None),
            batch.get("labels", None),
            batch.get("loss_mask", None),
            batch.get("position_ids", None),
            packed_seq_params,
            alignment=alignment,
            target_len=target_len,
            max_num_seqs=max_num_seqs,
            tail_padding_policy=resolve_thd_tail_padding_policy(config),
            padding_mask=batch.get("padding_mask", None),
            cp_size=1,
            cp_rank=0,
        )
        assert packed_seq_params.total_tokens in (None, target_len), (
            "Qwen3.5-VL decoder CUDA graph PackedSeqParams total_tokens changed "
            f"from {packed_seq_params.total_tokens} to {target_len}."
        )
        packed_seq_params.total_tokens = target_len
        packed_seq_params.pad_between_seqs = False

        batch = dict(batch)
        batch["input_ids"] = input_ids
        batch["labels"] = labels
        batch["loss_mask"] = loss_mask
        batch["position_ids"] = position_ids
        batch["packed_seq_params"] = packed_seq_params
        batch["padding_mask"] = padding_mask
        return batch

    def _graph_kwargs(self, kwargs, records):
        graph_kwargs = dict(kwargs)
        graph_kwargs['forward_step_func'] = self._forward_step_from_prepared_microbatch
        if isinstance(kwargs['data_iterator'], list):
            graph_kwargs['data_iterator'] = [iter(records)]
        else:
            graph_kwargs['data_iterator'] = iter(records)
        return graph_kwargs

    def _forward_step_from_prepared_microbatch(self, data_iterator, model):
        record = next(data_iterator)
        output_tensor = model(
            input_ids=record.static_inputs["input_ids"],
            position_ids=record.static_inputs["position_ids"],
            attention_mask=record.static_inputs.get("attention_mask", None),
            language_model_inputs=record.static_inputs,
        )
        return output_tensor, partial(loss_func, record.static_loss_mask)

    def _run_without_grad_finalization(self, config, graph_kwargs):
        saved_finalize = config.finalize_model_grads_func
        config.finalize_model_grads_func = None
        try:
            return self.forward_backward_func(**graph_kwargs)
        finally:
            config.finalize_model_grads_func = saved_finalize

    def _all_fsdp_params_for_pipeline(self, pipeline):
        params = []
        seen_params = set()
        buffer = getattr(pipeline, "buffer", None)
        for group in getattr(buffer, "parameter_groups", ()):
            for param in getattr(group, "params", ()):
                param_id = id(param)
                if param_id in seen_params:
                    continue
                seen_params.add(param_id)
                params.append(param)
        return params

    def _mark_fsdp_all_gather_buckets_empty_without_free(self, pipeline, *, bwd: bool) -> None:
        bucket_status = getattr(pipeline, "bucket_status", None)
        if not bucket_status:
            return

        status_type = type(next(iter(bucket_status.values())))
        empty_status = status_type.EMPTY
        bucket_can_be_released = getattr(pipeline, "bucket_can_be_released", {})
        for bucket_id in range(pipeline.num_buckets):
            bucket_key = pipeline.get_bucket_key(bucket_id, bwd)
            if bucket_key in bucket_can_be_released:
                bucket_can_be_released[bucket_key] = False
            if bucket_key in bucket_status:
                bucket_status[bucket_key] = empty_status

    def _refresh_eager_fsdp_all_gather_buckets(self, model) -> None:
        """Refresh graph-stable FSDP all-gather buckets outside CUDA graph capture.

        This diagnostic path keeps bucket storage alive across replay, but forces
        eager all-gather each iteration by marking bucket status EMPTY without
        freeing storage. The subsequent all-gather writes updated sharded weights
        into the same bucket addresses that CUDA graph replay reads.
        """
        found_pipeline = False
        for fsdp_module in _iter_megatron_fsdp_modules(model):
            pipeline = fsdp_module.all_gather_pipeline
            self._wait_fsdp_all_gather_pipeline(pipeline)
            params = self._all_fsdp_params_for_pipeline(pipeline)
            if not params:
                continue

            for bwd in (False, True):
                self._mark_fsdp_all_gather_buckets_empty_without_free(pipeline, bwd=bwd)
                fsdp_module.all_gather_and_wait_parameters_ready(
                    params=params,
                    prefetch=False,
                    wait_bucket_ready=True,
                    bwd=bwd,
                )
            found_pipeline = True

        if found_pipeline:
            torch.cuda.synchronize()

    def _synchronize_fsdp_all_gather_pipelines(self, model):
        """Drain eager-prelude FSDP all-gather work before CUDA graph execution.

        This intentionally does not call AllGatherPipeline.reset(), because reset()
        releases temporary bucket storage. Decoder CUDA graph replay bakes those
        bucket addresses, so the wrapper only waits valid eager work and resets the
        Python status tables back to "needs all-gather".
        """
        found_pipeline = False
        for fsdp_module in _iter_megatron_fsdp_modules(model):
            pipeline = fsdp_module.all_gather_pipeline
            self._wait_fsdp_all_gather_pipeline(pipeline)
            self._reset_fsdp_all_gather_pipeline_python_state(pipeline)
            found_pipeline = True

        if found_pipeline:
            torch.cuda.synchronize()

    def _discard_captured_fsdp_all_gather_work(self, model):
        """Drop FSDP Work handles created inside CUDA graph capture without waiting.

        Work handles returned during stream capture are not valid eager-side wait
        targets. Leaving them in AllGatherPipeline.param_gather_event_map makes the
        following eager decoder-input backward fail when Megatron-FSDP tries to wait
        on a captured all-gather. CUDA graph replay owns those operations; the Python
        pipeline state must be cleared back to an eager-ready state.
        """
        for fsdp_module in _iter_megatron_fsdp_modules(model):
            self._reset_fsdp_all_gather_pipeline_python_state(fsdp_module.all_gather_pipeline)

    def _assert_no_captured_fsdp_all_gather_work(self, model) -> None:
        communicating = 0
        events = 0
        for fsdp_module in _iter_megatron_fsdp_modules(model):
            pipeline = fsdp_module.all_gather_pipeline
            events += len(getattr(pipeline, "param_gather_event_map", {}))
            for status in getattr(pipeline, "bucket_status", {}).values():
                if getattr(status, "name", str(status)) == "COMMUNICATING":
                    communicating += 1

        if communicating or events:
            raise RuntimeError(
                "Qwen3.5-VL decoder fullCG eager FSDP all-gather experiment expected "
                "no FSDP all-gather work inside CUDA graph capture, but found "
                f"{events} event(s) and {communicating} communicating bucket status entries."
            )

    def _wait_fsdp_all_gather_pipeline(self, pipeline):
        while getattr(pipeline, "param_gather_event_map", None):
            bucket_id, bwd = next(iter(pipeline.param_gather_event_map))
            pipeline.wait_bucket_ready(bucket_id, bwd)

    def _reset_fsdp_all_gather_pipeline_python_state(self, pipeline):
        event_map = getattr(pipeline, "param_gather_event_map", None)
        if event_map is not None:
            event_map.clear()

        bucket_status = getattr(pipeline, "bucket_status", None)
        if not bucket_status:
            return

        status_type = type(next(iter(bucket_status.values())))
        empty_status = status_type.EMPTY
        preserved_status = status_type.PRESERVED
        bucket_can_be_released = getattr(pipeline, "bucket_can_be_released", {})
        parameter_groups = getattr(getattr(pipeline, "buffer", None), "parameter_groups", ())

        for bucket_id in range(pipeline.num_buckets):
            group = parameter_groups[bucket_id]
            is_unit_bucket = getattr(group, "fsdp_unit_id", None) is not None
            for bwd in (False, True):
                bucket_key = pipeline.get_bucket_key(bucket_id, bwd)
                if bucket_key in bucket_can_be_released:
                    bucket_can_be_released[bucket_key] = False
                if bucket_key in bucket_status:
                    bucket_status[bucket_key] = empty_status if is_unit_bucket else preserved_status

    @contextlib.contextmanager
    def _preserve_eager_fsdp_all_gather_buckets(self, model):
        """Disable FSDP bucket release while replay reads eager-refreshed buckets."""
        original_methods = []

        for fsdp_module in _iter_megatron_fsdp_modules(model):
            pipeline = fsdp_module.all_gather_pipeline
            instance_dict = getattr(pipeline, "__dict__", {})
            had_instance_method = "release_bucket" in instance_dict
            original_method = instance_dict.get("release_bucket", None)

            def release_bucket(bucket_id, bwd, lazy=False, pipeline=pipeline):
                bucket_key = pipeline.get_bucket_key(bucket_id, bwd)
                bucket_can_be_released = getattr(pipeline, "bucket_can_be_released", {})
                if bucket_key in bucket_can_be_released:
                    bucket_can_be_released[bucket_key] = False
                return None

            setattr(pipeline, "release_bucket", release_bucket)
            original_methods.append((pipeline, had_instance_method, original_method))

        try:
            yield
        finally:
            for pipeline, had_instance_method, original_method in reversed(original_methods):
                if had_instance_method:
                    setattr(pipeline, "release_bucket", original_method)
                else:
                    delattr(pipeline, "release_bucket")

    @contextlib.contextmanager
    def _fsdp_param_gather_sync_without_releasing_buckets(self, model):
        """Keep graph-owned FSDP buckets alive during gradient finalization."""
        original_methods = []

        for fsdp_module in _iter_megatron_fsdp_modules(model):
            instance_dict = getattr(fsdp_module, "__dict__", {})
            had_instance_method = "synchronize_param_gather" in instance_dict
            original_method = instance_dict.get("synchronize_param_gather", None)

            def synchronize_param_gather(module=fsdp_module):
                pipeline = module.all_gather_pipeline
                self._wait_fsdp_all_gather_pipeline(pipeline)
                if not self.eager_fsdp_all_gather:
                    self._reset_fsdp_all_gather_pipeline_python_state(pipeline)

            setattr(fsdp_module, "synchronize_param_gather", synchronize_param_gather)
            original_methods.append((fsdp_module, had_instance_method, original_method))

        try:
            yield
        finally:
            for fsdp_module, had_instance_method, original_method in reversed(original_methods):
                if had_instance_method:
                    setattr(fsdp_module, "synchronize_param_gather", original_method)
                else:
                    delattr(fsdp_module, "synchronize_param_gather")

    @contextlib.contextmanager
    def _fsdp_all_gather_on_capture_stream(self, model, capture_stream):
        """Launch captured FSDP all-gathers on capture-compatible streams."""
        original_streams = []
        for fsdp_module in _iter_megatron_fsdp_modules(model):
            pipeline = fsdp_module.all_gather_pipeline
            original_streams.append((pipeline, "ag_stream", pipeline.ag_stream))
            # Use torch.cuda.current_stream() inside torch.cuda.graph, which is the
            # graph capture stream. This avoids waiting on pre-existing side-stream work.
            pipeline.ag_stream = None

            if hasattr(pipeline, "outer_fsdp_group_param_gather_stream"):
                original_streams.append(
                    (
                        pipeline,
                        "outer_fsdp_group_param_gather_stream",
                        pipeline.outer_fsdp_group_param_gather_stream,
                    )
                )
                pipeline.outer_fsdp_group_param_gather_stream = capture_stream

        try:
            yield
        finally:
            for pipeline, attr, stream in reversed(original_streams):
                setattr(pipeline, attr, stream)

    def _zero_static_decoder_input_grads(self, records):
        for record in records:
            static_decoder_input = record.static_decoder_input
            if static_decoder_input is not None and static_decoder_input.grad is not None:
                static_decoder_input.grad.zero_()

    def _refresh_static_decoder_input_leaves(self, records):
        """Recreate decoder graph leaf inputs on the capture stream before first capture."""
        static_inputs_by_microbatch = self.static_store[_STAGE_TRAINING].inputs
        for microbatch, record in enumerate(records):
            static_decoder_input = record.static_decoder_input
            if (
                static_decoder_input is None
                or not static_decoder_input.requires_grad
                or microbatch >= len(static_inputs_by_microbatch)
            ):
                continue
            refreshed = static_decoder_input.detach().clone().requires_grad_(True)
            record.static_inputs["decoder_input"] = refreshed
            record.static_decoder_input = refreshed
            static_inputs_by_microbatch[microbatch]["decoder_input"] = refreshed

    def _cleanup_cuda_allocator_before_capture(self) -> None:
        """Release unreachable eager allocations before the first graph capture."""
        if not self.pre_capture_cleanup:
            return

        torch.cuda.synchronize()
        before_reserved = torch.cuda.memory_reserved()
        before_allocated = torch.cuda.memory_allocated()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info(
                "Qwen3.5-VL decoder fullCG pre-capture cleanup: "
                "allocated %.2f -> %.2f MB, reserved %.2f -> %.2f MB",
                before_allocated / (1024**2),
                torch.cuda.memory_allocated() / (1024**2),
                before_reserved / (1024**2),
                torch.cuda.memory_reserved() / (1024**2),
            )

    def _log_memory_probe(self, label: str, model=None) -> None:
        if not self.memory_probe:
            return

        torch.cuda.synchronize()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()
        max_reserved = torch.cuda.max_memory_reserved()
        device_used = total_bytes - free_bytes
        if _rank0():
            logger.info(
                "Qwen3.5-VL decoder fullCG memory probe | %s | "
                "allocated=%.2f MB reserved=%.2f MB max_allocated=%.2f MB "
                "max_reserved=%.2f MB device_used=%.2f MB",
                label,
                _mb(allocated),
                _mb(reserved),
                _mb(max_allocated),
                _mb(max_reserved),
                _mb(device_used),
            )
            if model is not None:
                logger.info(
                    "Qwen3.5-VL decoder fullCG FSDP probe | %s | %s",
                    label,
                    self._fsdp_pipeline_summary(model),
                )

    def _static_input_summary(self) -> str:
        seen_ptrs = set()
        tensor_count = 0
        total_bytes = 0
        for microbatch_inputs in self.static_store[_STAGE_TRAINING].inputs:
            for value in microbatch_inputs.values():
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    ptr = value.data_ptr()
                    if ptr in seen_ptrs:
                        continue
                    seen_ptrs.add(ptr)
                    tensor_count += 1
                    total_bytes += _tensor_nbytes(value)
        return f"tensors={tensor_count} bytes={total_bytes} mb={_mb(total_bytes):.2f}"

    def _fsdp_pipeline_summary(self, model) -> str:
        module_count = 0
        num_buckets = 0
        event_count = 0
        releasable_count = 0
        status_counts = {}
        temp_bytes = 0
        persistent_bytes = 0
        seen_ptrs = set()

        for fsdp_module in _iter_megatron_fsdp_modules(model):
            module_count += 1
            pipeline = fsdp_module.all_gather_pipeline
            num_buckets += getattr(pipeline, "num_buckets", 0)
            event_count += len(getattr(pipeline, "param_gather_event_map", {}))
            releasable_count += sum(
                1 for can_release in getattr(pipeline, "bucket_can_be_released", {}).values()
                if can_release
            )
            for status in getattr(pipeline, "bucket_status", {}).values():
                status_name = getattr(status, "name", str(status))
                status_counts[status_name] = status_counts.get(status_name, 0) + 1

            buffer = getattr(pipeline, "buffer", None)
            for group in getattr(buffer, "parameter_groups", ()):
                for attr in (
                    "model_weight_buffer",
                    "transpose_weight_buffer",
                    "hfsdp_helper_wbuf",
                    "hfsdp_helper_wtbuf",
                ):
                    dp_buffer = getattr(group, attr, None)
                    if dp_buffer is None:
                        continue
                    data = getattr(dp_buffer, "data", None)
                    if isinstance(data, torch.Tensor) and data.is_cuda:
                        ptr = data.data_ptr()
                        if ptr not in seen_ptrs:
                            seen_ptrs.add(ptr)
                            persistent_bytes += _tensor_nbytes(data)
                    allocator = getattr(dp_buffer, "temporary_bucket_allocator", None)
                    for bucket in getattr(allocator, "buckets", {}).values():
                        bucket_data = getattr(bucket, "data", None)
                        if isinstance(bucket_data, torch.Tensor) and bucket_data.is_cuda:
                            ptr = bucket_data.data_ptr()
                            if ptr not in seen_ptrs:
                                seen_ptrs.add(ptr)
                                temp_bytes += _tensor_nbytes(bucket_data)

        status_summary = ",".join(
            f"{name}:{count}" for name, count in sorted(status_counts.items())
        )
        return (
            f"modules={module_count} buckets={num_buckets} events={event_count} "
            f"releasable={releasable_count} statuses={status_summary or 'none'} "
            f"persistent_mb={_mb(persistent_bytes):.2f} temp_bucket_mb={_mb(temp_bytes):.2f}"
        )

    def _bridge_one_decoder_input_grad(self, record, model_for_call, config, microbatch):
        static_decoder_input = record.static_decoder_input
        if (
            record.bridge_batch is None
            or static_decoder_input is None
            or static_decoder_input.grad is None
        ):
            return
        prepared_inputs = self._prepare_language_model_inputs(
            model_for_call, record.bridge_batch, config, microbatch, track_grad=True
        )
        source_decoder_input = prepared_inputs.get("decoder_input")
        if source_decoder_input is None or not source_decoder_input.requires_grad:
            return
        source_decoder_input.backward(static_decoder_input.grad.detach())

    def _bridge_decoder_input_grads(self, records, config, model):
        if self.skip_decoder_input_grad_bridge:
            return

        model_for_call = model[0] if isinstance(model, list) else model
        no_sync_func = config.no_sync_func
        if no_sync_func is None:
            no_sync_func = contextlib.nullcontext

        if len(records) > 1:
            with no_sync_func():
                for microbatch, record in enumerate(records[:-1]):
                    self._bridge_one_decoder_input_grad(record, model_for_call, config, microbatch)

        if records:
            self._bridge_one_decoder_input_grad(
                records[-1], model_for_call, config, len(records) - 1
            )

    def _complete_fsdp_gradient_reductions_after_graph(self, model):
        """Finish graph-covered FSDP buckets that eager bridge backward partially marked.

        During CUDA graph capture Megatron-FSDP skips its normal post-backward
        grad-ready path. The recomputed eager prelude backward can still mark
        vision / embedding params ready in buckets that also contain graph-owned
        decoder params. Complete those buckets before final gradient sync resets
        the pipeline.
        """
        for fsdp_module in _iter_megatron_fsdp_grad_reduce_modules(model):
            pipeline = getattr(fsdp_module, "grad_reduce_pipeline", None)
            param_and_grad_buffer = getattr(fsdp_module, "param_and_grad_buffer", None)
            if pipeline is None or param_and_grad_buffer is None:
                continue
            bucket_ready_params = getattr(pipeline, "bucket_grad_ready_params", None)
            if bucket_ready_params is None:
                continue

            processed_bucket_groups = set()
            for bucket_id, ready_params in enumerate(bucket_ready_params):
                if not ready_params:
                    continue
                bucket_group_ids = tuple(
                    param_and_grad_buffer.bucket_to_bucket_group.get(bucket_id, [bucket_id])
                )
                if bucket_group_ids in processed_bucket_groups:
                    continue
                processed_bucket_groups.add(bucket_group_ids)

                params_to_reduce = []
                for group_bucket_id in bucket_group_ids:
                    group = param_and_grad_buffer.parameter_groups[group_bucket_id]
                    group_ready_params = bucket_ready_params[group_bucket_id]
                    if getattr(group, "main_grad_buffer", None) is None:
                        group_ready_params.clear()
                        continue

                    for param in group.params:
                        if not param.requires_grad or param in group_ready_params:
                            continue
                        self._ensure_fsdp_param_main_grad(fsdp_module, param)
                        params_to_reduce.append(param)

                if not params_to_reduce:
                    representative = next(iter(ready_params), None)
                    if representative is None:
                        continue
                    params_to_reduce = [representative]

                is_last_microbatch = getattr(fsdp_module, "is_last_microbatch", False)
                model_auto_sync = getattr(fsdp_module, "model_auto_sync", False)
                pipeline.reduce_gradients(
                    params_to_reduce,
                    suggested_queue_capacity=getattr(
                        fsdp_module, "suggested_RS_queue_capacity", None
                    ),
                    outer_fsdp_group_grad_reduce=(
                        getattr(
                            getattr(fsdp_module, "dist_index", None),
                            "use_hybrid_fsdp",
                            False,
                        )
                        and (is_last_microbatch or model_auto_sync)
                    ),
                )

    def _ensure_fsdp_param_main_grad(self, fsdp_module, param):
        param_and_grad_buffer = fsdp_module.param_and_grad_buffer
        group_id = param_and_grad_buffer.param_to_param_group[param]
        group = param_and_grad_buffer.parameter_groups[group_id]
        if not group.requires_grad:
            return

        grad_buffer = (
            group.hfsdp_helper_gbuf if group.hfsdp_helper_gbuf else group.main_grad_buffer
        )
        if grad_buffer.is_data_distributed:
            if not param.grad_added_to_main_grad:
                graph_param_ids = getattr(fsdp_module, "_cuda_graph_fused_wgrad_params", set())
                if (
                    id(param) in graph_param_ids
                    or id(getattr(param, "orig_param", param)) in graph_param_ids
                ):
                    param.grad_added_to_main_grad = True
                else:
                    param.main_grad = param.get_main_grad()
                    if param.grad is not None:
                        param.main_grad.copy_(to_local_if_dtensor(param.grad))
                        del param.grad
                    else:
                        param.main_grad.zero_()
        else:
            if not param.grad_added_to_main_grad:
                param.main_grad = param.get_main_grad()
                if param.grad is not None:
                    param.main_grad.add_(to_local_if_dtensor(param.grad))
                    del param.grad

        if param.grad_added_to_main_grad and param.grad is not None:
            del param.grad
        param.grad_added_to_main_grad = False

    def _finalize_model_grads(self, kwargs, finalize_model_grads_func, total_num_tokens):
        if finalize_model_grads_func is None:
            return
        model = kwargs['model'] if isinstance(kwargs['model'], list) else [kwargs['model']]
        config = get_model_config(model[0])
        finalize_model_grads_func(
            model,
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=kwargs.get("pg_collection", None),
            force_all_reduce=kwargs.get("force_all_reduce", False),
        )

    def reset_cuda_graph(self, stage=None):
        """Reset the captured decoder CUDA graph."""
        if stage is None or stage == _STAGE_TRAINING:
            if self.cuda_graph[_STAGE_TRAINING] is not None:
                del self.cuda_graph[_STAGE_TRAINING]
            self.cuda_graph[_STAGE_TRAINING] = None
            self.result[_STAGE_TRAINING] = None
            self.curr_iteration[_STAGE_TRAINING] = 0
        gc.collect()
