# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Decoder-only full-iteration CUDA graph support for Qwen3.5-VL training."""

import contextlib
import gc
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional

import torch

from examples.multimodal_dev.forward_step import get_batch, loss_func
from megatron.core import parallel_state
from megatron.core.full_cuda_graph import (
    _override_stale_capture_stream,
    _print_rank0,
    _use_pytorch_stale_stream_fix,
    get_graph_pool,
    get_shared_capture_stream,
)
from megatron.core.packed_seq_params import (
    PackedSeqParams,
    get_thd_padding_kwargs,
    pad_sequence_for_thd,
)
from megatron.core.tensor_parallel.random import get_all_rng_states
from megatron.core.transformer.cuda_graphs import set_current_microbatch
from megatron.core.utils import get_attr_wrapped_model, get_model_config


_STAGE_TRAINING = "training"


@dataclass
class _PreparedMicrobatch:
    """Static decoder inputs plus the eager tensor that receives bridge gradients."""

    static_inputs: Dict[str, Any]
    static_loss_mask: torch.Tensor
    source_decoder_input: Optional[torch.Tensor]
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
            if id(module) in seen_modules:
                continue
            pipeline = getattr(module, "grad_reduce_pipeline", None)
            param_and_grad_buffer = getattr(module, "param_and_grad_buffer", None)
            if (
                pipeline is None
                or param_and_grad_buffer is None
                or id(pipeline) in seen_pipelines
            ):
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

    def copy_microbatch(self, microbatch: int, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if microbatch == len(self.inputs):
            static_inputs = self._copy_inputs(inputs)
            self.inputs.append(static_inputs)
            return static_inputs

        assert microbatch < len(self.inputs)
        return self._update_inputs(self.inputs[microbatch], inputs)

    def _copy_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        static_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                static_inputs[key] = _clone_tensor(
                    value, requires_grad=key == "decoder_input" and value.requires_grad
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
        self.use_pytorch_stale_stream_fix = _use_pytorch_stale_stream_fix()
        self.debug_fsdp_grad_reduce = (
            os.getenv("MEGATRON_QWEN35_VL_DECODER_CG_DEBUG_FSDP", "0") == "1"
        )

    def __call__(self, *args, **kwargs):
        assert len(args) == 0, 'forward_backward_func does not accept positional args'
        assert all(
            key in kwargs
            for key in ['model', 'data_iterator', 'num_microbatches', 'seq_length', 'forward_only']
        )

        if kwargs['forward_only']:
            return self.forward_backward_func(*args, **kwargs)

        self._validate_supported_schedule(kwargs)

        stage = _STAGE_TRAINING
        iteration = self.curr_iteration[stage]
        if iteration < self.cuda_graph_warmup_steps:
            if self.use_pytorch_stale_stream_fix:
                self.result[stage] = self.forward_backward_func(*args, **kwargs)
            else:
                self.result[stage] = self._forward_backward_on_capture_stream(*args, **kwargs)
            self.curr_iteration[stage] += 1
            return self.result[stage]

        capture_stream = get_shared_capture_stream()
        records, total_num_tokens = self._prepare_microbatches(
            kwargs, static_copy_stream=capture_stream
        )
        self._zero_static_decoder_input_grads(records, stream=capture_stream)
        graph_kwargs = self._graph_kwargs(kwargs, records)
        config = get_model_config(
            kwargs['model'][0] if isinstance(kwargs['model'], list) else kwargs['model']
        )
        finalize_model_grads_func = config.finalize_model_grads_func

        self._synchronize_fsdp_all_gather_pipelines(kwargs['model'])

        if self.cuda_graph[stage] is None:
            _print_rank0(
                f"{stage} iteration {iteration}: Qwen3.5-VL decoder graph capture start"
            )
            torch.distributed.barrier()
            # Drop eager warmup outputs before capture. Overwriting them from inside the
            # capture context can release tensors from the previous eager autograd graph
            # while CUDA stream capture is active.
            self.result[stage] = None
            gc.collect()
            torch.cuda.empty_cache()
            self.cuda_graph[stage] = torch.cuda.CUDAGraph()
            for _, state in get_all_rng_states().items():
                self.cuda_graph[stage].register_generator_state(state)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            torch.cuda.synchronize()
            with self._fsdp_all_gather_on_capture_stream(kwargs['model'], capture_stream):
                with _override_stale_capture_stream(self.use_pytorch_stale_stream_fix):
                    with torch.autograd.set_multithreading_enabled(False):
                        with torch.cuda.graph(
                            self.cuda_graph[stage],
                            stream=capture_stream,
                            pool=get_graph_pool(self.use_single_mempool),
                            capture_error_mode="relaxed",
                        ):
                            self.result[stage] = self._run_without_grad_finalization(
                                config, graph_kwargs
                            )
            torch.cuda.synchronize()
            self._discard_captured_fsdp_all_gather_work(kwargs['model'])
            torch.distributed.barrier()
            _print_rank0(
                f"{stage} iteration {iteration}: Qwen3.5-VL decoder graph capture done"
            )
        else:
            self.cuda_graph[stage].replay()
            torch.cuda.current_stream().wait_stream(capture_stream)

        self._run_fsdp_root_pre_backward(kwargs['model'])
        self._bridge_decoder_input_grads(records, config)
        # Some TE-backed vision gradients publish their FSDP grad-ready hook after
        # the bridge backward has enqueued CUDA work. Drain that eager work before
        # letting FSDP's root post-backward hook complete the iteration boundary.
        torch.cuda.synchronize()
        self._run_fsdp_root_post_backward(kwargs['model'])
        self._complete_fsdp_grad_reduce_readiness(kwargs['model'])
        with self._fsdp_param_gather_sync_without_releasing_buckets(kwargs['model']):
            self._finalize_model_grads(kwargs, finalize_model_grads_func, total_num_tokens)
        self.curr_iteration[stage] += 1
        return self.result[stage]

    def _forward_backward_on_capture_stream(self, *args, **kwargs):
        """Run eager warmup on the same stream used for decoder graph capture."""
        capture_stream = get_shared_capture_stream()
        current_stream = torch.cuda.current_stream()
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            result = self.forward_backward_func(*args, **kwargs)
        current_stream.wait_stream(capture_stream)
        return result

    def _prepare_microbatches(self, kwargs, static_copy_stream=None):
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

            if config.enable_autocast:
                context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
            else:
                context_manager = contextlib.nullcontext()

            def prepare_and_copy():
                with context_manager:
                    prepared_inputs = model_for_call(
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
                static_inputs = self.static_store[_STAGE_TRAINING].copy_microbatch(
                    microbatch, prepared_inputs
                )
                return prepared_inputs, static_inputs

            if static_copy_stream is None:
                prepared_inputs, static_inputs = prepare_and_copy()
            else:
                current_stream = torch.cuda.current_stream()
                static_copy_stream.wait_stream(current_stream)
                with torch.cuda.stream(static_copy_stream):
                    prepared_inputs, static_inputs = prepare_and_copy()
                current_stream.wait_stream(static_copy_stream)
            static_loss_mask = static_inputs.get("loss_mask")
            if static_loss_mask is None:
                static_loss_mask = torch.ones_like(static_inputs["input_ids"], dtype=torch.float)
            total_num_tokens += static_loss_mask.contiguous().view(-1).float().sum().to(torch.int)
            records.append(
                _PreparedMicrobatch(
                    static_inputs=static_inputs,
                    static_loss_mask=static_loss_mask,
                    source_decoder_input=prepared_inputs.get("decoder_input"),
                    static_decoder_input=static_inputs.get("decoder_input"),
                )
            )
        return records, total_num_tokens

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
            pad_by_appending_dummy_seq=config.pad_packed_seq_by_appending_dummy_seq,
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

    def _zero_static_decoder_input_grads(self, records, stream=None):
        def zero_grads():
            for record in records:
                static_decoder_input = record.static_decoder_input
                if static_decoder_input is not None and static_decoder_input.grad is not None:
                    static_decoder_input.grad.zero_()

        if stream is None:
            zero_grads()
            return

        current_stream = torch.cuda.current_stream()
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            zero_grads()
        current_stream.wait_stream(stream)

    def _bridge_one_decoder_input_grad(self, record):
        source_decoder_input = record.source_decoder_input
        static_decoder_input = record.static_decoder_input
        if (
            source_decoder_input is None
            or static_decoder_input is None
            or static_decoder_input.grad is None
            or not source_decoder_input.requires_grad
        ):
            return
        source_decoder_input.backward(static_decoder_input.grad.detach())

    def _bridge_decoder_input_grads(self, records, config):
        no_sync_func = config.no_sync_func
        if no_sync_func is None:
            no_sync_func = contextlib.nullcontext

        if len(records) > 1:
            with no_sync_func():
                for record in records[:-1]:
                    self._bridge_one_decoder_input_grad(record)

        if records:
            self._bridge_one_decoder_input_grad(records[-1])

    def _run_fsdp_root_pre_backward(self, model):
        for fsdp_module in _iter_megatron_fsdp_grad_reduce_modules(model):
            pre_backward = getattr(fsdp_module, "pre_backward", None)
            if callable(pre_backward):
                pre_backward()

    def _run_fsdp_root_post_backward(self, model):
        for fsdp_module in _iter_megatron_fsdp_grad_reduce_modules(model):
            post_backward = getattr(fsdp_module, "post_backward", None)
            if callable(post_backward):
                post_backward()

    def _complete_fsdp_grad_reduce_readiness(self, model):
        """Close Megatron-FSDP buckets left partial by delayed eager hooks.

        The root post-backward hook handles normal graph/eager gradient
        accumulation. Some TE-backed eager vision hooks can still arrive after a
        bucket was drained, leaving just a delayed bias ready in an otherwise
        empty bucket. Complete only those non-empty partial buckets, then drain
        again before finalization checks the pipeline state.
        """
        total_completed_params = 0
        last_partial_after = []

        for pass_id in range(4):
            visited, completed_params, partial_before = (
                self._complete_fsdp_grad_reduce_readiness_pass(model)
            )
            total_completed_params += completed_params
            self._drain_fsdp_grad_reduce_queues(model)
            last_partial_after = self._collect_fsdp_partial_grad_reduce_buckets(model)

            if self.debug_fsdp_grad_reduce:
                _print_rank0(
                    "Qwen3.5-VL FSDP grad reduce completion "
                    f"pass={pass_id} visited={visited} "
                    f"completed_missing_params={completed_params} "
                    f"partial_before={partial_before[:4]} "
                    f"partial_after={last_partial_after[:4]}"
                )

            if not last_partial_after:
                return

        if self.debug_fsdp_grad_reduce:
            _print_rank0(
                "Qwen3.5-VL FSDP grad reduce completion still has partial buckets "
                f"after 4 passes; total_completed_missing_params={total_completed_params} "
                f"partial_after={last_partial_after[:8]}"
            )

    def _complete_fsdp_grad_reduce_readiness_pass(self, model):
        visited = 0
        completed_params = 0
        partial_before = []
        for fsdp_module in _iter_megatron_fsdp_grad_reduce_modules(model):
            visited += 1
            pipeline = getattr(fsdp_module, "grad_reduce_pipeline", None)
            param_and_grad_buffer = getattr(fsdp_module, "param_and_grad_buffer", None)
            if pipeline is None or param_and_grad_buffer is None:
                continue

            missing_params = []
            for bucket_id, ready_params in enumerate(pipeline.bucket_grad_ready_params):
                if bucket_id >= len(param_and_grad_buffer.parameter_groups):
                    continue
                group = param_and_grad_buffer.parameter_groups[bucket_id]
                if group.main_grad_buffer is None:
                    continue
                if not ready_params:
                    continue
                if len(ready_params) < len(group.params):
                    param_to_name = getattr(param_and_grad_buffer, "param_to_name", {})
                    partial_before.append(
                        (
                            bucket_id,
                            len(ready_params),
                            len(group.params),
                            [param_to_name.get(param, "<unnamed>") for param in ready_params],
                            [
                                param_to_name.get(param, "<unnamed>")
                                for param in group.params
                                if param not in ready_params
                            ],
                        )
                    )
                for param in group.params:
                    if param not in ready_params:
                        missing_params.append(param)

            if not missing_params:
                continue

            completed_params += len(missing_params)
            pipeline.reduce_gradients(
                missing_params,
                suggested_queue_capacity=getattr(
                    fsdp_module, "suggested_RS_queue_capacity", None
                ),
                outer_fsdp_group_grad_reduce=getattr(
                    getattr(fsdp_module, "dist_index", None), "use_hybrid_fsdp", False
                ),
            )

        return visited, completed_params, partial_before

    def _drain_fsdp_grad_reduce_queues(self, model):
        for fsdp_module in _iter_megatron_fsdp_grad_reduce_modules(model):
            pipeline = getattr(fsdp_module, "grad_reduce_pipeline", None)
            if pipeline is not None:
                pipeline.wait_for_previous_grad_reduce(0)

    def _collect_fsdp_partial_grad_reduce_buckets(self, model):
        partial = []
        for fsdp_module in _iter_megatron_fsdp_grad_reduce_modules(model):
            pipeline = getattr(fsdp_module, "grad_reduce_pipeline", None)
            param_and_grad_buffer = getattr(fsdp_module, "param_and_grad_buffer", None)
            if pipeline is None or param_and_grad_buffer is None:
                continue
            for bucket_id, ready_params in enumerate(pipeline.bucket_grad_ready_params):
                if bucket_id >= len(param_and_grad_buffer.parameter_groups):
                    continue
                group = param_and_grad_buffer.parameter_groups[bucket_id]
                if (
                    group.main_grad_buffer is not None
                    and 0 < len(ready_params) < len(group.params)
                ):
                    param_to_name = getattr(param_and_grad_buffer, "param_to_name", {})
                    partial.append(
                        (
                            bucket_id,
                            len(ready_params),
                            len(group.params),
                            [param_to_name.get(param, "<unnamed>") for param in ready_params],
                        )
                    )
        return partial

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
