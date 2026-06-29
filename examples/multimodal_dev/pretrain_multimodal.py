# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Standalone entry point for multimodal_dev model training (FSDP + EP).

This entry point is **model-agnostic**.  All model-specific logic (layer
specs, model construction, FLOPs metadata, dataset generation) is
delegated to factory functions registered in
:data:`multimodal_dev.models.MODEL_REGISTRY`.

Adding a new architecture only requires:

1. Creating a new model package under ``multimodal_dev/models/<arch>/``
   with the appropriate factory functions.
2. Registering an entry in ``MODEL_REGISTRY``.

No changes to this file are necessary.

Usage::

    torchrun --nproc_per_node=8 multimodal_dev/pretrain_multimodal.py \\
        --model-arch qwen35_vl \\
        --dataset-provider mock \\
        ... (other megatron args)
"""

import importlib
import json
import logging
import os
import sys
import traceback
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
)

import torch
from PIL import Image

from examples.multimodal_dev.arguments import add_multimodal_args
from examples.multimodal_dev.forward_step import forward_step
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
from megatron.training import get_args, pretrain
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.checkpointing import get_checkpoint_name

try:
    from megatron.energon import (
        DefaultTaskEncoder,
        LimitDataset,
        RepeatDataset,
        VQASample,
        WorkerConfig,
        get_loader,
        get_savable_loader,
        get_train_dataset,
        get_val_datasets,
    )
    from megatron.energon.task_encoder.base import stateless

    HAVE_ENERGON = True
except ImportError:
    DefaultTaskEncoder = object
    LimitDataset = None
    RepeatDataset = None
    VQASample = None
    WorkerConfig = None
    get_loader = None
    get_savable_loader = None
    get_train_dataset = None
    get_val_datasets = None
    HAVE_ENERGON = False

    def stateless(fn=None, **_kwargs):
        if fn is None:
            return lambda wrapped: wrapped
        return fn


logger = logging.getLogger(__name__)

_QWEN_VL_MIN_PIXELS = 256 * 28 * 28
_QWEN_VL_MAX_PIXELS = 1280 * 28 * 28


def _print_error_handler(exc: Exception, key: Optional[str]):
    print(
        f"The following exception occurred in the dataloader for sample {key} "
        "and is skipped",
        file=sys.stderr,
    )
    traceback.print_exc()


def _ensure_energon_available():
    if HAVE_ENERGON:
        return
    raise ImportError(
        "llava_pretrain_wds requires megatron-energon. Install "
        "`megatron-energon` or use --dataset-provider mock/cord_v2."
    )


def _normalize_data_path(data_path) -> str:
    if isinstance(data_path, (list, tuple)):
        if len(data_path) != 1:
            raise ValueError(
                "llava_pretrain_wds expects exactly one --data-path value, "
                f"got {list(data_path)!r}"
            )
        data_path = data_path[0]
    return os.fspath(data_path)


def _as_list(value) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _maybe_decode_json(value):
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return value


def _maybe_decode_image(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return _maybe_decode_image(value[0] if value else None)
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(BytesIO(value)).convert("RGB")
    if isinstance(value, torch.Tensor):
        raise TypeError(
            "llava_pretrain_wds expects PIL/encoded images from Energon; "
            "got a tensor image. Pass image_decode='pil' to get_train_dataset."
        )
    return value


def _get_sample_field(sample, *names):
    if isinstance(sample, dict):
        for name in names:
            if name in sample:
                return sample[name]
        return None
    for name in names:
        if hasattr(sample, name):
            return getattr(sample, name)
    return None


class Qwen35VLLlavaPretrainEnergonTaskEncoder(DefaultTaskEncoder):
    """Encode LLaVA-Pretrain Energon samples for multimodal_dev.

    The encoder intentionally returns a list of per-sample dictionaries from
    ``batch``. ``examples.multimodal_dev.forward_step.pack_or_pad_batch`` owns
    both BSHD padding and THD packing for this model path, so Energon is used
    for distributed WebDataset reading and checkpointable iteration only.
    """

    def __init__(
        self,
        processor_path: str,
        seq_length: int,
        image_token_id: Optional[int],
    ):
        super().__init__()

        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
        )
        self.seq_length = seq_length
        self.image_token_id = image_token_id

        tok = self.processor.tokenizer
        if tok.pad_token_id is not None:
            self.pad_token_id = int(tok.pad_token_id)
        elif tok.eos_token_id is not None:
            self.pad_token_id = int(tok.eos_token_id)
        else:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")

        if self.image_token_id is None:
            vocab = tok.get_vocab()
            for candidate in ("<|image_pad|>", "<|placeholder|>"):
                if candidate in vocab:
                    self.image_token_id = int(vocab[candidate])
                    break
            else:
                raise ValueError(
                    "Could not resolve image token id from tokenizer; "
                    "pass --image-token-id explicitly."
                )

        skipped = set(int(x) for x in (tok.all_special_ids or []))
        skipped.add(self.pad_token_id)
        skipped.add(int(self.image_token_id))
        self.skipped_token_ids = torch.tensor(sorted(skipped), dtype=torch.long)

    def _conversation_from_llava_turns(
        self,
        image: Image.Image,
        llava_conv: list,
    ) -> Tuple[list, str]:
        processor_conv = []
        assistant_text = ""
        image_attached = False

        for turn in llava_conv:
            role = turn.get("from") or turn.get("role")
            value = str(turn.get("value", turn.get("content", "")))
            if role in ("human", "user"):
                text = value.replace("<image>", "").strip()
                content = []
                if not image_attached:
                    content.append({"type": "image", "image": image})
                    image_attached = True
                if text:
                    content.append({"type": "text", "text": text})
                processor_conv.append({"role": "user", "content": content})
            elif role in ("gpt", "assistant"):
                assistant_text = value.strip()
                processor_conv.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": assistant_text}],
                    }
                )

        if not image_attached:
            raise ValueError("LLaVA pretrain sample does not contain a user image turn")
        if not assistant_text:
            raise ValueError("LLaVA pretrain sample does not contain an assistant answer")
        return processor_conv, assistant_text

    def _conversation_from_vqa_fields(self, sample) -> Tuple[list, str]:
        image = _maybe_decode_image(_get_sample_field(sample, "image", "jpg", "jpeg", "png"))
        if image is None:
            images = _as_list(_get_sample_field(sample, "images"))
            image = _maybe_decode_image(images[0] if images else None)
        if image is None:
            raise ValueError("LLaVA pretrain sample does not contain an image")

        raw_json = _maybe_decode_json(_get_sample_field(sample, "json"))
        if isinstance(raw_json, list):
            return self._conversation_from_llava_turns(image, raw_json)
        if isinstance(raw_json, dict) and "conversations" in raw_json:
            return self._conversation_from_llava_turns(image, raw_json["conversations"])

        contexts = _as_list(_get_sample_field(sample, "context", "question", "questions"))
        answers = _as_list(_get_sample_field(sample, "answers", "answer", "caption"))
        if not contexts or not answers:
            raise ValueError(
                "LLaVA pretrain sample must provide json conversations or "
                "VQA-style context/answers fields"
            )

        processor_conv = []
        assistant_text = ""
        for idx, (context, answer) in enumerate(zip(contexts, answers)):
            user_text = str(context).replace("<image>", "").strip()
            user_content = []
            if idx == 0:
                user_content.append({"type": "image", "image": image})
            if user_text:
                user_content.append({"type": "text", "text": user_text})
            processor_conv.append({"role": "user", "content": user_content})

            assistant_text = str(answer).strip()
            processor_conv.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                }
            )

        if not assistant_text:
            raise ValueError("LLaVA pretrain sample does not contain an assistant answer")
        return processor_conv, assistant_text

    def _mark_assistant_span(
        self,
        input_ids_list: List[int],
        assistant_text: str,
        loss_mask: torch.Tensor,
    ) -> bool:
        tokenizer = self.processor.tokenizer
        n = len(input_ids_list)
        variants = (
            assistant_text,
            assistant_text + "\n",
            assistant_text.strip(),
            assistant_text.strip() + "\n",
        )
        for variant in variants:
            span_tokens = tokenizer(
                variant,
                add_special_tokens=False,
            )["input_ids"]
            m = len(span_tokens)
            if m == 0 or m > n:
                continue
            for start in range(n - m, -1, -1):
                if input_ids_list[start : start + m] == span_tokens:
                    loss_mask[start : start + m] = 1.0
                    return True
        return False

    @stateless
    def encode_sample(self, sample):
        if VQASample is not None and isinstance(sample, VQASample):
            conversation, assistant_text = self._conversation_from_vqa_fields(sample)
        else:
            conversation, assistant_text = self._conversation_from_vqa_fields(sample)

        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        first_user = next(turn for turn in conversation if turn["role"] == "user")
        images = [
            item["image"]
            for item in first_user["content"]
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        batch = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            min_pixels=_QWEN_VL_MIN_PIXELS,
            max_pixels=_QWEN_VL_MAX_PIXELS,
        )

        input_ids = batch["input_ids"].squeeze(0)
        pixel_values = batch["pixel_values"].to(torch.bfloat16)
        image_grid_thw = batch["image_grid_thw"]

        if input_ids.shape[0] > self.seq_length:
            logger.warning(
                "Sample has %d tokens > seq_length=%d; truncating.",
                input_ids.shape[0],
                self.seq_length,
            )
            input_ids = input_ids[: self.seq_length]

        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        found = self._mark_assistant_span(
            input_ids.tolist(),
            assistant_text,
            loss_mask,
        )
        if not found:
            logger.warning("Assistant span not located; loss_mask will be all-zero.")

        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        labels[torch.isin(labels, self.skipped_token_ids)] = -100

        loss_mask = torch.cat(
            [loss_mask[1:], torch.zeros(1, dtype=loss_mask.dtype)],
        )
        labels[loss_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    def batch(self, samples: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        return list(samples)

    def encode_batch(self, batch_data):
        return batch_data


class EnergonDataloader:
    """Megatron-compatible wrapper for Megatron Energon loaders."""

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._iter = iter(_cyclic_iter(dataloader))

    def __next__(self):
        return next(self._iter)

    def __iter__(self):
        return iter(self._iter)

    def save_state(self):
        return self._dataloader.save_state_rank()


def _cyclic_iter(iterator: Iterable):
    while True:
        for item in iterator:
            yield item


def _is_first_or_last_stage(pp_size: int) -> bool:
    if pp_size == 1:
        return True
    pp_rank = get_pipeline_model_parallel_rank()
    return pp_rank in (0, pp_size - 1)


def _is_dataloader_rank() -> bool:
    is_first_tp_rank = get_tensor_model_parallel_rank() == 0
    pp_size = get_pipeline_model_parallel_world_size()
    return is_first_tp_rank and _is_first_or_last_stage(pp_size)


def qwen35_vl_llava_pretrain_energon_provider(train_val_test_num_samples):
    """Build Energon dataloaders for Qwen3.5-VL LLaVA-Pretrain WDS."""
    _ensure_energon_available()

    args = get_args()
    if args.dataloader_type != "external":
        raise ValueError(
            "llava_pretrain_wds uses Megatron Energon and requires "
            "--dataloader-type external so Megatron does not wrap the "
            "external loader in a second DataLoader."
        )
    if not getattr(args, "data_path", None):
        raise ValueError("llava_pretrain_wds requires --data-path")
    if not getattr(args, "hf_processor_path", None):
        raise ValueError("llava_pretrain_wds requires --hf-processor-path")

    if not _is_dataloader_rank():
        return None, None, None

    data_path = _normalize_data_path(args.data_path)
    seq_length = (
        getattr(args, "total_seq_length", None)
        or getattr(args, "seq_length", None)
        or 4096
    )
    task_encoder = Qwen35VLLlavaPretrainEnergonTaskEncoder(
        processor_path=args.hf_processor_path,
        seq_length=seq_length,
        image_token_id=getattr(args, "image_token_id", None),
    )

    worker_config = WorkerConfig(
        rank=parallel_state.get_data_parallel_rank(),
        world_size=parallel_state.get_data_parallel_world_size(),
        num_workers=args.num_workers,
        data_parallel_group=parallel_state.get_data_parallel_group(),
        worker_debug_path=None,
        worker_log_level=0,
    )

    train_ds = get_train_dataset(
        data_path,
        batch_size=args.micro_batch_size,
        task_encoder=task_encoder,
        virtual_epoch_length=max(train_val_test_num_samples[0], 1),
        max_samples_per_sequence=None,
        shuffle_buffer_size=100,
        worker_config=worker_config,
        packing_buffer_size=None,
        handler=_print_error_handler,
        image_decode="pil",
    )

    valid_datasets = None
    if getattr(args, "eval_iters", 0) > 0:
        valid_datasets = get_val_datasets(
            data_path,
            batch_size=args.micro_batch_size,
            task_encoder=task_encoder,
            worker_config=worker_config,
            packing_buffer_size=None,
            handler=_print_error_handler,
            image_decode="pil",
        )
        if not valid_datasets:
            raise ValueError(
                "llava_pretrain_wds requested eval_iters > 0 but Energon "
                "returned no validation datasets. Disable eval or add a val split."
            )
        valid_datasets = [
            LimitDataset(
                RepeatDataset(valid_ds, worker_config=worker_config),
                length=args.eval_iters * get_num_microbatches(),
                worker_config=worker_config,
                reset_after_epoch=True,
            )
            for valid_ds, _src_ds in valid_datasets
        ]

    train_dataloader = get_savable_loader(train_ds, worker_config=worker_config)
    if args.load is not None and getattr(args, "dataloader_save", None):
        dp_rank = parallel_state.get_data_parallel_rank()
        data_save_name = get_checkpoint_name(
            args.dataloader_save,
            args.iteration,
            pipeline_rank=0,
            basename=f"train_dataloader_dprank{dp_rank:03d}.pt",
        )
        if os.path.exists(data_save_name):
            try:
                dataset_state_dict = torch.load(data_save_name, map_location="cpu")
                train_dataloader.restore_state_rank(
                    dataset_state_dict["dataloader_state_dict"]
                )
                print(f"restored dataset state from {data_save_name}")
            except Exception as exc:
                print("loading dataset state failed. Skipping. " + str(exc))
        else:
            print(f"dataset state {data_save_name} does not exist")

    valid_dataloader = None
    if valid_datasets is not None:
        valid_dataloader = [
            EnergonDataloader(get_loader(valid_ds, worker_config=worker_config))
            for valid_ds in valid_datasets
        ]
    return EnergonDataloader(train_dataloader), valid_dataloader, None

def model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    **kwargs,
):
    """Build a multimodal model from ``--model-arch``.

    The language ``TransformerConfig`` is built from CLI args so that
    parallelism settings, precision, and fusion flags are inherited.
    Model-specific post-processing and construction are delegated to the
    registry factory functions.
    """
    args = get_args()
    model_arch = getattr(args, "model_arch", "qwen35_vl")

    from examples.multimodal_dev.models import MODEL_REGISTRY

    if model_arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model arch '{model_arch}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    registry = MODEL_REGISTRY[model_arch]

    # --- language config (generic + model-specific post-processing) ---
    language_config = core_transformer_config_from_args(args)
    post_language_config_fn = registry.get("post_language_config_fn")
    if post_language_config_fn is not None:
        post_language_config_fn(language_config, args)

    # --- vision config ---
    vision_config = registry["vision_config_fn"](
        num_layers_override=getattr(args, "vision_num_layers", None),
        variant=getattr(args, "model_variant", None),
    )
    vision_config.bf16 = language_config.bf16
    vision_config.fp16 = language_config.fp16

    if getattr(args, "recompute_vision", False):
        vision_config.recompute_granularity = "full"
        vision_config.recompute_method = "uniform"
        vision_config.recompute_num_layers = 1

    # --- vision FLOPs metadata ---
    vision_flops_fn = registry.get("vision_flops_fn")
    if vision_flops_fn is not None:
        vision_flops_fn(args, language_config, vision_config)

    # --- build model (fully delegated to the arch factory) ---
    model = registry["model_factory_fn"](
        args=args,
        language_config=language_config,
        vision_config=vision_config,
        **kwargs,
    )

    return model


def _resolve_provider_fn(provider_fn):
    """Resolve a provider that may be a dotted import path string."""
    if isinstance(provider_fn, str):
        module_path, func_name = provider_fn.rsplit(".", 1)
        provider_fn = getattr(
            importlib.import_module(module_path), func_name,
        )
    return provider_fn


def datasets_provider(train_val_test_num_samples):
    """Dataset provider dispatcher.

    Routes to the dataset factory registered for the current
    ``(--model-arch, --dataset-provider)`` combination.
    """
    args = get_args()
    model_arch = getattr(args, "model_arch", "qwen35_vl")
    provider = getattr(args, "dataset_provider", "mock")

    from examples.multimodal_dev.models import MODEL_REGISTRY

    if model_arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model arch '{model_arch}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    registry = MODEL_REGISTRY[model_arch]
    available = dict(registry.get("dataset_providers", {}))
    if model_arch == "qwen35_vl":
        available.setdefault(
            "llava_pretrain_wds",
            qwen35_vl_llava_pretrain_energon_provider,
        )

    if provider not in available:
        raise ValueError(
            f"Unknown dataset provider '{provider}' for arch "
            f"'{model_arch}'. Available: {list(available.keys())}"
        )

    provider_fn = _resolve_provider_fn(available[provider])
    return provider_fn(train_val_test_num_samples)


if __name__ == "__main__":
    datasets_provider.is_distributed = True

    args = parse_and_validate_args(
        extra_args_provider=add_multimodal_args,
        args_defaults={},
    )
    # multimodal_dev's model_provider builds the full model on every rank and
    # does not honor pre_process / post_process pipeline-stage flags. PP>1
    # would silently violate Megatron's pipeline-parallel contract.
    if args.pipeline_model_parallel_size > 1:
        raise ValueError(
            "multimodal_dev does not support pipeline_model_parallel_size > 1 "
            f"(got {args.pipeline_model_parallel_size}). The model provider "
            "builds the full model on every rank; pipeline-stage splitting is "
            "not wired through. Run with --pipeline-model-parallel-size 1."
        )
    full_config = pretrain_cfg_container_from_args(args)
    pretrain(
        full_config,
        datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
    )
