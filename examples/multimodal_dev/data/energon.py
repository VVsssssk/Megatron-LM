# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Energon dataloader provider for Qwen3.5-VL multimodal_dev training.

The provider reads Megatron-Energon WebDataset directories that expose
``VQASample`` records, such as the LLaVA-Pretrain conversion documented in
``examples/multimodal/README.md``. The task encoder converts each sample
to the same per-sample dict used by the native ``cord_v2`` provider:

``input_ids``, ``labels``, ``loss_mask``, ``pixel_values``,
``image_grid_thw``.

THD training packing is intentionally left to
``examples.multimodal_dev.forward_step.pack_or_pad_batch``. Energon may
still pre-combine short samples when ``--packing-buffer-size`` is set, but
the final Megatron ``PackedSeqParams`` are built in the forward step so the
model sees one consistent qwen3.5 VL data contract.
"""

import logging
import math
import os
import re
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
)
from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
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
from megatron.training import get_args
from megatron.training.checkpointing import get_checkpoint_name

try:
    from qwen_vl_utils import process_vision_info

    HAVE_QWEN_VL_UTILS = True
except ImportError:
    HAVE_QWEN_VL_UTILS = False

logger = logging.getLogger(__name__)

_IMAGE_PLACEHOLDER_RE = re.compile(r"<image(?:-\d+)?>")

# Qwen-VL recommended pixel-budget range. This matches cord_v2.py.
_QWEN_VL_MIN_PIXELS = 256 * 28 * 28
_QWEN_VL_MAX_PIXELS = 1280 * 28 * 28


def _print_error_handler(exc: Exception, key: Optional[str]):
    print(
        f"The following exception occurred in the dataloader for sample {key} and is skipped",
        flush=True,
    )
    traceback.print_exception(type(exc), exc, exc.__traceback__)


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _clean_user_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = _IMAGE_PLACEHOLDER_RE.sub("", text)
    return text.strip()


def _normalise_answer(answer: Any) -> str:
    if isinstance(answer, (list, tuple)):
        return "\n".join(str(x) for x in answer)
    return "" if answer is None else str(answer)


def _right_pad(tensor: torch.Tensor, target_len: int, value: int) -> torch.Tensor:
    pad = target_len - tensor.shape[0]
    if pad <= 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pad), value=value)


def _greedy_pack(samples: List[Dict[str, torch.Tensor]], max_length: int):
    """Greedily group encoded samples without exceeding ``max_length``."""
    groups: List[List[Dict[str, torch.Tensor]]] = []
    current: List[Dict[str, torch.Tensor]] = []
    current_len = 0
    for sample in sorted(
        samples, key=lambda sample: int(sample["input_ids"].numel()), reverse=True
    ):
        sample_len = int(sample["input_ids"].numel())
        if sample_len > max_length:
            raise ValueError(
                f"Sample length {sample_len} exceeds packing sequence length {max_length}."
            )
        if current and current_len + sample_len > max_length:
            groups.append(current)
            current = []
            current_len = 0
        current.append(sample)
        current_len += sample_len
    if current:
        groups.append(current)
    return groups


class Qwen35VLEnergonTaskEncoder(
    DefaultTaskEncoder[VQASample, Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]], Any]
):
    """Convert Energon ``VQASample`` records into qwen3.5 VL sample dicts."""

    def __init__(self, processor):
        super().__init__()
        args = get_args()
        self.processor = processor
        self.seq_length = (
            getattr(args, "total_seq_length", None) or getattr(args, "seq_length", 8192)
        )
        self.image_token_id = int(
            getattr(args, "image_token_id", QWEN35_VL_IMAGE_TOKEN_ID)
        )
        self.packing_seq_length = (
            getattr(args, "packing_seq_length", None)
            or getattr(args, "total_seq_length", None)
            or getattr(args, "seq_length", 8192)
        )
        self.packing_pad_to_multiple = int(
            getattr(args, "packing_pad_to_multiple", 1) or 1
        )
        self.min_pixels = int(
            getattr(args, "qwen_vl_min_pixels", None) or _QWEN_VL_MIN_PIXELS
        )
        self.max_pixels = int(
            getattr(args, "qwen_vl_max_pixels", None) or _QWEN_VL_MAX_PIXELS
        )

        tok = processor.tokenizer
        if tok.pad_token_id is not None:
            self.pad_token_id = int(tok.pad_token_id)
        elif tok.eos_token_id is not None:
            self.pad_token_id = int(tok.eos_token_id)
        else:
            raise ValueError(
                "Tokenizer has neither pad_token_id nor eos_token_id; cannot mask padding safely."
            )

        skipped = set(int(token_id) for token_id in (tok.all_special_ids or []))
        skipped.add(self.pad_token_id)
        skipped.add(self.image_token_id)
        self.skipped_token_ids = torch.tensor(sorted(skipped), dtype=torch.long)

    def _build_conversation(self, sample: VQASample):
        contexts = _ensure_list(getattr(sample, "context", None))
        answers = [
            _normalise_answer(answer)
            for answer in _ensure_list(getattr(sample, "answers", None))
        ]
        image = getattr(sample, "image", None)

        if not contexts:
            contexts = [""]
        if not answers:
            answers = [""]

        turns = []
        for idx, user_text in enumerate(contexts):
            user_content = []
            if idx == 0 and image is not None:
                user_content.append({"type": "image", "image": image})
            text = _clean_user_text(user_text)
            if text:
                user_content.append({"type": "text", "text": text})
            elif not user_content:
                user_content.append({"type": "text", "text": ""})
            turns.append({"role": "user", "content": user_content})

            if idx < len(answers):
                turns.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": answers[idx]}],
                    }
                )

        if len(answers) > len(contexts):
            for answer in answers[len(contexts) :]:
                turns.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": answer}],
                    }
                )

        return turns, [answer for answer in answers if answer]

    def _processor_inputs(self, conversation):
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        if HAVE_QWEN_VL_UTILS:
            images, videos = process_vision_info(conversation)
        else:
            images = [
                content["image"]
                for turn in conversation
                for content in turn.get("content", [])
                if isinstance(content, dict) and content.get("type") == "image"
            ]
            videos = None

        kwargs = {
            "text": [text],
            "images": images if images else None,
            "return_tensors": "pt",
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }
        if videos:
            kwargs["videos"] = videos
        return self.processor(**kwargs)

    def _find_answer_span(
        self,
        input_ids: Sequence[int],
        answer_text: str,
        start_idx: int,
    ):
        tokenizer = self.processor.tokenizer
        variants = (
            answer_text,
            answer_text + "\n",
            answer_text.strip(),
            answer_text.strip() + "\n",
        )
        n = len(input_ids)
        for variant in variants:
            if not variant:
                continue
            answer_ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
            m = len(answer_ids)
            if m == 0 or m > n:
                continue
            for start in range(start_idx, n - m + 1):
                if list(input_ids[start : start + m]) == answer_ids:
                    return start, start + m
        return -1, -1

    def _build_loss_mask(
        self,
        input_ids: torch.Tensor,
        answers: Sequence[str],
        sample_key: str,
    ) -> torch.Tensor:
        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        search_idx = 0
        ids = input_ids.tolist()
        for answer in answers:
            start, end = self._find_answer_span(ids, answer, search_idx)
            if start == -1:
                logger.warning(
                    "Assistant span not located for sample %s; answer will be masked out.",
                    sample_key,
                )
                continue
            loss_mask[start:end] = 1.0
            search_idx = end
        return loss_mask

    @stateless(restore_seeds=True)
    def encode_sample(self, sample: VQASample):
        conversation, answers = self._build_conversation(sample)
        batch = self._processor_inputs(conversation)

        input_ids = batch["input_ids"].squeeze(0)
        pixel_values = batch["pixel_values"].to(torch.bfloat16)
        image_grid_thw = batch["image_grid_thw"]

        if input_ids.shape[0] > self.seq_length:
            logger.warning(
                "Sample %s has %d tokens > seq_length=%d; truncating.",
                getattr(sample, "__key__", "<unknown>"),
                input_ids.shape[0],
                self.seq_length,
            )
            input_ids = input_ids[: self.seq_length]

        image_token_count = int((input_ids == self.image_token_id).sum().item())
        if image_token_count == 0:
            raise ValueError(
                f"Sample {getattr(sample, '__key__', '<unknown>')} has no qwen image tokens "
                "after preprocessing/truncation."
            )

        loss_mask = self._build_loss_mask(
            input_ids,
            answers,
            getattr(sample, "__key__", "<unknown>"),
        )

        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        labels[torch.isin(labels, self.skipped_token_ids)] = -100

        loss_mask = torch.cat(
            [loss_mask[1:], torch.zeros(1, dtype=loss_mask.dtype)]
        )
        labels[loss_mask == 0] = -100

        return {
            "input_ids": input_ids.long(),
            "labels": labels.long(),
            "loss_mask": loss_mask.float(),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw.int(),
        }

    def select_samples_to_pack(self, samples: List[Dict[str, torch.Tensor]]):
        return _greedy_pack(samples, int(self.packing_seq_length))

    @stateless
    def pack_selected_samples(self, samples: List[Dict[str, torch.Tensor]]):
        if not samples:
            raise ValueError("Cannot pack an empty sample group.")

        input_ids_list, labels_list, loss_mask_list = [], [], []
        for sample in samples:
            seqlen = int(sample["input_ids"].numel())
            target_len = (
                math.ceil(seqlen / self.packing_pad_to_multiple)
                * self.packing_pad_to_multiple
            )
            input_ids_list.append(
                _right_pad(sample["input_ids"], target_len, self.pad_token_id)
            )
            labels_list.append(_right_pad(sample["labels"], target_len, -100))
            loss_mask_list.append(_right_pad(sample["loss_mask"], target_len, 0))

        return {
            "input_ids": torch.cat(input_ids_list, dim=0),
            "labels": torch.cat(labels_list, dim=0),
            "loss_mask": torch.cat(loss_mask_list, dim=0),
            "pixel_values": torch.cat([sample["pixel_values"] for sample in samples], dim=0),
            "image_grid_thw": torch.cat([sample["image_grid_thw"] for sample in samples], dim=0),
        }

    def batch(self, samples: List[Dict[str, torch.Tensor]]):
        # Return a list of variable-length per-sample dicts. The qwen3.5 VL
        # forward step owns final BSHD padding or THD PackedSeqParams creation.
        return samples

    def encode_batch(self, batch):
        return batch


def _dataset_path(args) -> str:
    data_path = getattr(args, "data_path", None)
    if isinstance(data_path, (list, tuple)):
        data_path = data_path[0] if data_path else None
    if not data_path:
        raise ValueError("energon dataset provider requires --data-path <energon_wds_dir>")
    return data_path


def _split_part_has_shards(dataset_path: str, split_part: str):
    split_file = os.path.join(os.fspath(dataset_path), ".nv-meta", "split.yaml")
    if not os.path.exists(split_file):
        return None

    try:
        import yaml

        with open(split_file, "r", encoding="utf-8") as handle:
            split_config = yaml.safe_load(handle) or {}
    except Exception as exc:
        logger.warning(
            "Could not read Energon split metadata from %s: %s. Assuming split %s exists.",
            split_file,
            exc,
            split_part,
        )
        return None

    split_parts = split_config.get("split_parts") or {}
    return bool(split_parts.get(split_part))


def _should_build_validation_dataloaders(args, dataset_path: str) -> bool:
    eval_iters = int(getattr(args, "eval_iters", 0) or 0)
    needs_validation = bool(getattr(args, "full_validation", False) or eval_iters > 0)
    if not needs_validation:
        return False

    has_val_split = _split_part_has_shards(dataset_path, "val")
    if has_val_split is False:
        if getattr(args, "full_validation", False):
            raise ValueError(
                "Energon validation split 'val' is empty, but --full-validation was requested."
            )
        if eval_iters > 0:
            print(
                "Energon validation split 'val' is empty; setting eval_iters=0 and "
                "skipping validation dataloaders.",
                flush=True,
            )
            args.eval_iters = 0
        return False

    return True


def _is_first_or_last_stage(pp_size: int) -> bool:
    if pp_size == 1:
        return True
    pp_rank = get_pipeline_model_parallel_rank()
    return pp_rank in (0, pp_size - 1)


def _is_dataloader_rank() -> bool:
    is_first_tp_rank = get_tensor_model_parallel_rank() == 0
    pp_size = get_pipeline_model_parallel_world_size()
    return is_first_tp_rank and _is_first_or_last_stage(pp_size)


class EnergonDataloader:
    """Adapter that makes an Energon loader look like a cyclic Megatron loader."""

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._iter = iter(_cyclic_iter(dataloader)) if dataloader is not None else None

    def __next__(self):
        if self._iter is None:
            raise StopIteration
        return next(self._iter)

    def __iter__(self):
        return self

    def save_state(self):
        if self._dataloader is None:
            return {}
        return self._dataloader.save_state_rank()


def _cyclic_iter(iterator: Iterable):
    while True:
        for item in iterator:
            yield item


def _restore_loader_state_if_available(train_dataloader):
    args = get_args()
    if args.load is None or not getattr(args, "dataloader_save", None):
        return

    dp_rank = parallel_state.get_data_parallel_rank()
    data_save_name = get_checkpoint_name(
        args.dataloader_save,
        args.iteration,
        pipeline_rank=0,
        basename=f"train_dataloader_dprank{dp_rank:03d}.pt",
    )
    if not os.path.exists(data_save_name):
        print(f"dataset state {data_save_name} does not exist")
        return

    try:
        dataset_state_dict = torch.load(data_save_name, map_location="cpu")
        train_dataloader.restore_state_rank(dataset_state_dict["dataloader_state_dict"])
        print(f"restored dataset state from {data_save_name}")
    except Exception as exc:
        print("loading dataset state failed. Skipping. " + str(exc))


def train_valid_test_dataloaders_provider(train_val_test_num_samples):
    """Build Qwen3.5-VL Energon train/validation dataloaders."""
    from transformers import AutoProcessor

    args = get_args()
    if not _is_dataloader_rank():
        return None, None, None

    processor_path = getattr(args, "hf_processor_path", None) or getattr(
        args, "tokenizer_model", None
    )
    if processor_path is None:
        raise ValueError(
            "energon dataset provider requires --hf-processor-path or --tokenizer-model"
        )
    processor = AutoProcessor.from_pretrained(
        processor_path, trust_remote_code=True,
    )
    task_encoder = Qwen35VLEnergonTaskEncoder(processor)

    rank = parallel_state.get_data_parallel_rank()
    world_size = parallel_state.get_data_parallel_world_size()
    worker_config = WorkerConfig(
        rank=rank,
        world_size=world_size,
        num_workers=args.num_workers,
        data_parallel_group=parallel_state.get_data_parallel_group(),
        worker_debug_path=getattr(args, "energon_worker_debug_path", None),
        worker_log_level=getattr(args, "energon_worker_log_level", 0),
    )

    dname = _dataset_path(args)
    packing_buffer_size = getattr(args, "packing_buffer_size", None)
    train_dataset = get_train_dataset(
        dname,
        batch_size=args.micro_batch_size,
        task_encoder=task_encoder,
        virtual_epoch_length=getattr(args, "energon_virtual_epoch_length", 1000),
        max_samples_per_sequence=getattr(args, "energon_max_samples_per_sequence", 100),
        shuffle_buffer_size=getattr(args, "energon_shuffle_buffer_size", 100),
        worker_config=worker_config,
        packing_buffer_size=packing_buffer_size,
        handler=_print_error_handler,
        image_decode="pil",
    )

    valid_dataloaders = None
    if _should_build_validation_dataloaders(args, dname):
        val_datasets = get_val_datasets(
            dname,
            batch_size=args.micro_batch_size,
            task_encoder=task_encoder,
            worker_config=worker_config,
            packing_buffer_size=packing_buffer_size,
            handler=_print_error_handler,
            image_decode="pil",
        )
        val_datasets = [
            LimitDataset(
                RepeatDataset(val_ds, worker_config=worker_config),
                length=args.eval_iters * get_num_microbatches(),
                worker_config=worker_config,
                reset_after_epoch=True,
            )
            for val_ds, _src_ds in val_datasets
        ]
        valid_dataloaders = [
            EnergonDataloader(get_loader(val_ds, worker_config=worker_config))
            for val_ds in val_datasets
        ]

    train_dataloader = get_savable_loader(train_dataset, worker_config=worker_config)
    _restore_loader_state_if_available(train_dataloader)

    return EnergonDataloader(train_dataloader), valid_dataloaders, None


train_valid_test_dataloaders_provider.is_distributed = True
