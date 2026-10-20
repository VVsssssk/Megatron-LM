# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import patch

import pytest
import torch

from megatron.core.models.common.embeddings import RotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import CompressedSparseAttention
from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    fused_sparse_attention as fsa,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_csa import (
    _make_csa_submodules,
    _make_mla_config,
)


class _CapturedForward(Exception):
    """Stop after production layout construction, before the unrelated teacher loss."""


def _check_cp_padding_forward_modes():
    """Grad and checkpoint forward use identical valid lengths, including sequence tails."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("The pinned FlashMLA sparse kernels require SM100+")
    Utils.initialize_model_parallel(context_parallel_size=Utils.world_size)
    try:
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        config = _make_mla_config(
            num_layers=1,
            num_attention_heads=64,
            v_head_dim=512,
            csa_compress_ratios=[4],
            csa_window_size=8,
            dsa_indexer_n_heads=64,
            dsa_indexer_head_dim=128,
            dsa_indexer_loss_coeff=0.0,
        )
        config.context_parallel_size = Utils.world_size
        config.dsa_kernel_backend = "cudnn"
        config.deterministic_mode = True
        groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        rotary = RotaryEmbedding(
            config.qk_pos_emb_head_dim,
            rotary_percent=config.rotary_percent,
            rotary_base=config.rotary_base,
            cp_group=groups.cp,
        )
        csa = CompressedSparseAttention(
            config=config,
            submodules=_make_csa_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            pg_collection=groups,
            rotary_pos_emb=rotary,
            compress_ratio=4,
        ).cuda().train()
        local_rows = 256
        segment = local_rows * Utils.world_size // 2
        padded = torch.tensor([0, segment, segment * 2], dtype=torch.int32, device="cuda")
        actual = torch.tensor([0, segment - 1, segment * 2 - 2], dtype=torch.int32, device="cuda")
        packed = PackedSeqParams(
            qkv_format="thd",
            cp_partition_mode="contiguous",
            cu_seqlens_q=actual,
            cu_seqlens_kv=actual,
            cu_seqlens_q_padded=padded,
            cu_seqlens_kv_padded=padded,
            max_seqlen_q=segment,
            max_seqlen_kv=segment,
        )
        torch.manual_seed(31 + groups.cp.rank())
        query = torch.randn(local_rows, 64, 512, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(local_rows, 1, 1, 512, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(local_rows, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        qr = torch.randn(local_rows, 1, config.q_lora_rank, device="cuda", dtype=torch.bfloat16)
        boundary_hidden = torch.zeros(8, 1, config.hidden_size, device="cuda", dtype=x.dtype)
        boundary_kv = torch.zeros(8, 1, 1, 512, device="cuda", dtype=key.dtype)
        captured = []

        def capture(q, kv, indices, scale, **kwargs):
            captured.append(
                (q.detach().clone(), kv.detach().clone(), indices.detach().clone(), scale,
                 {k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                  for k, v in kwargs.items()})
            )
            raise _CapturedForward

        for grad_enabled in (False, True):
            with torch.set_grad_enabled(grad_enabled), patch.object(fsa, "_csa_fwd_flash_mla", capture):
                with pytest.raises(_CapturedForward):
                    csa._forward_thd_cp(query, key, x, qr, boundary_hidden, boundary_kv, packed)
        assert len(captured) == 2
        global_rows = groups.cp.rank() * local_rows + torch.arange(local_rows, device="cuda")
        padding = (global_rows == segment - 1) | (global_rows == segment * 2 - 1)
        outputs = []
        canonical_indices = []
        for q, kv, indices, scale, kwargs in captured:
            lengths = kwargs["topk_length"]
            assert torch.equal(
                lengths[padding], torch.zeros_like(lengths[padding])
            ), "padding forward length must be zero"
            assert torch.all(indices[padding] == -1)
            assert torch.all(lengths[~padding] > 0)
            # Isolate length semantics from the independent Top-K ordering issue.
            sentinel = torch.iinfo(indices.dtype).max
            ordered = torch.where(indices >= 0, indices, sentinel).sort(dim=-1).values
            ordered.masked_fill_(ordered == sentinel, -1)
            canonical_indices.append(ordered)
            with torch.no_grad():
                outputs.append(fsa._csa_fwd_flash_mla(q, kv, ordered, scale, **kwargs)[0])
        torch.testing.assert_close(captured[0][0], captured[1][0], rtol=0, atol=0)
        torch.testing.assert_close(captured[0][1], captured[1][1], rtol=0, atol=0)
        assert torch.equal(captured[0][4]["topk_length"], captured[1][4]["topk_length"])
        assert torch.equal(canonical_indices[0], canonical_indices[1])
        for mask in (padding, ~padding):
            torch.testing.assert_close(outputs[0][mask], outputs[1][mask], rtol=0, atol=0)
        assert all(torch.isfinite(output).all() for output in outputs)
        return {"rank": groups.cp.rank(), "local_rows": local_rows,
                "padding_rows": int(padding.sum()), "lengths_equal": True,
                "valid_output_equal": True, "padding_output_equal": True}
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.launch_on_gb200
def test_cp_padding_forward_modes():
    _check_cp_padding_forward_modes()
