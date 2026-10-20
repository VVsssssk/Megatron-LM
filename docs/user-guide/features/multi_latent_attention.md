<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Multi-Latent Attention

## Multi-Latent Attention Overview

Multi-Latent Attention (MLA) is an attention variant from the DeepSeek team. It uses multiple latent spaces to change how attention is computed. That design often lowers cost for large language models (LLMs) compared with standard attention and can shrink the KV cache. The DeepSeek-V2 technical report compares MLA to Multi-Head Attention (MHA) on quality and cache size.

## Enabling Multi-Latent Attention

To enable MLA in Megatron-LM, set the following on the command line:

- `--multi-latent-attention` to turn on MLA.
- Use `MLATransformerConfig` for MLA-specific model settings when you build the training configuration.

## DSv4 CSA packed context-parallel padding

For selected compressed attention in the DSv4 CSA context-parallel THD path,
provide both the true and padded cumulative sequence lengths when they differ.
True query lengths are used in both grad-enabled and checkpoint no-grad
forwards, independently of whether the indexer auxiliary loss is computed.
Padding queries use sink-only attention: all sparse indices are invalid and
the forward `topk_length` is zero. The fused indexer-loss function may subsequently
replace this with a safe backward placeholder while masking the padded gradients.
That internal placeholder must not change the initial forward's valid key range.

Matching length metadata does not by itself guarantee bitwise equality when
Top-K ordering or low-precision expert arithmetic differs.
