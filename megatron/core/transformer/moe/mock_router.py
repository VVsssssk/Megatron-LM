# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Deterministic MoE route mocks for benchmark load-shape experiments."""

import hashlib
import math
from dataclasses import dataclass
from typing import Optional

import torch


_HASH_MOD = 2**63 - 1


@dataclass(frozen=True)
class MockRouteStats:
    """Rank- and expert-level summary for a generated mock route."""

    tokens_per_expert: torch.Tensor
    tokens_per_ep_rank: torch.Tensor
    actual_rank_maxvio: float
    actual_concentration: float
    duplicate_topk_rate: float


@dataclass(frozen=True)
class MockRouteOutput:
    """Dense router-compatible mock route plus debug metadata."""

    topk_indices: torch.Tensor
    topk_probs: torch.Tensor
    routing_probs: torch.Tensor
    routing_map: torch.Tensor
    stats: MockRouteStats
    effective_maxvio: float
    effective_concentration: float
    hot_ep_ranks: torch.Tensor


def stable_hash64(*values) -> int:
    """Return a deterministic 63-bit seed from stable Python values."""

    hasher = hashlib.blake2b(digest_size=8)
    for value in values:
        encoded = str(value).encode("utf-8")
        hasher.update(len(encoded).to_bytes(4, "little", signed=False))
        hasher.update(encoded)
    return int.from_bytes(hasher.digest(), "little", signed=False) % _HASH_MOD


def make_mock_generator(device: torch.device, seed: int) -> torch.Generator:
    """Create an independent torch generator on ``device`` seeded with ``seed``."""

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) % _HASH_MOD)
    return gen


def _clamp_float(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _ep_group_id(ep_group: Optional[torch.distributed.ProcessGroup]) -> int:
    if (
        ep_group is None
        or not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
    ):
        return 0
    try:
        ranks = torch.distributed.get_process_group_ranks(ep_group)
    except Exception:
        return 0
    return min(int(rank) for rank in ranks) if ranks else 0


def _ep_rank(ep_group: Optional[torch.distributed.ProcessGroup]) -> int:
    if (
        ep_group is None
        or not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
    ):
        return 0
    try:
        return int(ep_group.rank())
    except Exception:
        try:
            return int(torch.distributed.get_rank(ep_group))
        except Exception:
            return 0


def _resolve_base_seed(config) -> int:
    """Resolve Megatron's training seed lazily without making core depend on training."""

    try:
        from megatron.training.global_vars import get_args

        args = get_args()
        return int(getattr(args, "seed"))
    except Exception:
        return int(getattr(config, "moe_router_mock_base_seed", 1234))


def _resolve_mock_step(config) -> int:
    try:
        from megatron.training.global_vars import get_args

        args = get_args()
        return int(getattr(args, "curr_iteration", getattr(args, "iteration", 0)))
    except Exception:
        return int(getattr(config, "moe_router_mock_global_step", 0))


def _resolve_mock_microbatch(config) -> int:
    try:
        from megatron.training.global_vars import get_args

        args = get_args()
        return int(getattr(args, "microbatch_id", getattr(args, "microbatch_idx", 0)))
    except Exception:
        return int(getattr(config, "moe_router_mock_microbatch_id", 0))


def _compute_integer_counts(
    probs: torch.Tensor, total: int, gen: torch.Generator
) -> torch.Tensor:
    """Convert probabilities to integer counts that sum exactly to ``total``."""

    if total == 0:
        return torch.zeros_like(probs, dtype=torch.long)
    expected = probs * float(total)
    counts = torch.floor(expected).to(torch.long)
    remainder = int(total - counts.sum().item())
    if remainder > 0:
        frac = expected - counts.to(expected.dtype)
        tie_break = torch.rand(frac.shape, generator=gen, device=frac.device, dtype=frac.dtype)
        _, indices = torch.topk(frac + tie_break * 1.0e-6, k=remainder)
        counts[indices] += 1
    elif remainder < 0:
        _, indices = torch.topk(counts.to(torch.float32), k=-remainder)
        counts[indices] -= 1
    return counts


def _duplicate_topk_rate(topk_indices: torch.Tensor) -> float:
    if topk_indices.numel() == 0 or topk_indices.shape[-1] <= 1:
        return 0.0
    sorted_indices = torch.sort(topk_indices, dim=-1).values
    duplicate_rows = (sorted_indices[..., 1:] == sorted_indices[..., :-1]).any(dim=-1)
    return float(duplicate_rows.float().mean().item())


def compute_mock_route_stats(
    topk_indices: torch.Tensor, num_experts: int, num_ep_rank: int
) -> MockRouteStats:
    """Compute debug stats for sparse top-k expert indices.

    ``actual_rank_maxvio`` is EP-rank-level ``max_e(T_e / T_avg)`` and is
    therefore bounded below by 1.0 for any non-empty route.
    """

    if num_ep_rank < 1:
        raise ValueError(f"num_ep_rank must be >= 1, got {num_ep_rank}.")
    if num_experts % num_ep_rank != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by num_ep_rank ({num_ep_rank})."
        )

    flat = topk_indices.reshape(-1).to(torch.long)
    tokens_per_expert = torch.bincount(flat.cpu(), minlength=num_experts).to(topk_indices.device)
    experts_per_rank = num_experts // num_ep_rank
    tokens_per_ep_rank = tokens_per_expert.reshape(num_ep_rank, experts_per_rank).sum(dim=1)

    total = float(tokens_per_ep_rank.sum().item())
    if total == 0.0:
        actual_rank_maxvio = 1.0
        actual_concentration = 0.0
    else:
        avg = total / float(num_ep_rank)
        actual_rank_maxvio = float(tokens_per_ep_rank.max().item()) / avg
        shares = tokens_per_ep_rank.to(torch.float32) / total
        hhi = float((shares * shares).sum().item())
        uniform_hhi = 1.0 / float(num_ep_rank)
        if num_ep_rank == 1:
            actual_concentration = 1.0
        else:
            actual_concentration = _clamp_float(
                (hhi - uniform_hhi) / (1.0 - uniform_hhi), 0.0, 1.0
            )

    return MockRouteStats(
        tokens_per_expert=tokens_per_expert,
        tokens_per_ep_rank=tokens_per_ep_rank,
        actual_rank_maxvio=actual_rank_maxvio,
        actual_concentration=actual_concentration,
        duplicate_topk_rate=_duplicate_topk_rate(topk_indices),
    )


class ImbalanceRouteMockGenerator:
    """Generate deterministic rank-level imbalanced MoE routes.

    The generator is stateless. Temporal behavior is derived from
    ``global_step``/``microbatch_id`` rather than an internal counter, so repeated
    forwards from activation recomputation do not advance hidden state.
    """

    def __init__(
        self,
        config,
        num_experts: int,
        num_ep_rank: int,
        topk: int,
        ep_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        if num_ep_rank < 1:
            raise ValueError(f"num_ep_rank must be >= 1, got {num_ep_rank}.")
        if num_experts % num_ep_rank != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by num_ep_rank ({num_ep_rank})."
            )
        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}.")
        self.config = config
        self.num_experts = int(num_experts)
        self.num_ep_rank = int(num_ep_rank)
        self.topk = int(topk)
        self.ep_group = ep_group
        self.ep_group_id = _ep_group_id(ep_group)
        self.local_ep_rank = _ep_rank(ep_group)
        self.experts_per_rank = self.num_experts // self.num_ep_rank

    def generate(
        self,
        num_tokens: int,
        global_step: Optional[int] = None,
        microbatch_id: Optional[int] = None,
        layer_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> MockRouteOutput:
        """Generate router-compatible tensors for the requested local token count."""

        if num_tokens < 0:
            raise ValueError(f"num_tokens must be >= 0, got {num_tokens}.")
        device = torch.device("cpu") if device is None else torch.device(device)
        global_step = _resolve_mock_step(self.config) if global_step is None else int(global_step)
        microbatch_id = (
            _resolve_mock_microbatch(self.config) if microbatch_id is None else int(microbatch_id)
        )
        layer_idx = 0 if layer_idx is None else int(layer_idx)
        base_seed = _resolve_base_seed(self.config)

        pattern = self._generate_pattern(base_seed, global_step, microbatch_id, layer_idx, device)
        local_seed = stable_hash64(
            base_seed,
            global_step,
            microbatch_id,
            layer_idx,
            self.ep_group_id,
            self.local_ep_rank,
            "moe_route_mock_local",
        )
        local_gen = make_mock_generator(device, local_seed)

        total_local_assignments = int(num_tokens) * self.topk
        counts = _compute_integer_counts(pattern["rank_probs"], total_local_assignments, local_gen)
        ranks = torch.arange(self.num_ep_rank, device=device, dtype=torch.long)
        flat_target_ranks = torch.repeat_interleave(ranks, counts)
        if flat_target_ranks.numel() > 0:
            perm = torch.randperm(flat_target_ranks.numel(), generator=local_gen, device=device)
            flat_target_ranks = flat_target_ranks[perm]
        target_ranks = flat_target_ranks.reshape(num_tokens, self.topk)

        offsets = torch.randint(
            self.experts_per_rank,
            (num_tokens, self.topk),
            generator=local_gen,
            device=device,
            dtype=torch.long,
        )
        topk_indices = target_ranks * self.experts_per_rank + offsets
        topk_indices = self._repair_duplicate_topk(topk_indices)
        topk_probs = torch.full(
            (num_tokens, self.topk), 1.0 / float(self.topk), device=device, dtype=dtype
        )

        routing_probs = torch.zeros((num_tokens, self.num_experts), device=device, dtype=dtype)
        routing_probs.scatter_(1, topk_indices, topk_probs)
        routing_map = torch.zeros((num_tokens, self.num_experts), device=device, dtype=torch.bool)
        routing_map.scatter_(1, topk_indices, True)

        stats = compute_mock_route_stats(topk_indices, self.num_experts, self.num_ep_rank)
        return MockRouteOutput(
            topk_indices=topk_indices,
            topk_probs=topk_probs,
            routing_probs=routing_probs,
            routing_map=routing_map,
            stats=stats,
            effective_maxvio=pattern["effective_maxvio"],
            effective_concentration=pattern["effective_concentration"],
            hot_ep_ranks=pattern["hot_ep_ranks"],
        )

    def _generate_pattern(
        self,
        base_seed: int,
        global_step: int,
        microbatch_id: int,
        layer_idx: int,
        device: torch.device,
    ) -> dict:
        pattern_seed = stable_hash64(
            base_seed,
            global_step,
            microbatch_id,
            layer_idx,
            self.ep_group_id,
            "moe_route_mock_pattern",
        )
        persistent_seed = stable_hash64(
            base_seed, layer_idx, self.ep_group_id, "moe_route_mock_persistent"
        )
        pattern_gen = make_mock_generator(device, pattern_seed)
        persistent_gen = make_mock_generator(device, persistent_seed)

        base_maxvio = float(getattr(self.config, "moe_router_mock_maxvio", 1.0))
        maxvio_jitter = float(getattr(self.config, "moe_router_mock_maxvio_jitter", 0.0))
        effective_maxvio = base_maxvio
        if maxvio_jitter > 0.0:
            jitter = torch.rand((), generator=pattern_gen, device=device).item() * 2.0 - 1.0
            effective_maxvio += jitter * maxvio_jitter

        period = int(getattr(self.config, "moe_router_mock_pattern_period", 0) or 0)
        call_idx = global_step * 1_000_003 + microbatch_id
        if period > 0 and maxvio_jitter > 0.0 and base_maxvio != 0.0:
            phase = 2.0 * math.pi * float(call_idx % period) / float(period)
            effective_maxvio *= 1.0 + (maxvio_jitter / abs(base_maxvio)) * math.sin(phase)
        effective_maxvio = _clamp_float(effective_maxvio, 1.0, float(self.num_ep_rank))

        base_concentration = float(getattr(self.config, "moe_router_mock_concentration", 1.0))
        concentration_jitter = float(
            getattr(self.config, "moe_router_mock_concentration_jitter", 0.0)
        )
        effective_concentration = base_concentration
        if concentration_jitter > 0.0:
            jitter = torch.rand((), generator=pattern_gen, device=device).item() * 2.0 - 1.0
            effective_concentration += jitter * concentration_jitter
        effective_concentration = _clamp_float(effective_concentration, 0.0, 1.0)

        consistency = _clamp_float(
            float(getattr(self.config, "moe_router_mock_consistency", 1.0)), 0.0, 1.0
        )
        persistent_scores = torch.rand(
            self.num_ep_rank, generator=persistent_gen, device=device, dtype=torch.float32
        )
        noise_scores = torch.rand(
            self.num_ep_rank, generator=pattern_gen, device=device, dtype=torch.float32
        )
        scores = consistency * persistent_scores + (1.0 - consistency) * noise_scores
        if period > 0:
            ranks = torch.arange(self.num_ep_rank, device=device, dtype=torch.float32)
            phase = 2.0 * math.pi * float(call_idx % period) / float(period)
            periodic_scores = torch.sin(phase + 2.0 * math.pi * ranks / float(self.num_ep_rank))
            scores = scores + 0.25 * periodic_scores

        num_hot = int(round(1.0 + (1.0 - effective_concentration) * (self.num_ep_rank - 1)))
        num_hot = max(1, min(self.num_ep_rank, num_hot))
        _, hot_ep_ranks = torch.topk(scores, k=num_hot, sorted=True)
        rank_probs = self._rank_probs(hot_ep_ranks, effective_maxvio, effective_concentration)

        return {
            "effective_maxvio": effective_maxvio,
            "effective_concentration": effective_concentration,
            "hot_ep_ranks": hot_ep_ranks,
            "rank_probs": rank_probs,
        }

    def _rank_probs(
        self, hot_ep_ranks: torch.Tensor, effective_maxvio: float, effective_concentration: float
    ) -> torch.Tensor:
        if self.num_ep_rank == 1:
            return torch.ones(1, device=hot_ep_ranks.device, dtype=torch.float32)

        uniform = torch.full(
            (self.num_ep_rank,),
            1.0 / float(self.num_ep_rank),
            device=hot_ep_ranks.device,
            dtype=torch.float32,
        )
        peak_prob = _clamp_float(
            effective_maxvio / float(self.num_ep_rank),
            1.0 / float(self.num_ep_rank),
            1.0,
        )

        raw = torch.zeros_like(uniform)
        primary = hot_ep_ranks[0]
        raw[primary] = 1.0
        if hot_ep_ranks.numel() > 1:
            tail = hot_ep_ranks[1:]
            tail_weight = max(1.0e-6, 1.0 - effective_concentration)
            raw[tail] = tail_weight

        tail_sum = float((raw.sum() - raw[primary]).item())
        if tail_sum > 0.0 and peak_prob < 1.0:
            raw[primary] = max(
                float(raw[primary].item()), peak_prob * tail_sum / (1.0 - peak_prob)
            )

        if raw.sum().item() == 0.0:
            shaped = uniform
        else:
            shaped = raw / raw.sum()

        shaped_max = float(shaped.max().item())
        uniform_prob = 1.0 / float(self.num_ep_rank)
        if shaped_max <= uniform_prob + 1.0e-12:
            return uniform

        alpha = (peak_prob - uniform_prob) / (shaped_max - uniform_prob)
        alpha = _clamp_float(alpha, 0.0, 1.0)
        probs = (1.0 - alpha) * uniform + alpha * shaped
        probs = probs / probs.sum()
        return probs

    def _repair_duplicate_topk(self, topk_indices: torch.Tensor) -> torch.Tensor:
        if self.topk <= 1 or self.experts_per_rank <= 1 or topk_indices.numel() == 0:
            return topk_indices

        repaired = topk_indices.clone()
        for kth in range(1, self.topk):
            for _ in range(self.experts_per_rank - 1):
                duplicate = (repaired[:, :kth] == repaired[:, kth : kth + 1]).any(dim=1)
                if not bool(duplicate.any().item()):
                    break
                rank_base = (
                    repaired[duplicate, kth] // self.experts_per_rank
                ) * self.experts_per_rank
                local_offset = (repaired[duplicate, kth] - rank_base + 1) % self.experts_per_rank
                repaired[duplicate, kth] = rank_base + local_offset
        return repaired


def generate_imbalance_route_mock(
    config,
    num_tokens: int,
    num_experts: int,
    num_ep_rank: int,
    topk: int,
    ep_group: Optional[torch.distributed.ProcessGroup],
    device: torch.device,
    dtype: torch.dtype,
    global_step: Optional[int] = None,
    microbatch_id: Optional[int] = None,
    layer_idx: Optional[int] = None,
) -> MockRouteOutput:
    """Convenience wrapper for callers that do not need to cache the generator."""

    generator = ImbalanceRouteMockGenerator(config, num_experts, num_ep_rank, topk, ep_group)
    return generator.generate(
        num_tokens=num_tokens,
        global_step=global_step,
        microbatch_id=microbatch_id,
        layer_idx=layer_idx,
        device=device,
        dtype=dtype,
    )
