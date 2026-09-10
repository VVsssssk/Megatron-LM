# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Transport contracts for runtime expert replica weights and gradients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True, slots=True)
class ReplicaTransportConfig:
    """Fixed storage and launch requirements shared by replica transports."""

    group: dist.ProcessGroup
    device: torch.device
    world_size: int
    num_local_home_experts: int
    num_local_replica_slots: int
    member_shapes: tuple[tuple[int, int], tuple[int, int]]
    weight_format: str
    rowwise_scale_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None
    columnwise_scale_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None
    grad_dtype: torch.dtype
    num_sms: int | None


@dataclass(frozen=True, slots=True)
class ReplicaWeightSource:
    """One projection's local source tensors plus an optional backend fast path."""

    data: tuple[torch.Tensor, ...]
    scales: tuple[torch.Tensor, ...] | None
    data_bases: torch.Tensor | None = None
    scale_bases: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class ReplicaGradDestination:
    """One projection's native gradient destinations and optional pointer table."""

    tensors: tuple[torch.Tensor, ...]
    bases: torch.Tensor | None = None


class ReplicaWeightTransport(ABC):
    """Move opaque expert storage without owning TE or optimizer semantics."""

    transport_name = "abstract"

    @property
    @abstractmethod
    def grad_dtype(self) -> torch.dtype:
        """Return the dtype of transport-owned replica gradient storage."""

    @abstractmethod
    def projection_views(self, projection_index: int) -> tuple[tuple[Any, ...], torch.Tensor]:
        """Return local replica weight components and gradient storage for a projection."""

    @abstractmethod
    def native_projection_grad_view(self, projection_index: int) -> torch.Tensor:
        """Return local staging for gradients reduced into native experts."""

    @abstractmethod
    def start_weight_sync(
        self,
        *,
        sources: tuple[ReplicaWeightSource, ...],
        experts_to_copy: torch.Tensor,
    ) -> Any:
        """Start an asynchronous owner-to-replica weight transfer."""

    @abstractmethod
    def wait_weight_sync(self, handle: Any) -> None:
        """Order the current stream after a weight-transfer handle."""

    @abstractmethod
    def start_grad_reduce(
        self,
        *,
        native_grads: tuple[ReplicaGradDestination, ...],
        experts_to_copy: torch.Tensor,
        projections: tuple[int, ...],
    ) -> Any:
        """Start replica-to-owner gradient reduction for selected projections."""

    @abstractmethod
    def wait_grad_reduce(self, handle: Any) -> None:
        """Order the current stream after a gradient-reduction handle."""

    def destroy(self) -> None:
        """Release layer-local transport resources."""


def create_replica_weight_transport(
    expert_dispatcher_type: str, config: ReplicaTransportConfig
) -> ReplicaWeightTransport:
    """Build the transport selected by the single expert-dispatch configuration axis."""
    if expert_dispatcher_type == "replica_hybridep":
        from megatron.core.transformer.moe.replica_peer_tma_transport import PeerTmaTransport

        return PeerTmaTransport(config)
    raise ValueError(
        "Unsupported replica expert-dispatch transport: "
        f"{expert_dispatcher_type!r}."
    )


def finalize_replica_weight_transports() -> None:
    """Release process-group-backed resources owned by every transport backend."""
    from megatron.core.transformer.moe.replica_peer_tma_transport import (
        finalize_peer_tma_transports,
    )

    finalize_peer_tma_transports()
