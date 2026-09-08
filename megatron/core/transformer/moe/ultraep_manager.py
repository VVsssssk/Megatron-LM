# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""UltraEP runtime ownership and layer registration for MCore MoE."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

from megatron.core import utils

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import ultra_ep

    HAVE_ULTRAEP = True
except ImportError:
    ultra_ep = None
    HAVE_ULTRAEP = False


class UltraEPEventHandle(Protocol):
    """Event handle returned by asynchronous UltraEP operations."""

    def current_stream_wait(self) -> None:
        """Make the current CUDA stream wait for the recorded operation."""


class UltraEPManager:
    """Own one UltraEP runtime per expert-parallel process group.

    Decoder layer IDs are one-based. GPT-style MTP layers occupy the IDs after
    the decoder stack so their placement and pointer slots cannot alias regular
    transformer layers.
    """

    def __init__(self, config: TransformerConfig, ep_group: torch.distributed.ProcessGroup) -> None:
        if not HAVE_ULTRAEP:
            raise ImportError(
                "moe_enable_ultraep=True requires the UltraEP package. Install "
                "https://github.com/Dots-Infra/UltraEP in the training container."
            )
        if config.num_moe_experts is None or config.moe_ffn_hidden_size is None:
            raise ValueError("UltraEP requires num_moe_experts and moe_ffn_hidden_size.")

        self.group = ep_group
        self.rank = utils.get_pg_rank(ep_group)
        self.num_ranks = utils.get_pg_size(ep_group)
        self.num_decoder_layers = config.num_layers
        self.num_mtp_layers = config.mtp_num_layers or 0
        self.num_layers = self.num_decoder_layers + self.num_mtp_layers
        self.num_local_master_experts = config.num_moe_experts // self.num_ranks
        self.num_local_redundant_experts = config.moe_num_redundant_experts_per_rank
        self.num_local_physical_experts = (
            self.num_local_master_experts + self.num_local_redundant_experts
        )
        self.num_global_logical_experts = config.num_moe_experts
        self.num_global_physical_experts = self.num_local_physical_experts * self.num_ranks
        self.local_physical_expert_indices = [
            self.rank * self.num_local_physical_experts + index
            for index in range(self.num_local_physical_experts)
        ]

        self.expert_fc1_numel = 2 * config.hidden_size * config.moe_ffn_hidden_size
        self.expert_fc2_numel = config.hidden_size * config.moe_ffn_hidden_size

        # PP is rejected by TransformerConfig in this phase, so a single slot is
        # sufficient unless activation checkpointing performs a second forward.
        self.max_microbatches = 3

        self.runtime = ultra_ep.Manager(
            group=ep_group,
            num_layers=self.num_layers,
            num_local_master_experts=self.num_local_master_experts,
            num_local_redundant_experts=self.num_local_redundant_experts,
            expert_fc1_numel=self.expert_fc1_numel,
            expert_fc2_numel=self.expert_fc2_numel,
            is_train=True,
            explicitly_destroy=True,
            max_microbatches=self.max_microbatches,
            weight_data_dtype=config.params_dtype,
            grad_dtype=torch.float32,
        )
        self._closed = False

        self.local_replica_fc1_weight_buffer = self.runtime.local_replica_fc1_weight_buffer
        self.local_replica_fc2_weight_buffer = self.runtime.local_replica_fc2_weight_buffer
        self.local_replica_fc1_grad_buffer = self.runtime.local_replica_fc1_grad_buffer
        self.local_replica_fc2_grad_buffer = self.runtime.local_replica_fc2_grad_buffer

    @property
    def signature(self) -> tuple[int, ...]:
        """Return configuration fields that must match for layers sharing the runtime."""
        return (
            self.num_decoder_layers,
            self.num_mtp_layers,
            self.num_local_master_experts,
            self.num_local_redundant_experts,
            self.expert_fc1_numel,
            self.expert_fc2_numel,
            self.max_microbatches,
        )

    def layer_id(self, layer_number: int, *, is_mtp_layer: bool = False) -> int:
        """Map an MCore layer number to a one-based UltraEP real layer ID."""
        upper_bound = self.num_mtp_layers if is_mtp_layer else self.num_decoder_layers
        layer_kind = "MTP" if is_mtp_layer else "decoder"
        if not 1 <= layer_number <= upper_bound:
            raise ValueError(
                f"UltraEP {layer_kind} layer_number must be in [1, {upper_bound}], "
                f"got {layer_number}."
            )
        return self.num_decoder_layers + layer_number if is_mtp_layer else layer_number

    def allocate_microbatch_slot(self, real_layer_id: int) -> int:
        """Allocate placement state for one real-layer/microbatch pair."""
        if not 1 <= real_layer_id <= self.num_layers:
            raise ValueError(
                f"UltraEP real layer ID must be in [1, {self.num_layers}], got {real_layer_id}."
            )
        return self.runtime.allocate_microbatch_slot(real_layer_id)

    def register_master_experts(
        self,
        real_layer_id: int,
        fc1_weights: list[torch.Tensor],
        fc2_weights: list[torch.Tensor],
        fc1_grads: list[torch.Tensor],
        fc2_grads: list[torch.Tensor],
    ) -> None:
        """Register final post-DDP master weight and gradient pointers for one layer."""
        if not 1 <= real_layer_id <= self.num_layers:
            raise ValueError(
                f"UltraEP real layer ID must be in [1, {self.num_layers}], got {real_layer_id}."
            )
        self.runtime.construct_local_master_ptr_pool(
            layer_id=real_layer_id,
            fc1_weights=fc1_weights,
            fc2_weights=fc2_weights,
            fc1_grads=fc1_grads,
            fc2_grads=fc2_grads,
        )

    def update_placement(self, virtual_layer_id: int, routing_map: torch.Tensor) -> None:
        """Compute physical expert placement from the logical routing map."""
        self.runtime.update_placement(virtual_layer_id, routing_map)

    def reroute(
        self, virtual_layer_id: int, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand logical routing tensors into the current physical expert space."""
        return self.runtime.reroute(virtual_layer_id, probs, routing_map)

    def weight_sync(self, virtual_layer_id: int, async_finish: bool) -> UltraEPEventHandle | None:
        """Copy master expert weights into replica slots for a placement."""
        return self.runtime.weight_sync(virtual_layer_id, async_finish=async_finish)

    def grad_reduce(self, virtual_layer_id: int, async_finish: bool) -> UltraEPEventHandle | None:
        """Reduce replica gradients back into their logical master experts."""
        return self.runtime.grad_reduce(virtual_layer_id, async_finish=async_finish)

    def close(self) -> None:
        """Flush profiling data and release the UltraEP runtime exactly once."""
        if self._closed:
            return
        self.runtime.destroy()
        self._closed = True


_ULTRAEP_MANAGER_REGISTRY: dict[int, UltraEPManager] = {}


def _manager_signature(
    config: TransformerConfig, ep_group: torch.distributed.ProcessGroup
) -> tuple[int, ...]:
    """Build the manager-sharing signature without constructing a runtime."""
    if config.num_moe_experts is None or config.moe_ffn_hidden_size is None:
        raise ValueError("UltraEP requires num_moe_experts and moe_ffn_hidden_size.")
    return (
        config.num_layers,
        config.mtp_num_layers or 0,
        config.num_moe_experts // utils.get_pg_size(ep_group),
        config.moe_num_redundant_experts_per_rank,
        2 * config.hidden_size * config.moe_ffn_hidden_size,
        config.hidden_size * config.moe_ffn_hidden_size,
        3,
    )


def get_or_create_ultraep_manager(
    config: TransformerConfig, ep_group: torch.distributed.ProcessGroup
) -> UltraEPManager:
    """Get the shared UltraEP runtime for an expert-parallel process group."""
    key = id(ep_group)
    manager = _ULTRAEP_MANAGER_REGISTRY.get(key)
    if manager is None:
        manager = UltraEPManager(config, ep_group)
        _ULTRAEP_MANAGER_REGISTRY[key] = manager
        return manager

    if manager.signature != _manager_signature(config, ep_group):
        raise ValueError(
            "MoE layers sharing an EP process group must use the same UltraEP configuration."
        )
    return manager


def clear_ultraep_manager_registry() -> None:
    """Drop Python references to cached UltraEP managers (primarily for tests)."""
    _ULTRAEP_MANAGER_REGISTRY.clear()


def destroy_ultraep_managers() -> None:
    """Flush and destroy all UltraEP runtimes before Python worker shutdown."""
    for manager in _ULTRAEP_MANAGER_REGISTRY.values():
        manager.close()
    _ULTRAEP_MANAGER_REGISTRY.clear()
