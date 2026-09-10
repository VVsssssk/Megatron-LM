# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""NVLink peer-memory/TMA transport for runtime expert replicas."""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from megatron.core.transformer.moe.replica_weight_transport import (
    ReplicaGradDestination,
    ReplicaTransportConfig,
    ReplicaWeightSource,
    ReplicaWeightTransport,
)
from megatron.core.transformer.moe.replica_weight_triton import (
    MAX_REPLICA_WEIGHT_SMS,
    compile_replica_weight_kernels,
    launch_replica_grad_reduce,
    launch_replica_weight_prefetch,
)
from megatron.core.utils import nvtx_decorator


@dataclass(frozen=True, slots=True)
class _PeerTmaWorkspaceConfig:
    world_size: int
    num_local_home_experts: int
    num_local_replica_slots: int
    member_shapes: tuple[tuple[int, int], tuple[int, int]]
    weight_format: str
    rowwise_scale_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None
    columnwise_scale_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None
    grad_dtype: torch.dtype
    num_sms: int


class _PeerTmaWorkspace:
    """Fixed symmetric arenas shared by compatible MoE layers on one EP group."""

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        device: torch.device,
        config: _PeerTmaWorkspaceConfig,
    ) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        self.group = group
        self.device = device
        self.config = config
        self.world_size = config.world_size
        self.num_local_home_experts = config.num_local_home_experts
        self.num_local_replica_slots = config.num_local_replica_slots
        self.member_shapes = config.member_shapes
        self.member_numels = tuple(math.prod(shape) for shape in config.member_shapes)
        self.weight_format = config.weight_format
        self.rowwise_scale_shapes = config.rowwise_scale_shapes
        self.columnwise_scale_shapes = config.columnwise_scale_shapes
        self.grad_dtype = config.grad_dtype
        self.num_sms = config.num_sms
        if device.index is None:
            raise ValueError("Replica peer-TMA transport requires an indexed CUDA device.")

        mxfp8 = self.weight_format == "mxfp8"
        self.scale_numels = tuple(numel // 32 for numel in self.member_numels) if mxfp8 else (0, 0)
        if mxfp8:
            assert self.rowwise_scale_shapes is not None
            assert self.columnwise_scale_shapes is not None
            for projection, scale_numel in enumerate(self.scale_numels):
                shapes = (
                    self.rowwise_scale_shapes[projection],
                    self.columnwise_scale_shapes[projection],
                )
                if any(math.prod(shape) != scale_numel for shape in shapes):
                    raise ValueError(
                        "Replica MXFP8 requires one unpadded E8M0 scale byte per 32 weight "
                        f"bytes; projection {projection} has member "
                        f"{self.member_shapes[projection]} and scale shapes {shapes}."
                    )
        arena_numel = self.num_local_replica_slots * sum(self.member_numels)
        try:
            # Backend selection is process-global and immutable after the first
            # symmetric allocation. Materialize the device communicator first.
            dist.barrier(group=group, device_ids=[device.index])
            if not group._get_backend(torch.device("cuda"))._comm_ptr():
                raise RuntimeError("ProcessGroupNCCL returned an invalid communicator pointer.")
            if symm_mem.get_backend(device) != "NCCL":
                symm_mem.set_backend("NCCL")
            self.weight_arena = symm_mem.empty(
                arena_numel + self.num_local_replica_slots * sum(self.scale_numels),
                dtype=torch.uint8 if mxfp8 else torch.bfloat16,
                device=device,
            )
            self.weight_handle = symm_mem.rendezvous(self.weight_arena, group)
            self.grad_arena = symm_mem.empty(arena_numel, dtype=self.grad_dtype, device=device)
            self.grad_handle = symm_mem.rendezvous(self.grad_arena, group)
        except RuntimeError as exc:
            raise RuntimeError(
                "Replica peer-TMA transport could not allocate PyTorch native symmetric memory "
                "for the EP group. This transport requires a single NVLink domain."
            ) from exc

        self.weight_arena.zero_()
        self.grad_arena.zero_()
        self.weight_grid_barrier = torch.zeros(1, dtype=torch.int32, device=device)
        self.grad_grid_barrier = torch.zeros(1, dtype=torch.int32, device=device)
        self.weight_stream = torch.cuda.Stream(device=device, priority=0)
        self.weight_stream_fallback = torch.cuda.Stream(device=device, priority=0)
        self.grad_stream = torch.cuda.Stream(device=device, priority=0)
        self._native_projection_grad_storage: dict[int, torch.Tensor] = {}
        self._destroyed = False

        compile_replica_weight_kernels(
            world_size=self.world_size,
            num_local_home_experts=self.num_local_home_experts,
            num_local_replica_slots=self.num_local_replica_slots,
            member_numels=self.member_numels,
            num_sms=self.num_sms,
            device_index=device.index,
            grad_dtype=self.grad_dtype,
            mxfp8=mxfp8,
        )
        # No rank may enter a device-side cross-rank barrier before all peers
        # have a launchable kernel.
        dist.barrier(group=group, device_ids=[device.index])

    def select_weight_stream(self, current_stream: torch.cuda.Stream) -> torch.cuda.Stream:
        """Return a preallocated weight stream distinct from the active graph stream."""
        for stream in (self.weight_stream, self.weight_stream_fallback):
            if stream.cuda_stream != current_stream.cuda_stream:
                return stream
        raise RuntimeError("Replica weight streams alias the active CUDA stream.")

    def validate(self, config: _PeerTmaWorkspaceConfig) -> None:
        """Reject heterogeneous layers instead of creating a shape-keyed memory pool."""
        if config != self.config:
            raise ValueError(
                "All replica-planned MoE layers on an EP group must share one weight shape and "
                f"launch configuration; expected {self.config}, got {config}."
            )

    def projection_views(self, projection_index: int) -> tuple[tuple[Any, ...], torch.Tensor]:
        """Return virtual runtime weights and gradients for one projection."""
        count = self.num_local_replica_slots
        member_numel = self.member_numels[projection_index]
        member_shape = self.member_shapes[projection_index]
        grad_offset = count * sum(self.member_numels[:projection_index])
        virtual_grad = self.grad_arena.narrow(0, grad_offset, count * member_numel).view(
            count, *member_shape
        )
        if self.weight_format == "bf16":
            weights = self.weight_arena.narrow(0, grad_offset, count * member_numel)
            return tuple(weights.view(count, *member_shape)), virtual_grad

        assert self.rowwise_scale_shapes is not None
        assert self.columnwise_scale_shapes is not None
        offset = count * sum(
            member + scale
            for member, scale in zip(
                self.member_numels[:projection_index], self.scale_numels[:projection_index]
            )
        )
        rowwise_data, columnwise_data = (
            self.weight_arena.narrow(0, offset, count * member_numel).view(count, *member_shape)
            for _ in range(2)
        )
        scales = self.weight_arena.narrow(
            0, offset + count * member_numel, count * self.scale_numels[projection_index]
        )
        rowwise_scale = scales.view(count, *self.rowwise_scale_shapes[projection_index])
        columnwise_scale = scales.view(count, *self.columnwise_scale_shapes[projection_index])
        return (
            tuple(
                (rowwise_data[i], rowwise_scale[i], columnwise_data[i], columnwise_scale[i])
                for i in range(count)
            ),
            virtual_grad,
        )

    def native_projection_grad_view(self, projection_index: int) -> torch.Tensor:
        """Return shared full-gradient staging for one projection."""
        cached = self._native_projection_grad_storage.get(projection_index)
        if cached is None:
            cached = torch.empty(
                (self.num_local_home_experts, *self.member_shapes[projection_index]),
                dtype=self.grad_dtype,
                device=self.device,
            )
            self._native_projection_grad_storage[projection_index] = cached
        return cached

    def destroy(self) -> None:
        """Release symmetric registrations while their NCCL group is still alive."""
        if self._destroyed:
            return
        torch.cuda.synchronize(self.device)
        self._native_projection_grad_storage.clear()
        self.weight_handle = None
        self.grad_handle = None
        self.weight_arena = None
        self.grad_arena = None
        self._destroyed = True


_peer_tma_workspaces: dict[tuple[int, int | None], _PeerTmaWorkspace] = {}


def _get_peer_tma_workspace(config: ReplicaTransportConfig) -> _PeerTmaWorkspace:
    if config.grad_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            "Replica gradients must use torch.float32 or torch.bfloat16, "
            f"got {config.grad_dtype}."
        )
    device_sms = torch.cuda.get_device_properties(config.device).multi_processor_count
    effective_sms = min(
        32 if config.num_sms is None else int(config.num_sms),
        MAX_REPLICA_WEIGHT_SMS,
        max(1, device_sms - 8),
    )
    if effective_sms <= 0:
        raise ValueError(f"Replica weight num_sms must be positive, got {config.num_sms}.")
    workspace_config = _PeerTmaWorkspaceConfig(
        world_size=config.world_size,
        num_local_home_experts=config.num_local_home_experts,
        num_local_replica_slots=config.num_local_replica_slots,
        member_shapes=config.member_shapes,
        weight_format=config.weight_format,
        rowwise_scale_shapes=config.rowwise_scale_shapes,
        columnwise_scale_shapes=config.columnwise_scale_shapes,
        grad_dtype=config.grad_dtype,
        num_sms=effective_sms,
    )
    key = (id(config.group), config.device.index)
    workspace = _peer_tma_workspaces.get(key)
    if workspace is None:
        workspace = _PeerTmaWorkspace(
            group=config.group, device=config.device, config=workspace_config
        )
        _peer_tma_workspaces[key] = workspace
    else:
        workspace.validate(workspace_config)
    return workspace


class PeerTmaTransport(ReplicaWeightTransport):
    """Use symmetric peer mappings and Triton TMA within one NVLink domain."""

    transport_name = "peer_tma"

    def __init__(self, config: ReplicaTransportConfig) -> None:
        self.config = config
        self.rank = dist.get_rank(group=config.group)
        self.workspace = _get_peer_tma_workspace(config)
        self._destroyed = False
        self._prefetch_done = torch.cuda.Event()
        self._grad_reduce_done = (torch.cuda.Event(), torch.cuda.Event())
        for event in (self._prefetch_done, *self._grad_reduce_done):
            event.record(torch.cuda.current_stream(config.device))

    @property
    def grad_dtype(self) -> torch.dtype:
        return self.workspace.grad_dtype

    def projection_views(self, projection_index: int) -> tuple[tuple[Any, ...], torch.Tensor]:
        return self.workspace.projection_views(projection_index)

    def native_projection_grad_view(self, projection_index: int) -> torch.Tensor:
        return self.workspace.native_projection_grad_view(projection_index)

    @torch.no_grad()
    @nvtx_decorator(message="replica_weight_push_start")
    def start_weight_sync(
        self,
        *,
        sources: tuple[ReplicaWeightSource, ...],
        experts_to_copy: torch.Tensor,
    ) -> torch.cuda.Event:
        workspace = self.workspace
        data_bases = tuple(source.data_bases for source in sources)
        if any(base is None for base in data_bases):
            raise ValueError("Peer-TMA transport requires device data pointer tables.")
        scale_bases = tuple(source.scale_bases for source in sources)
        if workspace.weight_format == "mxfp8" and any(base is None for base in scale_bases):
            raise ValueError("Peer-TMA MXFP8 transport requires device scale pointer tables.")
        current_stream = torch.cuda.current_stream(self.config.device)
        weight_stream = workspace.select_weight_stream(current_stream)
        weight_stream.wait_stream(current_stream)
        with torch.cuda.stream(weight_stream):
            launch_replica_weight_prefetch(
                sources=data_bases,
                scale_sources=scale_bases if workspace.weight_format == "mxfp8" else None,
                arena=workspace.weight_arena,
                peer_bases=workspace.weight_handle.buffer_ptrs_dev,
                signal_bases=workspace.weight_handle.signal_pad_ptrs_dev,
                experts_to_copy=experts_to_copy,
                grid_barrier=workspace.weight_grid_barrier,
                rank=self.rank,
                world_size=self.config.world_size,
                num_local_home_experts=self.config.num_local_home_experts,
                num_local_replica_slots=self.config.num_local_replica_slots,
                member_numels=workspace.member_numels,
                num_sms=workspace.num_sms,
            )
            self._prefetch_done.record(weight_stream)
        return self._prefetch_done

    @torch.no_grad()
    @nvtx_decorator(message="replica_weight_push_wait")
    def wait_weight_sync(self, handle: torch.cuda.Event) -> None:
        torch.cuda.current_stream(self.config.device).wait_event(handle)

    @torch.no_grad()
    @nvtx_decorator(message="replica_grad_reduce_start")
    def start_grad_reduce(
        self,
        *,
        native_grads: tuple[ReplicaGradDestination, ...],
        experts_to_copy: torch.Tensor,
        projections: tuple[int, ...],
    ) -> torch.cuda.Event:
        if len(projections) != 1:
            raise ValueError("Peer-TMA gradient reduction currently launches one projection.")
        native_grad_bases = tuple(destination.bases for destination in native_grads)
        if any(base is None for base in native_grad_bases):
            raise ValueError("Peer-TMA gradient reduction requires device pointer tables.")
        projection = projections[0]
        workspace = self.workspace
        current_stream = torch.cuda.current_stream(self.config.device)
        workspace.grad_stream.wait_stream(current_stream)
        with torch.cuda.stream(workspace.grad_stream):
            launch_replica_grad_reduce(
                arena=workspace.grad_arena,
                native_grads=native_grad_bases,
                peer_bases=workspace.grad_handle.buffer_ptrs_dev,
                signal_bases=workspace.grad_handle.signal_pad_ptrs_dev,
                experts_to_copy=experts_to_copy,
                grid_barrier=workspace.grad_grid_barrier,
                rank=self.rank,
                world_size=self.config.world_size,
                num_local_home_experts=self.config.num_local_home_experts,
                num_local_replica_slots=self.config.num_local_replica_slots,
                member_numels=workspace.member_numels,
                num_sms=workspace.num_sms,
                projections=projections,
            )
            self._grad_reduce_done[projection].record(workspace.grad_stream)
        return self._grad_reduce_done[projection]

    @torch.no_grad()
    @nvtx_decorator(message="replica_grad_reduce_wait")
    def wait_grad_reduce(self, handle: torch.cuda.Event) -> None:
        torch.cuda.current_stream(self.config.device).wait_event(handle)

    def destroy(self) -> None:
        """Drop layer-local completion objects while retaining the shared workspace."""
        if self._destroyed:
            return
        self._prefetch_done = None
        self._grad_reduce_done = ()
        self.workspace = None
        self._destroyed = True


def finalize_peer_tma_transports() -> None:
    """Release all peer-TMA symmetric windows before process-group teardown."""
    workspaces = list(_peer_tma_workspaces.values())
    for workspace in workspaces:
        workspace.destroy()
    _peer_tma_workspaces.clear()
    # NCCL symmetric-memory handles contain Python reference cycles.
    gc.collect()
