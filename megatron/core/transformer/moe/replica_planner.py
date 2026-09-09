# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Runtime replica weight movement for MoEScheduler expert dispatch.

``ReplicaWeightBridge`` consumes a planner-independent ``ReplicaPlan``. It
pushes owner weights into peer replica slots before expert compute and reduces
replica gradients into native wgrad staging after backward. Planning lives in
``moonep_moe_scheduler.py`` so Echo and external planners can reuse this bridge.
"""

import functools
import gc
import math
import weakref
from dataclasses import dataclass
from enum import Enum, auto

import torch
import torch.distributed as dist

from megatron.core.fp8_utils import is_mxfp8tensor
from megatron.core.transformer.moe.replica_weight_triton import (
    MAX_REPLICA_WEIGHT_SMS,
    compile_replica_weight_kernels,
    launch_replica_grad_reduce,
    launch_replica_weight_prefetch,
)
from megatron.core.utils import nvtx_decorator

try:
    from transformer_engine.pytorch.module.base import get_dummy_wgrad
except ImportError:
    get_dummy_wgrad = None

_MXFP8_COMPONENTS = (
    "_rowwise_data",
    "_rowwise_scale_inv",
    "_columnwise_data",
    "_columnwise_scale_inv",
)


def _discard_runtime_parameter_grad(parameter: torch.nn.Parameter) -> None:
    """Drop TE's throwaway leaf grad after its fused wgrad reaches runtime staging."""
    parameter.grad = None


@dataclass(frozen=True, slots=True)
class ReplicaPlan:
    """Transport-facing result of replica placement planning.

    Attributes:
        virtual_experts: Rank-major runtime expert ids with shape
            ``[num_tokens, router_topk]``.
            A native id is ``destination * num_runtime_experts_per_gpu +
            local_expert``; a replica id adds ``num_home_experts_per_gpu`` and
            the replica slot instead.
        experts_to_copy: Semantic expert ids assigned to each rank's replica
            slots, with shape ``[ep_size, num_replica_slots_per_gpu]``. Unused
            slots contain ``-1``.
    """

    virtual_experts: torch.Tensor
    experts_to_copy: torch.Tensor


@dataclass(frozen=True, slots=True)
class _ReplicaProjectionSpec:
    """Optimizer parameters and one runtime weight tensor per local expert."""

    parameters: tuple[torch.nn.Parameter, ...]
    source_tensors: tuple[torch.Tensor, ...]
    member_shape: tuple[int, int]
    weight_format: str
    gtp_leader: torch.nn.Parameter | None = None
    rowwise_scale_shape: tuple[int, ...] | None = None
    columnwise_scale_shape: tuple[int, ...] | None = None


def _parameter_storage(parameter: torch.nn.Parameter) -> torch.Tensor:
    """Return the BF16 storage used by a TE parameter, including tensor subclasses."""
    rowwise_data = getattr(parameter, "rowwise_data", None)
    return rowwise_data if rowwise_data is not None else parameter.data


def _bf16_ptrs(source: torch.Tensor, *, numel: int, device: torch.device, label: str) -> tuple[int]:
    """Validate one BF16 runtime member and return its pointer signature."""
    if (
        source.dtype != torch.bfloat16
        or source.device != device
        or source.numel() != numel
        or not source.is_contiguous()
    ):
        raise ValueError(
            f"{label} requires contiguous BF16 storage with {numel} elements on {device}; "
            f"got dtype={source.dtype}, shape={tuple(source.shape)}, device={source.device}."
        )
    return (source.data_ptr(),)


def _mxfp8_ptrs(
    source: torch.Tensor,
    *,
    member_shape: tuple[int, int],
    scale_shapes: tuple[tuple[int, ...] | None, tuple[int, ...] | None],
    device: torch.device,
    label: str,
    components: tuple[str, ...] = _MXFP8_COMPONENTS,
) -> tuple[int, ...]:
    """Validate the requested MXFP8 components of one member and return their pointers."""
    shapes = dict(
        zip(_MXFP8_COMPONENTS, (member_shape, scale_shapes[0], member_shape, scale_shapes[1]))
    )
    storage = tuple(getattr(source, name, None) for name in components)
    if (
        tuple(source.shape) != member_shape
        or source.device != device
        or any(
            tensor is None
            or tensor.dtype != torch.uint8
            or not tensor.is_contiguous()
            or tuple(tensor.shape) != shapes[name]
            for tensor, name in zip(storage, components)
        )
    ):
        actual = tuple(
            None if tensor is None else (tensor.dtype, tuple(tensor.shape)) for tensor in storage
        )
        raise ValueError(
            f"{label} requires contiguous MXFP8 {components} with shapes "
            f"{tuple(shapes[name] for name in components)} on {device}; got "
            f"shape={tuple(source.shape)}, device={source.device}, storage={actual}."
        )
    return tuple(tensor.data_ptr() for tensor in storage)


def _collect_replica_projection_specs(
    experts: torch.nn.Module, *, num_local_experts: int, backend_name: str
) -> tuple[list[_ReplicaProjectionSpec], torch.device]:
    """Collect independently allocated TE expert weights."""
    projection_specs = []
    device: torch.device | None = None
    for linear in (experts.linear_fc1, experts.linear_fc2):
        member_shape = (int(linear.out_features), int(linear.in_features))
        if getattr(linear, "single_grouped_weight", False):
            raise ValueError(
                f"{backend_name} requires discrete weight0..weightN expert parameters; "
                "moe_single_grouped_weight must be False."
            )
        parameters = tuple(
            linear.get_parameter(f"weight{index}") for index in range(num_local_experts)
        )
        gtp_members = tuple(
            bool(getattr(parameter, "is_gtp_weight_remat", False)) for parameter in parameters
        )
        if any(gtp_members) and not all(gtp_members):
            raise ValueError(
                f"{backend_name} requires every weight in a projection to use the same GTP layout."
            )
        gtp_leader = parameters[0] if all(gtp_members) else None
        if gtp_leader is not None:
            for index, parameter in enumerate(parameters):
                if tuple(parameter._unsharded_shape) != member_shape:
                    raise ValueError(
                        f"{backend_name} expected GTP expert {index} to materialize as "
                        f"{member_shape}, got {tuple(parameter._unsharded_shape)}."
                    )
        mxfp8_members = tuple(is_mxfp8tensor(parameter) for parameter in parameters)
        if any(mxfp8_members) and not all(mxfp8_members):
            raise ValueError(f"{backend_name} does not support mixed BF16 and MXFP8 experts.")
        projection_device = parameters[0].device
        if all(mxfp8_members):
            if gtp_leader is None:
                scales = (parameters[0]._rowwise_scale_inv, parameters[0]._columnwise_scale_inv)
                if any(scale is None for scale in scales):
                    raise ValueError(
                        f"{backend_name} MXFP8 weights require rowwise and columnwise scales."
                    )
                scale_shapes = tuple(tuple(scale.shape) for scale in scales)
                for index, parameter in enumerate(parameters):
                    _mxfp8_ptrs(
                        parameter,
                        member_shape=member_shape,
                        scale_shapes=scale_shapes,
                        device=projection_device,
                        label=f"{backend_name} expert {index}",
                    )
            else:
                quantizer = getattr(gtp_leader, "_gtp_gather_quantizer", None)
                if quantizer is None or not hasattr(quantizer, "get_scale_shape"):
                    raise ValueError(
                        f"{backend_name} GTP MXFP8 weights require a gather quantizer."
                    )
                scale_shapes = tuple(
                    tuple(quantizer.get_scale_shape(member_shape, columnwise=columnwise))
                    for columnwise in (False, True)
                )
            spec = _ReplicaProjectionSpec(
                parameters, parameters, member_shape, "mxfp8", gtp_leader, *scale_shapes
            )
        else:
            source_tensors = tuple(_parameter_storage(parameter) for parameter in parameters)
            for index, (parameter, source) in enumerate(zip(parameters, source_tensors)):
                _bf16_ptrs(
                    source,
                    numel=(
                        math.prod(parameter._sharded_padded_shape)
                        if gtp_leader is not None
                        else math.prod(member_shape)
                    ),
                    device=projection_device,
                    label=f"{backend_name} expert {index}",
                )
            spec = _ReplicaProjectionSpec(
                parameters, source_tensors, member_shape, "bf16", gtp_leader
            )
        if device is None:
            device = projection_device
        elif projection_device != device:
            raise ValueError(f"{backend_name} FC1 and FC2 weights must share one device.")
        projection_specs.append(spec)
    if device is None or device.type != "cuda":
        raise ValueError(f"{backend_name} expert weights must be CUDA tensors.")
    if len({spec.weight_format for spec in projection_specs}) != 1:
        raise ValueError(f"{backend_name} FC1 and FC2 weights must use one storage format.")
    return projection_specs, device


class _WeightDirection(Enum):
    FORWARD = auto()
    BACKWARD = auto()


@dataclass(slots=True)
class _DirectionalBinding:
    """Device pointer tables the push reads for one GEMM orientation."""

    data_bases: torch.Tensor
    scale_bases: torch.Tensor | None = None
    source_tensors: tuple[torch.Tensor, ...] | None = None
    source_ptrs: tuple[tuple[int, ...], ...] | None = None
    # Pinned staging for capture-safe GTP pointer updates; None without GTP.
    host_pointer_table: torch.Tensor | None = None


@dataclass(slots=True)
class _ReplicaProjection:
    """One projection and its stable native/virtual runtime storage.

    Expert backward writes native and replica wgrads into bridge-owned staging.
    The replica reduction accumulates the virtual contributions into the native
    staging, which is then handed to the optimizer parameters through autograd.
    """

    name: str
    device: torch.device
    weight_format: str
    parameters: tuple[torch.nn.Parameter, ...]
    gtp_leader: torch.nn.Parameter | None
    source_tensors: tuple[torch.Tensor, ...]
    forward: _DirectionalBinding
    backward: _DirectionalBinding
    native_grad_bases: torch.Tensor
    member_shape: tuple[int, int]
    rowwise_scale_shape: tuple[int, ...] | None
    columnwise_scale_shape: tuple[int, ...] | None
    virtual_weight: tuple[torch.Tensor, ...]
    virtual_grad: torch.Tensor
    native_grad: torch.Tensor
    runtime_parameters: tuple[torch.nn.Parameter, ...] | None = None
    source_storage_ptrs: tuple[tuple[int, ...], ...] | None = None
    native_grad_ptrs: tuple[int, ...] | None = None

    @property
    def member_numel(self) -> int:
        return math.prod(self.member_shape)

    def binding(self, direction: _WeightDirection) -> _DirectionalBinding:
        return self.backward if direction is _WeightDirection.BACKWARD else self.forward

    def _storage_ptrs(
        self, source: torch.Tensor, label: str, components: tuple[str, ...] = _MXFP8_COMPONENTS
    ) -> tuple[int, ...]:
        if self.weight_format == "bf16":
            return _bf16_ptrs(source, numel=self.member_numel, device=self.device, label=label)
        return _mxfp8_ptrs(
            source,
            member_shape=self.member_shape,
            scale_shapes=(self.rowwise_scale_shape, self.columnwise_scale_shape),
            device=self.device,
            label=label,
            components=components,
        )

    def bind_materialized_weights(
        self, materialized_weights: tuple[torch.Tensor, ...], direction: _WeightDirection
    ) -> None:
        """Bind one stable directional GTP gather without copying its payload."""
        if len(materialized_weights) != len(self.parameters):
            raise RuntimeError(
                f"GTP materialized {len(materialized_weights)} {self.name} weights, "
                f"expected {len(self.parameters)}."
            )
        binding = self.binding(direction)
        backward = direction is _WeightDirection.BACKWARD
        components = _MXFP8_COMPONENTS[2:] if backward else _MXFP8_COMPONENTS[:2]
        source_ptrs = tuple(
            self._storage_ptrs(
                source,
                f"GTP {direction.name.lower()} gather of {self.name} replica expert {index}",
                components,
            )
            for index, source in enumerate(materialized_weights)
        )
        if binding.source_ptrs is not None:
            if source_ptrs != binding.source_ptrs:
                raise RuntimeError(
                    f"Replica GTP {direction.name.lower()} all-gather storage of {self.name} "
                    "changed after direct binding; this would invalidate CUDA-graph source "
                    "pointers."
                )
        else:
            tables = tuple(
                table for table in (binding.data_bases, binding.scale_bases) if table is not None
            )
            if binding.host_pointer_table is None or len(tables) != len(source_ptrs[0]):
                raise RuntimeError(f"Replica GTP binding of {self.name} lost pointer storage.")
            binding.source_ptrs = source_ptrs
            for component, table in enumerate(tables):
                host_row = binding.host_pointer_table[component]
                host_row.copy_(
                    torch.tensor([ptrs[component] for ptrs in source_ptrs], dtype=torch.int64)
                )
                table.copy_(host_row, non_blocking=True)
        binding.source_tensors = materialized_weights

        if self.weight_format == "bf16":
            self.source_tensors = materialized_weights
            for parameter, source in zip(self.runtime_parameters or (), materialized_weights):
                parameter.data = source
            return
        # MXFP8 gathers alias their orientation into the stable native wrappers.
        for destinations in (self.source_tensors, self.runtime_parameters or ()):
            for destination, source in zip(destinations, materialized_weights):
                for field in components:
                    setattr(destination, field, getattr(source, field))
        if self.source_storage_ptrs is not None:
            offset = 2 if backward else 0
            self.source_storage_ptrs = tuple(
                ptrs[:offset] + update + ptrs[offset + 2 :]
                for ptrs, update in zip(self.source_storage_ptrs, source_ptrs)
            )

    def prepare_runtime_parameters(self, grad_dtype: torch.dtype) -> None:
        """Bind final DDP/GTP storage once, then validate pointer stability."""
        directional = self.gtp_leader is not None and self.weight_format == "bf16"
        sources = (
            tuple(_parameter_storage(parameter) for parameter in self.parameters)
            if self.gtp_leader is None and self.weight_format == "bf16"
            else self.source_tensors
        )
        if len(sources) != len(self.parameters):
            raise RuntimeError(
                f"Replica weight bridge {self.name} expected {len(self.parameters)} native "
                f"weights, got {len(sources)}."
            )
        storage_ptrs = tuple(
            self._storage_ptrs(source, f"Replica weight bridge {self.name} expert {index}")
            for index, source in enumerate(sources)
        )
        # Directional GTP BF16 storage is tracked per binding instead.
        if not directional and self.source_storage_ptrs is None:
            self.source_storage_ptrs = storage_ptrs
            if self.gtp_leader is None:
                bindings = (self.forward, self.backward)
                for direction, binding in enumerate(bindings):
                    host_table = binding.host_pointer_table
                    if host_table is None:
                        raise RuntimeError(
                            f"Replica pointer staging for {self.name} was not allocated."
                        )
                    component_offset = 2 * direction if self.weight_format == "mxfp8" else 0
                    for row, table in enumerate((binding.data_bases, binding.scale_bases)):
                        if table is None:
                            continue
                        component = component_offset + row
                        host_row = host_table[row]
                        host_row.copy_(
                            torch.tensor(
                                [ptrs[component] for ptrs in storage_ptrs], dtype=torch.int64
                            )
                        )
                        table.copy_(host_row, non_blocking=True)
        elif not directional and storage_ptrs != self.source_storage_ptrs:
            raise RuntimeError(
                f"Replica weight bridge {self.name} parameter storage changed after binding; "
                "this would invalidate CUDA-graph source pointers."
            )

        native_grads = tuple(self.native_grad)
        for index, grad in enumerate(native_grads):
            if (
                grad.dtype != grad_dtype
                or grad.device != self.device
                or grad.numel() != self.member_numel
                or not grad.is_contiguous()
            ):
                raise ValueError(
                    f"Replica weight bridge {self.name} native grad {index} must be contiguous "
                    f"{grad_dtype} with {self.member_numel} elements on {self.device}; got "
                    f"dtype={grad.dtype}, shape={tuple(grad.shape)}, device={grad.device}."
                )
        native_grad_ptrs = tuple(grad.data_ptr() for grad in native_grads)
        if self.native_grad_ptrs is None:
            self.native_grad_ptrs = native_grad_ptrs
            self.native_grad_bases.copy_(
                torch.tensor(native_grad_ptrs, dtype=torch.int64, device=self.device)
            )
        elif native_grad_ptrs != self.native_grad_ptrs:
            raise RuntimeError(
                f"Replica weight bridge {self.name} native-grad storage changed after binding; "
                "this would invalidate CUDA-graph destination pointers."
            )

        self.source_tensors = sources
        weights = sources + tuple(self.virtual_weight)
        grads = native_grads + tuple(self.virtual_grad)
        if self.runtime_parameters is None:
            self.bind_runtime_parameters(weights, grads)
        else:
            self.validate_runtime_parameters(weights, grads)

    def bind_runtime_parameters(self, weights, grads) -> None:
        """Create the stable native-then-virtual TE parameter sequence once."""
        runtime_parameters = []
        for weight, grad in zip(weights, grads):
            parameter = torch.nn.Parameter(weight, requires_grad=True)
            parameter.main_grad = grad
            parameter.grad_added_to_main_grad = True
            # TE's wgrad GEMM then rewrites the staging and every replica slot
            # on each backward, so the slots never need clearing.
            parameter.overwrite_main_grad = True
            parameter.register_post_accumulate_grad_hook(_discard_runtime_parameter_grad)
            runtime_parameters.append(parameter)
        self.runtime_parameters = tuple(runtime_parameters)

    def validate_runtime_parameters(self, weights, grads) -> None:
        """Validate that runtime parameters still alias the bound storage."""
        for parameter, weight, grad in zip(self.runtime_parameters, weights, grads):
            fields = ("data",) if self.weight_format == "bf16" else _MXFP8_COMPONENTS
            if any(
                getattr(parameter, field).data_ptr() != getattr(weight, field).data_ptr()
                for field in fields
            ):
                raise RuntimeError(
                    f"Replica weight bridge {self.name} runtime weight storage changed after "
                    "binding."
                )
            runtime_grad = getattr(parameter, "main_grad", None)
            if runtime_grad is None or runtime_grad.data_ptr() != grad.data_ptr():
                raise RuntimeError(
                    f"Replica weight bridge {self.name} runtime main-grad storage changed after "
                    "binding."
                )
            parameter.grad_added_to_main_grad = True
            parameter.overwrite_main_grad = True

    def destroy(self) -> None:
        for parameter in self.runtime_parameters or ():
            parameter.main_grad = None
        self.runtime_parameters = None


@dataclass(frozen=True, slots=True)
class _ReplicaWeightWorkspaceConfig:
    world_size: int
    num_local_home_experts: int
    num_local_replica_slots: int
    member_shapes: tuple[tuple[int, int], tuple[int, int]]
    weight_format: str
    rowwise_scale_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None
    columnwise_scale_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None
    grad_dtype: torch.dtype
    num_sms: int


class _ReplicaWeightWorkspace:
    """Fixed-shape symmetric arenas shared by every compatible MoE layer.

    The weight arena stores ``fc1 data, fc1 scales, fc2 data, fc2 scales`` with
    ``num_local_replica_slots`` members per section; MXFP8 keeps one scale section per
    projection because forward consumes rowwise and backward columnwise storage
    at disjoint times. The gradient arena stores ``fc1, fc2`` members.
    """

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        device: torch.device,
        config: _ReplicaWeightWorkspaceConfig,
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
            raise ValueError("Replica weight workspace requires an indexed CUDA device.")

        mxfp8 = self.weight_format == "mxfp8"
        self.scale_numels = tuple(numel // 32 for numel in self.member_numels) if mxfp8 else (0, 0)
        if mxfp8:
            for projection, scale_numel in enumerate(self.scale_numels):
                shapes = (
                    self.rowwise_scale_shapes[projection],
                    self.columnwise_scale_shapes[projection],
                )
                if any(math.prod(shape) != scale_numel for shape in shapes):
                    raise ValueError(
                        "Replica MXFP8 requires one unpadded E8M0 scale byte per 32 weight bytes; "
                        f"projection {projection} has member {self.member_shapes[projection]} and "
                        f"scale shapes {shapes}."
                    )
        arena_numel = self.num_local_replica_slots * sum(self.member_numels)
        try:
            # Symmetric-memory backend selection is process-global and becomes
            # immutable after the first allocation. NCCL window registration
            # requires the device-specific communicator to exist, so materialize
            # it once here, before training or graph capture.
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
                "Replica weights could not allocate PyTorch native symmetric memory for the EP "
                "group. The initial implementation requires a single NVLink domain."
            ) from exc

        self.weight_arena.zero_()
        self.grad_arena.zero_()
        self.weight_grid_barrier = torch.zeros(1, dtype=torch.int32, device=device)
        self.grad_grid_barrier = torch.zeros(1, dtype=torch.int32, device=device)
        self.weight_stream = torch.cuda.Stream(device=device, priority=0)
        # CUDA graph capture streams also come from PyTorch's stream pool and
        # may alias a stream allocated earlier. Keep a second candidate so the
        # weight branch never collapses onto the active planner stream.
        self.weight_stream_fallback = torch.cuda.Stream(device=device, priority=0)
        self.grad_stream = torch.cuda.Stream(device=device, priority=0)
        self._native_projection_grad_storage = {}
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
        # JIT time can vary substantially by rank on a cold cache. No rank may
        # enter the device-side cross-rank barrier until every peer has a
        # launchable kernel.
        dist.barrier(group=group, device_ids=[device.index])

    def select_weight_stream(self, current_stream: torch.cuda.Stream) -> torch.cuda.Stream:
        """Return a preallocated weight stream distinct from the active graph stream."""
        for stream in (self.weight_stream, self.weight_stream_fallback):
            if stream.cuda_stream != current_stream.cuda_stream:
                return stream
        raise RuntimeError("Replica weight streams alias the active CUDA stream.")

    def validate(self, config: _ReplicaWeightWorkspaceConfig) -> None:
        """Reject heterogeneous layers instead of creating a shape-keyed memory pool."""
        if config != self.config:
            raise ValueError(
                "All replica-planned MoE layers on an EP group must share one weight shape and "
                f"launch configuration; expected {self.config}, got {config}."
            )

    def projection_views(self, projection_index: int) -> tuple[tuple, torch.Tensor]:
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
        # The bridge wraps these raw views with source-matching TE metadata.
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
        # Handles own NCCL window registrations. Drop them before their backing
        # tensors and, critically, before model-parallel process-group teardown.
        self.weight_handle = None
        self.grad_handle = None
        self.weight_arena = None
        self.grad_arena = None
        self._destroyed = True


_replica_weight_workspaces = weakref.WeakValueDictionary()
_replica_weight_bridges = weakref.WeakSet()


def _get_replica_weight_workspace(
    *, group: dist.ProcessGroup, device: torch.device, num_sms: int | None, **config_fields
) -> _ReplicaWeightWorkspace:
    """Return the one fixed-shape workspace owned by an EP group and device."""
    if config_fields["grad_dtype"] not in (torch.float32, torch.bfloat16):
        raise ValueError(
            "Replica gradients must use torch.float32 or torch.bfloat16, "
            f"got {config_fields['grad_dtype']}."
        )
    device_sms = torch.cuda.get_device_properties(device).multi_processor_count
    effective_sms = min(
        32 if num_sms is None else int(num_sms), MAX_REPLICA_WEIGHT_SMS, max(1, device_sms - 8)
    )
    if effective_sms <= 0:
        raise ValueError(f"Replica weight num_sms must be positive, got {num_sms}.")
    config = _ReplicaWeightWorkspaceConfig(num_sms=effective_sms, **config_fields)
    key = (id(group), device.index)
    workspace = _replica_weight_workspaces.get(key)
    if workspace is None:
        workspace = _ReplicaWeightWorkspace(group=group, device=device, config=config)
        _replica_weight_workspaces[key] = workspace
    else:
        workspace.validate(config)
    return workspace


class ReplicaWeightBridge:
    """Dispatcher-independent asynchronous replica weight and gradient bridge."""

    def __init__(
        self,
        *,
        experts: torch.nn.Module,
        group: dist.ProcessGroup,
        num_experts: int,
        num_local_home_experts: int,
        num_local_replica_slots: int,
        grad_dtype: torch.dtype = torch.float32,
        num_sms: int | None = None,
    ) -> None:
        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self.num_local_home_experts = int(num_local_home_experts)
        self.num_local_replica_slots = int(num_local_replica_slots)
        self.num_runtime_experts = self.num_local_home_experts + self.num_local_replica_slots
        self.last_plan = None
        self._prefetch_plan = None
        self._completed_plan = None
        self._backward_plan = None
        self._grad_reduce_plan = None
        self._grad_reduce_started: set[int] = set()
        self._experts_ref = weakref.ref(experts)
        self._destroyed = False

        if self.num_local_replica_slots <= 0:
            raise ValueError(
                "Replica weights require at least one replica slot per rank, "
                f"got {self.num_local_replica_slots}."
            )
        if int(num_experts) != self.world_size * self.num_local_home_experts:
            raise ValueError(
                "Replica weights require an even expert distribution: "
                f"num_experts={num_experts}, world_size={self.world_size}, "
                f"num_local_home_experts={self.num_local_home_experts}."
            )
        projection_specs, self.device = _collect_replica_projection_specs(
            experts, num_local_experts=self.num_local_home_experts, backend_name="Replica-HybridEP"
        )
        self.weight_format = projection_specs[0].weight_format
        mxfp8 = self.weight_format == "mxfp8"
        self.workspace = _get_replica_weight_workspace(
            group=group,
            device=self.device,
            num_sms=num_sms,
            world_size=self.world_size,
            num_local_home_experts=self.num_local_home_experts,
            num_local_replica_slots=self.num_local_replica_slots,
            member_shapes=tuple(spec.member_shape for spec in projection_specs),
            weight_format=self.weight_format,
            rowwise_scale_shapes=(
                tuple(spec.rowwise_scale_shape for spec in projection_specs) if mxfp8 else None
            ),
            columnwise_scale_shapes=(
                tuple(spec.columnwise_scale_shape for spec in projection_specs) if mxfp8 else None
            ),
            grad_dtype=grad_dtype,
        )
        # PyTorch creates CUDA event handles lazily on first record. Materialize
        # every reusable event during binding, before graph capture or training.
        self.prefetch_done = torch.cuda.Event()
        self.grad_reduce_done = (torch.cuda.Event(), torch.cuda.Event())
        for event in (
            self.prefetch_done,
            *self.grad_reduce_done,
        ):
            event.record(torch.cuda.current_stream(self.device))

        def pointer_table() -> torch.Tensor:
            return torch.empty(self.num_local_home_experts, dtype=torch.int64, device=self.device)

        def binding(gtp: bool) -> _DirectionalBinding:
            del gtp
            # All pointer uploads stage through pinned host memory so the hot
            # path does not construct temporary CUDA tensors.
            components = 2 if mxfp8 else 1
            return _DirectionalBinding(
                pointer_table(),
                pointer_table() if mxfp8 else None,
                host_pointer_table=(
                    torch.empty(
                        (components, self.num_local_home_experts),
                        dtype=torch.int64,
                        pin_memory=True,
                    )
                    if components
                    else None
                ),
            )

        self.projections: list[_ReplicaProjection] = []
        for projection_index, spec in enumerate(projection_specs):
            virtual_storage, virtual_grad = self.workspace.projection_views(projection_index)
            gtp = spec.gtp_leader is not None
            if mxfp8:
                virtual_weight = _wrap_mxfp8(spec, virtual_storage, self.device)
                # GTP gather storage is aliased into distinct native wrappers over
                # the replica views before use, instead of a second full copy.
                source_tensors = (
                    _wrap_mxfp8(spec, virtual_storage, self.device) if gtp else spec.source_tensors
                )
            else:
                virtual_weight = virtual_storage
                source_tensors = () if gtp else spec.source_tensors
            forward = binding(gtp)
            backward = binding(gtp) if gtp or mxfp8 else forward
            self.projections.append(
                _ReplicaProjection(
                    name=f"FC{projection_index + 1}",
                    device=self.device,
                    weight_format=spec.weight_format,
                    parameters=spec.parameters,
                    gtp_leader=spec.gtp_leader,
                    source_tensors=source_tensors,
                    forward=forward,
                    backward=backward,
                    native_grad_bases=pointer_table(),
                    member_shape=spec.member_shape,
                    rowwise_scale_shape=spec.rowwise_scale_shape,
                    columnwise_scale_shape=spec.columnwise_scale_shape,
                    virtual_weight=virtual_weight,
                    virtual_grad=virtual_grad,
                    native_grad=self.workspace.native_projection_grad_view(projection_index),
                )
            )
        _replica_weight_bridges.add(self)

    @property
    def runtime_fc1_weights(self) -> tuple[torch.nn.Parameter, ...]:
        """Return stable native-then-virtual FC1 runtime parameters."""
        return self._runtime_weights(0)

    @property
    def runtime_fc2_weights(self) -> tuple[torch.nn.Parameter, ...]:
        """Return stable native-then-virtual FC2 runtime parameters."""
        return self._runtime_weights(1)

    def _runtime_weights(self, projection_index: int) -> tuple[torch.nn.Parameter, ...]:
        runtime_parameters = self.projections[projection_index].runtime_parameters
        if runtime_parameters is None:
            raise RuntimeError("Replica runtime weights were accessed before binding.")
        return runtime_parameters

    @property
    def source_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        """Return the optimizer-owned FC1 and FC2 parameters."""
        return tuple(parameter for p in self.projections for parameter in p.parameters)

    def prepare_runtime_parameters(self) -> None:
        """Late-bind final DDP/GTP storage and validate subsequent stability."""
        for projection in self.projections:
            projection.prepare_runtime_parameters(self.workspace.grad_dtype)

    def prepare_source_weights(self, direction: _WeightDirection) -> None:
        """Make plain weights ready and peek at GTP gathers for the transport push."""
        experts = self._experts_ref()
        if experts is None:
            raise RuntimeError("Replica experts were destroyed before prefetch.")
        experts.prepare_fused_impl_parameters()
        # Expert backward computes FC2 before FC1; keep GTP's linked gathers in
        # the order they will be consumed.
        backward = direction is _WeightDirection.BACKWARD
        for projection in reversed(self.projections) if backward else self.projections:
            leader = projection.gtp_leader
            if leader is None:
                continue
            peek = getattr(
                leader,
                "peek_group_for_backward" if backward else "peek_group_for_forward",
                None,
            )
            if callable(peek):
                materialized = peek()
            else:
                # Compatibility with GTP revisions predating the non-consuming
                # peek protocol. Those revisions consume at push time.
                materialized = (
                    leader.materialize_group_for_backward()
                    if backward
                    else leader.materialize_group_for_forward()
                )
            if not isinstance(materialized, (list, tuple)):
                materialized = (materialized,)
            projection.bind_materialized_weights(tuple(materialized), direction)
        self.prepare_runtime_parameters()

    def consume_source_weights(self, direction: _WeightDirection) -> None:
        """Consume GTP gathers at the GEMMs after the transport-only peek."""
        backward = direction is _WeightDirection.BACKWARD
        projections = reversed(self.projections) if backward else self.projections
        for projection in projections:
            leader = projection.gtp_leader
            if leader is None:
                continue
            peek = getattr(
                leader,
                "peek_group_for_backward" if backward else "peek_group_for_forward",
                None,
            )
            if not callable(peek):
                continue
            materialized = (
                leader.materialize_group_for_backward()
                if backward
                else leader.materialize_group_for_forward()
            )
            if not isinstance(materialized, (list, tuple)):
                materialized = (materialized,)
            # bind_materialized_weights verifies that the consuming gather uses
            # the same stable buffers read by the push.
            projection.bind_materialized_weights(tuple(materialized), direction)

    def consume_forward_source_weights(self) -> None:
        """Consume forward GTP gathers immediately before expert GEMMs."""
        self.consume_source_weights(_WeightDirection.FORWARD)

    def _validate_plan(self, plan: ReplicaPlan) -> None:
        """Validate fixed device metadata without extracting any CUDA values."""
        experts_to_copy = plan.experts_to_copy
        expected_shape = (self.world_size, self.num_local_replica_slots)
        if (
            experts_to_copy.dtype != torch.int32
            or experts_to_copy.device != self.device
            or tuple(experts_to_copy.shape) != expected_shape
            or not experts_to_copy.is_contiguous()
        ):
            raise ValueError(
                "Replica experts_to_copy must be contiguous int32 with shape "
                f"{expected_shape} on {self.device}."
            )

    @torch.no_grad()
    @nvtx_decorator(message="replica_weight_push_start")
    def start_prefetch(
        self, plan: ReplicaPlan, direction: _WeightDirection = _WeightDirection.FORWARD
    ) -> None:
        """Enqueue the owner push of FC1/FC2 weights without blocking the caller."""
        if self._prefetch_plan is not None:
            raise RuntimeError("Replica weight prefetch is already outstanding.")
        self._validate_plan(plan)
        # A GTP parameter stores only its local shard. Materialization consumes
        # any one-weight-ahead gather (or performs the cold synchronous gather)
        # and stages the full native experts before the push reads them.
        self.prepare_source_weights(direction)
        workspace = self.workspace
        bindings = tuple(projection.binding(direction) for projection in self.projections)
        current_stream = torch.cuda.current_stream(self.device)
        weight_stream = workspace.select_weight_stream(current_stream)
        weight_stream.wait_stream(current_stream)
        with torch.cuda.stream(weight_stream):
            launch_replica_weight_prefetch(
                sources=tuple(binding.data_bases for binding in bindings),
                scale_sources=(
                    tuple(binding.scale_bases for binding in bindings)
                    if self.weight_format == "mxfp8"
                    else None
                ),
                arena=workspace.weight_arena,
                peer_bases=workspace.weight_handle.buffer_ptrs_dev,
                signal_bases=workspace.weight_handle.signal_pad_ptrs_dev,
                experts_to_copy=plan.experts_to_copy,
                grid_barrier=workspace.weight_grid_barrier,
                rank=self.rank,
                world_size=self.world_size,
                num_local_home_experts=self.num_local_home_experts,
                num_local_replica_slots=self.num_local_replica_slots,
                member_numels=workspace.member_numels,
                num_sms=workspace.num_sms,
            )
            self.prefetch_done.record(weight_stream)
        self._prefetch_plan = plan

    @torch.no_grad()
    @nvtx_decorator(message="replica_weight_push_wait")
    def wait_prefetch(self, plan: ReplicaPlan) -> None:
        """Make the current stream wait for the outstanding push of ``plan``."""
        if self._prefetch_plan is None:
            # A repeated wait for the plan already resident is a no-op; anything
            # else means the planner never started the transport.
            if plan is None or plan is not self._completed_plan:
                raise RuntimeError("Replica weights require a started prefetch before use.")
        elif self._prefetch_plan is not plan:
            raise RuntimeError("Replica weight prefetch plan changed while outstanding.")
        torch.cuda.current_stream(self.device).wait_event(self.prefetch_done)
        self._completed_plan = plan
        self._prefetch_plan = None

    def wait_prefetch_for_backward(self, plan: ReplicaPlan) -> None:
        """Wait for the backward push and bind its plan to expert backward."""
        self.wait_prefetch(plan)
        self.consume_source_weights(_WeightDirection.BACKWARD)
        self._backward_plan = plan

    @torch.no_grad()
    @nvtx_decorator(message="replica_grad_reduce_start")
    def start_grad_reduce(self, plan: ReplicaPlan, projection: int) -> None:
        """Enqueue one projection's replica-gradient reduction."""
        if self._grad_reduce_plan is not None and self._grad_reduce_plan is not plan:
            raise RuntimeError("Replica gradient reduction is outstanding for another plan.")
        if projection in self._grad_reduce_started:
            raise RuntimeError(f"Replica gradient reduction of FC{projection + 1} started twice.")
        self._validate_plan(plan)
        workspace = self.workspace
        current_stream = torch.cuda.current_stream(self.device)
        workspace.grad_stream.wait_stream(current_stream)
        with torch.cuda.stream(workspace.grad_stream):
            launch_replica_grad_reduce(
                arena=workspace.grad_arena,
                native_grads=tuple(projection.native_grad_bases for projection in self.projections),
                peer_bases=workspace.grad_handle.buffer_ptrs_dev,
                signal_bases=workspace.grad_handle.signal_pad_ptrs_dev,
                experts_to_copy=plan.experts_to_copy,
                grid_barrier=workspace.grad_grid_barrier,
                rank=self.rank,
                world_size=self.world_size,
                num_local_home_experts=self.num_local_home_experts,
                num_local_replica_slots=self.num_local_replica_slots,
                member_numels=workspace.member_numels,
                num_sms=workspace.num_sms,
                projections=(projection,),
            )
            self.grad_reduce_done[projection].record(workspace.grad_stream)
        self._grad_reduce_plan = plan
        self._grad_reduce_started.add(projection)

    def start_fc2_grad_reduce(self) -> None:
        """Start FC2 reduction immediately behind its wgrad GEMM."""
        if self._backward_plan is None:
            raise RuntimeError("Replica FC2 gradient reduction needs the backward plan.")
        self.start_grad_reduce(self._backward_plan, 1)

    def start_pending_grad_reduces(self, plan: ReplicaPlan) -> None:
        """Start reductions not already issued by expert backward, FC2 first."""
        if self._grad_reduce_plan is not None and self._grad_reduce_plan is not plan:
            raise RuntimeError("Replica gradient reduction is outstanding for another plan.")
        for projection in (1, 0):
            if projection not in self._grad_reduce_started:
                self.start_grad_reduce(plan, projection)

    @torch.no_grad()
    @nvtx_decorator(message="replica_grad_reduce_wait")
    def wait_grad_reduce(self, plan: ReplicaPlan) -> tuple[torch.Tensor | None, ...]:
        """Finish both projection reductions and return source-parameter wgrads."""
        if self._grad_reduce_plan is not plan or self._grad_reduce_started != {0, 1}:
            raise RuntimeError("Replica gradient reduction of both projections must be started.")
        current_stream = torch.cuda.current_stream(self.device)
        for event in self.grad_reduce_done:
            current_stream.wait_event(event)
        self._grad_reduce_plan = None
        self._grad_reduce_started.clear()
        self._backward_plan = None

        # Expert backward computes FC2 before FC1. Preserve that reverse order
        # when handing full wgrads to GTP so its linked RS cascade remains valid.
        source_grads = [tuple(projection.native_grad) for projection in self.projections]
        for index in reversed(range(len(self.projections))):
            leader = self.projections[index].gtp_leader
            if leader is None:
                continue
            reduced = leader.wgrad_reduce_scatter(list(source_grads[index]))
            reduced = tuple(reduced) if isinstance(reduced, (list, tuple)) else (reduced,)
            if len(reduced) != len(source_grads[index]):
                raise RuntimeError(
                    "GTP returned a different number of reduced wgrads than source parameters."
                )
            source_grads[index] = reduced
        return tuple(grad for grads in source_grads for grad in grads)

    def destroy(self) -> None:
        """Detach layer-owned TE parameters from the shared symmetric arenas."""
        if self._destroyed:
            return
        experts = self._experts_ref()
        if experts is not None:
            experts._fused_ops = None
            experts._replica_weight_bridge = None
        for projection in self.projections:
            projection.destroy()
        self.projections.clear()
        self.last_plan = None
        self.workspace = None
        self._destroyed = True
        _replica_weight_bridges.discard(self)


def _wrap_mxfp8(
    spec: _ReplicaProjectionSpec, storage_views: tuple[tuple[torch.Tensor, ...], ...], device
) -> tuple[torch.Tensor, ...]:
    """Wrap raw arena views with the TE metadata of the matching source weights."""
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    return tuple(
        MXFP8Tensor(
            shape=spec.member_shape,
            dtype=source.dtype,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale,
            fp8_dtype=source._fp8_dtype,
            quantizer=source._quantizer,
            with_gemm_swizzled_scales=source._with_gemm_swizzled_scales,
            requires_grad=False,
            device=device,
        )
        for source, (rowwise_data, rowwise_scale, columnwise_data, columnwise_scale) in zip(
            spec.source_tensors, storage_views
        )
    )


def finalize_replica_weight_bridges() -> None:
    """Release replica weight contexts before their process group is destroyed."""
    from megatron.core.transformer.moe.moonep_moe_scheduler import (
        finalize_moonep_planner_workspaces,
    )

    workspaces = list(_replica_weight_workspaces.values())
    for bridge in list(_replica_weight_bridges):
        bridge.destroy()
    for workspace in workspaces:
        workspace.destroy()
    _replica_weight_workspaces.clear()
    finalize_moonep_planner_workspaces()
    # NCCLSymmetricMemory handles contain Python reference cycles. Collect them
    # now so their window deregistration runs before the process group is gone.
    gc.collect()


class _ReplicaBackwardHook(torch.autograd.Function):
    """Run one communication boundary while passing its tensor gradient through."""

    @staticmethod
    def forward(ctx, tensor, hook):
        ctx.hook = hook
        return tensor

    @staticmethod
    def backward(ctx, grad):
        ctx.hook()
        return grad, None


class _ReplicaWaitGradReduce(torch.autograd.Function):
    """Finalize replica gradients after all layer-input consumers ran backward."""

    @staticmethod
    def forward(ctx, hidden_states, *args):
        bridge, context = args[-2:]
        ctx.bridge = bridge
        ctx.context = context
        ctx.num_source_parameters = len(args) - 2
        return hidden_states

    @staticmethod
    def backward(ctx, grad_hidden_states):
        source_grads = ctx.bridge.wait_grad_reduce(ctx.context.plan)
        if len(source_grads) != ctx.num_source_parameters:
            raise RuntimeError(
                "Replica reduction returned a different number of wgrads than source parameters."
            )

        autograd_grads = []
        for parameter, source_grad in zip(ctx.bridge.source_parameters, source_grads):
            if source_grad is None or getattr(parameter, "is_gtp_weight_remat", False):
                autograd_grads.append(source_grad)
                continue

            main_grad = getattr(parameter, "main_grad", None)
            if main_grad is None or not hasattr(parameter, "grad_added_to_main_grad"):
                # AccumulateGrad may retain its input as parameter.grad. Give it
                # independent storage because the bridge reuses native staging.
                autograd_grads.append(source_grad.clone())
                continue

            if get_dummy_wgrad is None:
                raise RuntimeError("Replica fused wgrad accumulation requires Transformer Engine.")
            # Accumulate the completed wgrad directly in main_grad's dtype. Return a
            # parameter-dtype dummy so AccumulateGrad still invokes DDP's grad-ready hook;
            # grad_added_to_main_grad prevents DDP from accumulating the dummy again.
            main_grad.add_(source_grad)
            parameter.grad_added_to_main_grad = True
            autograd_grads.append(
                get_dummy_wgrad(
                    list(parameter.shape),
                    parameter.dtype,
                    zero=getattr(parameter, "zero_out_wgrad", False),
                )
            )

        return grad_hidden_states, *autograd_grads, None, None


def start_replica_weight_prefetch_before_combine_backward(
    combined_hidden: torch.Tensor, bridge: ReplicaWeightBridge, plan: ReplicaPlan
) -> torch.Tensor:
    """Start weight communication before transport-combine backward."""
    return _ReplicaBackwardHook.apply(
        combined_hidden, functools.partial(bridge.start_prefetch, plan, _WeightDirection.BACKWARD)
    )


def wait_replica_weight_prefetch_before_expert_backward(
    expert_output: torch.Tensor, bridge: ReplicaWeightBridge, plan: ReplicaPlan
) -> torch.Tensor:
    """Wait for weight communication immediately before expert backward."""
    return _ReplicaBackwardHook.apply(
        expert_output, functools.partial(bridge.wait_prefetch_for_backward, plan)
    )


def start_replica_grad_reduce_after_dispatch_backward(
    dispatch_input: torch.Tensor, bridge: ReplicaWeightBridge, plan: ReplicaPlan
) -> torch.Tensor:
    """Start all still-pending reductions after dispatch backward."""
    return _ReplicaBackwardHook.apply(
        dispatch_input, functools.partial(bridge.start_pending_grad_reduces, plan)
    )


def wait_replica_grad_reduce_at_layer_input(
    hidden_states: torch.Tensor, bridge: ReplicaWeightBridge, context
) -> torch.Tensor:
    """Wait after router, shared-expert, and latent-projection backward work."""
    return _ReplicaWaitGradReduce.apply(hidden_states, *bridge.source_parameters, bridge, context)
