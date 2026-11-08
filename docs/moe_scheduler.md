# MoE scheduler contracts

This document describes the backend and runtime stages of the MoE scheduler. The
existing MoonEP path (`moe_virtual_expert_load_balance`) continues to use its
current implementation and configuration.

`megatron/core/transformer/moe/moe_scheduler.py` defines an optional, backend
neutral boundary between route planning and expert placement:

1. A `MoELoadPlanner` returns a physical replica source table and opaque state.
2. An `ExpertDispatch` materializes those replicas before token dispatch.
3. The planner reroutes tokens to the physical expert IDs.
4. `MoEScheduler` validates the output and exposes capacity bounds to the token
   dispatcher. The caller finalizes per-forward dispatcher state afterward.

`MoESchedulerConfig` validates planner and transport names, expert counts, EP
size, replica slot counts, and the NCCL scratch budget. It does
not change `TransformerConfig` or enable a training path. A later integration
will translate the training configuration into this value when a concrete
planner is connected to `MoELayer`.

The source table has shape `[EP, local replica slots]` and integer dtype. It
contains physical source slots, not logical expert IDs. A dense rerouted map
has shape `[tokens, logical experts + replica slots]`; compact integer routes
are also accepted. The associated probabilities must have the same shape as
the map. Capacity bounds are host integers so allocation does not synchronize
the GPU.

## Replica execution

`replica_weight_transport.py` owns the transport interface and the immutable
placement, ownership, and transfer-handle records. The NCCL implementation in
`replica_nccl_transport.py` packs owner weights and returns replica gradients
through bounded scratch storage. A placement is a physical source-slot table;
it must remain stable until its backward finishes. Transport handles retain
their buffers until communication and unpacking have completed.

`replica_storage.py` fences consumers of shared storage before a later layer
may overwrite it. Replica weight prefetch runs on each forward and backward;
there is no layerwise replica cache or private per-layer weight arena. MXFP8
rowwise and columnwise views reuse the same storage.
The MXFP8 transport test checks the active direction after each sync; the
inactive view aliases that storage and does not retain an earlier generation.

`ReplicaExpertRuntime` binds native TE weights and transient replica slots,
starts weight prefetches, and adds replica gradients to the native owner before
the normal optimizer hooks run. `ReplicaExpertDispatch` attaches autograd
boundaries for prefetch, reduction, and placement-slot release. Multiple
microbatches retain distinct placement slots through backward. The TE grouped
MLP wrapper uses runtime weights only when a replica runtime is explicitly
bound; the existing MoonEP wrapper path remains separate.

The Peer-TMA transport, planner implementations, and `MoELayer`/CLI integration
are later stages. This branch can validate transport contracts and lifecycle
ordering independently, but it does not start a scheduler training run.
