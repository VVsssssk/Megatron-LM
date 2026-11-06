# MoE scheduler contracts

This document describes the first stage of the MoE scheduler integration. The
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
size, and replica slot counts without changing `TransformerConfig` or enabling
an unfinished training path. A later integration will translate the training
configuration into this value once a concrete planner and transport are ready.

The source table has shape `[EP, local replica slots]` and integer dtype. It
contains physical source slots, not logical expert IDs. A dense rerouted map
has shape `[tokens, logical experts + replica slots]`; compact integer routes
are also accepted. The associated probabilities must have the same shape as
the map. Capacity bounds are host integers so allocation does not synchronize
the GPU.
