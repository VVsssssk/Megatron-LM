# MoonEP virtual experts: non-GTP dev port

## Provenance and scope

- Dev base: `bb5dfd08f09ce06c5925af453fef06b3129f199d`.
- Source: NVIDIA/Megatron-LM PR #6892,
  `186b388abc6da2d58a1e0a3db47b6fc64819de7b`.
- Branch: `feat/moonep-pr6892-dev-no-gtp`.

This is a selective port, not a full merge of the main-targeting PR history.
The planner, Triton transport, unsharded BF16/MXFP8 runtime weights, gradient
reduction, expert op-fuser hooks, compact routing, and whole-MoE CUDA Graph support
are retained. GTP-specific implementation and tests are not imported.

No files under `megatron/core/tensor_parallel/` or
`tests/unit_tests/generalized_tensor_parallel/` change relative to the dev base.
The shared virtual-expert runtime has no GTP gather/peek/consume, persistent-wgrad,
or reduce-scatter integration. It rejects GTP-marked parameters before allocating
shared arenas. Existing dev code is not removed.

## Dev adaptations

- Preserve dev's fused-router keyword API and histogram-based quantile balancing;
  compact semantic indices use int64, independently of HybridEP's int16 wire ids.
- Preserve Hash-MoE input IDs and packed sequence metadata through the layer and
  router; return compact selected IDs/probabilities to virtual-expert planning.
- Preserve dev's expert-TP-expanded `router_topk` interface for ordinary dispatch;
  virtual experts require expert TP 1.
- Preserve `moe_hybridep_pad_variable_tokens`, existing dense-routing capability
  detection, and `num_of_experts` dispatch keyword.
- Preserve dev's static-capacity buffer lifetime and overload logging, rather
  than replacing these with old main implementations.
- Extend the shared CUDA Graph validator for virtual experts while retaining
  the ordinary HybridEP/drop-padding rules and rejecting partial MoE capture.
- Retain shared-expert scheduling and fused module post-forward hooks from dev.

Non-GTP planner, runtime-storage, kernel, full-layer parity, and configuration
tests are ported. Tests relying on main's old quantile algorithm are not imported:
dev retains its own histogram implementation and tests. GTP-only tests embedded
in otherwise shared test files are excluded; applicable storage tests use plain
parameters instead. A regression check covers explicit GTP rejection, and another
covers Hash-MoE compact IDs and differentiable probabilities.

## Verification performed locally

- Python 3.12 compilation of transformer sources and transformer tests: passed.
- isort and Black (Python 3.12 target): passed.
- `git diff --check`: passed.
- Tensor-parallel and GTP test trees unchanged from the dev base: verified.
- CPU API smoke: eager/local/TE graph config construction, compact-route expansion,
  routing-probability gradients, Hash-MoE gradients, and GTP rejection: passed.
- Fused-router int64 API contract checked on CPU with a mocked TE function;
  this does not validate the actual fused GPU kernel.
- Focused F821/F811 lint found only the two pre-existing unresolved type annotation
  names (`SummaryWriter`, `wandb`) in `moe_utils.py`; verified on the unmodified dev
  base as well. No new findings in the ported files.

GPU unit tests, multi-rank numerical parity, CUDA Graph replay, and training
throughput have not been run for this branch. Local checks do not establish GPU
correctness. The previous experiment results are not results for this port.
