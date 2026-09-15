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

## First GPU attempt and eager retry

The initial Lyris A/B runs at `7340ce87c` did not establish correctness:
OFF (3061333) logged one finite loss with NaN gradient norm, then NaN loss;
ON (3061339) failed the pre-CP one-dimensional THD token check before logging
an iteration. Neither run produced valid steady-state performance measurements.

The eager retry disables CUDA Graph and activation paged stashing in both recipes.
OFF also removes graph-only expert capacity padding. ON retains its required
virtual-expert rank budget and the overflow-checking runner (this runner is also
used when activation paged stashing is disabled).

Two integration fixes accompany this retry:

- Training now calls the overload tracker's `report()` on every rank when enabled,
  forwarding TensorBoard/W&B writers and per-layer logging, and appending its
  summary to the training log. Previously counters were collected but not reported.
- `PagedStashRunner.data_read` snapshots microbatch containers before each attempt.
  THD preparation replaces dictionary entries during CP slicing and reshaping;
  the saved retry input must retain the original one-dimensional packed layout.
  This is a shallow container copy, not a tensor clone or a disabled shape guard.

Regression tests cover enabled/disabled overload reporting and single-/multi-chunk
THD batch replay. At this stage GPU parity and successful training were not established.
Host-only smoke execution of the changed function bodies passed these checks;
this is not a run of the GPU/distributed pytest harness. Python compilation and
patch whitespace checks also passed.

## Eager retry follow-up

The eager OFF run reached finite first-step loss/gradients but overload reporting
failed because MTP depth 1 reused decoder layer 1's metric key (10 samples in
only 4 layer slots). Overload recording now uses decoder-offset MTP depth IDs,
including the depth provided by repeated/hybrid MTP routers.

The installed DeepEP `10d4dd7` ragged handle places `num_of_valid_tokens` at
index 10 and keeps `overflow_flag` last. The port incorrectly read index 10,
causing a false capacity retry on nonempty inputs. The overflow index now follows
the trailing-field contract, with regression coverage for legacy and ragged
handles and both overflow states. The preceding ON run's NaN was not considered
resolved solely by this API fix; the GPU validation below was required.
Host-only execution of the changed numbering function passed five cases; handle
index checks passed all four legacy/ragged and overflow/no-overflow combinations.
This is not a run of the distributed pytest suite.

## GPU validation of the eager fixes

Code commit `c827c106bd986535a11ca0ff65a4fc5411b0076a` completed both
Lyris jobs: OFF `3061947`, ON `3061959`, each 100/100 steps and SLURM exit 0:0.
Both retain 2 nodes x 4 GB200, TP1/PP1/EP8/CP4, MXFP8, THD16K,
4 decoder layers + 1 MTP, 128 experts, no forced router balance, no CUDA Graph,
and no activation paged stash. The image and DeepEP dependency were unchanged.

All 100 W&B samples of LM loss, MTP loss, gradient norm, and the three overload
metrics were finite in both runs. The previous NaN did not recur. Overload
logging has five distinct layer slots; existing W&B keys remain zero-based
(`_layer_0` through `_layer_4`, with MTP last).

| Metric | OFF | ON |
| --- | --- | --- |
| Final LM loss | 0.03506108 | 0.03461639 |
| Final gradient norm | 0.4211009 | 0.4201143 |
| Median iteration time, excluding first 8 | 317.4 ms | 347.3 ms |
| Median TFLOP/s/GPU, excluding first 8 | 326.7 | 298.6 |
| Mean avg_overload_factor, 100 steps | 1.112188 | 1.087938 |
| Mean max_overload_factor, 100 steps | 1.156771 | 1.118958 |
| Mean max_cum_overload_factor, 100 steps | 1.084146 | 1.079646 |

The LM loss curves have mean absolute difference 0.006046 and maximum absolute
difference 0.019493 (step 35); maximum relative difference is 2.395% (step 69).
This is a passing proxy functional/stability check, not bitwise or independent
kernel numerical parity. ON's median step is 9.42% slower in this small proxy,
despite lower overload; no performance improvement is claimed.

W&B project: `megatron-core-moe-dev/kuns-deepseek-v4-flash-proxy-gb200-moonep-pr6892-ab`.
Run IDs: OFF `dde8c76f91f64c8f84938e5278d28c14`,
ON `fd667ec10fc04945a558e07b619f42c0`.

## Padding-aware router CUDA Graph compatibility

Baseline job 3062611 reached two finite warmup steps but failed while capturing
`attn,moe_router,moe_preprocess`. The first unsupported operation was boolean
row selection in `TopKRouter._apply_expert_bias` for index-form routes:
`routing_map[~flat_mask]` creates a data-dependent shape and synchronizes.

Index-form routes now retain their fixed shape. Padded rows use safe expert
index zero and contribute zero count before the existing deterministic
`index_add_` or ordinary `scatter_add_`. Valid routes and repeated microbatch
accumulation are unchanged; padded sentinel indices cannot reach the scatter.
The input route tensor is not mutated. The existing boolean-map and MoonEP
conversion paths are unchanged. Padding validation and expert-bias updates
remain enabled; no graph scope, precision or router setting is disabled.

Dedicated tests cover absent/empty/mixed/all padding, ignored sentinel indices,
both accumulation modes, input immutability, repeated accumulation, and CUDA
Graph replay with changing padding contents at fixed addresses. GPU verification
is pending; adding these tests alone does not establish capture compatibility.
Host-only execution of the changed function passed eight counting, immutability
and repeated-accumulation cases. Compilation, isort, Black and patch whitespace
checks passed. The registered `mcore-ci-dev` image is x86_64, not compatible with
Lyris GB200's ARM hosts, so CI pytest execution is not claimed. The original
training image will be used for the model-level CUDA Graph integration check.
