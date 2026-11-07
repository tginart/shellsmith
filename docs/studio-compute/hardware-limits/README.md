# Studio compute and bandwidth investigation

The target is to explain the machine limits for this model, then approach those
limits without a change to the training objective, precision, or context. A fixed
speedup factor is not the acceptance criterion. This investigation is in progress.

The isolated worktree starts at `631f79c44da807775aa04da2e266f750a500e628`.
The original dirty checkout, stopped experiment, model assets, and historical
checkpoints remain preserved. All GPU controls hold the existing exclusive lease.
Each worker has a supervisor and a hard limit of at most one hour.

## Evidence and interpretation

- Studio reports an Apple M5 Ultra, 64 GPU cores, and 256 GiB unified memory.
  [Apple specifies 1.2 TB/s](https://www.apple.com/mac-studio/specs/) for this
  configuration. This is a physical bandwidth reference, not an achieved rate.
- Initial warm streaming controls reached approximately 1.09 TB/s of useful
  loads plus stores on 4 GiB arrays. These are useful byte counts, not hardware
  DRAM counters. The working set exceeds small on-chip caches.
- Large BF16 GEMMs reached approximately 97 dense-equivalent TFLOP/s. This is
  an empirical reference, not a proven physical arithmetic maximum. Count each
  multiply-add as two FLOPs. Smaller shapes need their own reference.
- Strict FP32 GEMMs reached approximately 23 TFLOP/s. The MLX default permits
  TF32 acceleration and reached approximately 79 TFLOP/s for FP32 arrays.
  The qualified Shellsmith service uses `MLX_ENABLE_TF32=0`; this setting must
  remain explicit in matched comparisons.
- Production-sized 8-bit GEMMs with 2048 rows reached 71–79 dense-equivalent
  TFLOP/s. Quantization instructions are excluded from this useful FLOP count.
  Isolated matvec rates include synchronization and may reuse cached weights;
  they do not establish full-model decode bandwidth.
- MLX 0.32.3's Metal implementation always falls back for attention VJP.
  Its D256 forward heuristic also rejects the small production query tiles.
  See the pinned [Metal dispatch source](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/scaled_dot_product_attention.cpp).
  Previous same-device graphs and controls confirm the actual dispatch.

Initial controls live under the chat artifact directory `mlx-hardware-limits/`.
Their raw timings, environment, device facts, execution receipts, and source
identities are separate from interpretation. The first inference diagnostic used
the MLX default precision and failed at its first batch-of-four report because
the harness omitted the `batched_decode` progress event. Its single-request
measurements are partial accelerated controls. They are not a qualified baseline.
The corrected baseline pins strict precision and handles both progress events.

## Cost model

Inventory the actual resident projections rather than use the model's rounded
27B label. Their 496 frozen projections contain 24,350,556,160 dense-equivalent
parameters in 25,872,465,920 stored bytes, including quantization metadata.
The frozen vocabulary head, embedding accesses, norms, recurrent state, KV
traffic, and adapter factors need separate accounts.

For `T` decoder tokens and `P` frozen projection parameters:

- Necessary projection forward and input backward: `4 T P` FLOPs. A frozen base
  needs an input gradient, but no base-weight gradient.
- Each complete recomputed projection forward adds `2 T P` FLOPs. Count actual
  graph use: some primal outputs can be pruned during a VJP.
- Necessary causal attention forward and backward:
  `12 layers heads head_dim T(T+1)/2` GEMM FLOPs. Add softmax, masks, layout
  operations, rectangle padding, and recomputation separately.
- Necessary frozen head forward and input backward: `4 S vocab width` FLOPs for
  `S` supervised positions. Account for head recomputation separately.
- LoRA forward and backward: `6 T rank sum(input_width + output_width)` FLOPs.
  Recurrent GDN arithmetic uses its own FP32 and bandwidth accounts.

Divide useful FLOPs and bytes by appropriate reference rates, then take the
larger time for each operation. Sum only operations that must execute in sequence.
Do not treat a whole-update FLOP ratio as measured GPU utilization. Do not add
VJP graph-build and evaluation subsets to the already inclusive VJP total.

For the preserved 57,538-token path, projections, attention, and the frozen head
alone require approximately 7.67 PFLOPs without recomputation. At the measured
large BF16 rate this corresponds to about 79 seconds of arithmetic. This is an
optimistic reference budget, not an achievable update-time prediction: it omits
LoRA, GDN, synchronization, layout, softmax, and memory effects, and uses an
empirical rate instead of a proven upper bound. The measured checkpoint-1 update
of 339 seconds still leaves a material gap to explain.

## Remaining work

1. Complete the production GDN candidate's matched full-batch gradient and
   optimizer comparison. The faster native attention candidate failed its
   numerical gates and remains disabled.
2. Complete inference phase attribution and test useful changes against actual
   sampled generation, prefix reuse, and the whole-weight reference.
3. Explain the remaining component-to-model gap and relate the final phase times
   to shape-matched empirical references.
4. Finish the retained-source audit, reproducible report, and focused PR.

The older High Command FlashAttention bridge is a research lead, not an accepted
replacement: its own qualification document reports failed historical BF16
parity and unresolved full-model gates. It cannot justify a production kernel
choice or a speed claim here.

## Completed controls

The corrected fixed-token inference baseline uses the actual preserved path,
fresh BF16 caches, seed-23 adapters, one warmup, and two measured repetitions.
Frozen-base hashes matched before and after the complete process. Each request
has 64 fixed completion tokens; the decode rate counts the 63 decoder advances.
These are in-process service-engine measurements, not HTTP latency.

| Prompt | Requests | Prefill seconds | Aggregate decode tokens/s |
| --- | ---: | ---: | ---: |
| 2,048 | 1 | 1.56 | 25.85 |
| 8,192 | 1 | 6.34 | 25.44 |
| 32,768 | 1 | 28.07 | 23.70 |
| 8,192 | 4 | 24.08 | 76.97 |

The vocabulary head occupies 1,350,860,800 stored bytes. The projection and
head weights alone total 27,223,326,720 bytes per model pass. Account for KV,
FP32 recurrent state, adapter reads, and activation traffic separately.

The command-buffer control (`MLX_MAX_OPS_PER_BUFFER=500`,
`MLX_MAX_MB_PER_BUFFER=1024`) was rejected. Decode became approximately 1.8%
slower at both batch sizes. Peak active memory increased by 15.10 GiB for one
request and 12.72 GiB for four. Behavior logprobs were identical.

Independent FP32 register-FMA controls reached 22.56 TFLOP/s. All sampled
outputs matched the CPU fused-operation reference exactly. Too few independent
chains exposed dependency latency; too many chains reduced throughput. This
supports the approximately 23 TFLOP/s strict-FP32 GEMM reference without an
assumption about an undocumented GPU clock.

At the actual GDN recurrence dimensions, complete forward-plus-VJP compilation
reduced the median from 20.43 to 16.72 ms. All measured output, state, and six
input-gradient errors were zero. Compiling only the primal reduced forward time
from 4.81 to 3.71 ms and left VJP time unchanged. This is an isolated recurrence
result, not an established full-update improvement.

At query length 2,048, key length 57,344, 24 query heads, four KV heads, and
head width 256, the actual attention tile GEMMs reached 54.4–55.0 TFLOP/s.
Comparing this shape against the 97 TFLOP/s large square GEMM alone would
overstate the attainable rate. Compiling the entire attention VJP reduced
550.99 to 541.08 ms while peak memory increased by 5.91 GiB; it was not selected.

Removing only the inner attention checkpoints retains the same query partitions
and arithmetic. Three measured repetitions followed one warmup for each case:

| Keys | Existing VJP ms | Direct VJP ms | Existing peak GiB | Direct peak GiB |
| ---: | ---: | ---: | ---: | ---: |
| 57,344 | 548.78 | 473.57 | 12.99 | 12.99 |
| 262,144 | 5,232.72 | 4,770.43 | 56.71 | 56.26 |

Raw gradient byte hashes matched at both lengths. These are attention-operation
peaks with no loaded model or retained reference gradients. Full-context
attention coverage does not establish full-model training qualification at that
length. The complete-path follow-up uses fresh processes in the order existing,
direct, direct, existing; all runs retain checkpoint interval one. Each process
performs a native full-path warmup and then measures one full update. A leased
coordinator enforces a one-hour deadline and compares all canonical gradients.

## Projection work and memory constraints

The read-only graph tool now traces both the regular 2,048-token replay unit and
its actual 194-token terminal unit. The terminal trace omits absent future carry
cotangents. It uses the actual-width four-layer representative decoder, frozen
8-bit projections, and canonical adapters. It loads no model assets and evaluates
no model arrays. CPU-default lazy graphs contain two explicitly GPU-specific
RoPE helper nodes; the tool does not execute them. These are typed graph facts,
not Metal instruction or DRAM counter measurements.

The final source-bound traces reproduce the earlier primitive counts and matrix
FLOP counts. Their reader extension, source files, runtime versions, inventory,
and tracer have recorded identities. The projection budget independently checks
its forward total against all 496 projections in the resident-model inventory.
For the complete longest path:

| Quantity | Dense-equivalent FLOPs |
| --- | ---: |
| Full-row projection forward/input-gradient reference | 5.604 PFLOPs |
| Executed projection graph, checkpoint interval one | 7.721 PFLOPs |
| Additional projection replay | 2.133 PFLOPs |

This is 2.75523 forward-equivalents after actual dead-edge pruning. The VJP
prunes the MLP down-projection primal. The final attention layer also has 14
prompt-only units with absent hidden cotangents. Their carry-only graph omits the
MLP and attention products, but still retains some Q-projection work. The budget
uses that observed graph rather than assuming ideal dead-code removal.

The independent check starts with `6 T P - 2 T P_down` and subtracts the measured
full-VJP/carry-only difference for those 14 final-layer units. Executed forward
and input-gradient graphs total 5.587 PFLOPs; additional primal replay contributes
2.133 PFLOPs. A mathematical projection minimum with sparse final-layer query,
output, and MLP rows is 5.552 PFLOPs. This minimum is not the implemented schedule.
Quantization instructions, padding, and non-matrix operations remain additional.

The recipe plan contains 32 distinct forward/input-gradient operations and 41,184
calls across all replay units and layers. It preserves actual BF16 metadata and
checks both `transpose=True` and `transpose=False` dimensions. The latter has not
been characterized by the initial forward-only projection controls. Its matched
measurement remains pending. The new control also compares a dequantized BF16
GEMM and records output error and additional weight memory. That comparison does
not qualify a dequantized training cache or change the production base.

Saving both BF16 MLP gate/up intermediates for every token and layer would require
256,415,105,024 bytes, approximately 238.8 GiB. Existing BF16 hidden boundaries
alone require another 35.1 GiB. Together they already exceed this machine's
256 GiB, before model weights, carries, optimizer state, or scratch. Thus a plan
that eliminates every repeated projection by retaining all intermediates is not
feasible for this workload.

The full-row causal attention reference is 1.953 PFLOPs; the mathematical
minimum with only supervised final-layer queries is 1.900 PFLOPs. Its 92 query
tiles per layer introduce only 1.65% extra rectangular pairs. The larger cost is
replay: units with a hidden cotangent have nine fallback matrix products across
forward and VJP, versus six necessary products. Removing the inner checkpoint
reduces nine to eight. The carry-only final-layer units omit attention VJP
products. After that correction, graph-equivalent totals are 2.940 and 2.615
PFLOPs. Metal fused forward kernels can skip causal cells and use padded tiles,
so these totals do not claim exact hardware instruction counts.

The strict baseline attributes 97.98% of its 339.90-second update to disjoint
GDN/attention forward and VJP layer timings. Inclusive VJP graph construction is
3.32 seconds; its nested primal-build measurements must not be added again.
Kernel-level costs still need shape-matched measurements before an attainable
whole-update target can be claimed.

The interrupted comparison remains preserved. Its completed strict baseline was
reused in a new leased comparison with fresh direct, direct, and baseline
processes. All four trials completed; every one of the 992 FP32 gradient tensors,
losses, and updated adapters matched bitwise. The frozen base remained unchanged.

| Variant | Measured update seconds | Mean seconds | Peak MLX GiB |
| --- | --- | ---: | ---: |
| Inner checkpoint | 339.90, 339.24 | 339.57 | 93.43 |
| Direct tile | 329.20, 329.18 | 329.19 | 93.43 |

The reduction is 10.38 seconds, or 3.06% of update time. The exact tested direct
callable is now retained in `studio_compute.numerics.ops`; query partitions,
precision, losses, and layer checkpoint settings stay the same. The existing
17-test BF16 grouped-replay/numerical suite also passed against a source snapshot
of this change on Studio. This does not complete the full-batch or inference
qualification required by the broader hardware-efficiency objective.

## Reproduction

`scripts/mlx_hardware_roofline.py --help` lists the bounded calibration,
inference, GDN, attention, shader, and checkpoint controls. Run it in the
qualified environment on Studio. Every invocation holds the existing lease,
uses a disposable worker, and records its source and process identity. Keep
`MLX_ENABLE_TF32=0` explicit for qualified comparisons; use a separate folder
when testing the default accelerated arithmetic.

`scripts/make_mlx_roofline_report.py` reads downloaded evidence and the prior
checkpoint comparison summary. It derives parameter, FLOP, and logical byte
counts and records the input hashes. Its output retains all interpretation
limits and marks the investigation as incomplete.

The CPU accounting tools are `scripts/build_mlx_graph_reader.py`,
`scripts/mlx_graph_reader.cpp`, and `scripts/trace_mlx_replay_costs.py`. The builder
uses the installed MLX headers and an isolated `nanobind==3.0.1` target. It does
not install packages into the qualified environment. Trace the regular unit and
then `--tokens 194 --key-prefix-chunks 28 --terminal` for this path's tail.

`scripts/mlx_projection_budget.py plan` combines those traces, the qualified
inference inventory, exact supervision layout, carry-only graph, and strict path
profile. Use `--carry-only` for that third trace. `calibrate` runs only after the
GPU lease is available, under a fresh supervised output folder and explicit
`MLX_ENABLE_TF32=0`. It records two warmups and seven samples by default.
`summarize` sums the measured operations by their graph multiplicities, while
retaining the distinction between a partial isolated-operation reference and a
verified complete-update time. Each command's `--help` describes its inputs.

`scripts/mlx_inference_profile.py` adds actual categorical generation at the same
four representative context/batch combinations. It records token materialization
through the engine's observer; generator yields occur after the next decode
advance and are reported separately. Its prefix controls compare cold, exact-hit,
and extended-prefix requests with the same seeds and require token/logprob parity.
Its final reference evaluates all 496 native frozen projections plus the head in
one pass. The full weight set exceeds cache, while one synchronization per pass
avoids charging an isolated synchronization to every small matvec. That reference
still omits layer dependencies, LoRA, norms, GDN, KV traffic, and token selection.
The sampled profile completed all 16 cases and 25 requests, with 1,600 actual
sampled tokens. Exact-hit and extended-prefix tokens and all logprobs matched
cold requests exactly. Frozen base hashes matched; swap remained 95.75 MiB.

| Prompt | Requests | True TTFT seconds | Aggregate decode tokens/s | Peak MLX GiB |
| --- | ---: | ---: | ---: | ---: |
| 2,048 | 1 | 1.550 | 25.97 | 32.06 |
| 8,192 | 1 | 6.345 | 25.43 | 32.48 |
| 32,768 | 1 | 28.053 | 23.62 | 33.97 |
| 8,192 | 4 | 24.288 | 75.20 | 41.03 |

An exact 8,192-token prefix hit produced its first token in 14.19 ms. Extending
that prefix to 10,240 tokens took 1.689 seconds, versus 7.990 seconds cold.
Model initialization took 10.68 seconds. An additional 8.94-second frozen-buffer
hash audit is separate benchmark instrumentation, not ordinary request work.

The independent whole-weight projection pass took 24.19 ms for batch one and
26.42 ms for batch four, with useful logical rates of 1.125 and 1.030 TB/s.
These are full-weight-set references with one synchronization per pass. They
omit layer dependencies and all nonprojection operations. Real 8K decode steps
were approximately 39.32 and 53.19 ms. The remaining 15.1/26.8 ms includes LoRA,
norms, GDN, KV accesses, sampling, and dependencies; it is not all avoidable work.

## Upstream attention-backward candidate

The official MLX 0.32.4 tag is commit
`36969c094354086e15d2a327a9c50feb20b9cf11`. Its
[Metal attention source](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/scaled_dot_product_attention.cpp)
adds a blocked native VJP with FP32 gradient accumulators and configurable query
and key blocks. This implementation is absent from the qualified 0.32.3 source.
Its component measurements are below. Full-model numerical qualification remains
open. The exact source hash and tag identity are preserved in the chat artifacts.

Both 0.32.4 wheels were prepared with `--target`, `--no-deps`, `--no-compile`, and
no pip cache in a fresh private Studio folder. The qualified environment reported
0.32.3 before and after preparation. Qualification covers GQA 24:4, head width
256, causal suffix offsets, short query tails, full-context attention, FP32-oracle
errors, and complete model gradients/Adam before any dependency change is accepted. The 262,144-key tiled control ends with
eight query rows, an important vector/full-attention dispatch boundary. Do not
load the read-only graph extension built against 0.32.3 into this candidate runtime.

`xcrun --find xctrace` reports that the profiling utility is unavailable on Studio.
No tooling installation was attempted. Current FLOP and traffic estimates remain
explicitly separate from hardware-counter measurements.

The 32 matched projection controls completed under strict precision. Their
weighted isolated-operation sum is 110.75 seconds, or 69.71 dense-equivalent
TFLOP/s, across the complete path's projection multiplicities. Forward and VJP
projection references are 28.86/55.30 seconds for GDN layers and 9.27/17.32 seconds
for attention layers. These are partial component references and must not be
mistaken for whole-update targets.

A dequantized BF16 reference sums to 88.33 seconds and would require another
48,701,112,320 weight bytes, about 45.4 GiB. Some component outputs differ, with
maximum relative L2 error 0.00495 across these controls. A blanket dequantized
cache is therefore not qualified; no such cache was enabled. Any selective cache
proposal still needs matched full-model gradients, optimizer results, memory
admission, lifecycle, and full-context evidence.


## Native attention component qualification

`scripts/mlx_attention_qualify.py` compares outputs and every Q/K/V derivative
with independent NumPy FP64 equations. Its finite-difference and causal-suffix
tests passed on Studio CPU. The unguarded 0.32.4 path failed one- and five-query
cases because the vector forward kernel does not return the required LSE.
An explicit ordinary-operation fallback for at most eight queries reproduces
the pinned BF16 output and gradients exactly. The guarded candidate passes all
six small cases and is closer to the FP64 references at the native cases.

`scripts/mlx_attention_large.py` retains all keys at both actual and maximum
context lengths. Seventeen-query anchors also use an explicit FP32 reference.
The component gates require relative L2 at most 0.02 and cosine at least 0.9999;
reference errors must not exceed max(1.25 times pinned error, twice BF16 unit
roundoff). These empirical component gates do not qualify a complete model.

| Queries | Keys | Pinned VJP ms | Native, old partitions ms | Native, whole query ms |
| ---: | ---: | ---: | ---: | ---: |
| 17 | 57,344 | 5.93 | 13.34 | 13.34 |
| 17 | 262,144 | 138.38 | 62.24 | 63.80 |
| 2,048 | 57,344 | 477.98 | 267.14 | 175.68 |
| 2,048 | 262,144 | 5,022.29 | 3,416.74 | 819.97 |

Whole-query peak active memory is 1.22/4.34 GiB for the two 2,048-query cases,
versus 12.99/56.26 GiB pinned. Reference tensors load after timing, so they do
not inflate these peaks. The native blocked backward bounds its own scratch.
Whole-query dispatch retains every query and key; it does not truncate context.

Two block-size alternatives also passed. BQ512/BK1024 measured 177.53/830.16 ms;
BQ1024/BK2048 measured 170.67/792.97 ms. Whole-query dispatch is the material
change. The first complete-model diagnosis uses upstream default blocks.

Before that complete-model trial, the phase model set a 250-second longest-path
target, with a 230–270-second range. It retains approximately 154.4 seconds of
GDN work and 6.85 seconds of other work. It scales the 112-second attention VJP
nonprojection residual by the measured 175.68/477.98 ratio, retains its
17.32-second projection reference, and forecasts 18–25 seconds forward attention.
This is a component-scaled estimate, not a measured optimized model result.

The candidate completed both its warmup and measured update. The measured update
took 237.30 seconds, with 78.98 GiB peak MLX memory and 81.02 GiB lifetime peak
physical footprint. Swap remained 95.75 MiB. The frozen base hash was unchanged;
all warmup and measured gradient bits matched within the candidate runtime.
Its measured phase totals were 37.77 seconds GDN forward, 108.50 seconds GDN VJP,
21.67 seconds attention forward, and 62.25 seconds attention VJP. These disjoint
layer totals cover 97.00% of measured update time; nested build times are subsets.

However, it fails the complete-model numerical gate. Every canonical tensor was
compared, including 496 nonzero B gradients and 496 exactly zero A gradients.
The whole-gradient relative L2 error is 1.26977 and cosine is 0.40482 against the
qualified baseline. The FP32 master-update delta cosine is 0.55264. Initial
adapter and sequence hashes match exactly. This is a substantial update-direction
change, not acceptable evidence for a harmless reduction-order difference.
The 237-second result is diagnostic; the retained qualified result remains
329.19 seconds. No native candidate dependency or dispatch change is enabled.

The baseline Adam state was reconstructed from the preserved exact gradient bits.
Its updated adapter hash matches the original full-path trial exactly. The scalar
gradient norm differs by 1.07e-10 after the tensor reload; this is recorded, with
a 1e-6 relative scalar-reduction limit. Gradient and published-adapter byte hashes
remain exact requirements. The first reconstruction's overstrict scalar equality
failure and the corrected source and receipt are both preserved.

An independent attention-precision reference now uses ordinary FP32 GQA operations
inside the original BF16 model. It retains the original runtime and all other
precision boundaries. It casts complete Q/K/V inputs once before query partitions,
so KV derivatives sum in FP32 before the original BF16 input boundary. This is
an attention reference within a complete model, not an all-FP32 model. Its six
small-shape output/derivative preflights match the independent FP64 equations with
maximum relative L2 3.09e-7. The complete measured reference took 514.82 seconds
and preserved the frozen base; warmup and measured gradients matched bitwise.
This is diagnostic higher precision, not a deployment or performance candidate.

Against this reference, the pinned whole-gradient relative L2 is 1.05603 with
cosine 0.41814. The native candidate is worse: relative L2 1.32882 and cosine
0.30383. It fails 167 per-tensor gradient/first-moment gates and 195 second-moment
gates. Its master-update cosine against the reference is 0.53205, versus 0.56122
pinned. Thus the reference does not rescue the native candidate. Even the original
BF16 model is sensitive to attention precision on this long path. This reference
alone does not prove which complete-model gradient is mathematically correct.
The evidence supports exact arithmetic-preserving optimizations; it does not
justify broader tolerances for a materially different optimizer direction.


## Exact compilation and sampling controls

The complete first GDN layer was measured with real model weights, the first
4,290 preserved input tokens, and synthetic cotangents. It covers an empty carry,
a carried 2,048-row unit, and a 194-row tail, for both initial and one-step
nonzero adapters. Every primal output, factor gradient, hidden gradient, and
carry gradient matched bitwise. Reused compiled functions also matched after
factors and hidden inputs changed. All eight A-gradient tensors are nonzero in
the nonzero-adapter cases. These are component gates, not a full-update proof.

For the carried 2,048-row unit, forward fell from 27.03 to 25.85 ms and full VJP
from 79.24 to 63.60 ms. Weighting the first, carried, and tail units over 48 GDN
layers predicts 22.73 seconds saved from the qualified 329.19-second path.
The separate complete-path forecast is 310 seconds, with a 300–320-second range
for trace/dispatch uncertainty. This does not replace the rejected native
attention candidate's 250-second target. The complete-path result is below.
The first compiled forward/VJP calls took 1.10/1.71 seconds; new same-shape
function objects after kernel warmup took approximately 28/76 ms. Cold compile
and warm trace costs must remain separate from steady-state operation time.

The sampler previously fetched each draw before scheduling its logprob gather.
The retained batched path schedules each request's independently keyed draw and
selected logprob before one evaluation. Finished rows remain inactive. The
actual-vocabulary control reduced four-request sampling from 1.317 to 0.482 ms,
with exact tokens/logprobs. This predicted 76.40 aggregate tokens/s from the
75.20 baseline at 8K context.

The complete actual-generation follow-up measured 76.35 tokens/s, a 1.54% gain.
Both measured batch samples, 76.30 and 76.40, exceed the two baseline samples,
75.26 and 75.13. Peak active memory remained 41.03 GiB. Across all 16 cases and
1,600 sampled tokens, tokens and every behavior logprob matched the baseline
exactly. Cold, exact-hit, and extended-prefix cases retained exact parity, and
the frozen base hash remained unchanged. Prefill variation is not claimed as a
sampling improvement.

The analogous single-request candidate had no dependable end-to-end gain and
was removed. Its method is restored exactly to the baseline. The retained batch
method is identical to the fully measured candidate. The candidate passed all
17 Studio BF16 numerical tests, including RNG isolation and mixed early stops,
and 17 local inference-contract/harness tests against the explicit worktree.


An additional dispatch diagnostic compared standalone attention with the primal
returned by its VJP on shared BF16 inputs. Pinned MLX0.32.3 differed at the
untiled 2,048-query/57,344-key shape (relative L2 0.00466), while 17- and 194-query
cases matched. Native MLX0.32.4 matched at 2,048 queries but differed at 17 and
194 queries. Its `force_fused=True` mode matched bitwise at all three shapes.
The pinned production path uses smaller late-context partitions, so the untiled
pinned result must not be projected onto every production unit. These observations
identify forward/replay dispatch differences; they do not establish complete-model
correctness or rescue the rejected native candidate.


The complete compiled-GDN candidate measured 324.37 seconds versus the qualified
direct-tile mean of 329.19 seconds, a 1.46% time reduction. Warmup took 324.78
seconds. Peak MLX memory was 93.43 GiB. Loss, all 992 gradient byte hashes, the
updated adapter hash, and the CPU-compared FP32 Adam moments/master updates all
match the baseline exactly. The exact compiled-unit library also passed all six
nonzero-adapter and changed-input cases. The production GDN path remains unchanged
while the forecast miss and full-batch effect are investigated.

The measured GDN forward/VJP totals are 42.61/104.04 seconds; attention is
40.13/131.11 seconds. The GDN VJP saves approximately seven seconds, rather than
the approximately 22-second whole-path saving predicted by the isolated control.
The 310-second forecast was missed and is not revised downward to claim success.

A separate paired working-set control uses the same compiled-unit library and
replay APIs. Each sample calls 48 units. Repeating one layer gives forward
1.287/1.245 seconds and VJP 3.828/3.081 seconds (baseline/compiled). Cycling through
all 48 distinct GDN layers gives forward 1.339/1.284 seconds and VJP 3.929/3.135
seconds. The large weight set changes the reference only slightly. Thus weight
reuse alone does not explain the complete-model gap. Actual adjoint layouts,
dtypes, and integration with gradient/carry accumulation remain to be measured.

The matched strict full-batch baseline now uses the preserved 22 paths and
444,145 decoder tokens, all 156,208 supervised positions, checkpoint interval
one, and one accumulated Adam step. Its supervisor is capped at one hour. It
records source/packed-sequence identities, reconstruction and initialization
times, extra hash-audit time, update phases, peak MLX and physical footprint,
swap, all gradients, and disposable optimizer diagnostics. The baseline forecast
is approximately 1,975 seconds (32.9 minutes), with 10% uncertainty from a phase
model scaled by token count and exact causal-pair count. This remains an estimate
until the complete update commits. No stopped experiment or policy is resumed.


## Complete preserved-batch baseline

The strict cold-process baseline completed all 22 packed paths, 444,145 decoder
rows, and 156,208 supervised positions. It accumulated all 992 canonical gradients
and committed exactly one disposable Adam update. The sum-reduced CISPO loss was
-0.4507699478417635. The frozen base remained unchanged. The model, histories,
credits, masks, precision, and checkpoint interval one match the path controls.

| Measurement | Baseline |
| --- | ---: |
| Complete update, including Adam | 1,901.56 s (31 min 42 s) |
| First update ready, including benchmark preparation | 1,922.31 s (32 min 2 s) |
| Extra frozen-buffer hash audit | 8.93 s |
| Model/adapter initialization | 10.79 s |
| Batch reconstruction and coverage audit | 0.91 s |
| Adam step | 1.67 s |
| Peak MLX active memory | 93.86 GiB |
| Lifetime peak physical footprint | 93.59 GiB |

Subtracting only the extra frozen-buffer audit gives 1,913.37 seconds until the
first update. This is an instrument-adjusted observation, not a second measured
run. Swap remained 87.75 MiB throughout. No historical experiment or policy ran.
The prior 1,974.95-second baseline forecast overestimated this run by 3.86%.

The disjoint layer timers cover 97.87% of the complete update. GDN forward/VJP
cost 310.63/860.88 seconds, or 61.61% together. Attention forward/VJP cost
156.71/532.75 seconds, or 36.26% together. Thus the shorter full batch spends a
larger fraction in GDN than the longest path. Nested graph-build/evaluation
measurements must not be added to these inclusive VJP totals.

The production GDN candidate now passes all 19 Metal numerical tests. The tests
include reused functions with two nonzero policies, exact complete gradients and
prefix gradients, and a switch back to native inference. Two CPU ownership and
exception checks also pass, within a 23-test local contract run. These source
checks do not establish the pending complete-batch numerical or speed gates.

Before the candidate full batch, the complete-path phase ratios predict 1,801.46
seconds for warm execution. An estimated 15–45 seconds for cold function/shape
traces gives an expected 1,816.46–1,846.46 seconds. This allowance is an explicit
estimate, not a measured cost. The prior 310-second longest-path forecast remains
a miss. The complete-batch forecast is preserved before the candidate run.

## Attention-adjacent GDN layout control

The complete-path profile shows slower GDN forward units immediately after
attention and slower GDN VJPs immediately before attention. Equalizing those
units with neighboring GDN units would explain at most 11.78 seconds. The pattern
alone does not identify the cause.

A separate bounded control used five real adjacent decoder layers, three causal
2,048-row segments, an actual attention-produced hidden adjoint, and a future-GDN
carry adjoint. The top cotangent was synthetic. The hidden inputs and all three
cotangents were already row-major. Explicit contiguous requests changed no bits
and provided no dependable gain: compiled GDN VJP medians were 66.32 ms without
copies and 66.54/66.30/67.15 ms with the copy combinations. Forward after attention
was 26.16 versus 26.07 ms. No layout copies are retained.

This rejects the tested layout explanation. It does not prove the cause of the
remaining complete-model gap. The component still omits full-model allocation,
gradient accumulation, and the complete attention working set.
