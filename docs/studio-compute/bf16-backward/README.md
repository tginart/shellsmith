# Faster BF16 backward on Mac Studio

The default execution profile is `bf16-fast`. It keeps the frozen backbone in
8-bit weights and uses BF16 tensors for activations, recurrent state, adapters,
loss, gradients, accumulation, and Adam state. `--execution-profile legacy`
selects the previous FP32 state and optimizer path. Fast Metal kernels use
internal float registers where useful; there are no FP32 master weights or
explicit FP32 activation tensors in the qualified BF16 graph.

Matched measurements give **196.49 seconds** for
the longest preserved path and **20.24 minutes**
for the complete preserved batch. Complete-update speedups are
**1.64x** and
**1.499x** respectively.
[The full report](performance.md) contains all trial times, first-update latency,
phase costs, memory, swap, counters, and explicit unmet objectives.
Total backward is **1.78x** faster
on the longest path and **1.73x**
on the batch. The report marks every unmet twofold objective explicitly,
including the 57K attention primitive's 1.81x result.

Full-batch gradient cosine relative to legacy is **0.200**, and Adam-update-delta
cosine is **0.408**. These changes are large. [Legacy diagnostics](legacy-diagnostics.json)
and [BF16 stage diagnostics](ablation-diagnostics.json) publish them explicitly.
The explicit saved-activation VJP closely matches its preceding BF16 stage
(cosine 0.99978); larger changes appear with numerical kernel/normalization
realizations. The requested memorization and independent derivative gates define
qualification. Long-rollout reward improvement is not tested.

## Reproduce the fixed learning gate

The actual 27B model uses canonical rank-16 adapters and the existing Adam
settings. The fixture asks for the secret phrase for Studio check Z7K4. The
arbitrary answer is ` violet otter 7391 cedar`, followed by EOS. The initial
model does not generate it. Only answer tokens contribute to loss.

Every attempt performs exactly ten optimizer updates. Both seeds, 23 and 91,
must generate the exact answer and reduce teacher-forced loss by at least 90%.
The complete legacy grid is preserved in [legacy-grid.json](legacy-grid.json):
both seeds pass at `3e-4` and `1e-3`; both fail at `3e-3`. The smallest successful
rate, `3e-4`, is frozen for every BF16 comparison. BF16 is never tuned separately.

```sh
export MLX_ENABLE_TF32=0 PYTHONPATH=src:scripts
python scripts/qualify_bf16_memorization.py \
    --model "$STUDIO_MODEL" --execution-profile legacy \
    --output "$FRESH_LEGACY_OUTPUT" --timeout-seconds 3600
python scripts/qualify_bf16_memorization.py \
    --model "$STUDIO_MODEL" --execution-profile bf16-fast \
    --calibration "$FRESH_LEGACY_OUTPUT/report.json" \
    --output "$FRESH_BF16_OUTPUT" --timeout-seconds 900
```

[The final learning evidence](memorization-default.json) publishes losses,
gradient norms, parameter changes, and greedy output before and after. Both
seeds pass. The ten-update zero-learning-rate control preserves the adapter and
initial output. A full checkpoint at update five, restore, and continuation to
update ten reproduce the uninterrupted adapter and optimizer state exactly.
BF16 selected-token logprobs reach the format's zero resolution on this fixture;
that does not establish zero true cross-entropy or broader learning quality.

Every earlier learning attempt remains published: initial setup errors in
[legacy-v1](attempts/legacy-v1.json) and [legacy-v2](attempts/legacy-v2.json);
[legacy calibration](legacy-memorization.json), [BF16 reference](bf16-memorization.json),
[GDN](gdn-memorization.json), [saved backward](attempts/saved-v28.json),
[initial complete kernels](final-memorization.json), [joint attention](memorization-joint.json),
[token inference](memorization-token.json), and [saved RMS statistics](memorization-statistics.json).
Historical evidence files are unchanged.

## Execution and restore identity

Stochastic rounding prevents small BF16 parameter updates from systematically
disappearing. The Adam kernel uses a dedicated counter RNG keyed by the saved
rounding seed, optimizer step, parameter identity, and element index. Sampling
uses a separate RNG. Tests check repeatability, mean small-update behavior,
zero-rate updates, and exact continuation. Parameters and both moments remain
BF16. Accumulated gradients and shared-prefix adjoints use balanced reductions.
See [Revisiting BFloat16 Training](https://arxiv.org/abs/2010.06192).

Capabilities, full checkpoints, and inference cache keys include the execution
profile and kernel build identity. That identity covers numerical Python/Metal
sources, pinned library versions and binary hash, GPU architecture, OS build,
and TF32 setting. Full-state restore rejects incompatible identities and tensor
dtypes. Selecting `legacy` does not bypass exact build-identity verification.
Existing model assets and historical checkpoints are never rewritten.

## GDN kernels

The BF16 recurrence uses 64-token chunks, direct unit-triangular solves applied
to the required right-hand sides, and device-resident forward/reverse state
scans. NAX contractions tile over value rows and key dimensions. Python does
not dispatch individual chunks. Reverse state propagation supplies carried-state
cotangents and Q/K/V, gate, and beta derivatives. Small unsupported tile shapes
use an explicit scalar implementation, with dispatch visible to qualification.
The implementation does not differentiate through a triangular inverse.

At the actual recurrence shape B=1, T=2048, Hk=16, Hv=48, Dk=Dv=128, warm
forward/VJP times are 2.57/7.28 ms versus 3.46/12.12 ms for the BF16 reference.
The complete VJP improves about 1.67x. [Component evidence](gdn-component.json)
includes warmups and distributions.

```sh
MLX_ENABLE_TF32=0 PYTHONPATH=src:scripts python scripts/benchmark_gdn_metal.py \
    --output "$FRESH_GDN_OUTPUT" --tokens 2048 --timeout-seconds 600
```

The independent sequential FP64 reference
checks all six derivatives, nonzero state, tails, convolution boundaries, and
nonzero adapters. A 4097-token test splits at 2048 and preserves BF16 outputs and
carry gradients exactly; maximum independent relative error is 1.31%.

A dedicated inference-only one-token recurrence removes the old 64-row padding.
It leaves training recurrence and backward unchanged. Complete decoder/head
latencies improve from 53.4 to 40.2 ms at B1/8K, 74.8 to 54.6 ms at B4/8K, and
77.7 to 66.3 ms at B1/57K. Incoming cache copies, prefill, and sampling are outside
that timer. Logit differences of 1.02–1.22% are diagnostic.
See [matched decode evidence](decode-token.json).

## Fused attention

The repository-owned MLX custom operator compiles bundled Metal sources against
MLX 0.32.3. It changes no installed runtime files. `attention_forward` returns
an output and tape; `attention_backward` consumes that tape. The production
specialization supports D256, 24 query heads, four KV heads, suffix-causal
offsets, tails including one/eight rows, and contexts through 262144 tokens.

Forward retains output and compact BF16 high/residual log-normalizer pairs.
Backward owns dQ tiles separately and shares score work between dK/dV owners.
Grouped-query reductions use a fixed order without global atomics. Score tiles
remain on chip. Backward recomputes those tiles intentionally, as in the
[FlashAttention approach](https://arxiv.org/abs/2307.08691); it does not rerun
projections or complete decoder layers. Native attention forward remains faster
for inference, where no training tape is required.

At Q=2048/K=57344, native/fused cached backward is 265.4/146.8 ms: **1.81x**,
below the twofold primitive objective. Complete VJP is 436.2/200.3 ms: **2.18x**.
At K=262144, cached backward is 1745.6/681.6 ms and complete VJP is
2394.7/933.6 ms, both about **2.56x**. These are distinct scopes.
[Matched component evidence](attention-joint.json) preserves both.

```sh
MLX_ENABLE_TF32=0 PYTHONPATH=src:scripts python scripts/benchmark_attention_metal.py \
    --output "$FRESH_ATTENTION_OUTPUT" --timeout-seconds 900
```

Independent FP64 checks cover tails, offsets, future-value exclusion, and
maximum-context derivatives. Maximum relative error at 262K is below 0.72%.

## Saved activation backward and memory

`forward_saved` returns hidden/carry outputs and an explicit layer tape.
`backward_saved` uses retained projections, small LoRA products, compact
attention residuals, and original carry producer identities. It applies frozen
8-bit projection transposes and adapter derivatives explicitly. Checkpoint-one
backward does not call a whole-layer VJP. Selected projection outputs execute
once. Existing KV allocations remain shared with their original producers.

The tape also retains two inverse-RMS statistic pairs per row for each layer's
input and post-mixer normalization. Forward and reconstruction use the same
pairs. Explicit normalization backward consumes them without recalculating
statistics. Independent tests cover widths 64 and 5120 and forbid statistics
recalculation during backward.

`--activation-budget-gib auto|N` admits optional residuals after resident state,
mandatory checkpoints/tapes/KV, scratch, and a further 16 GiB safety reserve.
Automatic admission respects the smaller recommended working-set/allocator
limit. `--checkpoint-layers auto|N` selects the smallest admitted interval or
honors a valid explicit override. Insufficient explicit memory fails.
Measured recomputation nanoseconds per unique BF16 byte determine optional
retention; identity ordering breaks ties deterministically. Branches reuse the
budget after their prior consumers finish.

Gate/up MLP tensors alone would require about 239 GiB on the longest path.
Unretained projections, local GDN/SwiGLU primal work, Q/K normalization/rotary,
attention gating, normalized outputs, and vocabulary-head work can therefore
remain deliberate reconstruction. [The performance report](performance.md)
quantifies retained bytes and logical operation calls. These counters describe
operator groups, not individual GPU instructions or compiler fusion.

## Independent tests and actual-model audits

```sh
MLX_ENABLE_TF32=0 PYTHONPATH=src:scripts python scripts/qualify_bf16_native.py \
    --output "$FRESH_NATIVE_OUTPUT" --timeout-seconds 600
MLX_ENABLE_TF32=0 PYTHONPATH=src:scripts python scripts/qualify_bf16_long_context.py \
    --output "$FRESH_LONG_CONTEXT_OUTPUT" --timeout-seconds 900
python scripts/build_graph_audit.py --nanobind-source "$NANOBIND_SOURCE" \
    --output "$FRESH_AUDIT_BUILD"
MLX_ENABLE_TF32=0 PYTHONPATH=src:scripts python scripts/audit_bf16_graphs.py \
    --model "$STUDIO_MODEL" --extension-dir "$FRESH_AUDIT_BUILD" \
    --output "$FRESH_AUDIT_OUTPUT" --timeout-seconds 600
```

The graph auditor is a read-only extension built outside the runtime against
pinned nanobind v3.0.1, including its robin-map submodule. The actual 27B default
profile passes training-layer forward/backward, loss/head, optimizer, balanced
sum, and real inference dtype audits. Computed FP32 tensor nodes and unexpected
training attention fallbacks fail the audit. Frozen constants and internal
register accumulation remain permitted. See [dtype-default.json](dtype-default.json),
[native-default.json](native-default.json), and [long-context-default.json](long-context-default.json).
Shader provenance and bundled permission notices ship in the wheel.

## Integrated timing and preservation

```sh
MLX_ENABLE_TF32=0 PYTHONPATH=src:scripts python scripts/qualify_bf16_updates.py \
    --source "$IMMUTABLE_CHECKOUT" --model "$STUDIO_MODEL" \
    --batch "$PRESERVED_BATCH" --suite longest --output "$FRESH_LONGEST_MATRIX"
MLX_ENABLE_TF32=0 PYTHONPATH=src:scripts python scripts/qualify_bf16_updates.py \
    --source "$IMMUTABLE_CHECKOUT" --model "$STUDIO_MODEL" \
    --batch "$PRESERVED_BATCH" --suite batch --output "$FRESH_BATCH_MATRIX"
```

Rebuild the published aggregate from the preserved raw matrices and release proof:

```sh
python scripts/summarize_bf16_qualification.py \
    --ablations docs/studio-compute/bf16-backward/evidence/longest-ablations.json.gz \
    --final docs/studio-compute/bf16-backward/evidence/longest-final.json.gz \
    --batch docs/studio-compute/bf16-backward/evidence/full-batch.json.gz \
    --attention docs/studio-compute/bf16-backward/attention-joint.json \
    --counters docs/studio-compute/bf16-backward/evidence/release-counters-longest.json.gz \
    --diagnostics docs/studio-compute/bf16-backward/legacy-diagnostics.json \
    --ablation-diagnostics docs/studio-compute/bf16-backward/ablation-diagnostics.json \
    --release-equivalence docs/studio-compute/bf16-backward/evidence/release-qualification.json.gz \
    --output "$FRESH_REPORT"
```

The published report preserves three longest trials per progressive stage and
two complete-batch trials per final/legacy profile. The complete batch covers
22 paths, 444145 decoder rows, and 156208 supervised tokens. Masks, reward and
credit values, CISPO semantics, complete context/carry gradients, and one
accumulated Adam update remain intact. Model bytes are hashed before/after.

The [longest-path forecast](cost-model-statistics.json) and
[full-batch forecast](cost-model-full-batch.json) precede final timings. Forecasts
are labeled as estimates and are never presented as measured speedups. A separate 2048-row
warmup includes first-call kernel setup and any required compilation, without an
optimizer step. OS compiler caches are retained. New shape
compilation can remain inside measured complete updates; compiler wall time is
not independently isolated. Warm component distributions provide steady-state
primitive times. First-ready latency includes preparation and warmup through
the in-memory Adam commit; durable service-checkpoint I/O is outside that scope.

Each GPU command owns the existing exclusive Studio lease and a hard deadline
of at most 60 minutes. Output folders are private and disposable. Qualification
keeps the experiment stopped and preserves model assets, historical checkpoints,
and the original dirty checkout. No training campaign starts.

The [final validation record](final-validation.json) and
[source/build evidence manifest](evidence-identity.json) bind these results to
the exact qualified implementation. All shipped source files match that snapshot.


## PR review corrections

The [review correction evidence](review-corrections.json) qualifies three fixes
from the Codex review of `4ff87f8`:

- Replay reservations protect gradient scratch and the 16 GiB safety margin while
  another request runs. Optional activation budgets remain unchanged.
- BF16 Adam bias correction uses the same representable decay constants as its
  moment kernel. Configured betas remain in checkpoint metadata. Betas that round
  to one fail before an update.
- Both numerical and runtime source inventories include `activation_budget.py`.
  Admission edits change the kernel build and full-state checkpoint identity.

The corrected source passes 94 portable admission, identity, storage, and recovery
checks and 80 native numerical checks. The actual 27B memorization fixture passes
for seeds 23 and 91 at the unchanged legacy-selected rate `3e-4`, with exact answer
output, the required loss reduction, exact checkpoint continuation, a zero-rate
control, and unchanged frozen weights. The actual-model BF16 dtype and dispatch
audit also passes. Every attempt and worker receipt is preserved in `evidence/`.

The matched performance tables and legacy-update diagnostics above describe the
original qualified snapshot. They were not rerun after the Adam correction;
updated parameter hashes must not be treated as identical to that snapshot.

Reproduce the corrected numerical and learning checks with the native,
memorization, and dtype audit commands above. Each command uses a fresh output
folder and the existing Studio GPU lease.


### Recommended working-set cap

A follow-up review found that concurrent requests could use a larger allocator
cap after replay admission used Metal's recommended working-set cap. BF16 request
admission, prefix-cache eviction, and cold inference batch capacity now share
`min(allocator, recommended working-set cap)`. Legacy retains its allocator policy.

[The follow-up evidence](review-working-set.json) records 325 portable Studio
checks, three subtests, and 80 native checks. One local native-only module skips;
the Studio native suite covers it. Boundary tests reject one byte above the
shared cap, cover explicit and automatic retention, and verify cache eviction
and batch capacity. Both fixed-rate actual-27B memorization seeds pass again,
including exact restore, zero-rate control, and frozen-base checks. Numerical
modules are unchanged from `5b95ca5`; no matched performance trial is repeated.
