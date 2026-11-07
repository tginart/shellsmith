# Rollout-segment masked training

The failed `studio-junior12-r1` batch contained 329 different response prompts.
Grouping identical whole prompts shared none of their growing histories. Training
now replays each exact causal path in a rollout once, with all of that path's
sampled supervision aligned to its token positions. Compaction starts a new
logical segment using the next request's actual captured prompt.

## Token provenance and objective

`MLXPlayerSampler.sample` renders the actual request with `Renderer.render`,
records the resulting `TokenPrompt`, and records sampled token IDs and behavior
logprobs directly from generation. Its two passes retain the exact thinking and
response IDs; a deterministic forced close has an explicit masked span.
`serialize_capture` persists these arrays. Hosted Tinker likewise retains the
renderer-specific prompt's exact IDs. The training builder reads those IDs; it
never decodes and re-encodes assistant text or invents a replacement prompt.

`PiCompactionAdapter` aligns consumed requests with the durable vat journal and
its successful compaction events. A summary generation keeps its existing credit
in the preceding segment. The actual compacted continuation starts the next;
summary replay/context receives no direct supervision. Earlier segments remain
in the same optimizer update. Split summary generators remain distinct actions.

For complete tokens `P + T`, model input is `P + T[:-1]`; sampled target `T[i]`
uses predictor row `len(P) - 1 + i`. Packing appends exact intervening prompt
suffixes and assigns zero mask/credit/weight to their tool, template and context
tokens. `source_spans` names each original sampled span in the unshifted full
path. Existing sampled masks, numerical weights, advantages and behavior
logprobs are copied without averaging. The rollout-wide denominator is computed
once before packing, excluding deterministic closes and compaction replay as
before. Zero-credit requests still provide context; their old skip semantics are
preserved. CISPO remains a sum with detached clipped ratios, and every segment
and rollout accumulates before one Adam step (.9/.95/1e-8 by default).

A later prompt must begin with every token in the preceding prompt **and**
completion to share the path. Seven captured requests rewrite earlier history;
these produce explicit `history_rewrites` audit entries and separate physical
paths, retaining both sets of losses. Strict callers can request
`on_incompatible="error"`. Transferring a loss to a reserialized history would
change its conditioning and is forbidden. The failed batch has 15 logical
segments and 22 physical paths; it had no compactions.

Studio gathers only enabled sampled rows for its vocabulary head. The decoder
still executes all context rows, and backward propagates through attention and
recurrent carries into earlier context. Masking never detaches context.
Entirely context-only paths have an exact zero gradient but still permit a
requested Adam step, including momentum evolution. Tinker receives the same
shifted full-path targets and zero advantages on context. Its optional reference
KL discount keeps per-response boundaries, then masks injected/context positions
before repacking. The reference shaping implementation is in the
[Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/rl/metrics.py).

## Memory and progress

At training handoff, Studio clears idle prefix snapshots and free allocator
buffers before admission. Active inference generators own independent copies
from `PrefixCache.get`; their working caches and reservations remain live.
There is one resident frozen quantized base, BF16 execution/LoRA/training KV,
existing FP32 Adam master/moments and the full 262,144-token served context.
No model assets, historical checkpoints or stopped experiment state are changed.

Progress carries whole-update `completed_segments/total_segments`,
`completed_tokens/total_tokens`, supervised counts, active/peak memory, retained
prefix-cache bytes, released cache bytes and accumulated phase execution times.
Whole-update completion counts advance after a physical path group; existing
inner `segment`/layer fields still identify recomputation chunks. A backward
receipt says `optimizer_committed=false`; the successful optimizer receipt says
true. The service publishes successful optimizer receipts after its existing
durable checkpoint boundary. Trainer state also names `training_outcome`
separately from controller lifecycle `status`.

## Reconstructed failed batch

[failed-batch.json](failed-batch.json) was reconstructed read-only from all 15
durable captures with the preserved experiment runtime and sealed credit
settings. The interrupted journal's request body had been pruned. The old source
hash is recorded in the report. The independent reconstruction matches every
count in the historical backward investigation.

| Work or coverage | Response replay | Packed paths |
| --- | ---: | ---: |
| Sequences | 329 | 22 |
| Shifted decoder input token occurrences | 4,844,072 | 444,145 |
| Supervised sampled positions | 156,208 | 156,208 |
| Longest shifted decoder input | 57,538 | 57,538 |

This is **90.83% fewer decoder token occurrences (10.91x less replay)**, not a
measured full-batch speedup. All 329 original conditioning prefixes, sampled
IDs, behavior logprobs, advantages and masks were compared exactly. The three
other sampled-tail positions are injected closes. Actual packed lengths are
listed in the report, from 2,836 to 57,538 tokens. Raw token payloads remain in
temporary evidence files and are not committed to this repository.

Reproduce the audit using a JSON/gzip bundle with `body.sequences` and ordered
`captures[].turns` (`request_id`, `kind`, `sequence_index` from the journal-aware
compaction adapter):

```sh
PYTHONPATH=src python scripts/analyze_rollout_segments.py failed-batch.json.gz \
  --output reconstruction.json
```

## Regression receipt

[validation.json](validation.json) records 2,843 passing tests, 19 skips and
595 passing subtests (376.22 seconds). After the final contiguous-head
preservation, all 17 native numerical tests passed again; the final compaction
and token contracts passed all nine checks. New Python files pass Ruff, and
`git diff --check` is clean. Source fingerprints and runtime versions are saved.

## Bounded matched measurements

[pre-change-benchmark.json](pre-change-benchmark.json) compares the exact
`replay.py` from base commit `c4f23b7` against the packed path on a four-layer
hybrid attention/GDN model with 8-bit frozen weights, BF16 execution/LoRA and
FP32 Adam. The original replay source SHA256 is recorded in the report. Both
paths use identical captured tokens, supervised positions, CISPO parameters,
initial factors and one Adam update per trial. One warmup is excluded; two
measured trials alternate execution order.

| Mean measured time | Original response replay | Packed path |
| --- | ---: | ---: |
| Forward/backward | 0.354 s | 0.117 s |
| Whole update, including Adam and diagnostic gradient snapshot | 0.439 s | 0.202 s |

The measured whole-update ratio is **2.17x on this fixture**. Replay occurrences
are 1,190 versus 332, covering the same 28 supervised targets. Gradient relative
L2 error is 0.01824, cosine 0.999834, and loss absolute difference 0.00001197.
Changed BF16 reduction/GEMM shapes cause rounding differences; equivalence is
numerical within tested tolerances, not bitwise. Maximum published factor
absolute difference after Adam is 0.00010014; near-zero gradient sign changes can
be amplified by the first Adam step. The frozen base hash is unchanged.

Earlier same-code per-response controls measured 2.24x and 1.97x whole-update
ratios ([first run](tiny-benchmark.json), [repeat](tiny-repeat.json)), with
identical gradient and factor differences. All reports are retained. These
short runs show a roughly 2x fixture improvement with timing variation, not a
production forecast.

The local numerical tests additionally compare individual response replay to
packed sparse-head replay with checkpoint intervals 1, 2 and 4; cover variable
lengths, tool context, signed per-token credit, reasoning/response weights,
injected closes, compaction masks, context VJPs and normalization. Native dense
forward/reference checks remain in the existing suite. Active inference is
paused across training/cache release and resumed with unchanged logprobs.
Hosted SDK fixtures verify shifted arrays, KL boundaries and one Adam step only
after successful backward. Hosted numerical execution itself is not measured.

```sh
git show c4f23b7:src/studio_compute/numerics/replay.py > /tmp/original-replay.py
PYTHONPATH=src python scripts/benchmark_rollout_segments.py \
  --reference-replay /tmp/original-replay.py --output tiny.json
# Optional, on an authorized idle Studio; use an outer process timeout:
PYTHONPATH=src python scripts/benchmark_rollout_segments.py \
  --model /path/to/Qwen3.8-27B-8bit --reference-replay /tmp/original-replay.py \
  --warmups 0 --trials 1 --output studio.json
```

## Approved Studio 27B check

[studio-benchmark.json](studio-benchmark.json) and
[studio-execution.json](studio-execution.json) record one approved trial on the
resident Qwen3.8-27B 8-bit base. The complete process finished in 43.40 seconds
under a 240-second hard cutoff, with the existing GPU lease, disposable factors
and optimizer state, BF16 execution/LoRA, FP32 Adam and 262,144-token context
capability. The source commit was `9dbe106`; the original replay source hash
matches the exact pre-change control above.

| Single measured trial | Original response replay | Packed path |
| --- | ---: | ---: |
| Forward/backward | 7.363 s | 2.279 s |
| Whole update, including Adam and diagnostic gradient snapshot | 9.068 s | 3.974 s |
| Peak active memory | 31.610 GiB | 32.263 GiB |
| Committed disposable optimizer steps | 1 | 1 |

The observed whole-update ratio is 2.28x for this short fixture: four responses,
1,190 versus 332 replay tokens, and the same 28 supervised positions. There was
**no warmup and one ordered trial**, with original replay first. Compilation and
order effects are therefore uncontrolled; this is not a steady-state throughput
estimate, a long-context memory result or a speedup claim for the failed batch.
Packed peak active memory was about 0.65 GiB higher in this trial. The second
path retains the first path's diagnostic gradient/factor snapshots for the
comparison, so these peaks do not isolate the paths' intrinsic memory needs.

The predeclared BF16 checks passed: gradient relative L2 0.036452, cosine
0.999362, and absolute loss difference 0.00007314. The maximum published-factor
absolute difference after Adam was 0.00010014. These are numerical comparisons
with rounding differences, not bitwise equality. The frozen base hash was
unchanged. Both disposable updates committed; no checkpoint or policy was saved
or published. The benchmark process exited and the GPU lease was verified
available afterward. The historical experiment remained stopped.

## Remaining limits

The full failed batch has not been replayed numerically or timed. The 90.83%
work reduction must not be interpreted as its wall-clock speedup or an ETA.
Attention backward/recomputation at 57K–262K tokens can remain expensive; no
long-context kernel speedup is claimed. Full 262K admission remains unchanged.
Seven actual history rewrites still require extra physical paths, so one path
per logical segment cannot preserve these historical conditionals.

On the tiny fixture, roughly 0.08 seconds of Adam/publication and diagnostic
snapshot overhead remains after packing. This identifies a fixed-cost floor,
not a long-context bottleneck measurement. On the short 27B trial, the Adam and
diagnostic-snapshot remainder was about 1.70 seconds on both paths; it forms a
fixed-cost floor for this small workload. No optimizer mathematics or defaults
were changed to improve these numbers. Steady-state 27B throughput, long-context
memory behavior and hosted Tinker numerical timing remain unmeasured.
Checkpoints retain their original numerical/runtime identity guards; no historical checkpoint migration
is attempted. No hours-long experiment,
monitor, campaign, merge or historical model/checkpoint mutation was performed.
