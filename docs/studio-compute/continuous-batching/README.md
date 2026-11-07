# Studio continuous inference

Implementation base: Shellsmith `36a33805252b3606beab2b9ee3838392f7b0a76e`.
Adaptation source: High Command `1f51929bf55c2f954c11c6774dc8e80fb8f2c063`,
from its `origin/main` on October 19, 2026. The original working checkouts were
preserved. Implementation lives in the managed worktree on
`codex/studio-continuous-batching`.

## Implementation

`mlx_request_cache.py` adapts immutable KV-prefix forks with private recurrent
state and append-only BF16 suffixes. Native packed 8-bit KV buffers receive a
private copy because the upstream writer mutates its capacity buffers.
`mlx_decode.py` batches projections, MLPs, and recurrent rows while attention
uses each request's own cache and absolute position. It keeps the existing
legacy and BF16 GDN arithmetic paths and native quantized attention dispatch.

`mlx_racing.py` extends High Command's finite race into a persistent FIFO. New
requests enter available slots at a materialized token boundary. Each boundary
prefills at most one configured chunk, advances existing pending tokens, and
draws one output token per ready row. Newly prefilled rows use their first
logits without a decode pad. EOS and independent length limits retire rows
immediately. The next boundary can refill their slots while survivors continue.
Requests for different immutable policies receive separate decoder calls.

`inference.py` owns reservations, policy leases, cache writers, RNG keys, and
receipts. Admission reserves a complete request peak, including snapshot copy
space. Reusable snapshots and idle policies yield to live inference or replay.
Temporary pressure defers admission; a request that cannot fit alone fails.
Cancellation and kernel failures release private caches, policy pins, and
reservations. A shared kernel error fails its affected rows. A journal or queue
failure stops the owner without overwriting an earlier successful receipt.

The service bridges the durable operation journal to one inference lane and
alternates that lane with the existing training lanes. Both samples and forced
logprob scores respect the inference limit. The model, training kernels,
optimizer, checkpoint serialization, and existing RPC payloads stay intact.
The existing runtime-source identity now also includes the new modules. It
continues to reject checkpoints from a different runtime identity.

Set `inference_concurrency` in `[service]` or pass `--inference-concurrency N`.
The default is eight; any positive integer is accepted. Episode-worker settings
do not control this limit. Memory pressure can admit fewer requests. Excess
requests remain queued without cache allocation. `prefill_tokens` remains a
global per-boundary chunk cap, between 1 and 2048.

`Client.cancel(id)` or `POST /v1/operations/<id>/cancel` requests cancellation.
Poll the same operation for its durable `interrupted` receipt. Cancellation does
not rewind a journaled RNG assignment. HTTP poll disconnection does not cancel
durable work. Token draws retain one independent categorical key stream per
request, temperature 1, top-p 1, and the existing behavior-logprob arithmetic.
The `batch_size` result reports that request's largest actual decode cohort.

## Hardware and workload

Validation uses the Tailnet Mac Studio at `studio-host`, identified by MLX as an
Apple M5 Ultra, with 256 GiB unified memory. Runtime versions are MLX 0.32.3,
mlx-lm 0.32.0, mlx-vlm 0.7.4, and transformers 5.17.0. The model is the resident
`Qwen3.8-27B-8bit` dense decoder, with BF16 inference and BF16 KV unless stated.
The existing GPU lease excludes competing service jobs during measurements.

The user authorized termination of Studio jobs. The pilot supervisor, trainer,
and service were stopped; its source, journal, episode records, and checkpoints
were retained. Benchmark state uses a separate directory under
`~/.local/state/shellsmith/continuous-batching-validation`.

The main replay uses twelve actual pilot conversations, three turns each,
without prompt padding or truncation. Prompts contain 2,540–4,659 tokens.
Recorded completion paths contain 25–96 measured tokens. Later turns enter only
after their own preceding turn completes. This gives 36 requests, 2,768 measured
completion tokens, queued load above the largest concurrency setting, and 24
requests with reusable prompt prefixes. Recorded token paths hold the numerical
comparison constant; this benchmark measures inference replay, not GM/tool time.

Run from the worktree with the qualified Studio Python environment:

```sh
PYTHONPATH=src MLX_ENABLE_TF32=0 python scripts/benchmark_studio_continuous.py \
    --model ~/models/Qwen3.8-27B-8bit \
    --records-root /path/to/pilot/train \
    --output /path/to/evidence.json \
    --sessions 12 --turns 3 --max-completion-tokens 96 \
    --repeats 2 --concurrency 1 2 4 8
```

The script records source identities, record hashes, all prompt/output lengths,
individual cache hits and latencies, actual decode batch histograms, admissions
with live survivors, queued decode boundaries, allocator/RSS/swap measurements,
and post-run resource counts. It uses one warmup and an unmodified singleton
reference. Each timed measurement starts with an empty prefix pool and retains
prefixes across its turns. `--allocator-gib` enables a constrained-memory run;
`--cache-precision 8bit` selects packed KV. `--distribution-parity` compares eight
native singleton continuation steps with the batched full-vocabulary distribution.

## Validation results

The [repeated replay evidence](replay-benchmark.json) completed both timed repeats
at each concurrency setting. Throughput below pools the two equal workloads by
their total elapsed time. The unmodified singleton reference measured 18.59
completion tokens/s over 148.89 seconds.

| Inference concurrency | Tokens/s | Relative to singleton | Peak active slots | Peak allocator GiB |
| --- | ---: | ---: | ---: | ---: |
| 1 | 18.31 | 0.985× | 1 | 37.72 |
| 2 | 25.30 | 1.361× | 2 | 37.90 |
| 4 | 32.84 | 1.766× | 4 | 38.11 |
| 8 | 37.04 | 1.993× | 8 | 38.65 |

The two repeats agree within 0.2%. Single-request queue overhead is about 1.5%.
At concurrency eight, each repeat has 258 eight-row decode calls, 299 decode
boundaries with queued work, and 28 admissions with live survivors. There are
396 decode calls overall, of which only 15 are single-row calls. Therefore at
least 284 queued boundaries execute multiple rows. The weighted histogram
contains exactly 2,732 row advances: 2,768 output tokens minus 36 first tokens.
Completed requests do not consume decode pads. All 36 requests complete, with
24 warm-prefix hits, zero live reservations/policy users afterward, and zero
swap growth at every concurrency.

Concurrency one matches singleton logprob records exactly. Other settings have
maximum absolute BF16 record-logprob error 0.25, 99th percentile 0.125, and mean
error about 0.006. The separate eight-row, eight-step full-vocabulary diagnostic
measures maximum KL divergence 0.00290 nats and mean 0.000385 nats, below the
predeclared 0.005 limit. The [final-source replay and KL receipt](final-benchmark.json)
measures 17.48 versus 11.98 tokens/s on eight conversations with two turns and
32-token completion caps: a 1.46× aggregate improvement on that shorter workload.

The [40 GiB admission run](memory-pressure.json) configures eight slots but admits
only two, with four requests initially queued. It retains all four reusable
prefixes, completes every request, and measures 13.98 versus 11.87 tokens/s.
Peak allocator use is 30.61 GiB, with zero swap growth, remaining reservations,
or policy pins. Admission waits before evicting snapshots when live reservations
already make admission impossible. This refinement fixed cache loss observed
during the first constrained-memory experiment.

The [packed 8-bit cache run](packed-cache.json) measures 16.27 versus 11.63
tokens/s at configured concurrency six with four conversations: a 1.40× gain.
It retains four prefix hits, preserves native packed cache storage, and has
maximum record-logprob error 0.125 and zero swap growth. Native fixtures also
check packed cache offsets after mixed-age decoder steps.

The [real 27B RPC probe](rpc-training-recovery.json) passes prefix reuse,
prefix extension, mixed prompts/output limits across eight actual decode rows,
active cancellation, inference during backward replay, a committed optimizer
update, unchanged frozen base and second-client state, and exact state plus
next-sample recovery in a fresh process. A 0.5 GiB cache cap retains 449,990,656
bytes and records two evictions. The probe sets `activation_budget_gib=1` to
leave inference headroom. Automatic replay activation admission can reserve that
headroom and keep new inference queued until replay releases its reservation;
the existing training-budget behavior stays intact.

[Native tests](native-tests.txt) pass all 49 tests, including legacy/BF16 parity,
variable ages, independent keys, packed caches, policy pins, optimizer/replay
regressions, and cancellation cleanup. [CPU/HTTP validation](cpu-validation.json)
passes 358 tests and three subtests, with two local MLX skips covered on Studio.
The final admission refinement passes another 33 targeted tests. Static checks
and the [bounded process receipt](rpc-process-receipt.json) pass. The Studio GPU
lease is available after validation.

[Hardware commands](hardware-validation.json) and [source provenance](source-provenance.json)
bind these results to their source identities. The larger repeated sweep
precedes receipt/error-path and admission refinements. Its decoder source and
prefill/sampling function ASTs match the final implementation. The final-source
native, replay/KL, memory, packed-cache, and RPC probes validate those refinements;
their runtime source hashes identify that validation snapshot. Later review fixes
and their source hashes appear in the receipt below.

## Review fixes

The [first review-pass receipt](review-pass-1.json) records three corrections:
policy reload defers after admission evicts the requested policy under live memory
pressure; distribution parity rejects fewer than nine first-turn target tokens
before model allocation; receipt timing includes policy, cache, and admission work.
The latter preserves decode-only timing and charges admission to the prefill phase.

The corrected source passes 365 CPU/HTTP tests and three subtests. Three local MLX
skips receive native coverage on Studio. The Tailnet Studio passes 90 native and
targeted tests, including the nine-target distribution probe. The policy regression
uses the production admission, eviction, and policy-lease code, then verifies a
clean retry after the live owner releases memory. Ruff and diff checks pass.
These corrections do not change decoder or sampling arithmetic. The throughput
figures above use external workload timers and remain attached to their recorded
source snapshots; this pass does not repeat the full-model throughput sweep.

## Scope and remaining gaps

This validates the dense Qwen decoder. It does not qualify Flash, MoE, QSA, or
multimodal models. Long-context admission estimates remain conservative,
especially for shared warm prefixes. The recorded replay covers early agent
turns, not 32K–256K prompts or full 32K completions. Throughput runs exclude
background training. Full-model measurements use `bf16-fast`; legacy numerical
parity is covered by native small-model fixtures. Hardware throughput limits
above eight are not measured. A bounded full-model RPC update/recovery probe
and native fixtures cover cooperative training behavior; a resumed live
four-update pilot is a separate campaign qualification.

BF16 row-count changes can change matrix reduction order and token choices near
a categorical boundary. Exact RNG streams are tested independently with fixed
logits. Teacher-forced parity tests compare the same output tokens, rather than
requiring identical sampled text after a numerical difference. The declared
absolute record-logprob tolerance is 0.25; full-vocabulary diagnostics use FP32
normalization and a maximum KL divergence limit of 0.005 nats. Neither diagnostic
changes production sampling or its stored BF16 logprobs.

Memory-pressure admission and injected failure cleanup do not establish safe
operation after physical device exhaustion or a driver failure. No long soak,
power-loss test, or physical OOM experiment is part of this change.
