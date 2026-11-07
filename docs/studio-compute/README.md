# Studio compute

`studio_compute` is a standalone text sampling and LoRA training service for the
existing Qwen3.8-27B 8-bit artifact on Apple Silicon. It does not import
Shellsmith's trainer, worlds, rewards or campaign logic. Shellsmith uses the common
backend interface with separate Studio and Tinker adapters; Tinker remains the
repository default.

Model admission checks the supplied 27B config fingerprint, 64-layer/5120-width
dense architecture and affine 8-bit/group-64 quantization before importing MLX
or loading weights. It uses the same approved config fingerprint as historical
MLX readiness. Training and inference compare discovery identity with their preflight snapshots
before creating a remote client or publishing immutable run artifacts. Capability preflight also checks the selected
model, context and advertised cache precision before publishing an immutable run
snapshot. Supplied policy references receive a read-only service lookup before
inference client initialization; unknown policies leave the campaign directory
reusable. Trainer and campaign cleanup acquire ownership only after successful
initialization, so a rejected competing create cannot suspend a live owner. Rollout finalization
closes registered episodes before draining memory, rejects new requests, and retains
in-flight handler ownership until their last exit releases policy pins. Late results
remain diagnostic JSONL evidence and cannot repopulate episode memory. Closed IDs
leave the lifecycle map after handlers finish. Standalone CLI endpoints retain
ad-hoc routing. Studio token-level outputs remain in compute receipts; normal and
recovered trainer metrics omit their logprob arrays from journal/state/checkpoint
metadata.

## Start the service and coordinator

On Studio, install the pinned optional runtime with `pip install '.[mlx]'` and
configure the model/state paths in [studio-service.toml](../../examples/studio-service.toml):

```sh
studio-compute serve --config examples/studio-service.toml
```

The default endpoint is `http://127.0.0.1:8766`. Remote coordinators use an SSH
tunnel to that loopback endpoint. Public and wildcard binds are rejected by both
the CLI and HTTP server factory. Install `.[studio-client]` on the coordinator;
it needs neither MLX nor the Studio model filesystem. The service holds an exclusive
lock on its canonical state root before engine construction, in addition to its
GPU lease; changing the GPU lease path cannot create a second state-root owner.
The service supplies a hash-verified tokenizer/template bundle. Malformed Studio policy references are
rejected locally before service discovery or inference-client creation. Clients publish complete immutable asset
directories atomically, so concurrent coordinator setup cannot expose partial files.
Tokenizer publication fsyncs each staged asset and its directory before rename,
then fsyncs the cache parent before returning success. Cache hits also complete the
parent barrier, allowing retry after a post-rename sync failure.

The [micro recipe](../../examples/studio-micro-train.toml) requests four attempts
on one world, four concurrent GMs, and at most one optimizer update:

```sh
shellsmith train --config examples/studio-micro-train.toml --dry-run
shellsmith train --config examples/studio-micro-train.toml --log-dir /path/to/private/run
```

These commands do not impose a whole-run deadline. A bounded launch must separately
own its service, trainer and GM descendants, preserve partial evidence, and stop
all owned work at its absolute deadline. The example's 7,200-second Player request
timeout, 900-second GM invocation ceiling and 180-second GM API inactivity
timeout are separate limits. The two-hour absolute run deadline still governs
all work and cleanup. Compute RPC polling has a
separate `[studio].operation_timeout_seconds` setting (43200 seconds by default);
sampling and token scoring retain the Player deadline. No failed or stopped run is
automatically authorized for retry, top-up, resume or extension.

## One context window, independent output budgets

Studio uses `context_window = 262144` for the rendered prompt/history, reasoning,
response and control IDs together. There is no fixed prompt partition. Native
reasoning budgets are low = 2048, medium = 4096 and xhigh = 8192, with an independent
`response_tokens = 32768`. The world-step ceiling is 256. A step is a world-resolved
sibling-call batch, not an individual token or shell command.

The medium thinking reserve is 36867 tokens: 4096 reasoning + up to three forced
closing IDs + 32768 response. Exact tokenizer admission reserves the selected turn's
maximum output. If it does not fit, generation fails before execution and the
client must compact; prompts are not silently truncated. Unused reasoning tokens
do not enlarge the response allowance. `max_tokens = 0` derives the reserve;
positive historical values remain combined ceilings. `thinking_tokens = 0` selects
the effort-specific budget; a positive value remains an explicit legacy override.
`plain_tokens = 0` follows the response limit.

The official served Qwen template owns effort instructions. Medium adds no extra
effort system instruction; low and xhigh use its corresponding instructions.
Reaching a reasoning limit records `reasoning_limit_exceeded`, closes thinking,
and continues the response pass. Injected close IDs have scored probabilities and
are masked from policy loss. A response limit records a diagnostic; rejected or
unconsumed responses retain diagnostics in `logs/generation_diagnostics.json`,
including when no trajectory is accepted. Observatory displays these notices.

Episode setup, sampling and auditing share one effort-to-mode mapping. Low,
medium and xhigh enable reasoning. Explicit off aliases disable it; unknown labels
fail explicitly. Pinned protocol expectations must agree with the recorded run
configuration. Supplied phase budgets replace the old fixed budget metadata.

## Resident execution and memory

A dedicated GPU owner loads one frozen quantized base. Training clients own
independent BF16 LoRA factors, FP32 master weights/Adam moments, and RNG streams.
Coordinator cleanup suspends its client, releasing accelerator state while
retaining the durable checkpoint and journaled RNG cursor. Restart leaves all
clients cold; training or implicit-seed sampling restores a client on demand.
Inference campaigns allocate a campaign-specific client and use a safe default
learning rate, ignoring `[train]` settings and any configured training client.
Explicit-seed sampling needs only its published adapter. An explicit `close` RPC
retires the client; `restore` can provision a new client directly from a checkpoint.
Published factors are immutable. Switching policies substitutes factor references
without merging or duplicating the base. Each rollout pins its policy revision.
Published references register checkpoint metadata and load factors on demand. A
four-entry LRU retains idle factor sets; memory admission can evict them sooner.
Active samples pin their factors across scheduler yields. Closing a client evicts
idle policies, while durable references remain available for later reload and
fresh-process recovery. Startup does not load historical policy factors.

Native MLX inference uses a persistent request queue with independent caches and
positions. Set `[service].inference_concurrency` or `--inference-concurrency`
(default 8) independently of episode-worker parallelism. There is no fixed
four-request ceiling. Memory admission can keep fewer requests active; excess
requests stay queued. Different prompt lengths, output limits, and cached
prefixes share decode rows for the same immutable policy. A boundary performs at
most one `prefill_tokens` chunk, then advances ready rows. EOS, cancellation, and
length limits release slots for new requests at the next boundary.

`POST /v1/operations/<id>/cancel` and `Client.cancel(id)` request sample cancellation.
Poll the same ID for its terminal `interrupted` receipt. Disconnecting an HTTP
poll does not cancel a durable operation. Runtime source hashes include the queue
and row decoder; the existing checkpoint identity checks remain in force.
See [continuous batching validation](continuous-batching/README.md) for measured
hardware evidence and remaining qualification gaps.

Grouped training shares compatible prefixes, tiles the vocabulary head,
recomputes activation groups, and differentiates explicit recurrent/attention
carries. Materialized segment/layer-group units yield to the round-robin scheduler,
which can offer sampling between training units.

The service TOML exposes replay segments (2048 tokens), checkpoint groups (8
layers), vocabulary tiles (128 rows), and prefill chunks (2048 tokens; currently
limited to at most 2048 by the scratch admission estimate). CLI flags override
TOML; unknown settings and invalid values fail before service startup.

Default host/GM headroom is 16 GiB, giving a 240 GiB allocator budget on a 256 GiB
Studio. Metal's approximately 223 GiB recommended working set is recorded
separately. Admission accounts for resident weights, adapters/optimizer state,
active caches, retained prefixes, replay storage and scratch. The automatic prefix
KV pool uses remaining admitted capacity and evicts for active work. The 2 GiB
Metal freed-buffer cache is separate from this pool. Zero prefix-cache capacity
disables retention; an explicit capacity caps it.

Activations, adapter factors and qualification attention KV use BF16. Optimizer
state and native recurrent state retain FP32. Supported attention caches also have
an 8-bit inference option; training carries remain differentiable. Metrics record
actual precision, allocator peak, RSS, swap growth, cache reuse and timings.

Model-specific layer math, factor targets, quantization/layout and carry estimates
live behind the model interface. Numerical extraction attribution and upstream
license notices are in [numerics](../../src/studio_compute/numerics/NOTICE.md).
Flash execution is deferred; its architectural target remains a 262144-token
window with 8-bit dense/shared weights, 4-bit routed experts and initially 4-bit
n-gram embeddings with a 6-bit option. Multimodal training, speculative decoding
and unmodified Tinker SDK wire compatibility are also deferred.

## Portable operations and durability

`GET /v1/capabilities` returns model/runtime identity, context, supported losses and
cache precisions, plus physical/allocator/working-set information. `GET
/v1/tokenizer` returns the tokenizer/template bundle. `POST /v1/operations` accepts
`id`, `client`, `kind` and `body`; `GET /v1/operations/<encoded-id>?wait=30` polls
its asynchronous receipt. The Python `studio_compute.client.Client` supplies
blocking and asynchronous helpers.

Operations are `create`, `close`, `suspend`, `sample`, `logprobs`, `forward_backward`,
`optim_step`, `save`, `restore`, `publish` and `inspect`. Numerical requests use
portable token IDs and aligned arrays. Sampling returns IDs, behavior logprobs,
actual counts, a pinned policy revision and execution metrics. V1 supports CISPO
and weighted cross-entropy, not arbitrary executable callbacks. Forward metrics
include `loss_kind`, the generic `loss`, and the matching objective-specific loss
key; weighted cross-entropy is not reported as CISPO. Temperature and
top-p are both 1 for exact behavior probabilities.

Mutation IDs are durable: identical input returns the original operation and
conflicting input fails. Mutations serialize within a training client. A create
interrupted before any durable checkpoint/client commit can safely retry its
original ID; interrupted optimizer operations are never blindly re-executed. Generated
operation IDs use stable digests when labels exceed the RPC limit, including durable
update IDs and their backward/optimizer suffixes. Full-state
checkpoint files and directories are fsynced and atomically renamed before a
successful optimizer receipt commits. Slow serialization, hashing and fsync run
outside the global polling lock. Sampler saves reuse the durable weight reference.
State saves and restores share immutable adapter/master/moment files through hard
links and write fresh RNG state, avoiding redundant full optimizer payloads.
Committed optimizer versions remain retained for receipts and restore references;
checkpoint storage still grows with actual updates and requested snapshots.
Lazy policy registration retains the verified manifest's adapter digest. Cold loads
and loads after eviction rehash the file before decoding factors; corruption cannot
redefine an immutable policy ID. Resident pinned factors retain their original data.

Publication requires the matching durable
revision. Compute identity includes replay segment size, vocabulary tile size,
checkpoint layer groups, and prefill chunk size; changing these rejects recovery
under the prior identity. Restart verifies checkpoint hashes/runtime identity, reconstructs
committed receipts and marks unfinished work interrupted. Corruption fails closed;
uncommitted gradients cannot become a restored backward pass.
Recovery removes abandoned checkpoint staging directories with the exact generated
name format, without following staging symlinks or removing committed checkpoints.

Studio inference client names include the run label and canonical campaign output
path. Training mutation IDs (including trainer segment updates), sampling and token-logprob
operation IDs include the selected-client backend namespace,
since the service receipt table uses globally unique IDs. Independent output
directories may reuse a label without sharing receipts or RNG cursors; retries
and path aliases for the same output retain stable IDs. Reusing a nonempty
campaign output remains forbidden.

Studio checkpoints and policies are opaque `studio-state:` and `studio-policy:`
references. Shellsmith commits its own cursor and metrics before switching to the
new published policy. Historical `compute.backend = "mlx"` artifacts and frozen
runtimes retain their original format; they are not silently migrated.

Optimizer admission includes old and prospective FP32 state, BF16 publication,
and a conservative bound for unfused Adam arithmetic intermediates. Backward
checks that its following optimizer step can fit, then retains the optimizer
reservation across RPCs until step, abort, restore or client closure. Interleaved
clients and disposable caches cannot consume that reserved headroom. Restore
checks the transient blank-plus-loaded state and conversion/validation workspace
before allocating a cold client or loading checkpoint tensors. These are
conservative byte estimates; CPU admission fixtures do not measure native Metal
peaks. Neither check changes optimizer precision or objective semantics.

Restore acquisition rejects a client claimed by a live coordinator, including
between backward and optimizer, without entering rollback or clearing gradients.
A durable run-owned initialization journal separates uncertain pending submissions
from completed sessions. Pending retries reuse their operation/release IDs; a later
session rotates its attempt and restores the checkpoint again, including RNG and
volatile gradients. A process-held coordinator lock excludes simultaneous use of
one output. Release requests fence against the current acquisition receipt so an
intervening owner cannot be suspended. Journal writes precede release/restore RPCs. Initialization and
release submissions carry the discovered compute identity, checked by the service
before recording an intent or allocating state. Managed sampling passes capture
that acquisition token; suspension or a replacement acquisition rejects late
passes before RNG assignment or lazy client allocation. Raw RPC clients may still
sample cold durable clients when they intentionally omit an acquisition fence.
Canonical run output is part of every backend operation namespace: separate run
directories sharing a label and client cannot inherit one another's acquisition
receipt. Path aliases retain the same namespace and directory lock.
Explicit suspend/close releases the claim; a fresh service starts unclaimed after
interrupting volatile work. The owning backend explicitly releases before its own
controlled restore and never releases a client it failed to acquire.

Terminal request bodies are compacted while preserving their input hash, original
result and optimizer-to-backward recovery link. Queued/running work retains its
body; safe interrupted-create retries restore it from the verified retry input.
Recovery also compacts older terminal rows. SQLite reuses freed pages; this does
not delete checkpoint/result evidence or immediately shrink its file high-water mark.
Runtime/cache/service/CLI/storage/protocol/HTTP source hashes bind the engine
identity used by checkpoint verification, policy references and coordinator preflight.

## Objective and editable configuration

CISPO uses sum reduction and rollout normalization once, before compute submission.
Adam defaults are beta1 = 0.9, beta2 = 0.95 and epsilon = 1e-8, with no weight decay
or additional gradient-norm clipping. These semantics do not use High Command's
averaging or optimizer defaults. The training TOML exposes optimizer settings,
credit coefficients/floors/caps, request/artifact retries, sampling and compaction
alongside rollout/update settings. The service TOML owns execution and memory
settings. Aggregate update token guards are zero in the micro recipe, while every
sequence still must fit context and memory admission. The HTTP boundary also has a
128 MiB canonical JSON request limit, advertised in model capabilities. The portable
client checks exact serialized envelopes before RPC; Shellsmith preflights the
captured update before writing its pending-update marker or advancing its cursor.
Oversized batches fail locally with their byte count and preserve captures; V1 does
not split updates or implement chunked gradient accumulation. Training client IDs
must be 1–200 characters and are checked before immutable run setup. Studio train
and inference seeds must fit uint32; residual credit caps must be positive. These
checks also run before immutable artifacts or rollout collection.

Acceptance checks simulator/factual consistency and capture integrity separately
from task success. Protocol checks cover ordered proxy/Pi projections, bridge
requests when present, world action correlation, exactly-once returned tool results,
and compaction journal alignment. Malformed/duplicate capture evidence fails.
Model-authored truncation or schema errors can remain trainable negative behavior
when the capture and simulated world are intact. GM success or reward alone cannot
override corrupted capture evidence.

The local MLX adapter also defaults `thinking_tokens` to zero, selecting the
2K/4K/8K reasoning budget by effort. Positive historical overrides and explicit
total ceilings remain effective. Its 32K profile needs a compaction reserve above
12,291 tokens for xhigh with a 4K response (for example, 16,384); insufficient
headroom fails configuration preflight. Historical example TOMLs retain their
explicit 4K thinking and 8K total limits, and frozen runtimes are unchanged.
Standalone MLX episodes forward the complete `[retry]` section to their GM/Pi
smoke handoff, including disabled Player retries and GM artifact attempt limits.

## Validation and limits

CPU fixtures cover durable retries/conflicts, interrupted work, checkpoint
corruption and optimizer/publication failures, remote-client independence,
configuration/budget/template behavior, protocol corruption and Tinker regressions.
Metal-only BF16 fixtures in `tests/test_studio_numerics.py` cover full-prompt
replay/carry gradients, masks, optimizer recovery and frozen-base invariance; they
skip explicitly on hosts without the MLX runtime.

For a matched light benchmark, see `scripts/benchmark_studio.py` and its pinned
High Command reference under `benchmarks/high_command_342d`. The standalone
`scripts/qualify_studio.py` constructs direct full-context fixtures and recovery
checks. These tools are opt-in; this PR does not launch qualification or a junior
run, and configured 262144 capacity is not proof that a given run exercised it.

The compact [protocol review](protocol-review.json) records an offline recheck of
four completed micro-r3 trajectories. Correcting only their wrong reasoning
expectation in temporary copies makes three acceptable; the fourth retains a
terminal-output failure. The original run committed zero updates, and its audits,
policies/checkpoints and raw archives remain unchanged outside this PR. No training
update with the corrected runtime is claimed. A future run requires a fresh frozen
runtime, and learning quality requires its own evaluation.

### Diagnostic and GM account guards

Pinned GM logins use the checked command-backed credential provider for both
Bladerunner registries and file-backed ChatGPT caches. Each credential fetch
verifies the exact token snapshot against the account pin; same-account refresh
remains valid and replacement accounts fail closed. The selected login owns
refresh. Tokens are not copied into flags, environment variables or audit logs.
An expired owning login must be refreshed there.

The bounded diagnostic wrapper records `startup_exit` and fails if a child exits
before its readiness file appears, including exit status zero. Fast successful
children that publish readiness are accepted. The runtime smoke suspends its
live isolation-control client before restoring the checkpoint. These changes
were tested with CPU/local process fixtures; no real runtime probe was launched.

## Run reliability

See [monitoring and GM timeout controls](runtime-reliability.md) for isolated
telemetry, terminal status reconciliation, and separate 900-second invocation /
180-second API inactivity limits. The micro recipe uses medium GM reasoning and
a 32K Player response cap. These changes require a fresh frozen runtime; existing
run enclosures and their two-hour absolute deadlines are preserved.

Local sampling now records throughput and actual request occupancy at the compute
boundary. Read the live counters with `studio-compute metrics` (or
`studio-compute metrics --endpoint http://HOST:8766`); the same data is available
from `GET /v1/metrics` and `studio_compute.client.Client.metrics()`. The response
includes current active/queued sampling requests, cumulative generated tokens,
and the latest interval's tokens per second, time-weighted average, and peak
concurrency. Before the first interval, `latest_interval` is null.

The service appends nominal one-second measurements to
`STATE_ROOT/inference-telemetry.jsonl`, with a final partial interval on clean
shutdown. Each row records its actual monotonic duration, generated token count,
tokens/second, current/average/peak active and queued requests, and completed or
failed sampling counts. Active requests have entered local sampling and have not
finished generation, including prefill and pauses between cooperative GPU turns;
this differs from the number of rows executing in a single GPU batch. A batch
member leaves the active count when it reaches EOS or its output limit. Queued
requests have not started sampling. Configured rollout parallelism and maximum
GPU batch size remain capacity settings, rather than measured activity.

Throughput is generated output tokens divided by the interval's full wall time,
including prefill, training interleaving, and idle time. It excludes prompt tokens,
forced-token scoring, and training targets. Tokens already generated by failed
requests count as work, and retries of a durable operation do not count again.
Disk logging runs on a separate thread, with failures isolated from computation
and exposed through `logging_errors`. Service restarts receive a new session ID;
crash-tail intervals and failed writes may be missing and are never fabricated.

The offline report reads `state/inference-telemetry.jsonl` and aligns throughput
and concurrency charts over identical measured windows. Choose the target window
with `scripts/render_studio_run_report.py --throughput-window-seconds 1` (or 10,
60, or another positive duration), alongside the existing `--run-root` and
`--output` arguments. Windows combine whole recorded intervals, report actual
duration, and preserve gaps/restarts. Active and queued averages are weighted by
time; peak concurrency preserves short requests that finish between monitor polls.
Historical runs without these intervals remain explicitly unmeasured. Freeze the
updated source for new runs; decode progress instrumentation changes the engine's
source identity, so old frozen checkpoints are not silently migrated.
