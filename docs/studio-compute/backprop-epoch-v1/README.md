# Prepared catalog epoch and combined Backprop qualification

The epoch is **prepared, not launched**. The historical junior12 experiment remains stopped. The original dirty checkout and PR #17 checkout are preserved. This branch starts at merged PR #17, `6a95f9adf7769b796fedaabe543cf4c107f85b7a`, and selectively ports the original telemetry module, its tests, configuration, dependency, and trainer hooks. It retains current main's numerical code, packed segments, acquisition ownership, preflight fences, and checkpoint recovery.

## Prepared run

From the repository root:

```sh
PYTHONPATH=src python scripts/plan_catalog_epoch.py
```

The approved manifest, dated 2026-10-18, selects 177 unique worlds. `hero_runs/studio_catalog_epoch_v1.json` pins the manifest SHA-256 and all selected world bytes, in manifest order. `examples/studio-catalog-epoch.toml` preserves the existing reward/credit/CISPO/mask/context-gradient/sampling/Adam recipe. The seed is 20261019. The target is one pass: 1,416 **logical attempts**, not 1,416 accepted episodes. Each world has eight initial and maximum attempts; existing within-attempt protocol/provider retries remain. Provider outages halt the stream; this plan does not add backup groups or acceptance top-ups.

Two ready world groups accumulate into each Adam update. The final group contains one world. There are at most 89 committed updates; skipped groups can reduce this number. Eight rollout/GM slots share the existing GPU batch ceiling of four compatible inference requests. An outage or manual interruption can stop the pass early. The schedule does not cycle worlds.

Service settings remain `bf16-fast`, the frozen Qwen3.8-27B 8-bit backbone, BF16 KV, automatic allocator admission, a 16 GiB optional saved-activation cap, and automatic prefix-cache eviction. Host headroom is 16 GiB; on the 256 GiB Studio the allocator ceiling is approximately 240 GiB. BF16 admission also respects the lower device recommended working-set limit (222.72 GiB in the failed automatic-retention probe). The Metal freed-buffer cache is 2 GiB. Layer-checkpoint interval remains automatic; replay segments and prefill chunks are 2,048 tokens, and head blocks are 128 rows. Admission can select smaller batches or reject an oversized operation; this small qualification does not establish every catalog trajectory's memory fit.

`epoch-plan.json` records all update groups and estimates 118.5 hours (4.94 days). This assumes 30 minutes per eight-attempt world and the prior merged full-batch BF16 median of 1,214.22 seconds per update. A 10–60 minute collection assumption gives 59.5–207.0 hours (2.48–8.63 days), before outage, startup, checkpoint, and publication overhead. Collection time is a planning assumption; the update timing comes from prior evidence and is not a fresh performance claim. The 16 GiB retention cap can increase reconstruction work, so the current profile can take longer than that prior timing. Live ETA uses measured elapsed time per newly completed world, includes update overhead, and is explicitly named `train/eta_seconds_estimate`.

## Launch, stop, recovery — separate future user action

These commands are instructions only. Run them on Studio from a checkout of `origin/main` after this PR merges. Use the existing Python and pinned Backprop SDK; do not replace installed MLX. The default Codex profile at `~/.codex` was verified as authenticated. The inherited `CODEX_HOME=~/.local/share/shellsmith/gm-codex` was unauthenticated, so the launch environment selects the working profile explicitly. Credentials stay in place.

Once per fresh checkout, install the lockfile-pinned episode runner:

```sh
npm --prefix src/shellsmith/episode/vat_runner ci --ignore-scripts --no-audit --no-fund
```

Run this environment setup in **each terminal**, from that checkout root:

```sh
export CODEX_HOME=~/.codex
export PYTHONPATH=src:~/.local/share/shellsmith/pilot-backprop-0b57d68
PY=~/.local/share/shellsmith/studio-v1/env/bin/python
RUN=~/.local/state/shellsmith/catalog-epoch-v1
mkdir -p "$RUN"
```

Terminal 1 starts the service with isolated state and the existing GPU lease:

```sh
"$PY" -m studio_compute.cli serve \
    --config examples/studio-catalog-epoch-service.toml --state-root "$RUN/service"
```

After the service is ready, Terminal 2 checks readiness without an epoch launch:

```sh
"$PY" -m shellsmith train \
    --config examples/studio-catalog-epoch.toml --log-dir "$RUN/train" --dry-run
```

The full campaign starts only with this separate future user action in Terminal 2:

```sh
"$PY" -m shellsmith train \
    --config examples/studio-catalog-epoch.toml --log-dir "$RUN/train"
```

Use Ctrl-C in the trainer terminal, allow cleanup to finish, then Ctrl-C in the service terminal. Preserve the entire service and train directories. Restart the same service command, then the identical trainer command and log directory to recover. Keep source, pinned worlds, recipe, model, and runtime identity unchanged. Do not delete acquisition journals, metrics receipts, or service checkpoints. An ambiguous optimizer update is reconciled by its durable operation ID. Resume restores completed offsets plus sparse settled offsets; it does not infer completed work from telemetry acknowledgments.

Backprop `published-*.jsonl` files record SDK-journal acceptance, not proof of remote visibility. `backprop.json` records every SDK sync directory. Use the SDK's `python -m backprop sync <sync_dir>` with the same existing connection to retry unacknowledged uploads. Resume also replays durable metrics, outcomes, and policy-publication receipts. Update rows retain their source segment; episode replay reads the segment from the episode ID. Legacy metric rows without source provenance leave the segment unset. A crash between SDK append and receipt persistence is at-least-once; `train/update_id`, publication identity, and episode revision identity support reconciliation. Committed updates are absolute counters, never a sum of delivery acknowledgments.

## Telemetry semantics and bounds

The 128-item publication queue is nonblocking. A full queue increments a deferred counter; authoritative update/outcome/publication files remain for replay. Progress samples are globally coalesced to one per configured interval (15 seconds for this epoch); errors bypass sampling. Measured GM turn counts and wall seconds include coalesced events and are labelled per segment. Detailed local episode artifacts remain authoritative. Background health polls every 15 seconds and persists `telemetry-health.jsonl`.

The live HTTP metrics route reads existing CPU counters and durable active-operation progress. It never asks the GPU owner to inspect tensors. The qualification observer requires a numerical backward phase or a positive elapsed counter for a backward phase; forward progress alone does not qualify. It reports generated-token service throughput, active/queued requests, admitted batch sizes, and forward/backward/accumulation progress. Active request count is not batch occupancy. Service throughput includes prefill, idle time, and cooperative scheduling; it is not isolated GPU decode-kernel throughput. Episode completion tokens/second includes GM and tool time. MLX and RSS peaks are process-lifetime peaks; they are not reset per update.

The combined RPC receipt retains `backward_execution_metrics` and `optimizer_execution_metrics` separately. Legacy top-level execution fields retain optimizer semantics. Forward/backward phase counters, activation retention, reconstruction, cache occupancy/eviction, MLX active/peak memory, process RSS, and swap remain in operation progress/receipts and sanitized update records. Checkpoint durations and durable policy-publication durations are separate. Full checkpoint timing uses a durable publication receipt after the checkpoint seals its exact metrics row. Fresh MLX and Studio runs persist revision-zero sampler/state checkpoint timings and the initial policy-publication duration before telemetry starts; startup replay uploads them. Provider-down slots marked `episode_not_started` do not count as attempted, failed, rejected, accepted, or discarded episodes. Episode outcomes retain audit/reward components, acceptance, failures, policy identity, counts, reasons, and text trajectories.

Text trajectory uploads are capped at 131,072 UTF-8 bytes for the complete serialized preview envelope; full local artifacts remain available. Table objects are released after publication rather than retained for the whole campaign. Credentials, raw replay arrays, and model weights are excluded. Console capture is disabled in the prepared epoch. Backlog, queue depth, deferred events, logging failures, SDK upload-error state, and last live timestamp diagnose telemetry health. A stale timestamp without finalization indicates an interrupted process; no automatic external alert is installed.

## Qualification evidence

The portable matrix passed **524 tests**, with zero skips, failures, or errors. `portable-tests.json` records the exact command. The [live PR checks](https://github.com/tginart/shellsmith-dev/pull/18/checks) are authoritative for full CI on the exact reviewed head. Both bounded real-27B probes and the Backprop readback passed. The live observer verified numerical backward work; the summary records measured phase evidence. `qualification-summary.json` and `backprop-readback.json` contain compact receipts. Raw GPU reports and live samples stay outside the source diff; the summary preserves their hashes and disposable Studio location.

The exact qualified implementation is `7235e72212304932a3002ed77c6ca64638f486f3`, scoped source SHA-256 `73a83f372bd7497abf141be29fcbf5775961e48b48b84e2cfda88b3402f16994`. `source-identity.json` lists all 193 runtime files, six qualification helpers, and the fixed calibration file. All 200 files were hash-verified on Studio. Later evidence and documentation commits do not change those qualified file bytes. `verified-source.json` identifies the unchanged installed numerical dependencies and reused pinned Backprop SDK.

The initial automatic-retention probe failed interleaved inference admission: 243,511,541,760 required bytes exceeded the 239,143,780,352-byte recommended working-set bound. The paired epoch service profile caps optional retained activations at 16 GiB. This capped profile passed interleaved inference, one BF16 update, exact recovery, and eviction. This is a memory-profile qualification; the numerical code remains unchanged.

The disposable Studio source is `/tmp/shellsmith-backprop-review-7235e72-20261019/source`. The memorization probe completed in about 141 seconds with a 3,600-second hard deadline. The RPC probe completed in about 101 seconds with a 3,500-second hard deadline. Each GPU probe owned the shared lease, serially. The final check found the lease available and all historical experiments stopped. Model files, historical checkpoints, and installed MLX remain unchanged.

Backprop initialization initially exceeded a 30-second history-read limit. The explicit initialization limit is now 300 seconds; publication and resume passed with this setting. The verified qualification run (`shellsmith-qualification-7235e72-20261019` in project `/shellsmith-qualification`) contains all **35 expected events and 5 downloaded, hash-verified tables**, exactly one committed update after resume, and visible synthetic failure/interruption fixtures. The test uploads contain real 27B receipts, source identity, and measured counters; they omit credentials, weights, and raw replay arrays.

Reproduce serially in fresh disposable output directories:

```sh
PY=~/.local/share/shellsmith/studio-v1/env/bin/python
PYTHONPATH=src "$PY" scripts/qualify_bf16_memorization.py \
    --model ~/models/Qwen3.8-27B-8bit \
    --execution-profile bf16-fast \
    --calibration docs/studio-compute/bf16-backward/legacy-memorization.json \
    --output /tmp/fresh-memorization --timeout-seconds 3600
# Start the CPU-only observer in another terminal before the RPC probe:
# PYTHONPATH=src "$PY" scripts/observe_studio_metrics.py --output /tmp/fresh-live.json
PYTHONPATH=src "$PY" scripts/run_bounded_studio.py --ready-immediate \
    --receipt /tmp/fresh-rpc-execution.json --seconds 3500 -- \
    "$PY" scripts/smoke_studio_runtime.py \
    --model ~/models/Qwen3.8-27B-8bit \
    --output-root /tmp/fresh-rpc --port 8773 --activation-budget-gib 16
PYTHONPATH=src:~/.local/share/shellsmith/pilot-backprop-0b57d68 \
    "$PY" scripts/qualify_backprop_stack.py \
    --rpc-report /tmp/fresh-rpc/report.json \
    --memorization /tmp/fresh-memorization/report.json \
    --live-observation /tmp/fresh-live.json \
    --source-identity docs/studio-compute/backprop-epoch-v1/source-identity.json \
    --output /tmp/fresh-publication --run-id shellsmith-qualification-UNIQUE
```

The first probe uses fixed seeds 23 and 91, the frozen calibrated learning rate, ten updates, exact restore at update five, and the zero-rate control. The RPC probe exercises the real production HTTP service, one committed BF16 CISPO/Adam update, client isolation, eight queued inference requests, actual compatible batching, checkpoint/policy publication, fresh-process exact state and next-sample recovery, and bounded cache eviction. These are qualification fixtures, not catalog training episodes.

The publication probe uploads the actual RPC receipts and fixed-learning checks under `shellsmith-qualification`. The failed episode and interrupted controller are explicitly synthetic fixtures. It resumes the same run, requires exactly one committed update row, reads expected stable event IDs from the remote Foundry stream, and downloads content-addressed table bytes with hash verification. An SDK acknowledgment alone cannot pass this probe. The historical 5,796 acknowledged events are not used as current qualification evidence.

## Coverage matrix

| Area | Coverage and limit |
| --- | --- |
| Identity and recipe | Run config, source/model/kernel execution identities, manifest/world byte pins, seed, policy references. |
| Episodes and rewards | Attempted/accepted/rejected/failed/discarded episode rows; skipped-world counters; audit components and bounded text trajectories. Episode revisions supersede previous acceptance. |
| Timings | Collection, measured GM turns, service inference/queue occupancy, backward phases/accumulation, separate optimizer receipt, sampler/full checkpoint and policy-publication duration. GM bootstrap/judge totals and pure GPU decode-kernel time are not separately aggregated. |
| Live compute and memory | Existing active-operation progress, generated-token service rate, queued/active requests, admitted batch occupancy; MLX/RSS/swap/cache/activation/reconstruction receipts. Peaks are process lifetime; no update-local peak claim. |
| Progress | Absolute committed updates, durable world cursor and sparse settled offsets, remaining worlds, measured-progress ETA estimate. |
| Publication and interruptions | Bounded queue, sanitized durable sources, SDK backlog/error state, heartbeat timestamp, failed/interrupted finalization, restart/replay tests and remote readback. No external unattended alert or multi-day soak is claimed. |
| Model qualification | Real 27B BF16 production update plus fixed memorization/restore. The full 177-world campaign and a new full-context/full-batch performance suite are intentionally not run. |
