# Generative world runtime prototype

> **Historical design reference:** the package map, hosted endpoint examples,
> and conductor/training commands below describe the original monorepo, not
> supported standalone interfaces. The conductor is not shipped in Shellsmith.
> See [the root README](../README.md) for the current standalone layout, defaults,
> and milestone availability. The vat-runner development commands at the end
> are updated for this checkout.

This package implements the no-training runtime documented in:

- [`GAME_MASTER_PROTOTYPE.md`](GAME_MASTER_PROTOTYPE.md)
- [`WORLD_FILE_SPEC.md`](WORLD_FILE_SPEC.md)

## Package map

Two scopes, mirrored in the filesystem:

```text
generative_world/
  contracts.py          Shared four-section Markdown world file grammar.
  prompts/              Game Master AGENTS.md + boot/turn/final templates.
  worlds/            Draft and checked-in task Markdown worlds.
  docs/                 Design notes and harness audits.
  viewer/               Read-only episode artifact browser (Observatory).
  hero_runs/            Checked-in hero slate JSON.
  artifacts/            Checked-in smoke/campaign logs (optional).

  episode/              ONE Player ↔ Game Master rollout
    smoke.py            Live smoke CLI entrypoint.
    orchestrator.py     Episode lifecycle / durable state machine.
    ...

  campaign/             MANY tasks / MANY episodes (one durable enclosure)
    draft_catalog.py    Flavor inventory + authoring QA over drafts.
    hero_slate.py       Fixed task × Game Master assignment queue.
    rollout_campaign.py One-shot multi-task tryout queue + scoreboard.
    stream_campaign.py  open / wave / finalize for multi-wave streams.
    checkout_batch.py   Bounded-parallel checkout runner.
    provider_outage.py  Game Master outage classify + Sol↔Opus / terra↔Sonnet remap.

  conductor/            Full training-run scripting (above campaign)
    conductor.py        Hero stream + Tinker GRPO/optim/checkpoints.
    stream_policy.py    AdaptivePolicy, additional_attempts, world decision table.
    stream_engine.py    Per-world adaptive lifecycle inside one campaign.
    tinker_collect.py   Injectable Tinker rollout_runner + CaptureStore.
    to_tinker_bridge.py lives in player/rl/tinker/ (Accepted captures → Tinker training datums).
```

### Product model: campaign = stream of world files

A **campaign** is one durable folder / Observatory filter. It runs a **stream**
of world files. For each world file:

1. launch N rollouts that share one `RolloutGroupIdentity`;
2. that set is **one GRPO group** (group ⊂ campaign — never “campaign = one group”);
3. apply adaptive + coordination rules (top-ups, skip, Game Master fallback, halt).

```text
CAMPAIGN (Observatory filter / one folder)
  mode: inference | tinker_train
  stream cursor: next_world_offset
  worlds:
    task_A  → group_A  (rollouts …)  [adaptive lifecycle]
    task_B  → group_B  (rollouts …)
  coordination: provider halt / fallback cutover / stop stream
  if mode=tinker_train:
    after enough trainable groups → GRPO → optim → checkpoints → continue
```

| Mode | Collection | Training |
|------|------------|----------|
| **inference** | Stream + scoreboard (`campaign_report.json`) | None (tryout scripts) |
| **tinker_train** | Same stream via campaign waves + Tinker Player proxy | Conductor: GRPO, CISPO, optim, sampler/state checkpoints |

**Hard rule:** Conductor never calls `run_live_smoke` for collection. Every
episode goes through a campaign job (default runner or injectable Tinker runner).

### Adaptive controls (inside the campaign stream)

| Knob | Default | Meaning |
|------|---------|---------|
| `initial_attempts` | 4 | First batch of rollouts for a world |
| `maximum_attempts` | 8 | Hard cap per world (group) |
| `minimum_trainable` | 4 | Need this many accepted rollouts for a trainable group |
| `minimum_reward_range` | 0.15 | If accepted≥min but reward spread is too low → top up |
| `low_spread_top_up` | 4 | Extra attempts when spread is too low (capped) |
| `groups_per_update` | 2 | Trainable groups before one Tinker optim step |
| `provider_outage` | `halt` \| `fallback` | Stream-level infra policy |

Per-world lifecycle: initial wave → evaluate → top-up (same task folder,
continuing `rollout_NN`) / group_ready / skip_insufficient / discard-and-redo
on backup Game Master / halt stream. Fallback **never mixes models** in one GRPO
group: discard the failed group entirely, then redo the same world file as a
fresh group on the backup.

`campaign` calls `episode` (smoke / audit). It does not own the turn loop or
optimizer steps. `conductor` opens one campaign enclosure for the hero run and
owns the resumable training loop; tryout scripts may still call
`rollout_campaign` directly (one-shot wrapper over the same enclosure idea).

Provider outage policy (`--provider-outage halt|fallback`, default `halt`) lives
in `campaign/provider_outage.py`. Health is consecutive hard fails per
`gm_runtime` (default: 5 fails → 5 min cooldown → second streak marks down),
not error-string markers. Historical bidirectional Game Master backup metadata:

| Primary | Backup |
|---------|--------|
| `gpt-5.6-sol` (codex-cli) | `opus` (claude-cli) |
| `gpt-5.6-terra` (codex-cli) | `sonnet` (claude-cli) |

The map is symmetric (Opus→Sol, Sonnet→terra) and remains inspectable in
historical hero slates and offline audits. It does not enable those backups:
**GM execution is Codex-only**. Claude targets are rejected before execution,
not enabled by an explicit model, provider override, or fallback selection.

The vat protocol and bridge can be tested without a model. A Node child suspends
its virtual tool executor, Python obtains a world result, and only the public
tool response resolves the Pi Promise. The actual temporary workspace remains
`AGENTS.md`-only throughout the episode.

After boot, the orchestrator initializes `world/consistency_ledger.json` from
Sol's private commitments. Every later response must review the exact current
revision and report its claims, new commitments, conflicts, and executed checks.
The bridge appends that receipt before releasing the observation; Sol cannot edit
the ledger directly. Any conflict makes the rollout untrainable and terminates it.

The live path uses an existing Codex CLI login and installed Pi SDK; it never
installs dependencies. The only intentionally unfinished lifecycle feature is
process-crash recovery. A failed run still copies its staging audit directory
before raising, but it cannot yet resume from that cursor.

Model-generated world commands run under a custom Codex permission profile that
can read and write only the episode workspace plus the minimal system runtime and
may use the public internet. Virtual commands from Player are never host commands:
the Game Master decides whether to synthesize them, compute against state under
`world/simulation_root`, or fetch public sources and materialize them there. Public
network access does not permit reading private host data, credentials, or the vat.
This is the enforced host-data boundary; a unique cwd by itself is not isolation.

## Live smoke episode

Requirements:

- an authenticated `codex` CLI with `gpt-6.1-sol` access;
- the installed Pi SDK/CLI used by this workspace;
- HTTPS access to the OpenRouter endpoint, `https://openrouter.ai/api/v1`; and
- `OPENROUTER_API_KEY` in the environment for the default
  `qwen/qwen3.8-27b` Player. Tailscale and the legacy Lambda endpoint are not
  required by the standalone defaults.

Run one Sol/low-reasoning world against the live Player model:

```bash
python -m shellsmith.episode.smoke \
  --output-dir logs/tinker/generative_world_smoke/$(date -u +%Y%m%dT%H%M%SZ)
```

The defaults use a 65,536-token embedded context and at most 100 world-resolved
tool actions. A missing or malformed world artifact is retried on the same
Codex thread. Finished audits include the world file, generated vat `AGENTS.md`,
both session IDs, every intercepted action/result, raw Codex and Pi logs, the
final judgment, and the terminal episode state.

**Claude GM execution is disabled** in both host and Docker modes. Only the
Codex GM can execute, including through programmatic launchers. Legacy Claude
configuration, transcript helpers, and backup metadata are offline-only.

**Pi GM execution is disabled** in both host and Docker modes. Pi's approved
shell and read tools lack the filesystem/process confinement needed to protect
provider credentials and auth files. Filtering only the shell environment would
not provide that boundary. Use the confined Codex GM shown above; the previous
Pi-as-GM comparison walkthrough is no longer supported. Historical Pi transcript
parsing remains available for offline audits, not for resuming GM execution.

The **Pi Player SDK remains supported**. Its virtual tools still route through
the GM rather than executing model-requested commands on the host. This Player
role is distinct from the disabled Pi GM adapter.

Production rollout generation should run 4–8 independent episodes in parallel.
Actions inside one episode remain serialized so its world state has one clear
order.

Run four checkout episodes with four-way parallelism:

```bash
python -m shellsmith.campaign.checkout_batch \
  --count 4 \
  --parallelism 4 \
  --output-dir logs/tinker/generative_world_batches/checkout-4x
```

Each rollout receives `acceptance_audit.json`; the batch directory receives
`batch_report.json` and `batch_report.md`. Acceptance deliberately depends on
simulator integrity, factual consistency, and the host boundary—not task reward.
Reasoning scores, executed tests, and task success remain separate measurements.
For checkout, an atomic-fix claim is counted only when the terminal judge declares
it, the safe-fix and verification criteria each earn at least 80% of their weight,
and the transcript contains an executed passing test command.

## Adaptive RFT conductor

A hero Tinker run is **one campaign enclosure** under `--run-dir` (same layout
Observatory groups by campaign). The conductor streams the hero slate: for each
world file it runs the adaptive lifecycle via campaign waves (initial attempts,
top-ups into the same `{task}/rollout_NN/` folder, skip, or provider
discard-and-redo on backup Game Master). After `groups_per_update` trainable
groups, it builds Tinker datums, runs CISPO + optim, saves due checkpoints, and
continues the stream. Updates are sequential; inside an update, worlds are
scheduled greedily under dual episode caps (`--parallelism` total,
`--parallelism-per-provider` per `gm_runtime`).

`conductor.py` / `stream_policy.py` start with four rollouts per task and add
1–4 more when fewer than four are trainable or accepted rewards lack useful
spread. It never exceeds eight attempts. Group identities include task content, world runtime, model, provider,
reasoning effort, the complete simulator prompt bundle, turn budget, and the
Player renderer/token/sampling controls. Changing any of these values requires
a new run directory, so retries cannot mix differently sampled trajectories in
one relative-reward group.

Inference tryouts (`rollout_campaign` scripts 01 / 01b / 02) stay one-shot
multi-task campaigns with `mode=inference` — scoreboard only, no optim.

```bash
python -m shellsmith.conductor \
  --hero-slate backend/research/player/simulator/generative_world/hero_runs/hero_slate.json \
  --run-dir logs/tinker/conductor/baby-001 \
  --model-name ... \
  --tinker-base-url ...
```

### Operational run layout and inspection

Each conductor invocation owns one durable `--run-dir` that **is** the campaign
root:

```text
<run-dir>/                          # = Observatory campaign
├── campaign_manifest.json          # mode=tinker_train, slate hash, …
├── campaign_report.json            # rewritten after each wave
├── conductor_state.json
├── capture_index.json              # episode_id → durable capture / policy refs
├── pending_update.json             # stable update transaction until state commit
├── <world file-id>/
│   ├── rollout_01/ …               # initial attempts
│   └── rollout_05/ …               # adaptive top-ups (same group)
└── live_proxy/
    └── <episode-id>/proxy_logs/
```

Top-ups append under the same task folder; they do not mint a new campaign id.
The older `step_XXXX/task/rollout` layout is no longer the primary collection
path.

A world that fell back to the backup Game Master stores its redo group under
`<world file-id>__backup_<backup-model>/` instead. Two identities therefore exist
per settled world, and `conductor_state.json` records both on each step task row:

| Field | Meaning |
|-------|---------|
| `world_id` | **Stream** identity — the slate row. Unchanged by a fallback. |
| `rollout_world_id` | **Storage** identity — the folder that owns `rollout_NN`. Becomes `<task>__backup_<model>` after a fallback. |
| `gm_runtime` / `gm_model` / `gm_provider` | The simulator that actually produced those rollouts. |
| `original_gm_model` | The pre-fallback model, set only when `gm_fallback_applied` is true. |

`--retry` resumes from the **storage** identity. A fallen-back group tops up
inside its backup folder on the backup simulator, so one GRPO group never mixes
two Game Master models.

Every completed `rollout_NN` contains:

```text
episode.json                       Lifecycle status and failure, if any
run_config.json                    Embedded/world models and runtime configuration
scenario/                          Frozen task, boot result, and vat instructions
world/turns/000000/request.json    Player reasoning and tool action
world/turns/000000/response.json   Observation, turn reward, and assessment
world/final/response.json          Final reward and rubric judgment
acceptance_audit.json              Deterministic training-admission decision
smoke_summary.json                 Compact successful-rollout summary
policy_provenance.json             Which base/LoRA checkpoint sampled Player
logs/vat.jsonl                     Raw Pi/Player protocol transcript
logs/tinker_trainable_capture.json Rebuildable prompt_token_ids + tokens/logprobs
logs/durable_outcome.json          Finished-job outcome for in-place resume skip
logs/codex/                      Raw GM audit logs (legacy Claude logs: offline only)
```

Active episodes first use unique macOS temporary directories named approximately
`/private/var/folders/.../T/<episode-id>-world-<random>/`. Pi also receives a
separate `shellsmith-vat-<random>/workspace` temporary directory. These are working
directories, not durable storage. The conductor copies completed episode
artifacts into `--run-dir`; incomplete live work may therefore be visible only
under the temporary path until the episode settles.

`conductor_state.json` is the quickest machine-readable status view. It records
the run status, next world offset, completed RFT/SFT/optimizer update counts,
per-step group outcomes and optimizer metrics, plus every persistent
`tinker://` checkpoint handle.

Use the read-only Observatory dashboard to browse final and turn-level rewards.
It defaults to the same tree every `campaign/run_scripts/*.sh` recipe writes:

```bash
python -m shellsmith.viewer.app
# → logs/tinker/generative_world_campaigns/
```

Open `http://127.0.0.1:8734/`. The dashboard recursively discovers completed
`episode.json` files and exposes filtering by conductor run directory, task,
model, rollout status, reward, trainability, integrity, and acceptance.
Selecting a run directory also recalculates the summary cards over only that
run. Expanding a trajectory step
shows Player's action and reasoning, the simulated observation, turn reward,
quality/efficiency scores, completion estimate, and simulator commentary.
Training-update metrics are written live under each conductor run directory:

- `{run_dir}/tensorboard/` — TensorBoard event files (default on when training;
  disable with `--no-tensorboard`, or override with `--tensorboard-log-dir`)
- `{run_dir}/metrics.jsonl` — one JSON row per CISPO wave (same scalars)

Open TensorBoard with:

```bash
tensorboard --logdir <run_dir>/tensorboard
```

Scalars include campaign acceptance counts, reward mean/range, advantage
magnitude, and numeric Tinker `optimizer_metrics` under `train/`.

### Live Tinker bridge boundary

[`../../rl/tinker/to_tinker_bridge.py`](../../rl/tinker/to_tinker_bridge.py) is the in-memory RFT preparation boundary. Production
rollouts must sample Player through the existing Tinker proxy so each model
request keeps its live prompt object, sampled tokens, and behavior-policy
logprobs. Accepted captures are then written to
`logs/tinker_trainable_capture.json` (prompt token ids + tokens/logprobs) so an
in-place resume can rebuild CISPO datums after a crash.

**In-place resume:** point `--run-dir` at an existing campaign root that already
has `campaign_manifest.json` and `conductor_state.json`. Full optimizer-state
checkpoints remain controlled by `--state-checkpoint-every`. Each recorded full
checkpoint carries a conductor snapshot (update counters, committed step count,
world offset, and SFT cursor). Resume loads the newest such checkpoint, rewinds
the conductor to that exact boundary, and replays later committed updates in
order from `logs/tinker_trainable_capture.json`; it does not resample worlds or
keep counters newer than the loaded optimizer state.

`pending_update.json` has a stable update ID and base counters. Normal updates,
checkpoint replay, and pending recovery share the same RFT → due-SFT → checkpoint
→ atomic conductor-state commit path. The pending file is removed only after the
final conductor state is durable. A stale pending file whose update ID is already
committed is cleared without another optimizer call. Missing captures, checkpoint
snapshots, mismatched training settings, or inconsistent offsets/counters stop
resume explicitly rather than approximating recovery. New campaigns still
require an empty `--run-dir`.

The collector maps every live record either to one scored world action or
explicitly to an unscored final/compaction turn. The bridge then:

1. requires deterministic acceptance for every rollout;
2. rejects mixed conductor group keys;
3. centers/standardizes terminal rewards within the group;
4. passes signed `turn_reward` through as `reward_delta`;
5. treats missing quality/efficiency as zero reasoning correction; and
6. emits real `tinker.types.Datum` objects with prompt masking.

Completed Lambda/Pi calibration rollouts cannot be converted after the fact:
their text logs do not contain the live Tinker prompt object or behavior-policy
logprobs. The bridge fails rather than inventing either value.

## Large draft campaign

The authoring campaign targets at least 50 worlds in each of three flavors:
agentic API/tool investigations, pinned public-repository repairs, and recent-paper
audits. Paper drafts carry a private error-bearing/control label so the set can be
kept balanced without revealing the class to the Player.

Inspect the current counts and structural/provenance checks:

```bash
python -m shellsmith.campaign.draft_catalog
```

Require the full 50-per-flavor campaign gate and a near-even paper split:

```bash
python -m shellsmith.campaign.draft_catalog \
  --campaign-gate \
  --minimum-per-flavor 50 \
  --paper-balance-tolerance 1
```

After authoring, run one globally scheduled calibration rollout per draft with at
most twelve independent Sol/Pi pairs. Jobs are round-robin across tasks rather than
serializing one task group at a time:

```bash
python -m shellsmith.campaign.rollout_campaign \
  --from-draft-catalog \
  --rollouts-per-task 1 \
  --parallelism 12 \
  --exclude-task public_repo_go_jose_multisig_key_selection_001 \
  --output-dir logs/tinker/generative_world_campaigns/initial-calibration
```

Campaigns default to a two-hour Pi-trajectory timeout because evidence-heavy
tasks can exceed 30 minutes even when every individual Sol turn is healthy. The
same ceiling protects each individual Sol process call. Override it with
`--timeout-seconds`; these wall-time guards are independent of the 100-action
environment budget.

Every rollout receives the deterministic acceptance audit. The campaign report
groups reward spread, trainability, and reasoning quality by task so subsequent
authoring can produce easier or harder variants from measured behavior.

## Vat runner development

The runner is plain ESM JavaScript so its protocol tests do not require a build:

```bash
npm test --prefix src/shellsmith/episode/vat_runner
npm run check --prefix src/shellsmith/episode/vat_runner
```

Do not run `npm install` implicitly from episode orchestration. Provision the
dependency as an explicit environment setup step before a live rollout.

The first authored environment is
[`../worlds/generative_world/checkout_idempotency_001.md`](../worlds/checkout_idempotency_001.md).

Parent: [`../README.md`](../README.md)
