# Shellsmith

Train terminal agents in a **vat**. A real coding-agent harness (Pi) runs with
a workspace holding only `AGENTS.md`; every `bash`/`read`/`write`/`edit` it
issues is answered by a **Game Master**: itself a tool-using terminal agent
(Codex CLI) that, for each action, invents the observation, computes it exactly,
lazily materializes files, runs the agent's code for real, or fetches real web
data, while keeping a private consistency ledger. The same Game Master thread
scores every turn. Scored rollout groups train the Player with CISPO on Tinker
(the default) or the standalone MLX Studio compute service.

Positioning against prior work: [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md).

## Vocabulary

| Term | Meaning |
| --- | --- |
| **Game Master (GM)** | The world simulator agent (term from Concordia, arXiv 2312.03664). Default: Codex CLI `gpt-6.1-sol`. |
| **Player** | The policy acting inside the vat. Default: Tinker `Qwen/Qwen3.8-27B` (LoRA). |
| **Vat** | The Player's harness whose tools are all answered by the GM. Any real workspace mutation makes the episode untrainable. |
| **World file** | One Markdown file: World / Player / Reward / Termination conditions. |
| **Campaign** | A durable stream of worlds; each world gets one GRPO group of episodes. |

## Architecture map

```text
worlds/*.md  ──► campaign (stream of worlds, adaptive top-ups, halt on GM outage)
                    │  one world = one GRPO group of N episodes
                    ▼
                 episode: Player (Pi in the vat) ◄──tool results── Game Master (Codex)
                    │       per-turn reward, reasoning score, final reward, untrainable flag
                    ▼
                 credit: captured tokens/logprobs ──► per-turn advantages ──► token batches
                    ▼
                 train: CISPO forward_backward + optim_step, checkpoints, metrics
```

```text
src/shellsmith/
  world_file.py     four-section World file parser
  paths.py          checkout anchors (worlds/, prompts/) and the default temp log root
  episode/          one Player <-> GM episode: orchestrator, bridge, ledger, audits, vat
    gm_process/          GM runtimes (codex is the only one in use; claude, pi)
    vat_runner/          Node Pi-SDK runner for the vat
  prompts/          GM AGENTS.md and boot/turn/final/repair prompts
  campaign/         stream campaigns, GM outage circuit, `shellsmith campaign` (inference.py)
  credit/           per-turn credit and backend-neutral token batches
  policy/           private OpenAI-compatible Player endpoint, captures, episode harvest
  mlx/              native dense LoRA generation, differentiable replay, Adam, full checkpoints
  jobs/             detached Studio supervisor and SSH job control
  train/            `shellsmith train`: config, CISPO loop, adaptive per-world policy, shared rollout stack
  sandbox/          opt-in Docker launcher and in-container entry point
  viewer/           Observatory run viewer (optional [viewer] extra)
src/studio_compute/ standalone resident training/sampling service and portable RPC client
worlds/             177 promoted World files (manifest.json) + qualification variants
hero_runs/          checked-in hero slates
docker/             opt-in sandbox image (Node + Pi + Codex)
docs/               design notes, audits, related work
```

## World catalog

[`worlds/manifest.json`](worlds/manifest.json) admits 177 worlds: the original
151 tasks, 23 seed-grounded construction/analysis/workflow tasks, and three
simulator-operation tasks. Five additional files under `worlds/mlx_context/`
are context-qualification variants of existing tasks and are excluded from
ordinary catalog sampling. Promotion records contract review and runtime
selection; it does not certify a solve rate, reward spread, or learning gain.

`[worlds] select = "manifest"`, `"random"`, and `"ids"` select from this catalog.
Set `count = 177` for the full catalog; the default count remains eight.
Each selected world contributes one adaptively collected rollout group.
The trainer consumes the queue once; `train.steps` limits updates, not epochs.

The three simulator-operation worlds evaluate the Player as an operator of a
fictional inner simulator. They use ordinary intercepted tools and persistent
world state. Inner response schemas and simulator roles are task data; actual
outer GM instructions, private scores, and orchestration state remain private.
Player-authored inner mistakes affect task reward and do not by themselves make
the outer episode untrainable. Use the explicit
[`meta-simulator-train.toml`](examples/meta-simulator-train.toml) recipe to choose
this curriculum deliberately. Historical flavor-discovery hero slates continue
to exclude it.

See the [catalog guide](worlds/README.md),
[seed coverage](worlds/seeds/README.md), and
[promotion audit](docs/world-catalog-audit/2026-10-18/report.html).

## Quickstart (development)

```bash
pip install -e ".[dev]"
pytest -q
(cd src/shellsmith/episode/vat_runner && npm ci --ignore-scripts && npm test)
```

Requires Python ≥ 3.11 and Node ≥ 22.19. Run output defaults to
`<system temp>/shellsmith-<uid>/` on POSIX (a separate directory per user),
or `<user temp>/shellsmith/` on Windows.

## Episodes

```bash
shellsmith episode --world-file worlds/checkout_idempotency_001.md  # protected host default
shellsmith episode --docker --world-file worlds/checkout_idempotency_001.md
```

The default Codex GM requires an existing ChatGPT login, and the default
OpenRouter Player requires `OPENROUTER_API_KEY`. **Codex is the only supported
GM runtime** in host and Docker modes; Claude/Pi GM execution is disabled.
The separate Pi Player SDK remains supported.
`--skip-player-models-preflight` supports chat-only Player endpoints while still
requiring a credential.

**Docker security requirement:** Codex keeps its restricted tool-filesystem
profile inside the container. The engine must support nested unprivileged user
namespaces and a compatible confined LSM policy. The shipped narrowly extended
Docker seccomp policy does not override AppArmor or SELinux; Docker's default
AppArmor policy denies the sandbox's mounts. A no-model Codex probe requires
both workspace writes and denial of outside synthetic credentials before any
model call. Incompatible engines fail closed. Shellsmith never enables privileged
containers, disables an LSM, or adds broad capabilities automatically.
Protected `--host` execution is available on systems without that capability.
Build with `docker/build.sh`; freshness covers both `src/` and the Docker recipe.
World input, credentials, and writable artifacts use container-owned storage
(no host bind mounts). Artifact export assigns ownership to the invoking host
user without disabling daemon user-namespace mappings. The launcher resolves
and pins a **local** Docker endpoint before staging credentials; SSH/TCP contexts
(including loopback TCP) are rejected. POSIX Docker launches must run in a
main-thread worker process so SIGTERM can stop, export, and remove the container.
SIGKILL and an unavailable daemon cannot provide that cleanup guarantee.

Docker supports **file-backed ChatGPT OAuth** only. It reads
`SHELLSMITH_CODEX_AUTH_PATH`, or `CODEX_HOME/auth.json`, or `~/.codex/auth.json`.
An existing keyring/auto login may have no file: use the default host backend,
or explicitly log in with `codex -c 'cli_auth_credentials_store="file"' login`
in a normal persistent Codex home (not a run directory). Use file mode for other
Codex invocations sharing that home too; this is a new login, not a keyring export.
The launcher locks the source across Docker execution, retrieves refreshed OAuth
from the stopped container before removal, and atomically updates the private
host file while preserving unrelated fields. Do not run an unrelated Codex client
against the same file concurrently: those clients do not share Shellsmith's lock.
Detected concurrent changes preserve returned OAuth in a private recovery file
beside the original credentials and report a conflict, rather than overwriting
another login. Credentials and recovery files are never episode artifacts.

`--timeout-seconds` sets each GM turn deadline; `--player-prompt-timeout-seconds`
sets the complete Player deadline (default 12h). The Docker deadline includes
that trajectory, up to three bootstrap and three judgement attempts, and five
minutes for I/O (default 15h 5m). `--docker-timeout-seconds` can extend it.
On timeout or interruption, execution stops before exporting completed artifacts
and sanitized in-flight logs under `artifacts/partial/`; the original failure
is still raised. Recovery errors are reported, never treated as episode success.

## Tinker Player (optional)

```bash
pip install -e ".[tinker]"
shellsmith episode --player tinker --world-file worlds/checkout_idempotency_001.md
```

The Tinker SDK needs configured credentials (`TINKER_API_KEY` or a stored
`tinker auth login`). `--tinker-model` is restricted to the compatible
`Qwen/Qwen3.8-27B` and `Qwen/Qwen3.8-27B:peft:262144` models; the endpoint also
checks the service's working-context limit. Tinker episodes require the confined
Codex GM (host or Docker); the unconfined Claude GM is not supported for Tinker.
The generated Player credential is passed directly to the trusted Player child
(or Docker's create subprocess), never installed in the host process environment.
Episode URLs use opaque IDs, independent of output
folder names. After the endpoint starts, full prompt-token IDs, sampled tokens,
and log probabilities are saved in `player_records.json`, including after a late
episode failure. Successful episodes also produce `capture_summary.json` after
the current acceptance audit; use a fresh output directory for every run.

## Training

Tinker remains the default backend and requires Tinker credentials. Select
`compute.backend = "studio"` to use the independently runnable Studio service:
one frozen resident base, separate LoRA clients, durable optimizer receipts, and
immutable sampled policy revisions. Shellsmith can run on a CPU-only machine
through an SSH tunnel; it needs neither MLX nor Studio model paths. The
[Studio guide](docs/studio-compute/README.md) describes the contract, memory
budget, generation limits, and qualification procedure.

The [prepared catalog epoch](docs/studio-compute/backprop-epoch-v1/README.md) pins 177 worlds, eight attempts per world, and two world groups per accumulated update. It includes Backprop recovery/readback evidence and a separate service memory profile. The epoch is not started.

```bash
# On Studio, with the pinned MLX runtime:
studio-compute serve --model /path/to/Qwen3.8-27B-8bit --state-root /path/to/studio-state
# On the coordinator, with .[studio-client] and a forwarded service endpoint:
shellsmith train --config examples/studio-micro-train.toml --dry-run
shellsmith train --config examples/studio-micro-train.toml --log-dir /path/to/private/run
```

The historical `compute.backend = "mlx"` implementation, frozen runtimes, and
checkpoint format remain available independently. Its setup and job commands
are documented in [the MLX guide](docs/MLX_TRAINING.md):

```bash
shellsmith train --config examples/mlx-studio.toml --dry-run
shellsmith job submit --config examples/mlx-studio.toml --request-id my-mlx-run
shellsmith job status JOB_ID
shellsmith job logs JOB_ID
```

The historical job supervisor retains jobs after SSH disconnects. Its readiness
checks run before loading weights. Previous runs and stopped qualification
evidence remain in the [MLX evidence guide](docs/mlx-qualification/README.md);
they do not establish the new service's qualification.

```bash
shellsmith train --config train.toml --log-dir /path/to/private/run
# Reuse the same directory and config to resume from its full checkpoint.
```

A small `train.toml`:

```toml
[worlds]
select = "manifest"  # manifest order, seeded random, explicit ids, or hero
count = 4

[train]
steps = 2
groups_per_update = 2
checkpoint_every = 1

[player]
max_tokens = 32768  # maximum total generation, not a target
```

For an immutable hero stream, use `[worlds] select = "hero"` and
`hero_slate = "hero_runs/first_hero_baby_terra_4.json"`. The actual slate loader
preserves the complete world order and each world's GM assignment; catalog
`count`/`seed` do not truncate or reshuffle it. `groups_per_update` must match
the slate. Historical slates assigning Claude/Pi are rejected before model
work: training requires confined Codex, never a silent substitute GM.

Configuration and the tokenizer-specific minimum thinking budget are validated
before paid work. Training requires at least two trainable rollouts per group.
Explicit run IDs must be single non-dot path segments using only letters,
digits, `.`, `_`, or `-`. Composed episode/record/staging names are checked
against destination filesystem byte limits before model work, including the
restored run ID and current resume-segment width. Selected hero simulators
receive the same validation as the global GM settings.
Tinker training supports `provider_outage = "halt"`, not
unconfined-GM fallback. Each run directory has one exclusive trainer owner;
concurrent attempts fail instead of overwriting optimizer state. Automatically
generated run IDs include a random suffix.
Full optimizer checkpoints are saved on cadence and at the end of the world
stream. Metrics publish with a recoverable state boundary; resume reconciles the
log with the restored full checkpoint and archives rolled-back or repaired
bytes under `metrics_recovery/`. Corrupt committed history fails closed.
Failures during reference KL scoring retain the prior world cursor and can be
retried; interrupted optimizer submissions remain fail-closed without a full
checkpoint. Resume keeps the original run ID and allows relocating `log_dir`
and increasing `[train].steps`; the accepted target cannot be lowered, even
after a failed extension. It rejects changed source, manifest, world contents,
prompts, hero assignments,
selected Pi SDK, or Codex executable instead of silently continuing a different
experiment. SDK and GM launches use the snapshotted canonical entry paths;
supported Codex npm installations also fingerprint their native executable.
This is bounded local provenance, not a fingerprint of every transitive runtime
dependency. Selected world IDs must be unique, endpoint-safe, and distinct from
campaign bookkeeping paths. World and GM prompt bytes are captured before
authentication. Admitted worlds run from read-only staged world copies; prompt
rendering uses the same frozen bundle as snapshot/group identity, including model
profile tails and repair templates. Read-only prompt copies are archived with
source/hash provenance on admission. Editing the source cannot change later
rollouts in that active run. The Player endpoint uses the same private credential
handoff as `episode --player tinker`.

## Training telemetry (Backprop or W&B)

`shellsmith train` uses [Backprop](https://github.com/tginart/backprop) by default
for Tinker, MLX, and Studio runs. W&B is the other supported provider, selected
explicitly with `[telemetry].provider = "wandb"`. Installation includes Backprop's
SDK and system metrics collector. The shared trainer logs configuration and source
provenance, optimizer and loss metrics, rewards, world decisions, token usage,
throughput, CPU/memory
telemetry, console output, live episode progress, failures, and text/tool trajectories
for accepted and rejected rollouts. Studio's remote memory/allocator measurements
and MLX's runtime measurements are logged alongside training metrics.

For a local default, put Backprop's connection JSON at `.backprop/config.json`
(Git-ignored) and authenticate with `backprop.login()`. For a remote process or
an explicit connection, use:

```bash
export BACKPROP_CONFIG=/path/to/backprop/backend/state.json
export FOUNDRY_TOKEN=...  # or BACKPROP_TOKEN, or a saved backprop.login()
shellsmith train --config examples/studio-micro-train.toml
```

The default `auto` mode goes online when a connection file exists
(`BACKPROP_CONFIG`, `connection_file`, or `.backprop/config.json`) or when both
`BACKPROP_HOST` and `BACKPROP_STREAM_DATASET_RID` are set. Otherwise it journals
offline. Each run prints its Backprop URL or offline sync command and records all
journal locations in `<run.log_dir>/backprop.json`. Upload an offline journal later
with the exact printed command, normally
`python -m backprop sync /path/printed/by/training`. An offline run with a
connection file prefixes it with `BACKPROP_CONFIG=<file>` so the upload reaches that
deployment. Prime RL is not used by the trainer.

To select W&B, install its optional SDK and authenticate with `wandb login` or
`WANDB_API_KEY`, then set the provider in your training TOML:

```bash
pip install -e '.[wandb]'
```

```toml
[telemetry]
provider = "wandb"
```

W&B logs the same metrics, progress and text trajectories. Its `auto` mode uses
the SDK's normal online default and honors `WANDB_MODE` (`dryrun` means offline;
`disabled` turns telemetry off, as below); `mode = "offline"` uses local W&B storage.
Run URLs and sync directories are recorded in `wandb.json`, and offline runs print
a `python -m wandb sync ...` command. Backprop connection files and Foundry
credentials belong to the Backprop provider. Selecting W&B
uses W&B authentication and its optional `WANDB_BASE_URL` instead.

Optional TOML settings (these are the defaults):

```toml
[telemetry]
provider = "backprop"  # backprop (default) | wandb
mode = "auto"  # online, offline, or disabled also supported
connection_file = ""  # optional connection JSON; BACKPROP_CONFIG takes precedence
project = "shellsmith"
entity = ""  # inherit SDK setting
log_trajectories = true
capture_console = true
system_metrics = true
system_interval_seconds = 15.0
init_timeout_seconds = 300.0
```

Trajectories appear in `rollouts/trajectories` as incremental tables containing
episode/world/policy IDs, acceptance, reward, errors, and the text/tool trace,
including model reasoning and final judgments. A trajectory row is capped at 128 KiB;
a longer one uploads as a truncated preview, and the full trace stays in the run's
local episode logs. Repeated request payloads
and token/logprob arrays are excluded from telemetry uploads, along with **all
model weights, adapters, optimizer checkpoints, and workspace archives**. There
are no model upload/watch hooks.

The local checkpoint journal and episode recovery files remain authoritative for
resuming training. The selected provider receives metrics after durable
publication, and missed metrics/finished episodes are replayed on resume. Logging failures retain these
sources and warn without changing rollout acceptance or optimizer commits. Online
resumes retain the run ID; offline invocations have separate sync journals recorded
in `backprop.json` or `wandb.json`. Remote history is append-only: segment markers identify restored
training steps if recovery rewinds to an older checkpoint. Publication is at least
once across a crash between SDK journaling and a local receipt; stable update/episode
IDs identify such duplicates. Switching from offline to online replays durable
metrics and trajectories automatically; avoid also syncing old offline journals if
you want to avoid duplicate history. Progress/console events exist in their original
journals and require syncing those journals to retain their full history.
While telemetry is disabled (`mode = "disabled"` or W&B `auto` with
`WANDB_MODE=disabled`), the trainer writes no policy/checkpoint publication
receipts, so enabling it later replays metrics and episodes but not those
publications.

## Inference campaigns

`shellsmith campaign` uses the same world selection, local preflight, private
Player endpoint, and adaptive collection as training, but submits no optimizer
updates. `--sampler-path` selects a trained Tinker sampler and `--adapter-path`
selects an exported MLX adapter; otherwise it evaluates base weights. `[train]`
settings do not control inference. A campaign holds the
same exclusive directory lock as training, from its fresh-directory check through
finalization; a competing invocation cannot replace its snapshot or start model
work in that directory.
`[world_overrides.<world_id>]` may override the adaptive attempt/trainability
settings; every merged policy is validated before model work. Admission windows
use those merged initial-attempt counts in queue order, and finish their top-ups
before admitting later worlds. Outage thresholds
are configured by `[adaptive].outage_failure_threshold` and
`outage_cooldown_seconds`; an outage halts rather than switches GMs.
`campaign_summary.json` reports per-world top-ups and separates completed,
started-but-incomplete, and not-started worlds after an interruption.

## Commands

```bash
shellsmith episode                      # one episode on this machine (default world, Sol GM, OpenRouter Player)
shellsmith episode --player tinker      # Player sampled from Tinker; writes a token-exact capture
shellsmith episode --docker             # same, inside the pinned Docker image (docker/build.sh)
shellsmith train --config run.toml      # Tinker CISPO loop; rerun the same log dir to resume
shellsmith campaign --config run.toml   # adaptive per-world collection over many worlds, no training
shellsmith view                         # Observatory over $TMPDIR/shellsmith
```

Episodes run on this machine by default: the Player only ever sees the vat,
and the Codex GM is confined by Codex's own sandbox profile. `--docker` adds
pinned CLI versions and an isolated filesystem. The Codex GM is the only GM
in use for now; a GM outage halts a run. Config keys are documented in
`src/shellsmith/train/config.py`; defaults live in `src/shellsmith/config.py`.

## Status

Milestones M0–M5 provide the standalone package, vocabulary, episode CLI,
Tinker Player, trainer, inference campaigns, and Observatory viewer.
Defaults live in `src/shellsmith/config.py`.
