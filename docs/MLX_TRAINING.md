# MLX training on the Studio

The current qualification examples select the v2 slates for the expanded catalog.
Those inputs have not been executed. The v1 run and readiness records below are
historical evidence; they do not qualify the revised worlds or shared prompts.
The original v1 JSON is preserved, and its manifest pins reject a current-catalog
launch. Use the matching historical source revision to reproduce that plan.

Shellsmith's first MLX profile trains the dense Qwen3.8 27B 8-bit model on one
Apple Silicon Mac. The entire episode/campaign/training job belongs to the
Studio, including the Player, Codex Game Master, optimizer and checkpoints.
The coordinator submits and inspects bounded SSH requests; it can disconnect
after submission. No persistent remote RPC daemon is required.

The numerical reference is [High Command at
10e830e9](https://github.com/tginart/high-command/tree/10e830e9a5becb9c2387bd37a378fd7dad1d3590).
Shellsmith adapts its dense numerical components and retains Shellsmith's
world acceptance, turn credit, compaction and streamed-update contracts.
Historical High Command benchmarks do not qualify Shellsmith's throughput.

## Prepare the selected machine

Set the destination with `[compute].host`, `--host`, or `SHELLSMITH_STUDIO_HOST`
(for example `user@studio-host`, an SSH alias, or the Studio's tailnet name
`studio-host.<tailnet>.ts.net`). Execution home and Python default to
`~/.local/state/shellsmith` and `~/.local/share/shellsmith/studio-v1/env/bin/python`
on that host. SSH requires the existing trusted host
key and disables connection multiplexing; it never accepts a new host key
implicitly. Confirm
Apple Silicon, macOS 26 or newer, at least 256 GiB of detected physical unified memory for the v1 128 GiB MLX budget, sufficient free disk and SSH access.
The supplied model directory is `~/models/Qwen3.8-27B-8bit`.
Setup does not download weights. This profile requires the model's actual
8-bit/group-64 assets; metadata labels alone do not establish compatibility.

Run the setup script explicitly on the Studio from a checkout:

```sh
bash scripts/setup_mlx_studio.sh ~/.local/share/shellsmith/studio-v1
```

An optional second argument clones an existing environment into the dedicated
prefix without modifying that source environment. The script uses Python 3.12
and the complete artifact-hashed numerical lock inherited from the pinned
reference, then installs Node 22.22.0, Codex 0.159.2 and Pi 0.84.1 in account-owned paths.
Unmanaged environments are refused. Conda uses explicit conda-forge channels;
the script does not accept third-party channel terms on the user's behalf.

Authenticate Codex on the Studio in an interactive session:

```sh
export PATH="$HOME/.local/share/shellsmith/studio-v1/env/bin:$HOME/.local/share/shellsmith/studio-v1/node/bin:$HOME/.local/share/shellsmith/studio-v1/tools/node_modules/.bin:$PATH"
codex -c 'cli_auth_credentials_store="file"' login --device-auth
```

Complete the browser/device authorization yourself. Credentials stay in the
Studio account's normal Codex storage; source bundles, job records and artifact
mirrors do not copy that storage. When explicitly authorized to reuse an existing
file-based local login, the [official Codex authentication guide](https://learn.chatgpt.com/docs/auth)
also supports transferring its cached credentials directly over SSH to the
Studio's normal `~/.codex/auth.json`. Keep that file at mode `0600`, outside the
checkout and job artifacts, and verify it with the dedicated Codex CLI's
`login status`. The authorized transfer for the current qualification returned
`Logged in using ChatGPT` with Codex 0.159.2.

Keep the dedicated runtime PATH above for
local commands. Existing inference serving must be stopped through
its own owner before exclusive training, then restarted deliberately when
training is finished. The GPU lease serializes cooperating Shellsmith
processes; it does not claim to control arbitrary inference services.

MLX jobs expose only the approved Python installation read-only to the confined
GM. The launcher supplies its absolute interpreter path and existing private
`world/runtime_scratch` directory for virtual environments and build caches.
World computations and durable state remain in `world/simulation_root`.
Readiness checks the interpreter before loading weights. The Studio's system
`/usr/bin/python3` currently requires unavailable macOS developer tools; the
dedicated runtime avoids that launcher. Existing Tinker permissions are unchanged.

## Check and submit

The concrete canary configuration is `examples/mlx-studio.toml`. Its
execution-host paths are explicit. A check inspects packages, Metal, model
metadata, token limits, the Pi SDK and Codex login without loading weights:

```sh
shellsmith job check --config examples/mlx-studio.toml
shellsmith job submit --config examples/mlx-studio.toml --request-id studio-canary-001
```

Keep the returned job ID and retain the same request ID after a timeout or
dropped response. Repeating a submission returns the original job; changed
configuration under the same request ID is rejected. The local private receipt
also retains the exact original source bundle, so later checkout edits do not
change a retried launch. `--host`, `--home` and `--python` can override the initial
endpoint; historical control uses the recorded endpoint and refuses redirection.

The source bundle includes required Shellsmith source, prompts, worlds and hero
slates. It excludes `.git`, environments, node_modules, run output and credential
directories. Each file is hashed. Runtime, configuration, interpreter and complete
installed dependency identity are frozen before execution and checked again at
worker start and resume. The worker prepares Pi's dependency tree from the
frozen npm lockfile. Relative hero-slate paths must belong to the bundled
checkout; model/SDK paths refer to the selected execution machine.

```sh
shellsmith job status JOB_ID
shellsmith job logs JOB_ID --offset 0
shellsmith job stop JOB_ID --request-id stop-canary-001
shellsmith job resume JOB_ID --request-id resume-canary-001
shellsmith job sync JOB_ID --local-dir /private/tmp/shellsmith-observatory/JOB_ID
shellsmith view --runs-root /private/tmp/shellsmith-observatory
```

Logs use byte offsets so polling need not repeat the complete log. Metadata sync
copies private JSON/JSONL/text artifacts read-only in checksum-verified pages,
including individual captures or totals larger than 32 MiB. It retries bounded
inventory changes, preserves the previous mirror on failure, and retains local
notes in a managed mirror for the same job. Use a new empty private directory for
the first sync. Historical frozen runtimes retain their original one-shot limit.
Credentials, hidden files and tensor weights are excluded. Large binary
recovery checkpoints remain on the Studio. Source snapshots and historical job
homes remain available for control after the coordinator checkout changes.

## Training and recovery contracts

The profile admits 32,768 total prompt-plus-generation tokens. It never silently
truncates over-budget prompts. Thinking, response and compaction budgets are
explicit in the configuration. The Player is sequential; all canonical
attention, GDN and MLP factors use rank 16 and scale 2. Model base tensors,
embeddings and the vocabulary head remain frozen.

Training uses BF16 computations, FP32 accumulated gradients and FP32 master Adam
parameters/moments, with betas .9/.95, epsilon 1e-8 and no weight decay or
additional clipping. CISPO uses a sum over supplied normalized token advantages.
Shellsmith's logical-rollout token normalization is applied exactly once;
request length, compaction sequence count and group count introduce no further
averages. Every numerical job sets `MLX_ENABLE_TF32=0` before importing MLX;
that execution policy is recorded with the job. Prompt, injected forced-close and compaction-replay targets carry zero
advantage. Native segmented differentiable replay recomputes each complete
request; inference caches are not reused as training graphs.

Stop targets only the recorded process birth/boot identity. The detached worker
records heartbeat, trainer identity and terminal status independently of SSH.
A missing worker/reboot becomes `interrupted`, never an automatic new launch.
Explicit resume requires a complete full optimizer checkpoint and matching
sampler weights, original model/source/dependency identity and available GPU
lease. Adapter exports alone are not recovery points. Checkpoint publication
and trainer cursor/metrics recovery preserve the last committed update boundary.
Stop first sends SIGINT and waits up to 30 seconds. A long native replay may
require termination after that deadline; recovery then uses the previous complete
checkpoint. V1 does not promise cooperative cancellation inside a Metal update.

## Qualification

The [offline numerical evidence](mlx-qualification/README.md) records native
decoder parity, real-model resource measurements and checkpoint recovery on the
Studio. It does not establish live world acceptance or learning quality.
The Studio passes [current public readiness](mlx-qualification/studio-runtime-readiness.json)
with `ready: true`, no blockers, and `codex_authenticated: true`. The
[initial authenticated canary](mlx-qualification/studio-live-canary-initial.json)
passed native protocol checks but was correctly rejected when the GM could not
execute Python. It was stopped with its complete baseline checkpoint intact.
After the [real OS confinement proof](mlx-qualification/studio-gm-python-confinement.json),
live qualification restarted under job `job-8c40c2bc05b5c650a77d62de` and its GM
executed the approved interpreter successfully. Accepted live updates and heldout
evaluation remain pending. The train and heldout
slates pin the current approved `worlds/manifest.json` and every selected raw
world SHA-256; loading either slate rejects changed hashes, IDs or catalog paths.
`hero_runs/mlx_qualification_evidence_v1.json` records all eight inputs, their
slate hashes, the active job, and the earlier authentication-blocked history.

`examples/mlx-studio-qualification-train.toml` schedules four training worlds,
two groups per update and two updates with rollout parallelism two. The heldout
configuration selects four different worlds, initially attempts 16 episodes per
world, permits at most 24 attempts and requires at least 16 accepted episodes
per world for each policy. The qualification compares base and the sampler
published after update two, using the same simulator assignment and token
budgets. It reports reward, acceptance, truncation/compaction, replay drift,
latency and peak memory separately; incomplete accepted quotas fail qualification.

```sh
shellsmith job check --config examples/mlx-studio-qualification-train.toml
shellsmith job submit --config examples/mlx-studio-qualification-train.toml \
  --request-id studio-qualification-train-001
shellsmith job submit --command campaign \
  --config examples/mlx-studio-qualification-eval.toml \
  --request-id studio-qualification-base-001
shellsmith job submit --command campaign \
  --config examples/mlx-studio-qualification-eval.toml \
  --adapter-path /ABSOLUTE/STUDIO/SAMPLER/CHECKPOINT \
  --request-id studio-qualification-trained-001
```

Read the final state checkpoint row's `sampler_path` from the training job's
`trainer_state.json` for the last command. Campaign jobs freeze and verify that
sampler's model/runtime identity, tensor hashes and manifest hash before launch.
Base campaigns omit `--adapter-path`. Campaigns are stopped through the same job
controls; a new request starts a new campaign rather than resuming one.

Each world row and the overall `campaign_summary.json` contain `evaluation`
statistics. Accepted rewards report sample count, mean, standard error and an
approximate 95% normal interval; that interval assumes independent episodes.
Acceptance reports every attempted episode with a 95% Wilson interval. Discarded
fallback attempts contribute latency and token cost, while their rewards are
excluded from the retained accepted population. Provider-down slots that never
launched are reported separately and contribute no attempted-episode denominator
or cost. Missing token/latency
measurements are explicit. Compare base and trained results per heldout world
before aggregating, verify at least 16 retained accepted episodes in each cell,
and retain both rejection rates and simulator identities beside reward changes.
These intervals describe sampling uncertainty; they do not establish a training
improvement without the planned matched base/trained evidence.

Portable tests cover dropped replies, request reuse, source integrity,
destination pinning, private artifact sync, lease contention and real detached
process completion/stop. Numerical tests compare tiny FP32 replay to the
unmodified dense decoder and verify Shellsmith's sum loss and Adam semantics.
Real-model qualification must confirm finite updates, frozen base assets, exact
state restoration, next-update reproducibility and accepted live multi-turn
episodes before calling the backend ready for routine training.

Flash Next/MoE, full-weight training, multiple Macs, custom Metal optimizations,
FP8 hidden boundaries, speculative decode and multimodal training are outside
this release. Behavior-versus-replay drift is telemetry: finite alignment and
usable importance signal are hard requirements, while deliberately different
sampling distributions must not be mistaken for replay corruption.
