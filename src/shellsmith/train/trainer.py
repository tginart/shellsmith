"""``shellsmith train``: the thin Tinker CISPO loop over streamed worlds.

Architecture map
----------------
    run_train(config)
      log_dir (printed first) <- config [run].log_dir or <temp>/shellsmith/runs/<run_id>
      start or resume
          trainer_state.json absent  -> new LoRA training client, base-weights sampler
          trainer_state.json present -> resume from the last *full* checkpoint:
                                        optimizer state + its sampler weights + its
                                        step / world offset; a new campaign segment
      config_snapshot.json  config, source/manifest/world/prompt hashes, selected GM assignments
      PlayerEndpoint (Tinker)  <- every episode's Pi samples here
      for each step until [train].steps or the world stream runs out:
          collect_update_worlds         adaptive per-world policy (stream_engine):
                                        4 rollouts, top-ups, skip, GM-outage redo
          GROUP_READY worlds -> CaptureStore -> build_tinker_group (per-turn credit)
          optional base-reference KL ([train].kl_penalty_coef, default 0)
          forward_backward(loss="cispo") + optim_step(Adam)
          save sampler weights -> endpoint serves the new policy
          every checkpoint_every steps and on exhaustion/orderly halt: save full state
          trainer_state.json commits metrics boundary/last row; fsynced metrics.jsonl append

Invariants
----------
- Updates are sequential; inside an update several worlds run concurrently
  under the campaign's parallelism caps.
- A step with no trainable group submits nothing and is recorded as skipped.
- GM models never mix in one group (capture store + identity key).
- The run directory is never reused for a different config: resume requires
  the same config snapshot.

Why this exists:
    Replaces both the deprecated legacy conductor and the legacy
    ``rl.train`` GW branch with one small loop (plan §6).

Main callers:
    ``shellsmith train`` (cli.py).
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable

from shellsmith.campaign.draft_catalog import classify
from shellsmith.credit.to_tinker_bridge import build_tinker_group
from shellsmith.credit.turn_credit import TurnAdvantageConfig
from shellsmith.paths import CATALOG_MANIFEST, PROJECT_ROOT
from shellsmith.policy.sampler import PlayerSamplingConfig, TinkerPlayerSampler
from shellsmith.policy.tinker_setup import require_long_context_async
from shellsmith.process_io import write_json
from shellsmith.train.config import TrainConfig, validate_train_config
from shellsmith.train.backend import TokenBatch, player_profile, profile_hash
from shellsmith.train.durable_metrics import HEAD_KEY, MetricsJournal
from shellsmith.train.telemetry import (
    TrainingTelemetry, is_telemetry_enabled, persist_publication, require_telemetry,
)
from shellsmith.train.harness_identity import codex_identity, pi_sdk_identity
from shellsmith.train.name_budget import (
    validate_component_name,
    validate_training_names,
)
from shellsmith.train.rollout_stack import (
    episode_args,
    open_rollout_stack,
    stream_config,
    summarize_world,
    world_task_runs,
)
from shellsmith.train.stream_engine import collect_update_worlds
from shellsmith.train.stream_policy import StreamPolicyConfig, WorldAction
from shellsmith.train.world_selection import (
    FrozenWorld,
    SelectedWorld,
    WorldStream,
    freeze_worlds,
    select_world_stream,
)
from shellsmith.world_file import load_world_file

STATE_FILE = "trainer_state.json"
SNAPSHOT_FILE = "config_snapshot.json"
METRICS_FILE = "metrics.jsonl"


# --- small durable-file helpers ----------------------------------------------


def _git_sha() -> str:
    manifest = _runtime_manifest()
    if manifest is not None:
        return manifest.get("git_sha") or f"runtime:{manifest['source_sha256']}"
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Cannot read the Shellsmith git SHA: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _runtime_manifest() -> dict[str, Any] | None:
    path = PROJECT_ROOT / "runtime_manifest.json"
    if not path.is_file():
        return None
    from shellsmith.jobs.bundle import verify

    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("source_sha256"), str):
        raise RuntimeError("Frozen runtime has no content identity.")
    return verify(PROJECT_ROOT, raw["source_sha256"])


# --- world selection ------------------------------------------------------------


def select_worlds(config: TrainConfig) -> list[Path]:
    """Compatibility wrapper for callers that need only the selected paths."""
    return [world.world_file for world in select_world_stream(config).worlds]


def classify_world(path: Path) -> str:
    """The world flavor recorded on campaign scoreboards."""
    return classify(load_world_file(path), path=path)


def _source_sha256(*, frozen_worlds: tuple[FrozenWorld, ...] = ()) -> str:
    """Hash local source too: a dirty checkout can change without changing HEAD."""
    manifest = _runtime_manifest()
    if manifest is None:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "ls-files",
                "-co",
                "--exclude-standard",
                "-z",
                "--",
                "src/shellsmith",
                "src/studio_compute",
                "pyproject.toml",
            ],
            capture_output=True,
            check=True,
        )
        names = sorted(set(result.stdout.split(b"\0")) - {b""})
    else:
        names = sorted(
            name.encode()
            for name in manifest["files"]
            if name == "pyproject.toml"
            or name.startswith(("src/shellsmith/", "src/studio_compute/"))
        )
    digest = hashlib.sha256()
    prompt_bytes = {
        prompt.path: prompt.contents
        for world in frozen_worlds
        for prompt in world.prompt_inputs.files
    }
    for name in names:
        path = PROJECT_ROOT / name.decode()
        content = prompt_bytes[path] if path in prompt_bytes else path.read_bytes()
        digest.update(name + b"\0" + content + b"\0")
    return digest.hexdigest()


def snapshot(
    config: TrainConfig,
    world_stream: WorldStream,
    *,
    frozen_worlds: tuple[FrozenWorld, ...] | None = None,
) -> dict[str, Any]:
    frozen_worlds = (
        freeze_worlds(world_stream.worlds) if frozen_worlds is None else frozen_worlds
    )
    profile = player_profile(config)
    identities = [
        {
            "world_file": str(frozen.source_path),
            **frozen.group_identity(
                gm_runtime=frozen.selected.gm_runtime,
                gm_model=frozen.selected.gm_model,
                gm_provider=frozen.selected.gm_provider,
                gm_reasoning_effort=frozen.selected.gm_reasoning_effort,
                max_world_steps=config.gm.max_world_steps,
                player_model=profile["base_model"],
                renderer_name=profile["thinking_renderer"],
                max_tokens=profile["max_tokens"],
                temperature=config.player.temperature,
                top_p=config.player.top_p,
                **(
                    {
                        "backend": config.compute.backend,
                        "profile_sha256": profile_hash(profile),
                    }
                    if config.compute.backend in {"mlx", "studio"}
                    else {}
                ),
            ).to_jsonable(),
        }
        for frozen in frozen_worlds
    ]
    # Normalize tuples to lists before comparing with a snapshot read from JSON.
    extra = {}
    if config.compute.backend == "studio":
        from studio_compute.client import Client
        from studio_compute.protocol import digest

        capabilities = Client(config.studio.endpoint).capabilities()
        from shellsmith.train.studio import validate_capabilities
        validate_capabilities(config.studio, capabilities)
        identity = {
            **capabilities["identity"],
            "backend": "studio",
            "identity_sha256": digest(capabilities["identity"]),
        }
        extra = {"player_profile": profile, "policy_identity": identity}
    elif config.compute.backend == "mlx":
        from shellsmith.mlx.identity import model_identity, policy_identity
        from shellsmith.jobs.readiness import installed_dependencies

        extra = {
            "player_profile": profile,
            "policy_identity": policy_identity(
                config.mlx, model_identity(config.mlx.model_path)
            ),
            "runtime_dependencies": installed_dependencies(),
            "python_runtime": {
                "version": platform.python_version(),
                "executable": sys.executable,
                "implementation": platform.python_implementation(),
            },
        }
    return json.loads(
        json.dumps(
            {
                "config": config.to_jsonable(),
                "git_sha": _git_sha(),
                "source_sha256": _source_sha256(frozen_worlds=frozen_worlds),
                "pi_sdk": pi_sdk_identity(config.player.pi_sdk_module_path),
                "codex": codex_identity(),
                "manifest_path": str(CATALOG_MANIFEST.resolve()),
                "manifest_sha256": hashlib.sha256(
                    CATALOG_MANIFEST.read_bytes()
                ).hexdigest(),
                "worlds": [identity["world_id"] for identity in identities],
                "world_identities": identities,
                "world_stream": world_stream.provenance,
                **extra,
            }
        )
    )


def _resume_key(config_json: dict[str, Any]) -> dict[str, Any]:
    """Immutable settings; run ID is checked separately, target is monotonic."""
    normalized = json.loads(json.dumps(config_json))
    normalized.pop("telemetry", None)
    return normalized | {
        "run": normalized["run"] | {"run_id": None, "log_dir": None},
        "train": normalized["train"] | {"steps": None},
    }


def _resume_config(
    *,
    config: TrainConfig,
    state: dict[str, Any],
    saved: dict[str, Any],
    current: dict[str, Any],
    log_dir: Path,
) -> TrainConfig:
    """Check immutable experiment inputs locally, then recover the original ID."""
    run_id = state.get("run_id")
    if (
        not isinstance(run_id, str)
        or not run_id
        or saved["config"]["run"]["run_id"] != run_id
    ):
        raise RuntimeError(
            f"Inconsistent saved run_id in {log_dir}; use a new log_dir."
        )
    if not config.run_id_generated and config.run.run_id != run_id:
        raise RuntimeError(
            f"Explicit run_id {config.run.run_id!r} does not match saved run_id {run_id!r}."
        )
    if _resume_key(saved["config"]) != _resume_key(current["config"]):
        raise RuntimeError(
            f"Config changed since {log_dir} started; use a new log_dir."
        )
    targets = [
        saved["config"]["train"]["steps"],
        state.get("target_steps", 0),
        state.get("step", 0),
        state.get("rft_updates_completed", 0),
    ]
    targets.extend(
        row.get("step", 0)
        for row in state.get("checkpoints", [])
        if row.get("kind") == "state"
    )
    if any(type(target) is not int or target < 0 for target in targets):
        raise RuntimeError(f"Invalid saved training target/checkpoint in {log_dir}.")
    minimum_target = max(targets)
    if config.train.steps < minimum_target:
        raise RuntimeError(
            f"[train].steps may only extend monotonically: requested {config.train.steps}, "
            f"saved target/current checkpoint requires at least {minimum_target}."
        )
    if saved | {"config": None} != current | {"config": None}:
        raise RuntimeError(
            f"Source, manifest, worlds, prompts or hero metadata changed since {log_dir} started; "
            "use a new log_dir."
        )
    return replace(config, run=replace(config.run, run_id=run_id))


# --- Tinker step -------------------------------------------------------------------


def datum_without_mask(*, deps: Any, datum: Any) -> Any:
    """Drop the action ``mask``: hosted CISPO rejects it (advantages already zero the prompt)."""
    from .tinker import datum_without_mask as convert

    return convert(deps=deps, datum=datum)


async def submit_update(
    *,
    deps: Any,
    training_client: Any,
    reference_client: Any | None,
    datums: list[Any],
    config: TrainConfig,
    before_submit: Callable[[], None] | None = None,
    segments: list[Any] | None = None,
) -> dict[str, Any]:
    """Apply optional KL, run one CISPO forward_backward + Adam step, return metrics.

    ``before_submit`` runs synchronously after KL/local setup, immediately before
    the first training RPC. Direct callers need not provide a durability hook.

    Main callers:
        ``run_train`` once per step with every banked group's datums.
    """
    train = config.train
    if config.compute.backend in {"mlx", "studio"}:
        batch = TokenBatch(tuple(segments if segments is not None else (item.datum for item in datums)))
        counts = {
            "prompt": sum(len(x.prompt_token_ids) for x in batch.sequences),
            "completion": sum(len(x.sampled_tokens) for x in batch.sequences),
        }
        for kind, count in counts.items():
            limit = getattr(train, f"max_update_{kind}_tokens")
            if limit and count > limit:
                raise ValueError(
                    f"Update {kind} token budget exceeded: {count} > {limit}; "
                    "no training RPC submitted and no sequences truncated."
                )
        if config.compute.backend == "studio" and hasattr(training_client, "validate_update"):
            await asyncio.to_thread(
                training_client.validate_update, batch, learning_rate=train.learning_rate,
                clip_low=train.cispo_clip_low, clip_high=train.cispo_clip_high,
            )
        if before_submit is not None:
            before_submit()
        result = await training_client.update(
            batch,
            learning_rate=train.learning_rate,
            clip_low=train.cispo_clip_low,
            clip_high=train.cispo_clip_high,
        )
        if config.compute.backend == "studio":
            from shellsmith.train.studio import compact_update_metrics
            return compact_update_metrics(result)
        return result
    from .tinker import TinkerBackend

    adapter = (
        training_client
        if isinstance(training_client, TinkerBackend)
        else TinkerBackend(
            deps=deps,
            training_client=training_client,
            reference_client=reference_client,
        )
    )
    return await adapter.update_raw(datums, config=config, before_submit=before_submit, sequences=segments)


async def save_checkpoint(
    *,
    training_client: Any,
    kind: str,
    name: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Save sampler weights (``kind='sampler'``) or full optimizer state (``'state'``)."""
    if hasattr(training_client, "save_checkpoint"):
        return await training_client.save_checkpoint(
            kind=kind, name=name, metadata=metadata
        )
    future = await (
        training_client.save_weights_for_sampler_async(name)
        if kind == "sampler"
        else training_client.save_state_async(name)
    )
    saved = await future.result_async()
    path = getattr(saved, "path", None)
    if not isinstance(path, str) or not path:
        raise RuntimeError(f"Tinker {kind} checkpoint {name!r} returned no path.")
    return path


def latest_full_checkpoint(state: dict[str, Any]) -> dict[str, Any] | None:
    """The newest ``state`` checkpoint row (with its step, world offset and sampler path)."""
    rows = [row for row in state.get("checkpoints", []) if row.get("kind") == "state"]
    return rows[-1] if rows else None


async def _ensure_full_checkpoint(
    *,
    training_client: Any,
    state: dict[str, Any],
    sampler_path: str | None,
    metrics_head: dict[str, Any] | None = None,
) -> str | None:
    """Cover committed updates and the settled cursor, including no-datum exits."""
    if state["step"] == 0 and "policy_identity" not in state:
        return None
    if sampler_path is None:
        raise RuntimeError(
            "Cannot save optimizer state without matching sampler weights."
        )
    checkpoint = latest_full_checkpoint(state)
    position = {
        "step": state["step"],
        "next_world_offset": state["next_world_offset"],
        "settled_world_offsets": list(state.get("settled_world_offsets", [])),
        "sampler_path": sampler_path,
    }
    if "policy_identity" in state:
        position.update(
            backend=state.get("compute_backend", "mlx"),
            policy_identity=state["policy_identity"],
        )
        position["committed_group_ids"] = list(state.get("committed_group_ids", []))
        position.update(
            {
                key: state[key]
                for key in ("runtime_dependencies", "python_runtime")
                if key in state
            }
        )
        if HEAD_KEY in state:
            position[HEAD_KEY] = (
                metrics_head if metrics_head is not None else state[HEAD_KEY]
            )
    if checkpoint is not None and all(
        checkpoint.get(key) == value for key, value in position.items()
    ):
        return None
    # A later skipped update can advance the cursor without changing the step.
    name = (
        f"{state['run_id']}-step{state['step']:05d}-world{state['next_world_offset']:05d}"
        f"-state{len(state['checkpoints']):05d}"
    )
    if "policy_identity" in state:
        name += f"-s{len(state['segments']) - 1:03d}"
    path = await save_checkpoint(
        training_client=training_client, kind="state", name=name, metadata=position
    )
    state["checkpoints"].append(
        {"kind": "state", "name": name, "path": path, **position}
    )
    return path


async def _recover_studio_pending(
    *, backend, state, state_path, journal, telemetry_enabled: bool = True,
):
    """Reconcile the service's committed receipt before collecting another world."""
    pending = state.get("pending_studio_update")
    if not pending:
        return
    try:
        operation = await backend.operation_status(pending["id"] + ":optimizer")
    except ValueError:
        operation = None
    if operation and operation["status"] in {"queued", "running"}:
        # initialize() serialized a restore behind this client's prior work, so
        # a still-running operation would indicate a conflicting coordinator.
        raise RuntimeError(
            "Studio optimizer remains active after serialized restore; refusing cursor advancement."
        )
    if operation and operation["status"] == "succeeded":
        result = operation["result"]
        revision = result["optim_metrics"]["optimizer_steps"]
        if revision != pending["step"] + 1:
            raise RuntimeError(
                "Studio receipt does not match the pending Shellsmith update."
            )
        forward = await backend.operation_status(pending["id"] + ":backward")
        if forward["status"] != "succeeded":
            raise RuntimeError(
                "Committed Studio update has no matching backward receipt."
            )
        await backend.restore_state(
            result["checkpoint"],
            request_id=pending["id"] + ":recover:" + str(len(state["segments"])),
        )
        state.update(
            step=revision,
            rft_updates_completed=revision,
            next_world_offset=pending["next_world_offset"],
            settled_world_offsets=pending["settled_world_offsets"],
            optimizer_update_in_flight=False,
        )
        state["committed_group_ids"].extend(pending["group_ids"])
        from studio_compute.client import combine_update_receipts

        metrics = {
            **pending["metrics"],
            **combine_update_receipts(forward["result"], result),
            "optimizer_updates_committed": revision,
            "committed_group_ids": pending["group_ids"],
            "recovered_compute_receipt": pending["id"],
            "seconds": time.time() - pending["started"],
        }
        from shellsmith.train.studio import compact_update_metrics
        metrics = compact_update_metrics(metrics)
        name = pending["id"].replace(":", "-") + "-recovered"
        sampler_path = await backend.save_checkpoint(kind="sampler", name=name)
        state["checkpoints"].append(
            {
                "kind": "sampler",
                "name": name,
                "path": sampler_path,
                "rft_update": revision,
            }
        )
        metrics["sampler_checkpoint"] = sampler_path
        state["active_policy_ref"] = sampler_path
        await _ensure_full_checkpoint(
            training_client=backend,
            state=state,
            sampler_path=sampler_path,
            metrics_head=journal.preview_head(metrics),
        )
        state.pop("pending_studio_update")
        journal.publish(
            row=metrics,
            state_path=state_path,
            state=state,
            checkpoint=latest_full_checkpoint(state),
        )
        publication_started = time.monotonic()
        if await backend.publish_policy(revision=revision) != sampler_path:
            raise RuntimeError(
                "Recovered Studio policy differs from its durable publication."
            )
        persist_publication(state_path.parent, {
            "step": revision, "policy_ref": sampler_path,
            "seconds": time.monotonic() - publication_started,
            "segment": len(state["segments"]), "context": "recovery",
        }, enabled=telemetry_enabled)
    else:
        # No optimizer receipt committed. initialize() restored the previous full
        # state; the next segment gets fresh operation IDs and reuses captures.
        state.pop("pending_studio_update")
        write_json(state_path, state)


# --- the loop ------------------------------------------------------------------------


def _episode_args(
    config: TrainConfig, world: SelectedWorld | None = None
) -> argparse.Namespace:
    """Compatibility wrapper; shared campaign/trainer implementation."""
    return episode_args(config, world)


def _stream_config(config: TrainConfig) -> StreamPolicyConfig:
    """Compatibility wrapper for training stream-policy consumers."""
    return stream_config(
        config,
        campaign_mode="tinker_train",
        groups_per_update=config.train.groups_per_update,
    )


async def run_train(config: TrainConfig, *, deps: Any = None) -> dict[str, Any]:
    """Run/resume as the exclusive owner of this run directory."""
    from shellsmith.train.run_lock import lock_train_run

    validate_train_config(config)
    require_telemetry(config)
    with lock_train_run(config.resolved_log_dir()):
        if config.compute.backend == "mlx":
            from shellsmith.jobs.lease import GpuLease

            with GpuLease(config.compute.gpu_lease_root or None):
                return await _run_train_locked(config, deps=None)
        if deps is None and config.compute.backend == "tinker":
            from shellsmith.policy.deps import load_tinker_deps

            deps = load_tinker_deps()
        return await _run_train_locked(config, deps=deps)


async def _run_train_locked(config: TrainConfig, *, deps: Any) -> dict[str, Any]:
    """Inspect state, collect, update and finalize only while holding ownership."""
    with ExitStack() as resources:
        return await _run_train_owned(config, deps=deps, resources=resources)


async def _run_train_owned(
    config: TrainConfig, *, deps: Any, resources: ExitStack
) -> dict[str, Any]:
    world_stream = select_world_stream(config)
    task_runs = world_task_runs(config, world_stream.worlds)
    log_dir = config.resolved_log_dir()
    print(f"shellsmith: log dir {log_dir}", flush=True)
    frozen_worlds = freeze_worlds(world_stream.worlds)
    snap = snapshot(config, world_stream, frozen_worlds=frozen_worlds)
    state_path = log_dir / STATE_FILE
    snapshot_path = log_dir / SNAPSHOT_FILE
    checkpoint = None
    saved = None
    resume_from_base = False
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        saved = json.loads(snapshot_path.read_text(encoding="utf-8"))
        config = _resume_config(
            config=config, state=state, saved=saved, current=snap, log_dir=log_dir
        )
        checkpoint = latest_full_checkpoint(state)
        if checkpoint is None:
            # A failed optimizer RPC may have committed remotely while step is
            # still zero. Only an explicit durable no-update marker is safe;
            # missing markers and any checkpoint/update history fail closed.
            resume_from_base = (
                all(
                    type(state.get(key)) is int and state[key] == 0
                    for key in ("step", "rft_updates_completed")
                )
                and state.get("checkpoints") == []
                and state.get("optimizer_update_in_flight") is False
            )
            if not resume_from_base:
                raise RuntimeError(
                    f"{log_dir} has no full checkpoint to resume from safely; start a new log_dir."
                )
    else:
        entries = list(log_dir.iterdir()) if log_dir.exists() else []
        if entries:
            # This is the only publication window before any collection or
            # optimizer work. Do not adopt directories containing other data.
            allowed = {snapshot_path}
            # Telemetry is separate from optimizer authority. Startup can leave
            # only zero-update initialization observations before state exists.
            observation_dir = log_dir / "observability"
            if observation_dir.is_dir():
                metadata = json.loads((observation_dir / "metadata.json").read_text())
                if metadata.get("run_id") != json.loads(snapshot_path.read_text())["config"]["run"]["run_id"]:
                    raise RuntimeError("Startup telemetry belongs to a different run.")
                for line in (observation_dir / "events.jsonl").read_text().splitlines():
                    record = json.loads(line)
                    if record.get("train/optimizer_updates_committed") != 0 or record.get("kind") not in {"phase", "gap", "health"}:
                        raise RuntimeError("Startup telemetry contains post-initialization work; refusing adoption.")
                allowed.update({observation_dir, log_dir / "backprop", log_dir / "backprop.json",
                                log_dir / "wandb", log_dir / "wandb.json", log_dir / "telemetry-health.jsonl"})
            if config.compute.backend == "studio":
                from .studio_initialization import INITIALIZATION_FILE
                allowed.add(log_dir / INITIALIZATION_FILE)
            if (
                set(entries) - allowed
                or not snapshot_path.is_file()
                or snapshot_path.is_symlink()
            ):
                raise RuntimeError(
                    f"log_dir {log_dir} is not empty and has no {STATE_FILE}."
                )
            saved = json.loads(snapshot_path.read_text(encoding="utf-8"))
            config = _resume_config(
                config=config,
                state={"run_id": saved["config"]["run"]["run_id"]},
                saved=saved,
                current=snap,
                log_dir=log_dir,
            )
        state = {
            "run_id": config.run.run_id,
            "status": "running",
            "step": 0,
            "rft_updates_completed": 0,
            "optimizer_update_in_flight": False,
            "next_world_offset": 0,
            "settled_world_offsets": [],
            "checkpoints": [],
            "segments": [],
            "group_identities": {},
        }

    if config.compute.backend == "mlx" and (
        type(state.get("initial_checkpoint_attempt", 0)) is not int
        or state.get("initial_checkpoint_attempt", 0) < 0
    ):
        raise RuntimeError(
            "Invalid initial MLX checkpoint attempt counter; refusing startup."
        )

    # Resume may restore a different run ID and grow the segment's digit width.
    # Budget the actual campaign names before any external sampler/SDK access.
    segment = len(state["segments"])
    segment_root = log_dir / "segments" / f"{segment:03d}"
    campaign_id = f"{config.run.run_id}-s{segment:03d}"
    validate_training_names(
        config=config,
        worlds=world_stream.worlds,
        campaign_id=campaign_id,
        segment_root=segment_root,
    )
    for frozen in frozen_worlds:
        frozen.validate_names(segment_root)
    if config.compute.backend == "mlx":
        # Native artifacts live on this filesystem; budget retry suffixes and
        # the largest reachable update/cursor before allocating the model.
        max_step = min(config.train.steps, state["step"] + len(task_runs))
        count = len(state["checkpoints"]) + 2 * len(task_runs) + 2
        names = (
            f"{config.run.run_id}-step00000-initial{state.get('initial_checkpoint_attempt', 0) + 1:05d}-state00000",
            f"{config.run.run_id}-step{max_step:05d}-world{len(task_runs):05d}-state{count:05d}-s{segment:03d}",
        )
        for name in names:
            validate_component_name(
                name,
                label="MLX checkpoint directory",
                directory=log_dir / "checkpoints",
            )

    if snap["pi_sdk"]["module_sha256"] is None:
        raise RuntimeError(
            "The pinned Pi SDK is not installed; run npm ci --ignore-scripts in the vat_runner directory or configure player.pi_sdk_module_path."
        )
    if snap["codex"]["executable_sha256"] is None:
        raise RuntimeError(
            "Codex executable not found on PATH; install Codex before training."
        )

    local_session = None
    sampler = None
    if config.compute.backend == "studio":
        from .studio import StudioBackend

        # Discovery is read-only. Check before snapshots, target ratcheting,
        # journal repair, remote initialization, or a remote cleanup callback.
        local_session = await asyncio.to_thread(StudioBackend, config)
        if local_session.policy_identity != snap["policy_identity"]:
            raise RuntimeError("Studio model/runtime identity changed between snapshot and backend discovery.")
    if config.compute.backend == "tinker":
        sampler = TinkerPlayerSampler(
            config=PlayerSamplingConfig(
                base_model=config.player.base_model,
                thinking_renderer=config.player.thinking_renderer,
                plain_renderer=config.player.plain_renderer,
                temperature=config.player.temperature,
                top_p=config.player.top_p,
                max_tokens=config.player.max_tokens,
                thinking_anomaly_attempts=config.retry.thinking_anomaly_attempts,
            ),
            deps=deps,
            sampling_client=None,
        )
        # Tokenizer-specific: positivity alone cannot guarantee room for a forced
        # reasoning close. Validate once, before creating any paid training client.
        await asyncio.to_thread(sampler.validate_thinking_budget)

    # Preserve the immutable snapshot while durably ratcheting the accepted
    # target *before* SDK setup. Even a failed restore/extension may not later
    # lower that target. A fresh/snapshot-only ordinary retry still publishes
    # no state until setup, retaining the safe M4 publication window.
    previous_target = max(
        state.get("target_steps", 0), saved["config"]["train"]["steps"] if saved else 0
    )
    state["target_steps"] = config.train.steps
    if saved is not None and config.train.steps > previous_target:
        write_json(state_path, state)

    # Local config, confinement, resume and budget checks precede SDK authentication.
    # Restore metrics against the same full checkpoint (or untouched base state),
    # never the later attempted step/cursor retained by failure finalization.
    metrics_journal = MetricsJournal.recover(
        log_dir / METRICS_FILE, state=state, checkpoint=checkpoint
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    if not snapshot_path.is_file():
        write_json(snapshot_path, snap)
    telemetry = resources.enter_context(TrainingTelemetry(config, snapshot=snap, state=state, segment=segment))
    if local_session is not None:
        local_session.observation_phase = telemetry.phase
    telemetry.phase("restore" if checkpoint else "initialization")
    sampler_path = None
    if config.compute.backend in {"mlx", "studio"}:
        if not snapshot_path.is_file():
            write_json(snapshot_path, snap)
        if config.compute.backend == "studio":
            await local_session.initialize(checkpoint)
            resources.callback(local_session.close)
        else:
            from shellsmith.mlx import create_session

            local_session = await create_session(
                config, checkpoint=checkpoint, checkpoint_root=log_dir / "checkpoints"
            )
        if config.compute.backend != "studio":
            resources.callback(local_session.close)
        if local_session.policy_identity != snap["policy_identity"]:
            raise RuntimeError(
                "MLX model/runtime identity changed between snapshot and session creation."
            )
        training_client = local_session
        sampler = local_session.sampler
        sampling_client = None
        state["compute_backend"] = config.compute.backend
        state["policy_identity"] = local_session.policy_identity
        state["active_policy_ref"] = local_session.policy_ref
        state["committed_group_ids"] = (
            list(checkpoint.get("committed_group_ids", [])) if checkpoint else []
        )
        state.update(
            {
                key: snap[key]
                for key in ("runtime_dependencies", "python_runtime")
                if key in snap
            }
        )
        if checkpoint is not None:
            sampler_path = checkpoint["sampler_path"]
            state.update(
                step=checkpoint["step"],
                rft_updates_completed=checkpoint["step"],
                next_world_offset=checkpoint["next_world_offset"],
                optimizer_update_in_flight=False,
                settled_world_offsets=list(checkpoint.get("settled_world_offsets", [])),
                status="running",
            )
            state["resumes"] = state.get("resumes", 0) + 1
        else:
            # Publish proof of untouched base weights before any checkpoint
            # side effects. Failed baseline attempts retain diagnostic artifacts
            # and safely retry under a fresh immutable artifact name.
            attempt = state.get("initial_checkpoint_attempt", 0) + 1
            state.update(
                status="running",
                baseline_pending=True,
                initial_checkpoint_attempt=attempt,
            )
            write_json(state_path, state)
            try:
                sampler_name = f"{config.run.run_id}-step00000-initial{attempt:05d}"
                checkpoint_started = time.monotonic()
                sampler_path = await save_checkpoint(
                    training_client=local_session, kind="sampler", name=sampler_name
                )
                sampler_checkpoint_seconds = time.monotonic() - checkpoint_started
                position = {
                    "step": 0,
                    "next_world_offset": 0,
                    "settled_world_offsets": [],
                    "sampler_path": sampler_path,
                    "backend": config.compute.backend,
                    "policy_identity": local_session.policy_identity,
                    "committed_group_ids": [],
                    HEAD_KEY: state[HEAD_KEY],
                    **{
                        key: state[key]
                        for key in ("runtime_dependencies", "python_runtime")
                        if key in state
                    },
                }
                state_name = f"{sampler_name}-state00000"
                checkpoint_started = time.monotonic()
                initial_state = await save_checkpoint(
                    training_client=local_session,
                    kind="state",
                    name=state_name,
                    metadata=position,
                )
                persist_publication(log_dir, {
                    "step": 0, "policy_ref": sampler_path,
                    "sampler_checkpoint_seconds": sampler_checkpoint_seconds,
                    "state_checkpoint_seconds": time.monotonic() - checkpoint_started,
                    "segment": segment, "kind": "state_checkpoint",
                    "context": "baseline",
                }, enabled=is_telemetry_enabled(config))
                state["checkpoints"].extend(
                    [
                        {
                            "kind": "sampler",
                            "name": sampler_name,
                            "path": sampler_path,
                            "rft_update": 0,
                        },
                        {
                            "kind": "state",
                            "name": state_name,
                            "path": initial_state,
                            **position,
                        },
                    ]
                )
                state.update(active_policy_ref=sampler_path, baseline_pending=False)
                write_json(state_path, state)
                publication_started = time.monotonic()
                if await local_session.publish_policy(revision=0) != sampler_path:
                    raise RuntimeError(
                        "MLX initial published revision differs from its saved sampler."
                    )
                persist_publication(log_dir, {
                    "step": 0, "policy_ref": sampler_path,
                    "seconds": time.monotonic() - publication_started,
                    "segment": segment, "context": "baseline",
                }, enabled=is_telemetry_enabled(config))
            except BaseException:
                state["status"] = "failed"
                write_json(state_path, state)
                raise
    else:
        from .tinker import TinkerBackend

        training_client, sampling_client = await TinkerBackend.create(
            deps=deps,
            config=config,
            sampler=sampler,
            checkpoint=checkpoint,
            require_context=require_long_context_async,
            before_training=lambda: (
                write_json(snapshot_path, snap) if not snapshot_path.is_file() else None
            ),
        )
        if checkpoint is not None:
            sampler_path = checkpoint["sampler_path"]
            state.update(
                step=checkpoint["step"],
                rft_updates_completed=checkpoint["step"],
                next_world_offset=checkpoint["next_world_offset"],
                optimizer_update_in_flight=False,
                settled_world_offsets=list(checkpoint.get("settled_world_offsets", [])),
                status="running",
            )
            state["resumes"] = state.get("resumes", 0) + 1
            print(
                f"shellsmith: resuming at step {state['step']} from {checkpoint['path']}",
                flush=True,
            )
        else:
            if not snapshot_path.is_file():
                write_json(snapshot_path, snap)
            if resume_from_base:
                state["status"] = "running"
                state["resumes"] = state.get("resumes", 0) + 1
        reference_client = training_client.reference_client
    if local_session is not None:
        reference_client = None
        if config.compute.backend == "studio" and checkpoint is not None:
            await _recover_studio_pending(
                backend=local_session,
                state=state,
                state_path=state_path,
                journal=metrics_journal,
                telemetry_enabled=is_telemetry_enabled(config),
            )
            sampler_path = (
                state["active_policy_ref"]
                if state["step"] != checkpoint["step"]
                else checkpoint["sampler_path"]
            )
            state["active_policy_ref"] = sampler_path
            publication_started = time.monotonic()
            published = await local_session.publish_policy(revision=state["step"])
            persist_publication(log_dir, {
                "step": state["step"], "policy_ref": published,
                "seconds": time.monotonic() - publication_started,
                "segment": segment, "context": "resume",
            }, enabled=is_telemetry_enabled(config))

    state["segments"].append(
        {"index": segment, "root": str(segment_root), "start_step": state["step"]}
    )
    write_json(state_path, state)
    stack = None
    halted: dict[str, Any] | None = None
    primary_error: BaseException | None = None
    try:
        stack = open_rollout_stack(
            config,
            deps=deps,
            sampling_client=sampling_client,
            root=segment_root,
            campaign_id=campaign_id,
            campaign_mode=(config.compute.backend + "_train")
            if local_session
            else "tinker_train",
            groups_per_update=config.train.groups_per_update,
            state=state,
            prepared_sampler=sampler,
            pi_sdk_module_path=Path(snap["pi_sdk"]["module_path"]),
            codex_identity=snap["codex"],
            task_runs=task_runs,
            frozen_worlds=frozen_worlds,
        )
        stack.runner = telemetry.wrap_runner(stack.runner)
        capture_store = stack.capture_store
        while (
            state["step"] < config.train.steps
            and state["next_world_offset"] < len(task_runs)
            and halted is None
        ):
            step = state["step"]
            started = time.time()
            telemetry.phase("collection")
            telemetry._health()
            update = await collect_update_worlds(
                task_runs=task_runs,
                start_offset=state["next_world_offset"],
                groups_per_update=config.train.groups_per_update,
                base_config=stack.stream,
                settled_world_offsets=set(state.get("settled_world_offsets", [])),
                classify_flavor=stack.classify_world,
                collect_one=stack.collect_one,
            )
            collection_seconds = time.time() - started
            telemetry.phase("data_preparation")
            datums: list[Any] = []
            segments: list[Any] = []
            collected_group_ids: list[str] = []
            world_rows: list[dict[str, Any]] = []
            for settled in update.worlds:
                result = settled.result
                row = summarize_world(result)
                if result.decision.action == WorldAction.GROUP_READY:
                    captures = capture_store.captures_for_outcomes(result.outcomes)
                    group_id = f"step{step}:{result.world_id}"
                    bridge = build_tinker_group(
                        # Keep credit portable; only the adapter builds SDK tensors.
                        deps=None, group_id=group_id, step=step, rollouts=captures,
                        advantage_std_floor=config.credit.advantage_std_floor,
                        advantage_cap=config.credit.advantage_cap,
                        turn_config=TurnAdvantageConfig(**{
                            name: getattr(config.credit, name)
                            for name in TurnAdvantageConfig.__dataclass_fields__
                        }),
                    )
                    collected_group_ids.append(group_id)
                    datums.extend(bridge.datums)
                    segments.extend(bridge.segments)
                    row["training_segments"] = len(bridge.segments)
                    row["history_rewrites"] = list(bridge.history_rewrites)
                    row["datums"] = len(bridge.datums)
                elif result.decision.action == WorldAction.HALT_STREAM:
                    halted = {
                        "world_id": result.world_id,
                        "reason": result.decision.reason,
                    }
                world_rows.append(row)
            metrics: dict[str, Any] = {
                "step": step,
                "segment": segment,
                "worlds": world_rows,
                "datums": len(datums),
                "next_world_offset": update.next_world_offset,
                "settled_world_offsets": update.settled_world_offsets,
            }
            metrics["collection_seconds"] = collection_seconds
            if datums:
                if config.compute.backend == "studio":
                    from studio_compute.protocol import bounded_identifier
                    update_id = bounded_identifier(
                        f"{local_session.operation_prefix}:segment:{segment}:update:{step + 1}", limit=160
                    )
                    # Compute the exact RPC ID before size preflight, without
                    # publishing an ambiguity marker or advancing the cursor.
                    local_session.update_request_id = update_id

                def mark_submission() -> None:
                    # Keep the prior cursor through KL/local setup failures. Once
                    # an RPC may run, publish its cursor and ambiguity marker together.
                    state["next_world_offset"] = update.next_world_offset
                    state["settled_world_offsets"] = update.settled_world_offsets
                    state["optimizer_update_in_flight"] = True
                    if config.compute.backend == "studio":
                        state["pending_studio_update"] = {
                            "id": update_id,
                            "step": step,
                            "started": started,
                            "metrics": dict(metrics),
                            "group_ids": collected_group_ids,
                            "next_world_offset": update.next_world_offset,
                            "settled_world_offsets": update.settled_world_offsets,
                        }
                    write_json(state_path, state)

                telemetry.phase("forward_backward")
                metrics.update(
                    await submit_update(
                        deps=deps,
                        training_client=training_client,
                        reference_client=reference_client,
                        datums=datums,
                        config=config,
                        before_submit=mark_submission,
                        segments=segments,
                    )
                )
                state["step"] += 1
                state["rft_updates_completed"] = state["step"]
                state["optimizer_update_in_flight"] = False
                telemetry.phase("checkpoint")
                if local_session is not None:
                    state["committed_group_ids"].extend(collected_group_ids)
                    metrics["committed_group_ids"] = collected_group_ids
                name = f"{config.run.run_id}-step{state['step']:05d}"
                if local_session is not None:
                    name += f"-s{segment:03d}"
                checkpoint_started = time.monotonic()
                sampler_path = await save_checkpoint(
                    training_client=training_client, kind="sampler", name=name
                )
                metrics["sampler_checkpoint_seconds"] = time.monotonic() - checkpoint_started
                state["checkpoints"].append(
                    {
                        "kind": "sampler",
                        "name": name,
                        "path": sampler_path,
                        "rft_update": state["step"],
                    }
                )
                if local_session is None:
                    await training_client.publish_policy(revision=state["step"])
                metrics["sampler_checkpoint"] = sampler_path
            else:
                state["next_world_offset"] = update.next_world_offset
                state["settled_world_offsets"] = update.settled_world_offsets
                metrics["skipped"] = "no trainable group in this update"
            last = (
                state["step"] >= config.train.steps
                or state["next_world_offset"] >= len(task_runs)
                or halted is not None
            )
            metrics["optimizer_updates_committed"] = state["rft_updates_completed"]
            state_checkpoint_seconds = None
            if local_session is not None:
                # The immutable native envelope carries the exact upcoming
                # journal boundary, including skipped-world cursor progress.
                metrics["seconds"] = round(time.time() - started, 1)
            if (
                local_session is not None
                or last
                or (datums and state["step"] % config.train.checkpoint_every == 0)
            ):
                checkpoint_started = time.monotonic()
                state_ckpt = await _ensure_full_checkpoint(
                    training_client=training_client,
                    state=state,
                    sampler_path=sampler_path,
                    metrics_head=metrics_journal.preview_head(metrics)
                    if local_session is not None
                    else None,
                )
                if state_ckpt is not None:
                    state_checkpoint_seconds = time.monotonic() - checkpoint_started
                if state_ckpt is not None and local_session is None:
                    metrics["state_checkpoint"] = state_ckpt
            if local_session is None:
                metrics["seconds"] = round(time.time() - started, 1)
            if local_session is not None and datums:
                state["active_policy_ref"] = sampler_path
            if config.compute.backend == "studio":
                state.pop("pending_studio_update", None)
            metrics_journal.publish(
                row=metrics,
                state_path=state_path,
                state=state,
                checkpoint=latest_full_checkpoint(state),
            )
            telemetry.enqueue("update", json.loads(json.dumps(metrics)))
            if state_checkpoint_seconds is not None:
                # Checkpoint duration is known only after the immutable metrics seal.
                receipt = {"step": state["step"], "policy_ref": sampler_path,
                           "state_checkpoint_seconds": state_checkpoint_seconds,
                           "segment": segment, "kind": "state_checkpoint"}
                if persist_publication(
                    log_dir, receipt, enabled=is_telemetry_enabled(config)
                ):
                    telemetry.enqueue("publication", receipt)
            if local_session is not None and datums:
                telemetry.phase("publication")
                publication_started = time.monotonic()
                published = await local_session.publish_policy(revision=state["step"])
                if published != sampler_path:
                    raise RuntimeError(
                        "MLX published revision differs from the durable sampler checkpoint."
                    )
                publication = {"step": state["step"], "policy_ref": published,
                               "seconds": time.monotonic() - publication_started,
                               "segment": segment}
                if persist_publication(
                    log_dir, publication, enabled=is_telemetry_enabled(config)
                ):
                    telemetry.enqueue("publication", publication)
            print(
                f"shellsmith: step {step} datums={len(datums)} offset={state['next_world_offset']}",
                flush=True,
            )
        state["status"] = "halted" if halted else "complete"
        telemetry.phase("finalization", outcome=state["status"])
        state["halt"] = halted
        # Controller lifecycle is separate from whether training committed Adam.
        state["training_outcome"] = ("optimizer_updates_committed" if state["rft_updates_completed"] > 0
                                     else "no_optimizer_updates_committed")
    except BaseException as caught:
        primary_error = caught
        state["status"] = "failed"
        telemetry.phase("finalization", outcome="failed")
        raise
    finally:
        try:
            if stack is not None:
                stack.close(
                    status=state["status"]
                    if state["status"] != "running"
                    else "failed",
                    halt=halted,
                )
        except BaseException as caught:
            state["status"] = "failed"
            state["finalization_error"] = {
                "type": type(caught).__name__,
                "message": str(caught),
            }
            if primary_error is None:
                raise
            primary_error.add_note(f"Campaign finalization also failed: {caught}")
        finally:
            write_json(state_path, state)
    return state
