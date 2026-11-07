"""``shellsmith campaign``: adaptive per-world inference, without optimizer work.

Resolve and validate the immutable world/GM queue, overrides, composed names,
SDK/Codex bindings and Player token budget before authentication or paid work.
The shared rollout stack then runs the same episodes and capture audits as
training, admitting a small window so top-ups precede later worlds.

``campaign_summary.json`` retains actual completed results even if a sibling
is interrupted. Started-but-incomplete worlds are separate from worlds never
started. A HALT is terminal for this campaign even though the training-oriented
sparse collector cursor deliberately leaves that world unfinished.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from shellsmith.policy.sampler import PlayerSamplingConfig, TinkerPlayerSampler
from shellsmith.policy.tinker_setup import require_long_context_async
from shellsmith.campaign.evaluation import summarize_evaluation
from shellsmith.process_io import write_json
from shellsmith.train.config import TrainConfig, validate_train_config
from shellsmith.train.name_budget import validate_training_names
from shellsmith.train.rollout_stack import (
    open_rollout_stack,
    summarize_world,
    world_task_runs,
)
from shellsmith.train.stream_engine import (
    UpdateWorldResult,
    WorldCollectResult,
    collect_update_worlds,
)
from shellsmith.train.stream_policy import StreamPolicyConfig, WorldAction
from shellsmith.train.world_selection import freeze_worlds, select_world_stream

SNAPSHOT_FILE = "config_snapshot.json"
SUMMARY_FILE = "campaign_summary.json"


def world_window(
    config: TrainConfig, task_runs: list[tuple[Path, argparse.Namespace]] | None = None
) -> int:
    """Size the next queue prefix using merged initial batches, allowing one oversized world.

    Without a queue, retain the uniform-default capacity estimate for existing
    callers. Campaign admission always supplies the preflighted, merged queue.
    """
    if task_runs is None:
        return max(1, config.adaptive.parallelism // config.adaptive.initial_attempts)
    attempts = 0
    count = 0
    for _, task_args in task_runs:
        initial = int(task_args.initial_attempts)
        if count and attempts + initial > config.adaptive.parallelism:
            break
        attempts += initial
        count += 1
    return count


def campaign_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare attempts with each world's merged initial count, not the global default."""
    decisions: dict[str, int] = {}
    for row in rows:
        decisions[row["decision"]] = decisions.get(row["decision"], 0) + 1
    return {
        "worlds": len(rows),
        "decisions": decisions,
        "worlds_topped_up": sum(
            1 for row in rows if row["attempts"] > row["initial_attempts"]
        ),
        "worlds_on_backup_gm": sum(1 for row in rows if row["gm_fallback_applied"]),
        "episodes": sum(row["attempts"] + row["discarded"] for row in rows),
        "accepted_episodes": sum(row["accepted"] for row in rows),
    }


async def run_inference_campaign(
    config: TrainConfig,
    *,
    deps: Any = None,
    sampler_path: str | None = None,
    adapter_path: Path | None = None,
) -> dict[str, Any]:
    """Run a campaign as the exclusive owner of its output directory."""
    from shellsmith.train.run_lock import lock_train_run

    validate_train_config(config, campaign_mode="inference")
    if config.compute.backend == "studio" and sampler_path is not None:
        key = sampler_path.removeprefix("studio-policy:")
        if not sampler_path.startswith("studio-policy:") or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Studio inference requires an opaque published policy reference.")
    # Share the hardened sidecar with trainers: neither kind of writer may
    # inspect or replace the other's provenance at the same canonical path.
    # The lock itself never populates log_dir and survives through finalization.
    if adapter_path is not None and config.compute.backend != "mlx":
        raise ValueError("MLX adapters require the MLX compute backend.")
    if sampler_path is not None and config.compute.backend == "mlx":
        raise ValueError("Tinker sampler references cannot be loaded by MLX.")
    if config.compute.backend == "tinker" and deps is None:
        from shellsmith.policy.deps import load_tinker_deps

        deps = load_tinker_deps()
    with lock_train_run(config.resolved_log_dir(kind="campaigns")):
        if config.compute.backend == "mlx":
            from shellsmith.jobs.lease import GpuLease

            holder: dict[str, Any] = {}
            with GpuLease(config.compute.gpu_lease_root or None):
                try:
                    return await _run_inference_campaign_locked(
                        config,
                        deps=None,
                        sampler_path=None,
                        adapter_path=adapter_path,
                        session_holder=holder,
                    )
                finally:
                    if "session" in holder:
                        holder["session"].close()
        holder = {}
        try:
            return await _run_inference_campaign_locked(
                config, deps=deps, sampler_path=sampler_path, session_holder=holder
            )
        finally:
            if "session" in holder:
                await asyncio.to_thread(holder["session"].close)


async def _run_inference_campaign_locked(
    config: TrainConfig,
    *,
    deps: Any,
    sampler_path: str | None,
    adapter_path: Path | None = None,
    session_holder: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Preflight, publish and collect only while holding directory ownership."""
    # The shared M4 snapshot is trainer-owned. Import it only after the campaign
    # module is initialized, not through an import-time campaign -> trainer ->
    # rollout_stack dependency. Selection and collection use their shared APIs
    # directly, never the trainer's training-mode path compatibility wrapper.
    from shellsmith.train.trainer import snapshot

    validate_train_config(config, campaign_mode="inference")
    log_dir = config.resolved_log_dir(kind="campaigns")
    print(f"shellsmith: log dir {log_dir}", flush=True)
    if log_dir.exists() and any(log_dir.iterdir()):
        raise RuntimeError(
            f"log_dir {log_dir} is not empty; a campaign needs a new log_dir."
        )
    world_stream = select_world_stream(config, campaign_mode="inference")
    task_runs = world_task_runs(config, world_stream.worlds, campaign_mode="inference")
    root = log_dir / "campaign"
    validate_training_names(
        config=config,
        worlds=world_stream.worlds,
        campaign_id=config.run.run_id,
        segment_root=root,
    )
    frozen_worlds = freeze_worlds(world_stream.worlds)
    for frozen in frozen_worlds:
        frozen.validate_names(root)
    snap = snapshot(config, world_stream, frozen_worlds=frozen_worlds) | {
        "sampler_path": sampler_path
    }
    if snap["pi_sdk"]["module_sha256"] is None:
        raise RuntimeError(
            "The pinned Pi SDK is not installed; configure player.pi_sdk_module_path."
        )
    if snap["codex"]["executable_sha256"] is None:
        raise RuntimeError(
            "Codex executable not found on PATH; install Codex before starting a campaign."
        )
    player = config.player
    state: dict[str, Any] = {
        "rft_updates_completed": 0,
        "checkpoints": [],
        "group_identities": {},
    }
    if config.compute.backend == "studio":
        from shellsmith.train.studio import StudioBackend

        from studio_compute.protocol import digest
        backend = await asyncio.to_thread(
            StudioBackend, config, client_id="inference:" + digest({
                "run_id": config.run.run_id, "output": str(log_dir.resolve()),
            })
        )
        if backend.policy_identity != snap["policy_identity"]:
            raise RuntimeError("Studio model or execution identity changed after campaign preflight.")
        if sampler_path is not None:
            policy = await asyncio.to_thread(backend.client.policy_info, sampler_path)
            if policy["identity"] != backend.capabilities["identity"]:
                raise RuntimeError("Studio identity changed during policy lookup.")
        await backend.initialize(inference=True)
        if session_holder is not None:
            session_holder["session"] = backend
        if sampler_path is None:
            sampler_path = await backend.publish_policy(revision=0)
        backend.policy_ref = sampler_path
        sampler, sampling_client = backend.sampler, None
        state.update(
            policy_identity=backend.policy_identity, active_policy_ref=sampler_path
        )
        snap.update(policy_ref=sampler_path)
    elif config.compute.backend == "mlx":
        from shellsmith.mlx import create_session
        from shellsmith.mlx.checkpoint import verify_checkpoint

        if adapter_path is not None:
            verify_checkpoint(
                adapter_path, kind="sampler", expected_identity=snap["policy_identity"]
            )
        session = await create_session(config, checkpoint_root=log_dir / "checkpoints")
        if session_holder is not None:
            session_holder["session"] = session
        if session.policy_identity != snap["policy_identity"]:
            raise RuntimeError("MLX model or runtime changed after campaign preflight.")
        if adapter_path is not None:
            session.load_adapter(adapter_path)
        sampler, sampling_client = session.sampler, None
        sampler_path = session.policy_ref if adapter_path is not None else None
        state.update(
            policy_identity=session.policy_identity,
            active_policy_ref=session.policy_ref,
        )
        snap.update(
            adapter_path=str(adapter_path.resolve()) if adapter_path else None,
            policy_ref=session.policy_ref,
        )
    else:
        sampler = TinkerPlayerSampler(
            config=PlayerSamplingConfig(
                base_model=player.base_model,
                thinking_renderer=player.thinking_renderer,
                plain_renderer=player.plain_renderer,
                temperature=player.temperature,
                top_p=player.top_p,
                max_tokens=player.max_tokens,
                thinking_anomaly_attempts=config.retry.thinking_anomaly_attempts,
            ),
            deps=deps,
            sampling_client=None,
        )
        await asyncio.to_thread(sampler.validate_thinking_budget)
        from shellsmith.train.tinker import TinkerBackend

        backend, sampling_client = await TinkerBackend.for_sampling(
            deps=deps,
            config=config,
            sampler=sampler,
            policy=sampler_path,
            require_context=require_long_context_async,
            before_setup=lambda: write_json(log_dir / SNAPSHOT_FILE, snap),
        )
    if config.compute.backend in {"mlx", "studio"}:
        write_json(log_dir / SNAPSHOT_FILE, snap)
    window = world_window(config, task_runs)
    stack = open_rollout_stack(
        config,
        deps=deps,
        sampling_client=sampling_client,
        root=root,
        campaign_id=config.run.run_id,
        campaign_mode="inference",
        groups_per_update=window,
        state=state,
        pinned_sampler_path=sampler_path,
        prepared_sampler=sampler,
        pi_sdk_module_path=Path(snap["pi_sdk"]["module_path"]),
        codex_identity=snap["codex"],
        task_runs=task_runs,
        frozen_worlds=frozen_worlds,
    )

    settled: dict[int, UpdateWorldResult] = {}
    started: set[str] = set()
    status = "running"
    halt: dict[str, Any] | None = None

    async def collect_one(
        world_file: Path,
        task_args: argparse.Namespace,
        world_config: StreamPolicyConfig,
        flavor: str,
    ) -> WorldCollectResult:
        started.add(task_args.world_id)
        return await stack.collect_one(world_file, task_args, world_config, flavor)

    def retain_completed(item: UpdateWorldResult) -> None:
        nonlocal halt
        # This is the collector's actual result, including its decision and
        # discarded outcomes. Never reconstruct a result from a partial report.
        settled[item.stream_offset] = item
        if item.result.decision.action == WorldAction.HALT_STREAM and halt is None:
            halt = {
                "world_id": item.result.world_id,
                "reason": item.result.decision.reason,
            }

    def write_summary() -> dict[str, Any]:
        rows = [
            summarize_world(item.result)
            | {
                "initial_attempts": item.task_args.initial_attempts,
                "evaluation": summarize_evaluation(
                    item.result.outcomes,
                    discarded_outcomes=item.result.discarded_outcomes,
                ),
            }
            for _, item in sorted(settled.items())
        ]
        completed_ids = {row["world_id"] for row in rows}
        incomplete_ids = [
            world.world_id
            for world in world_stream.worlds
            if world.world_id in started and world.world_id not in completed_ids
        ]
        summary = {
            "run_id": config.run.run_id,
            "status": status,
            "halt": halt,
            "sampler_path": sampler_path,
            "worlds_selected": len(task_runs),
            "worlds_started": len(started),
            "worlds_completed": len(rows),
            "worlds_not_started": len(task_runs) - len(started),
            "worlds_incomplete": len(incomplete_ids),
            "incomplete_world_ids": incomplete_ids,
            "world_window": window,
            "totals": campaign_totals(rows),  # Peak actual window size.
            "worlds": rows,
            "group_identities": state["group_identities"],
            "evaluation": summarize_evaluation(
                (
                    outcome
                    for item in settled.values()
                    for outcome in item.result.outcomes
                ),
                discarded_outcomes=(
                    outcome
                    for item in settled.values()
                    for outcome in item.result.discarded_outcomes
                ),
            ),
            "backend": config.compute.backend,
            "policy_identity": state.get("policy_identity"),
        }
        write_json(log_dir / SUMMARY_FILE, summary)
        return summary

    try:
        offset = 0
        window_end = 0
        sparse_offsets: set[int] = set()
        while offset < len(task_runs) and halt is None:
            if offset >= window_end:
                current_window = world_window(config, task_runs[offset:])
                window_end = offset + current_window
                window = max(window, current_window)
            # The shared collector refills skipped groups. Bound its entire
            # queue, not just its ready-group target, so replacements cannot
            # precede this window's top-ups. Keep absolute offsets and the same
            # end until even a sparse/cancelled gap in this window is settled.
            update = await collect_update_worlds(
                task_runs=task_runs[:window_end],
                start_offset=offset,
                groups_per_update=window_end - offset,
                base_config=stack.stream,
                classify_flavor=stack.classify_world,
                collect_one=collect_one,
                settled_world_offsets=sparse_offsets,
                on_world_completed=retain_completed,
            )
            # M4's cursor leaves a HALT unfinished for training resume. It need
            # not advance, so check the real decision before the progress guard.
            if halt is not None:
                break
            if update.next_world_offset <= offset:
                raise RuntimeError(
                    f"World stream did not advance past offset {offset}."
                )
            offset = update.next_world_offset
            sparse_offsets = set(update.settled_world_offsets)
            write_summary()
        status = "halted" if halt else "complete"
    except (asyncio.CancelledError, KeyboardInterrupt):
        status = "interrupted"
        raise
    except BaseException:
        status = "failed"
        raise
    finally:
        try:
            summary = write_summary()
        except BaseException as publication_error:
            # Final publication is part of the operation, not a successful
            # epilogue. Never finalize a failed write as a complete campaign.
            # Preserve an earlier failure/interruption if collection failed too.
            if status not in {"failed", "interrupted"}:
                status = (
                    "interrupted"
                    if isinstance(
                        publication_error, (asyncio.CancelledError, KeyboardInterrupt)
                    )
                    else "failed"
                )
            try:
                # A transient replace failure can still leave truthful durable
                # status. A persistent I/O fault cannot; propagate the original
                # publication error and keep the report non-successful anyway.
                write_summary()
            except BaseException:
                pass
            raise
        finally:
            # A full disk or failed atomic summary replacement must not leave
            # the private Player endpoint serving after this invocation exits.
            stack.close(status=status, halt=halt)
    print(
        f"shellsmith: campaign {status}: {json.dumps(summary['totals'], sort_keys=True)}",
        flush=True,
    )
    return summary
