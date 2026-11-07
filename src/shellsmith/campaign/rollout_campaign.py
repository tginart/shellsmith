"""Campaign layer: run many no-training episodes and write a scoreboard.

What this is (plain English)
----------------------------
One smoke is one Player ↔ GM episode (``shellsmith.episode.smoke``).
This module queues **many** of those episodes across tasks, runs them under a
shared concurrency limit, audits each, and aggregates reward / acceptance stats.

It does not train. The trainer may call ``run_campaign`` for calibration and
top-ups before Tinker handoff.

Pipeline
--------
::

    select world files (--world-file and/or --from-draft-catalog)
            |
            v
    build_jobs  (round-robin across tasks × attempts)
            |
            v
    parallel run_live_job
      - episode.smoke.run_live_smoke
      - episode.acceptance_audit.write_audit
            |
            v
    campaign_report.json / .md  (atomic checkpoints)

Scope: lives under ``shellsmith.campaign`` (many episodes).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from shellsmith.cli_args import GMArgumentParser
from shellsmith.config import (
    DEFAULT_GM_API_IDLE_TIMEOUT_SECONDS,
    DEFAULT_MAX_WORLD_STEPS,
    DEFAULT_GM_API_KEY_ENV,
    DEFAULT_GM_MODEL,
    DEFAULT_GM_REASONING_EFFORT,
    DEFAULT_GM_RUNTIME,
    DEFAULT_GM_TURN_TIMEOUT_SECONDS,
    GM_PROVIDER_BY_RUNTIME,
)
from shellsmith.paths import DEFAULT_LOG_ROOT
from shellsmith.episode.acceptance_audit import (
    write_audit,
)
from shellsmith.episode.failure import (
    outcome_failure_fields,
    read_episode_failure_detail,
    serialize_failure,
)
from shellsmith.episode.progress import (
    emit_simple,
    install_stdout_progress,
)
from shellsmith.campaign.draft_catalog import (
    DEFAULT_DRAFT_ROOT,
    NON_RUNNABLE,
    classify,
)
from shellsmith.campaign.provider_outage import (
    PROVIDER_OUTAGE_MODES,
    ProviderHealthEvent,
    ProviderHealthTracker,
    apply_backup_to_job_fields,
)
from shellsmith.world_file import (
    load_world_file,
)
from shellsmith.episode.smoke import (
    _parser as smoke_parser,
    run_live_smoke,
)


DEFAULT_OUTPUT_PARENT = DEFAULT_LOG_ROOT / "campaigns"
# Two eight-attempt worlds can run every episode at once.
MAX_PARALLELISM = 16


@dataclass(frozen=True)
class RolloutJob:
    """One independent task/attempt pair in the global campaign queue."""

    world_file: Path
    world_id: str
    flavor: str
    attempt: int
    episode_id: str
    output_dir: Path
    gm_runtime: str
    gm_model: str
    gm_provider: str
    gm_reasoning_effort: str
    gm_fallback_applied: bool = False
    original_gm_model: str | None = None


RolloutRunner = Callable[[RolloutJob, argparse.Namespace], Awaitable[dict[str, Any]]]


def fallback_output_dir(job: RolloutJob, *, backup_runtime: str) -> Path:
    """Return a distinct sibling directory path for a provider-fallback retry.

    Main callers:
        ``run_campaign`` when a GM runtime is marked down.

    Why this exists:
        A cancelled smoke run copies its partial diagnostic archive to
        ``job.output_dir``. ``run_live_smoke`` correctly refuses to overwrite
        that directory, so a backup provider must use a distinct directory.
        Keeping the original directory also preserves the cancelled run's logs.
    """
    return job.output_dir.with_name(
        f"{job.output_dir.name}__fallback_{backup_runtime.strip().lower()}"
    )


def not_started_outcome(job: RolloutJob) -> dict[str, Any]:
    """Build the scoreboard row for an episode that was never launched.

    Main callers:
        ``run_campaign_wave`` and inference ``run_campaign``, when the job's
        GM runtime is already marked DOWN and no process is started.

    Why this exists:
        A "provider is down, episode not started" row still belongs on the
        campaign scoreboard as evidence, but it is **not** a settled episode
        outcome. ``episode_not_started`` marks that difference so a later
        resume re-attempts the attempt slot instead of inheriting a permanent
        failure. Without the marker, one transient outage burns those
        ``rollout_NN`` slots forever: every resume replays the same synthetic
        failure, the group can never reach ``minimum_trainable``, and the world
        deterministically skips on every future run.
    """
    return _annotate_outcome(
        job,
        {
            "episode_id": job.episode_id,
            "world_id": job.world_id,
            "flavor": job.flavor,
            "attempt": job.attempt,
            "rollout_dir": str(job.output_dir),
            "error": (
                f"RuntimeError: GM provider {job.gm_runtime!r} "
                "is marked down; episode not started."
            ),
            "audit": None,
            "episode_not_started": True,
        },
    )


def outcome_is_settled(outcome: Mapping[str, Any]) -> bool:
    """True when a recorded row represents an episode that actually ran.

    Main callers:
        ``run_campaign_wave`` resume filtering.
    """
    return outcome.get("episode_not_started") is not True


def _parser() -> argparse.ArgumentParser:
    parser = GMArgumentParser(
        description="Run and audit shellsmith rollouts across tasks."
    )
    parser.add_argument("--world-file", type=Path, action="append")
    parser.add_argument("--from-draft-catalog", action="store_true")
    parser.add_argument("--draft-root", type=Path, default=DEFAULT_DRAFT_ROOT)
    parser.add_argument(
        "--flavor",
        action="append",
        choices=("agentic", "public_repo", "paper"),
        help="Restrict catalog selection; may be repeated.",
    )
    parser.add_argument(
        "--exclude-task",
        action="append",
        default=[],
        help="Exact world file ID to exclude; may be repeated.",
    )
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--rollouts-per-task", type=int, default=1)
    parser.add_argument("--parallelism", type=int, default=8)
    parser.add_argument(
        "--parallelism-per-provider",
        type=int,
        default=4,
        help=(
            "Max in-flight episodes per gm_runtime (claude/codex/pi). "
            "Must be between 1 and --parallelism."
        ),
    )
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--campaign-id")
    parser.add_argument(
        "--gm-runtime", choices=tuple(GM_PROVIDER_BY_RUNTIME), default=DEFAULT_GM_RUNTIME
    )
    parser.add_argument("--gm-model", help="GM model (Codex defaults to Sol; required for other runtimes).")
    parser.add_argument("--gm-provider", help="Provider override (default follows --gm-runtime).")
    parser.add_argument("--gm-api-key-env", default=DEFAULT_GM_API_KEY_ENV)
    parser.add_argument("--gm-reasoning-effort", default=DEFAULT_GM_REASONING_EFFORT)
    parser.add_argument("--gm-api-idle-timeout-seconds", type=float, default=DEFAULT_GM_API_IDLE_TIMEOUT_SECONDS)
    parser.add_argument("--max-world-steps", type=int, default=DEFAULT_MAX_WORLD_STEPS)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_GM_TURN_TIMEOUT_SECONDS,
        help="Deadline for each individual GM process turn.",
    )
    parser.add_argument(
        "--player-prompt-timeout-seconds",
        type=float,
        default=43_200.0,
        help="Deadline for the complete Player trajectory (default: 12h).",
    )
    parser.add_argument(
        "--provider-outage",
        choices=PROVIDER_OUTAGE_MODES,
        default="halt",
        help=(
            "When GM provider infra fails repeatedly: halt remaining jobs, "
            "or remap remaining jobs to the bidirectional backup GM "
            "(Sol↔Opus, terra↔Sonnet)."
        ),
    )
    parser.add_argument(
        "--player-base-url",
        help="Optional Player/OpenAI-compatible base URL (defaults to smoke default).",
    )
    parser.add_argument("--player-provider")
    parser.add_argument("--player-model")
    parser.add_argument("--player-api-key-env")
    parser.add_argument("--player-thinking")
    parser.add_argument("--player-context-window", type=int)
    parser.add_argument("--player-max-tokens", type=int)
    return parser


def select_worlds(args: argparse.Namespace) -> list[tuple[Path, str]]:
    """Resolve explicit and catalog task selection, rejecting ambiguous emptiness."""
    selected: dict[Path, str] = {}
    for raw_path in args.world_file or []:
        path = raw_path.expanduser().resolve()
        spec = load_world_file(path)
        selected[path] = classify(spec, path=path)

    if args.from_draft_catalog:
        root = args.draft_root.expanduser().resolve()
        allowed_flavors = set(args.flavor or ("agentic", "public_repo", "paper"))
        for path in sorted(root.glob("*.md")):
            if path.name in NON_RUNNABLE:
                continue
            spec = load_world_file(path)
            flavor = classify(spec, path=path)
            if flavor in allowed_flavors:
                selected[path] = flavor

    excluded = set(args.exclude_task)
    selected_ids = {path.stem for path in selected}
    unknown_exclusions = sorted(excluded - selected_ids)
    if unknown_exclusions:
        raise RuntimeError(
            "--exclude-task contains IDs that were not selected: "
            + ", ".join(unknown_exclusions)
        )
    result = [
        (path, flavor)
        for path, flavor in selected.items()
        if path.stem not in excluded
    ]
    result.sort(key=lambda item: (item[1], item[0].stem))
    paths_by_id: dict[str, list[Path]] = {}
    for path, _ in result:
        paths_by_id.setdefault(path.stem, []).append(path)
    duplicate_ids = {
        task_id: paths
        for task_id, paths in paths_by_id.items()
        if len(paths) > 1
    }
    if duplicate_ids:
        details = "; ".join(
            f"{task_id}: {', '.join(str(path) for path in paths)}"
            for task_id, paths in sorted(duplicate_ids.items())
        )
        raise RuntimeError(
            "Selected task environments have duplicate IDs, which would collide "
            f"in episode IDs and output paths: {details}"
        )
    if args.limit_tasks is not None:
        if args.limit_tasks < 1:
            raise RuntimeError("--limit-tasks must be at least one.")
        result = result[: args.limit_tasks]
    if not result:
        raise RuntimeError(
            "No worlds selected. Pass --world-file or --from-draft-catalog."
        )
    return result


def build_jobs(
    *,
    campaign_id: str,
    output_dir: Path,
    selected: list[tuple[Path, str]],
    rollouts_per_task: int,
    shuffle_seed: int,
    gm_runtime: str,
    gm_model: str,
    gm_provider: str,
    gm_reasoning_effort: str,
) -> list[RolloutJob]:
    """Build a round-robin queue so early global slots span different tasks."""
    if rollouts_per_task < 1:
        raise RuntimeError("--rollouts-per-task must be at least one.")
    task_ids = [world_file.stem for world_file, _ in selected]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("Selected task environment IDs must be unique.")
    ordered = list(selected)
    if shuffle_seed:
        random.Random(shuffle_seed).shuffle(ordered)
    jobs: list[RolloutJob] = []
    for attempt in range(1, rollouts_per_task + 1):
        for world_file, flavor in ordered:
            world_id = world_file.stem
            jobs.append(
                RolloutJob(
                    world_file=world_file,
                    world_id=world_id,
                    flavor=flavor,
                    attempt=attempt,
                    episode_id=f"{campaign_id}-{world_id}-{attempt:02d}",
                    output_dir=output_dir.joinpath(
                        world_id, f"rollout_{attempt:02d}"
                    ),
                    gm_runtime=gm_runtime,
                    gm_model=gm_model,
                    gm_provider=gm_provider,
                    gm_reasoning_effort=gm_reasoning_effort,
                )
            )
    return jobs


def _smoke_args(args: argparse.Namespace, job: RolloutJob) -> argparse.Namespace:
    smoke_args = smoke_parser().parse_args([])
    smoke_args.world_file = job.world_file
    smoke_args.output_dir = job.output_dir
    smoke_args.episode_id = job.episode_id
    smoke_args.gm_runtime = job.gm_runtime
    smoke_args.gm_model = job.gm_model
    smoke_args.gm_provider = job.gm_provider
    smoke_args.gm_api_key_env = args.gm_api_key_env
    smoke_args.gm_reasoning_effort = job.gm_reasoning_effort
    smoke_args.max_world_steps = args.max_world_steps
    smoke_args.timeout_seconds = args.timeout_seconds
    smoke_args.gm_api_idle_timeout_seconds = getattr(args, "gm_api_idle_timeout_seconds", DEFAULT_GM_API_IDLE_TIMEOUT_SECONDS)
    smoke_args.player_prompt_timeout_seconds = getattr(
        args, "player_prompt_timeout_seconds", 43_200.0
    )
    # Optional Player routing — defaults remain smoke parser defaults
    # when the campaign CLI does not set these attributes.
    if getattr(args, "player_base_url", None) is not None:
        smoke_args.player_base_url = args.player_base_url
    if getattr(args, "player_provider", None) is not None:
        smoke_args.player_provider = args.player_provider
    if getattr(args, "player_model", None) is not None:
        smoke_args.player_model = args.player_model
    if getattr(args, "player_api_key_env", None) is not None:
        smoke_args.player_api_key_env = args.player_api_key_env
    if getattr(args, "player_thinking", None) is not None:
        smoke_args.player_thinking = args.player_thinking
    if getattr(args, "player_context_window", None) is not None:
        smoke_args.player_context_window = args.player_context_window
    if getattr(args, "player_max_tokens", None) is not None:
        smoke_args.player_max_tokens = args.player_max_tokens
    return smoke_args


def _annotate_outcome(job: RolloutJob, outcome: dict[str, Any]) -> dict[str, Any]:
    """Stamp world fields, acceptance, and hard-error classification.

    ``provider_infrastructure_failure`` is true only when the owning failure
    boundary explicitly attributes the error to GM. Local vat, audit, and
    orchestration failures must not poison the provider circuit.

    ``accepted_count`` semantics:
        - If the runner already set ``accepted_count`` (Tinker path: 1 only when
          a CaptureStore capture was built), keep that value. Never overwrite a
          capture failure with audit-only acceptance — that lies to GROUP_READY
          and crashes ``captures_for_outcomes``.
        - Otherwise derive 0/1 from ``accepted_for_training`` (simulation / live
          smoke jobs that have no capture store).
    """
    error = outcome.get("error")
    audit = outcome.get("audit")
    audit_accepted = (
        isinstance(audit, dict) and audit.get("accepted_for_training") is True
    )
    annotated = dict(outcome)
    provider_attributable = outcome.get("provider_attributable")
    if provider_attributable is None:
        # Compatibility for injected/custom runners that still return only
        # ``error``. First-party runners always emit explicit attribution.
        provider_attributable = bool(error)
    annotated["provider_attributable"] = bool(provider_attributable)
    annotated["provider_infrastructure_failure"] = (
        bool(error) and provider_attributable is True
    )
    if "accepted_count" in outcome:
        raw = outcome["accepted_count"]
        if raw not in (0, 1):
            raise RuntimeError(
                "Rollout outcome accepted_count must be 0 or 1; "
                f"got {raw!r} for episode {outcome.get('episode_id')!r}."
            )
        annotated["accepted_count"] = int(raw)
    else:
        annotated["accepted_count"] = 1 if audit_accepted else 0
    annotated["gm_runtime"] = job.gm_runtime
    annotated["gm_model"] = job.gm_model
    annotated["gm_provider"] = job.gm_provider
    annotated["gm_fallback_applied"] = job.gm_fallback_applied
    if job.original_gm_model is not None:
        annotated["original_gm_model"] = job.original_gm_model
    return annotated


async def run_live_job(job: RolloutJob, args: argparse.Namespace) -> dict[str, Any]:
    """Run and audit one production episode without aborting sibling jobs."""
    emit_simple(
        episode_id=job.episode_id,
        phase="campaign_job_start",
        message=(
            f"campaign job starting world={job.gm_model} "
            f"attempt={job.attempt}"
        ),
        world_id=job.world_id,
        gm_model=job.gm_model,
        attempt=job.attempt,
    )
    error: str | None = None
    failure_detail: dict[str, Any] | None = None
    try:
        await run_live_smoke(_smoke_args(args, job))
    except Exception as caught:
        failure_detail = serialize_failure(caught, stage="campaign_job")
        error = str(failure_detail["error"])

    persisted_detail = read_episode_failure_detail(job.output_dir.joinpath("episode.json"))
    if persisted_detail is not None:
        failure_detail = persisted_detail
        error = str(persisted_detail.get("error") or error)

    audit: dict[str, Any] | None = None
    if job.output_dir.joinpath("episode.json").is_file():
        try:
            audit = write_audit(job.output_dir)
        except Exception as caught:
            audit_detail = serialize_failure(
                caught,
                stage="acceptance_audit",
                component="acceptance_audit",
            )
            audit_error = str(audit_detail["error"])
            error = f"{error}; audit: {audit_error}" if error else f"audit: {audit_error}"
            if failure_detail is None:
                failure_detail = audit_detail
    raw_outcome: dict[str, Any] = {
        "episode_id": job.episode_id,
        "world_id": job.world_id,
        "flavor": job.flavor,
        "attempt": job.attempt,
        "rollout_dir": str(job.output_dir),
        "error": error,
        "audit": audit,
    }
    raw_outcome.update(outcome_failure_fields(failure_detail))
    outcome = _annotate_outcome(job, raw_outcome)
    emit_simple(
        episode_id=job.episode_id,
        phase="campaign_job_done",
        message=(
            f"campaign job finished accepted={outcome.get('accepted_count')} "
            f"error={outcome.get('error') is not None}"
        ),
        world_id=job.world_id,
        accepted_count=outcome.get("accepted_count"),
        error=outcome.get("error"),
    )
    return outcome


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


def summarize_campaign(
    *,
    campaign_id: str,
    outcomes: list[dict[str, Any]],
    parallelism: int,
    status: str = "running",
    provider_outage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate outcomes by task while preserving every rollout audit."""
    task_ids = sorted({str(outcome["world_id"]) for outcome in outcomes})
    tasks: list[dict[str, Any]] = []
    for task_id in task_ids:
        rows = [outcome for outcome in outcomes if outcome["world_id"] == task_id]
        accepted = [
            row
            for row in rows
            if isinstance(row.get("audit"), dict)
            and row["audit"].get("accepted_for_training") is True
        ]
        rewards = [
            float(row["audit"]["final_task"]["reward"])
            for row in accepted
            if isinstance(row["audit"].get("final_task", {}).get("reward"), (int, float))
        ]
        qualities = [
            float(row["audit"]["reasoning"]["mean_quality"])
            for row in accepted
            if isinstance(row["audit"].get("reasoning", {}).get("mean_quality"), (int, float))
        ]
        efficiencies = [
            float(row["audit"]["reasoning"]["mean_efficiency"])
            for row in accepted
            if isinstance(row["audit"].get("reasoning", {}).get("mean_efficiency"), (int, float))
        ]
        tasks.append(
            {
                "world_id": task_id,
                "flavor": rows[0]["flavor"],
                "attempt_count": len(rows),
                "accepted_count": len(accepted),
                "infrastructure_failure_count": sum(row.get("error") is not None for row in rows),
                "provider_infrastructure_failure_count": sum(
                    row.get("provider_infrastructure_failure") is True for row in rows
                ),
                "reward": {
                    "values": rewards,
                    "mean": _mean(rewards),
                    "population_stddev": (
                        round(statistics.pstdev(rewards), 6)
                        if len(rewards) > 1
                        else 0.0 if rewards else None
                    ),
                    "minimum": min(rewards) if rewards else None,
                    "maximum": max(rewards) if rewards else None,
                },
                "reasoning": {
                    "mean_quality": _mean(qualities),
                    "mean_efficiency": _mean(efficiencies),
                },
            }
        )

    report: dict[str, Any] = {
        "schema": "shellsmith.rollout_campaign.v1",
        "campaign_id": campaign_id,
        "status": status,
        "parallelism": parallelism,
        "task_count": len(tasks),
        "rollout_count": len(outcomes),
        "accepted_count": sum(task["accepted_count"] for task in tasks),
        "infrastructure_failure_count": sum(
            task["infrastructure_failure_count"] for task in tasks
        ),
        "tasks": tasks,
        "rollouts": outcomes,
    }
    if provider_outage is not None:
        report["provider_outage"] = provider_outage
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Shellsmith campaign: {report['campaign_id']}",
        "",
        f"- Status: {report.get('status', 'unknown')}",
        f"- Tasks: {report['task_count']}",
        f"- Rollouts: {report['rollout_count']}",
        f"- Accepted: {report['accepted_count']}",
        f"- Infrastructure failures: {report['infrastructure_failure_count']}",
        f"- Global parallelism: {report['parallelism']}",
        (
            f"- Per-provider parallelism: {report['parallelism_per_provider']}"
            if report.get("parallelism_per_provider") is not None
            else "- Per-provider parallelism: (unset)"
        ),
        "",
        "| Task | Flavor | Accepted | Reward mean | Reward SD | Quality | Efficiency |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in report["tasks"]:
        lines.append(
            f"| {task['world_id']} | {task['flavor']} | "
            f"{task['accepted_count']}/{task['attempt_count']} | "
            f"{task['reward']['mean']} | {task['reward']['population_stddev']} | "
            f"{task['reasoning']['mean_quality']} | "
            f"{task['reasoning']['mean_efficiency']} |"
        )
    return "\n".join(lines) + "\n"


def _write_text_atomic(path: Path, text: str) -> None:
    """Replace one campaign checkpoint without exposing a partial report.

    Main callers:
        ``write_report`` after each completed rollout.

    Why this exists:
        A long live campaign may be interrupted at any point. The most recent
        report must remain valid JSON/Markdown rather than a half-written file.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_report(output_dir: Path, report: dict[str, Any]) -> None:
    """Atomically checkpoint all completed outcomes seen by the scheduler."""
    campaign_report_json: Path = output_dir.joinpath("campaign_report.json")
    _write_text_atomic(
        campaign_report_json,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    campaign_report_md: Path = output_dir.joinpath("campaign_report.md")
    _write_text_atomic(campaign_report_md, _markdown(report))


async def run_campaign(
    args: argparse.Namespace,
    *,
    rollout_runner: RolloutRunner = run_live_job,
) -> tuple[Path, dict[str, Any]]:
    """Execute the global queue under dual caps (total + per gm_runtime).

    Whole inference campaign = one update: schedule every job greedily subject
    to ``--parallelism`` and ``--parallelism-per-provider``.

    On repeated GM hard failures for one runtime (consecutive-fail streak →
    cooldown → second trip), either halt remaining jobs or remap them to the
    bidirectional backup GM (see ``campaign.provider_outage``).
    """
    # Lazy import avoids a circular import with stream_campaign.
    from shellsmith.campaign.stream_campaign import (
        DualCapPool,
        validate_parallelism_caps,
    )

    install_stdout_progress()
    parallelism = int(args.parallelism)
    parallelism_per_provider = int(
        getattr(args, "parallelism_per_provider", min(4, parallelism))
    )
    validate_parallelism_caps(
        parallelism=parallelism,
        parallelism_per_provider=parallelism_per_provider,
    )
    if args.timeout_seconds <= 0:
        raise RuntimeError("--timeout-seconds must be positive.")
    player_prompt_timeout_seconds = float(
        getattr(args, "player_prompt_timeout_seconds", 43_200.0)
    )
    if player_prompt_timeout_seconds <= args.timeout_seconds:
        raise RuntimeError(
            "--player-prompt-timeout-seconds must be greater than "
            "--timeout-seconds so an outer vat deadline cannot mask the "
            "GM turn that actually timed out."
        )
    outage_mode = getattr(args, "provider_outage", "halt")
    if outage_mode not in PROVIDER_OUTAGE_MODES:
        raise RuntimeError(f"Unsupported --provider-outage mode: {outage_mode!r}.")
    campaign_id = args.campaign_id or datetime.now(UTC).strftime(
        "worlds-%Y%m%dT%H%M%SZ"
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_OUTPUT_PARENT.joinpath(campaign_id)
    ).expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError(f"Campaign output directory already exists: {output_dir}.")
    selected = select_worlds(args)
    jobs = build_jobs(
        campaign_id=campaign_id,
        output_dir=output_dir,
        selected=selected,
        rollouts_per_task=args.rollouts_per_task,
        shuffle_seed=args.shuffle_seed,
        gm_runtime=args.gm_runtime,
        gm_model=args.gm_model,
        gm_provider=args.gm_provider,
        gm_reasoning_effort=args.gm_reasoning_effort,
    )
    output_dir.mkdir(parents=True)
    manifest = {
        "schema": "shellsmith.rollout_campaign_manifest.v1",
        "campaign_id": campaign_id,
        "parallelism": parallelism,
        "parallelism_per_provider": parallelism_per_provider,
        "rollouts_per_task": args.rollouts_per_task,
        "timeout_seconds": args.timeout_seconds,
        "player_prompt_timeout_seconds": player_prompt_timeout_seconds,
        "provider_outage": outage_mode,
        "jobs": [
            {
                "episode_id": job.episode_id,
                "world_id": job.world_id,
                "flavor": job.flavor,
                "attempt": job.attempt,
                "world_file": str(job.world_file),
                "output_dir": str(job.output_dir),
                "gm_runtime": job.gm_runtime,
                "gm_model": job.gm_model,
                "gm_provider": job.gm_provider,
            }
            for job in jobs
        ],
    }
    manifest_path: Path = output_dir.joinpath("campaign_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pool = DualCapPool(total=parallelism, per_provider=parallelism_per_provider)
    health = ProviderHealthTracker()
    health_events: list[dict[str, Any]] = []

    async def bounded(job: RolloutJob) -> dict[str, Any]:
        if health.is_down(job.gm_runtime):
            annotated = not_started_outcome(job)
            result = await health.observe_outcome_async(annotated)
            annotated["provider_health_event"] = result.event.value
            annotated["provider_health_status"] = result.status.value
            return annotated
        try:
            await health.wait_until_available(job.gm_runtime)
        except RuntimeError:
            # A sibling marked this runtime DOWN while we waited out a cooldown.
            # Same "never started" case as above: no slot taken, no episode run.
            # Raising here would abort the queue before it can halt or remap.
            if not health.is_down(job.gm_runtime):
                raise
            annotated = not_started_outcome(job)
            result = await health.observe_outcome_async(annotated)
            annotated["provider_health_event"] = result.event.value
            annotated["provider_health_status"] = result.status.value
            return annotated
        async with pool.slot(job.gm_runtime):
            try:
                outcome = await rollout_runner(job, args)
            except Exception as caught:
                failure_detail = serialize_failure(
                    caught,
                    stage="rollout_runner",
                    component="campaign_runner",
                    # Custom legacy runners historically represented provider
                    # calls. Explicit boundary annotations override this default.
                    provider_attributable=True,
                )
                outcome = {
                    "episode_id": job.episode_id,
                    "world_id": job.world_id,
                    "flavor": job.flavor,
                    "attempt": job.attempt,
                    "rollout_dir": str(job.output_dir),
                    "error": str(failure_detail["error"]),
                    "audit": None,
                }
                outcome.update(outcome_failure_fields(failure_detail))
            annotated = _annotate_outcome(job, outcome)
            result = await health.observe_outcome_async(annotated)
            annotated["provider_health_event"] = result.event.value
            annotated["provider_health_status"] = result.status.value
            annotated["provider_consecutive_failures"] = result.consecutive_failures
            annotated["provider_trip_count"] = result.trip_count
            if result.event in {
                ProviderHealthEvent.ENTERED_COOLDOWN,
                ProviderHealthEvent.MARKED_DOWN,
            }:
                health_events.append(
                    {
                        "event": result.event.value,
                        "runtime": result.runtime,
                        "status": result.status.value,
                        "trip_count": result.trip_count,
                        "episode_id": job.episode_id,
                        "error": annotated.get("error"),
                        "cooldown_remaining_seconds": result.cooldown_remaining_seconds,
                    }
                )
            return annotated

    job_by_id: dict[str, RolloutJob] = {job.episode_id: job for job in jobs}
    job_order = {job.episode_id: index for index, job in enumerate(jobs)}
    pending: dict[str, asyncio.Task[dict[str, Any]]] = {
        job.episode_id: asyncio.create_task(bounded(job)) for job in jobs
    }
    outcomes: list[dict[str, Any]] = []
    provider_outage_info: dict[str, Any] | None = None
    status = "running"
    remapped_episode_ids: list[str] = []
    report: dict[str, Any] = summarize_campaign(
        campaign_id=campaign_id,
        outcomes=[],
        parallelism=parallelism,
        status=status,
    )
    report["parallelism_per_provider"] = parallelism_per_provider
    report["provider_health"] = health.snapshot()

    def _checkpoint() -> dict[str, Any]:
        ordered_outcomes = sorted(
            outcomes,
            key=lambda outcome: job_order.get(str(outcome["episode_id"]), 10**9),
        )
        built = summarize_campaign(
            campaign_id=campaign_id,
            outcomes=ordered_outcomes,
            parallelism=parallelism,
            status=status,
            provider_outage=provider_outage_info,
        )
        built["parallelism_per_provider"] = parallelism_per_provider
        built["provider_health"] = health.snapshot()
        write_report(output_dir, built)
        return built

    def _down_runtimes() -> list[str]:
        return sorted(
            key
            for key, row in health.snapshot().items()
            if row.get("status") == "down"
        )

    try:
        while pending:
            done, _ = await asyncio.wait(
                set(pending.values()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                episode_id = next(
                    key for key, value in pending.items() if value is task
                )
                del pending[episode_id]
                outcomes.append(await task)
            report = _checkpoint()
            down = _down_runtimes()
            if not down:
                continue

            # Provider marked DOWN after cooldown + second failure streak.
            remaining_ids = list(pending.keys())
            failing_models = sorted(
                {
                    str(item.get("gm_model"))
                    for item in outcomes
                    if str(item.get("gm_runtime") or "").lower() in down
                    and item.get("gm_model")
                }
            )
            infra_count = sum(
                item.get("provider_infrastructure_failure") is True
                for item in outcomes
            )
            if outage_mode == "halt":
                for leftover in pending.values():
                    leftover.cancel()
                if pending:
                    await asyncio.gather(
                        *pending.values(), return_exceptions=True
                    )
                pending.clear()
                status = "halted_provider_unavailable"
                provider_outage_info = {
                    "mode": outage_mode,
                    "action": "halt",
                    "down_runtimes": down,
                    "infrastructure_failure_count": infra_count,
                    "failing_gm_models": failing_models,
                    "cancelled_episode_ids": remaining_ids,
                    "remapped_episode_ids": remapped_episode_ids,
                    "health_events": list(health_events),
                    "provider_health": health.snapshot(),
                }
                report = _checkpoint()
                break

            # Fallback mode with nothing left to remap: keep running/finish.
            # Do not overwrite a successful remap with halted_provider_unavailable.
            if not remaining_ids:
                continue

            # fallback mode: cancel unfinished on down runtimes, remap once, relaunch.
            remap_ids = [
                eid
                for eid in remaining_ids
                if job_by_id[eid].gm_runtime.strip().lower() in down
            ]
            other_ids = [eid for eid in remaining_ids if eid not in remap_ids]
            for leftover_id in remap_ids:
                pending[leftover_id].cancel()
            if remap_ids:
                await asyncio.gather(
                    *(pending[eid] for eid in remap_ids),
                    return_exceptions=True,
                )
            for leftover_id in remap_ids:
                del pending[leftover_id]

            remapped_jobs: list[RolloutJob] = []
            remap_pairs: list[dict[str, str]] = []
            try:
                for leftover_id in remap_ids:
                    job = job_by_id[leftover_id]
                    backup, original_model = apply_backup_to_job_fields(
                        gm_runtime=job.gm_runtime,
                        gm_model=job.gm_model,
                        gm_provider=job.gm_provider,
                        gm_fallback_applied=job.gm_fallback_applied,
                    )
                    remapped = replace(
                        job,
                        output_dir=fallback_output_dir(
                            job, backup_runtime=backup.runtime
                        ),
                        gm_runtime=backup.runtime,
                        gm_model=backup.model,
                        gm_provider=backup.provider,
                        gm_fallback_applied=True,
                        original_gm_model=original_model,
                    )
                    job_by_id[leftover_id] = remapped
                    remapped_jobs.append(remapped)
                    remapped_episode_ids.append(leftover_id)
                    remap_pairs.append(
                        {
                            "episode_id": leftover_id,
                            "from_model": original_model,
                            "to_model": backup.model,
                            "to_runtime": backup.runtime,
                            "original_output_dir": str(job.output_dir),
                            "fallback_output_dir": str(remapped.output_dir),
                        }
                    )
            except RuntimeError as remap_error:
                for leftover in list(pending.values()):
                    leftover.cancel()
                if pending:
                    await asyncio.gather(*pending.values(), return_exceptions=True)
                pending.clear()
                status = "halted_provider_unavailable"
                provider_outage_info = {
                    "mode": outage_mode,
                    "action": "halt",
                    "reason": str(remap_error),
                    "down_runtimes": down,
                    "infrastructure_failure_count": infra_count,
                    "failing_gm_models": failing_models,
                    "cancelled_episode_ids": remaining_ids,
                    "remapped_episode_ids": remapped_episode_ids,
                    "health_events": list(health_events),
                    "provider_health": health.snapshot(),
                }
                report = _checkpoint()
                break

            provider_outage_info = {
                "mode": "fallback",
                "action": "remapped_remaining",
                "down_runtimes": down,
                "infrastructure_failure_count": infra_count,
                "failing_gm_models": failing_models,
                "remaps": remap_pairs,
                "remapped_episode_ids": list(remapped_episode_ids),
                "continued_episode_ids": other_ids,
                "health_events": list(health_events),
                "provider_health": health.snapshot(),
            }
            for remapped in remapped_jobs:
                pending[remapped.episode_id] = asyncio.create_task(
                    bounded(remapped)
                )
            report = _checkpoint()

        if status == "running":
            status = (
                "completed_with_provider_fallback"
                if remapped_episode_ids
                else "completed"
            )
        report = _checkpoint()
        return output_dir, report
    except BaseException:
        for task in pending.values():
            task.cancel()
        if pending:
            await asyncio.gather(*pending.values(), return_exceptions=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir, report = asyncio.run(run_campaign(args))
    print(
        json.dumps(
            {"output_dir": str(output_dir), **report},
            ensure_ascii=False,
            indent=2,
        )
    )
    halted = report.get("status") == "halted_provider_unavailable"
    return 1 if halted or report["infrastructure_failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
