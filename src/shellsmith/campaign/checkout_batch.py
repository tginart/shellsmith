"""Run and audit a parallel checkout-idempotency evaluation batch.

Architecture map:
    one batch CLI
      -> N independent `run_live_smoke` coroutines
      -> semaphore enforces requested parallelism
      -> each durable rollout receives `acceptance_audit.json`
      -> aggregate reward/variance/integrity/atomicity report in JSON + Markdown

Each episode remains internally serialized. Parallelism exists only across
independent Codex/Pi pairs, preserving a single state order within every world.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shellsmith.cli_args import GMArgumentParser
from shellsmith.config import (
    DEFAULT_GM_API_IDLE_TIMEOUT_SECONDS,
    DEFAULT_MAX_WORLD_STEPS,
    DEFAULT_GM_MODEL,
    DEFAULT_GM_REASONING_EFFORT,
    DEFAULT_GM_RUNTIME,
    GM_PROVIDER_BY_RUNTIME,
)
from shellsmith.paths import DEFAULT_LOG_ROOT
from shellsmith.episode.acceptance_audit import (
    write_audit,
)
from shellsmith.campaign.provider_outage import (
    PROVIDER_OUTAGE_MODES,
    ProviderHealthEvent,
    ProviderHealthTracker,
    apply_backup_to_job_fields,
)
from shellsmith.episode.failure import (
    outcome_failure_fields,
    read_episode_failure_detail,
    serialize_failure,
)
from shellsmith.episode.smoke import (
    DEFAULT_WORLD,
    _parser as smoke_parser,
    run_live_smoke,
)
from shellsmith.episode.progress import (
    install_stdout_progress,
)


DEFAULT_BATCH_PARENT = DEFAULT_LOG_ROOT / "batches"


def _parser() -> argparse.ArgumentParser:
    parser = GMArgumentParser(
        description="Run checkout shellsmith rollouts concurrently."
    )
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-id")
    parser.add_argument("--world-file", type=Path, default=DEFAULT_WORLD)
    parser.add_argument(
        "--gm-runtime", choices=tuple(GM_PROVIDER_BY_RUNTIME), default=DEFAULT_GM_RUNTIME
    )
    parser.add_argument("--gm-model", help="GM model (Codex defaults to Sol; required for other runtimes).")
    parser.add_argument("--gm-provider", help="Provider override (default follows --gm-runtime).")
    parser.add_argument("--gm-reasoning-effort", default=DEFAULT_GM_REASONING_EFFORT)
    parser.add_argument("--gm-api-idle-timeout-seconds", type=float, default=DEFAULT_GM_API_IDLE_TIMEOUT_SECONDS)
    parser.add_argument("--max-world-steps", type=int, default=DEFAULT_MAX_WORLD_STEPS)
    parser.add_argument(
        "--provider-outage",
        choices=PROVIDER_OUTAGE_MODES,
        default="halt",
        help=(
            "When a GM runtime is marked down (consecutive fails → "
            "cooldown → second trip): halt remaining rollouts, or remap "
            "remaining ones to the bidirectional backup."
        ),
    )
    parser.add_argument(
        "--audit-existing",
        action="store_true",
        help="Recompute audits and aggregate reports for an existing batch directory.",
    )
    return parser


def _smoke_args(
    batch_args: argparse.Namespace,
    *,
    episode_id: str,
    output_dir: Path,
    gm_runtime: str,
    gm_model: str,
    gm_provider: str,
    gm_reasoning_effort: str,
) -> argparse.Namespace:
    args = smoke_parser().parse_args([])
    args.episode_id = episode_id
    args.output_dir = output_dir
    args.world_file = batch_args.world_file
    args.gm_runtime = gm_runtime
    args.gm_model = gm_model
    args.gm_provider = gm_provider
    args.gm_reasoning_effort = gm_reasoning_effort
    args.max_world_steps = batch_args.max_world_steps
    args.gm_api_idle_timeout_seconds = getattr(batch_args, "gm_api_idle_timeout_seconds", DEFAULT_GM_API_IDLE_TIMEOUT_SECONDS)
    return args


def _annotate_batch_outcome(
    *,
    episode_id: str,
    rollout_dir: Path,
    error: str | None,
    audit: dict[str, Any] | None,
    gm_runtime: str,
    gm_model: str,
    gm_provider: str,
    gm_fallback_applied: bool,
    original_gm_model: str | None,
    failure_detail: dict[str, Any] | None,
) -> dict[str, Any]:
    accepted = (
        isinstance(audit, dict) and audit.get("accepted_for_training") is True
    )
    outcome: dict[str, Any] = {
        "episode_id": episode_id,
        "rollout_dir": str(rollout_dir),
        "error": error,
        "audit": audit,
        "provider_attributable": (
            failure_detail is not None
            and failure_detail.get("provider_attributable") is True
        ),
        "provider_infrastructure_failure": (
            bool(error)
            and failure_detail is not None
            and failure_detail.get("provider_attributable") is True
        ),
        "accepted_count": 1 if accepted else 0,
        "gm_runtime": gm_runtime,
        "gm_model": gm_model,
        "gm_provider": gm_provider,
        "gm_fallback_applied": gm_fallback_applied,
    }
    if original_gm_model is not None:
        outcome["original_gm_model"] = original_gm_model
    outcome.update(outcome_failure_fields(failure_detail))
    return outcome


def _aggregate(
    batch_id: str,
    outcomes: list[dict[str, Any]],
    *,
    status: str = "completed",
    provider_outage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    audits = [item["audit"] for item in outcomes if isinstance(item.get("audit"), dict)]
    rewards = [
        float(audit["final_task"]["reward"])
        for audit in audits
        if isinstance(audit.get("final_task", {}).get("reward"), (int, float))
    ]
    actions = [int(audit["action_count"]) for audit in audits]
    qualities = [
        float(audit["reasoning"]["mean_quality"])
        for audit in audits
        if isinstance(audit.get("reasoning", {}).get("mean_quality"), (int, float))
    ]
    efficiencies = [
        float(audit["reasoning"]["mean_efficiency"])
        for audit in audits
        if isinstance(audit.get("reasoning", {}).get("mean_efficiency"), (int, float))
    ]
    atomic = [
        audit.get("final_task", {}).get("atomic_fix")
        for audit in audits
        if isinstance(audit.get("final_task", {}).get("atomic_fix"), bool)
    ]
    contradiction_count = sum(
        int(audit.get("factual_consistency", {}).get("contradiction_count") or 0)
        for audit in audits
    )

    def mean(values: list[float | int]) -> float | None:
        return round(statistics.fmean(values), 6) if values else None

    report: dict[str, Any] = {
        "schema": "shellsmith.checkout_batch.v1",
        "batch_id": batch_id,
        "status": status,
        "requested_rollouts": len(outcomes),
        "completed_audits": len(audits),
        "infrastructure_failure_count": sum(item.get("error") is not None for item in outcomes),
        "accepted_for_training_count": sum(
            audit.get("accepted_for_training") is True for audit in audits
        ),
        "simulator_integrity_failure_count": sum(
            audit.get("simulator_integrity", {}).get("passed") is not True
            for audit in audits
        ),
        "factual_contradiction_rollout_count": sum(
            audit.get("factual_consistency", {}).get("passed") is not True
            for audit in audits
        ),
        "factual_contradiction_count": contradiction_count,
        "reward": {
            "values": rewards,
            "mean": mean(rewards),
            "population_stddev": (
                round(statistics.pstdev(rewards), 6)
                if len(rewards) > 1
                else 0.0 if rewards else None
            ),
            "minimum": min(rewards) if rewards else None,
            "maximum": max(rewards) if rewards else None,
        },
        "action_count": {
            "values": actions,
            "mean": mean(actions),
            "minimum": min(actions) if actions else None,
            "maximum": max(actions) if actions else None,
        },
        "reasoning": {
            "mean_rollout_quality": mean(qualities),
            "mean_rollout_efficiency": mean(efficiencies),
        },
        "executed_verification_rollout_count": sum(
            audit.get("executed_verification", {}).get("executed") is True
            for audit in audits
        ),
        "atomic_fix": {
            "evaluated_count": len(atomic),
            "success_count": sum(value is True for value in atomic),
            "success_rate": (
                round(sum(value is True for value in atomic) / len(atomic), 6)
                if atomic
                else None
            ),
        },
        "rollouts": outcomes,
    }
    if provider_outage is not None:
        report["provider_outage"] = provider_outage
    return report


def _markdown(report: dict[str, Any]) -> str:
    reward = report["reward"]
    atomic = report["atomic_fix"]
    lines = [
        f"# Checkout batch: {report['batch_id']}",
        "",
        f"- Rollouts: {report['completed_audits']}/{report['requested_rollouts']} audited",
        f"- Accepted for training: {report['accepted_for_training_count']}",
        f"- Reward mean ± population SD: {reward['mean']} ± {reward['population_stddev']}",
        f"- Simulator integrity failures: {report['simulator_integrity_failure_count']}",
        f"- Factual contradiction rollouts: {report['factual_contradiction_rollout_count']}",
        f"- Executed-verification rollouts: {report['executed_verification_rollout_count']}",
        f"- Genuinely atomic fixes: {atomic['success_count']}/{atomic['evaluated_count']}",
        "",
        "| Episode | Reward | Actions | Integrity | Facts | Verification | Atomic fix |",
        "| --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for item in report["rollouts"]:
        audit = item.get("audit") or {}
        lines.append(
            "| {episode} | {reward} | {actions} | {integrity} | {facts} | "
            "{verification} | {atomic} |".format(
                episode=item["episode_id"],
                reward=audit.get("final_task", {}).get("reward", "—"),
                actions=audit.get("action_count", "—"),
                integrity="pass" if audit.get("simulator_integrity", {}).get("passed") else "FAIL",
                facts="pass" if audit.get("factual_consistency", {}).get("passed") else "FAIL",
                verification=(
                    "yes" if audit.get("executed_verification", {}).get("executed") else "no"
                ),
                atomic=(
                    "yes"
                    if audit.get("final_task", {}).get("atomic_fix") is True
                    else "no"
                    if audit.get("final_task", {}).get("atomic_fix") is False
                    else "unknown"
                ),
            )
        )
    return "\n".join(lines) + "\n"


def _write_batch_report(output_dir: Path, report: dict[str, Any]) -> None:
    batch_report_json: Path = output_dir.joinpath("batch_report.json")
    batch_report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    batch_report_md: Path = output_dir.joinpath("batch_report.md")
    batch_report_md.write_text(_markdown(report), encoding="utf-8")


def audit_existing_batch(output_dir: Path, *, batch_id: str) -> dict[str, Any]:
    """Rebuild deterministic audits after rules change or an interrupted report write."""
    output_dir = output_dir.expanduser().resolve()
    outcomes: list[dict[str, Any]] = []
    manifest_path: Path = output_dir.joinpath("batch_manifest.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("rollouts") if isinstance(manifest, dict) else None
        if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
            raise RuntimeError(f"Malformed batch manifest: {manifest_path}.")
        expected = []
        for item in entries:
            episode_id = item.get("episode_id")
            directory = item.get("directory")
            if not isinstance(episode_id, str) or not isinstance(directory, str):
                raise RuntimeError(f"Malformed batch manifest rollout entry: {item!r}.")
            rollout_dir: Path = output_dir.joinpath(directory)
            if rollout_dir.parent != output_dir:
                raise RuntimeError(
                    f"Batch manifest rollout directory escapes its root: {directory!r}."
                )
            expected.append((episode_id, rollout_dir))
    else:
        expected = [(path.name, path) for path in sorted(output_dir.glob("rollout_*"))]

    for episode_id, rollout_dir in expected:
        if not rollout_dir.joinpath("episode.json").is_file():
            outcomes.append(
                {
                    "episode_id": episode_id,
                    "rollout_dir": str(rollout_dir),
                    "error": "Missing durable episode.json; rollout did not initialize or archive.",
                    "audit": None,
                }
            )
            continue
        try:
            audit = write_audit(rollout_dir)
            outcomes.append(
                {
                    "episode_id": audit.get("episode_id") or episode_id,
                    "rollout_dir": str(rollout_dir),
                    "error": None,
                    "audit": audit,
                }
            )
        except Exception as error:
            outcomes.append(
                {
                    "episode_id": episode_id,
                    "rollout_dir": str(rollout_dir),
                    "error": f"{type(error).__name__}: {error}",
                    "audit": None,
                }
            )
    if not outcomes:
        raise RuntimeError(f"No completed rollout directories found under {output_dir}.")
    report = _aggregate(batch_id, outcomes)
    _write_batch_report(output_dir, report)
    return report


async def run_batch(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Run independent live episodes with bounded cross-episode concurrency."""
    install_stdout_progress()
    if args.count < 1:
        raise RuntimeError("--count must be at least one.")
    if args.parallelism < 1:
        raise RuntimeError("--parallelism must be at least one.")
    outage_mode = getattr(args, "provider_outage", "halt")
    if outage_mode not in PROVIDER_OUTAGE_MODES:
        raise RuntimeError(f"Unsupported --provider-outage mode: {outage_mode!r}.")
    batch_id = args.batch_id or datetime.now(UTC).strftime("checkout-%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir if args.output_dir is not None else DEFAULT_BATCH_PARENT.joinpath(batch_id)
    ).expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError(f"Batch output directory already exists: {output_dir}.")
    output_dir.mkdir(parents=True)
    manifest = {
        "schema": "shellsmith.checkout_batch_manifest.v1",
        "batch_id": batch_id,
        "requested_rollout_count": args.count,
        "parallelism": args.parallelism,
        "provider_outage": outage_mode,
        "rollouts": [
            {
                "episode_id": f"{batch_id}-{index:02d}",
                "directory": f"rollout_{index:02d}",
            }
            for index in range(1, args.count + 1)
        ],
    }
    manifest_path: Path = output_dir.joinpath("batch_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    semaphore = asyncio.Semaphore(args.parallelism)

    @dataclass
    class _Job:
        index: int
        episode_id: str
        rollout_dir: Path
        gm_runtime: str
        gm_model: str
        gm_provider: str
        gm_reasoning_effort: str
        gm_fallback_applied: bool = False
        original_gm_model: str | None = None

    jobs = [
        _Job(
            index=index,
            episode_id=f"{batch_id}-{index:02d}",
            rollout_dir=output_dir.joinpath(f"rollout_{index:02d}"),
            gm_runtime=args.gm_runtime,
            gm_model=args.gm_model,
            gm_provider=args.gm_provider,
            gm_reasoning_effort=args.gm_reasoning_effort,
        )
        for index in range(1, args.count + 1)
    ]
    job_by_id = {job.episode_id: job for job in jobs}
    health = ProviderHealthTracker()
    health_events: list[dict[str, Any]] = []

    async def _finish_outcome(
        job: _Job,
        *,
        error: str | None,
        audit: dict[str, Any] | None,
        failure_detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        outcome = _annotate_batch_outcome(
            episode_id=job.episode_id,
            rollout_dir=job.rollout_dir,
            error=error,
            audit=audit,
            gm_runtime=job.gm_runtime,
            gm_model=job.gm_model,
            gm_provider=job.gm_provider,
            gm_fallback_applied=job.gm_fallback_applied,
            original_gm_model=job.original_gm_model,
            failure_detail=failure_detail,
        )
        result = await health.observe_outcome_async(outcome)
        outcome["provider_health_event"] = result.event.value
        outcome["provider_health_status"] = result.status.value
        outcome["provider_consecutive_failures"] = result.consecutive_failures
        outcome["provider_trip_count"] = result.trip_count
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
                    "error": error,
                    "cooldown_remaining_seconds": result.cooldown_remaining_seconds,
                }
            )
        return outcome

    async def worker(job: _Job) -> dict[str, Any]:
        if health.is_down(job.gm_runtime):
            return await _finish_outcome(
                job,
                error=(
                    f"RuntimeError: GM provider {job.gm_runtime!r} "
                    "is marked down; episode not started."
                ),
                audit=None,
            )
        await health.wait_until_available(job.gm_runtime)
        smoke_args = _smoke_args(
            args,
            episode_id=job.episode_id,
            output_dir=job.rollout_dir,
            gm_runtime=job.gm_runtime,
            gm_model=job.gm_model,
            gm_provider=job.gm_provider,
            gm_reasoning_effort=job.gm_reasoning_effort,
        )
        error: str | None = None
        failure_detail: dict[str, Any] | None = None
        audit: dict[str, Any] | None = None
        try:
            async with semaphore:
                await run_live_smoke(smoke_args)
        except Exception as caught:
            failure_detail = serialize_failure(
                caught,
                stage="checkout_batch_job",
                provider_attributable=True,
            )
            error = str(failure_detail["error"])
            persisted_detail = read_episode_failure_detail(
                job.rollout_dir.joinpath("episode.json")
            )
            if persisted_detail is not None:
                failure_detail = persisted_detail
                error = str(persisted_detail.get("error") or error)
        if job.rollout_dir.joinpath("episode.json").is_file():
            try:
                audit = write_audit(job.rollout_dir)
            except Exception as caught:
                if failure_detail is None:
                    failure_detail = serialize_failure(
                        caught,
                        stage="acceptance_audit",
                        component="acceptance_audit",
                    )
                    error = str(failure_detail["error"])
        return await _finish_outcome(
            job,
            error=error,
            audit=audit,
            failure_detail=failure_detail,
        )

    pending: dict[str, asyncio.Task[dict[str, Any]]] = {
        job.episode_id: asyncio.create_task(worker(job)) for job in jobs
    }
    outcomes: list[dict[str, Any]] = []
    status = "running"
    provider_outage_info: dict[str, Any] | None = None
    remapped_episode_ids: list[str] = []

    while pending:
        done, _ = await asyncio.wait(
            set(pending.values()), return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            episode_id = next(key for key, value in pending.items() if value is task)
            del pending[episode_id]
            outcomes.append(await task)
        down = sorted(
            key
            for key, row in health.snapshot().items()
            if row.get("status") == "down"
        )
        if not down:
            continue

        remaining_ids = list(pending.keys())
        infra_count = sum(
            item.get("provider_infrastructure_failure") is True
            for item in outcomes
        )
        failing_models = sorted(
            {
                str(item.get("gm_model"))
                for item in outcomes
                if str(item.get("gm_runtime") or "").lower() in down
                and item.get("gm_model")
            }
        )
        if outage_mode == "halt" or not remaining_ids:
            for leftover in pending.values():
                leftover.cancel()
            if pending:
                await asyncio.gather(*pending.values(), return_exceptions=True)
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
            break

        for leftover in pending.values():
            leftover.cancel()
        if pending:
            await asyncio.gather(*pending.values(), return_exceptions=True)
        pending.clear()
        remapped_jobs: list[_Job] = []
        remap_pairs: list[dict[str, str]] = []
        try:
            for leftover_id in remaining_ids:
                job = job_by_id[leftover_id]
                backup, original_model = apply_backup_to_job_fields(
                    gm_runtime=job.gm_runtime,
                    gm_model=job.gm_model,
                    gm_provider=job.gm_provider,
                    gm_fallback_applied=job.gm_fallback_applied,
                )
                job.gm_runtime = backup.runtime
                job.gm_model = backup.model
                job.gm_provider = backup.provider
                job.gm_fallback_applied = True
                job.original_gm_model = original_model
                remapped_jobs.append(job)
                remapped_episode_ids.append(leftover_id)
                remap_pairs.append(
                    {
                        "episode_id": leftover_id,
                        "from_model": original_model,
                        "to_model": backup.model,
                        "to_runtime": backup.runtime,
                    }
                )
        except RuntimeError as remap_error:
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
            break
        provider_outage_info = {
            "mode": "fallback",
            "action": "remapped_remaining",
            "down_runtimes": down,
            "infrastructure_failure_count": infra_count,
            "failing_gm_models": failing_models,
            "remaps": remap_pairs,
            "remapped_episode_ids": list(remapped_episode_ids),
            "health_events": list(health_events),
            "provider_health": health.snapshot(),
        }
        for remapped in remapped_jobs:
            pending[remapped.episode_id] = asyncio.create_task(worker(remapped))

    if status == "running":
        status = (
            "completed_with_provider_fallback"
            if remapped_episode_ids
            else "completed"
        )
    report = _aggregate(
        batch_id,
        outcomes,
        status=status,
        provider_outage=provider_outage_info,
    )
    _write_batch_report(output_dir, report)
    return output_dir, report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.audit_existing:
        if args.output_dir is None:
            raise RuntimeError("--audit-existing requires --output-dir.")
        output_dir = args.output_dir.expanduser().resolve()
        batch_id = args.batch_id or output_dir.name
        report = audit_existing_batch(output_dir, batch_id=batch_id)
        print(json.dumps({"output_dir": str(output_dir), **report}, ensure_ascii=False, indent=2))
        return 0
    output_dir, report = asyncio.run(run_batch(args))
    print(json.dumps({"output_dir": str(output_dir), **report}, ensure_ascii=False, indent=2))
    halted = report.get("status") == "halted_provider_unavailable"
    return 1 if halted or report["infrastructure_failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
