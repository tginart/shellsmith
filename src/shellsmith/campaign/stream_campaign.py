"""Open / wave / finalize API for one durable campaign enclosure.

Architecture map
----------------
    open_campaign(root, spec)
        -> create campaign_manifest.json once
        -> DualCapPool(total, per_provider) on the handle
    run_campaign_wave(handle, jobs, runner, ...)
        -> execute a batch of RolloutJobs into the same root
        -> skip episode_ids already on the scoreboard or with durable_outcome.json
        -> each new job acquires DualCapPool slot keyed by gm_runtime
        -> append outcomes; refresh campaign_report.json
        -> returns this wave's outcome rows
    finalize_campaign(handle, status, ...)
        -> set terminal status on the scoreboard

Why this exists
---------------
Tryouts can keep using one-shot ``run_campaign`` (create dir, fire all jobs).
Trainer hero streams need multiple waves (initial collect, adaptive top-ups,
optim pauses) without minting a new campaign id per GRPO group.

Concurrency
-----------
One shared DualCapPool per campaign:
    total in-flight episodes ≤ parallelism
    per gm_runtime in-flight ≤ parallelism_per_provider
Acquire waits until both caps have free capacity, then takes both together.

Main callers:
    Trainer stream engine; tests; future one-shot wrapper around waves.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncIterator

from shellsmith.campaign.provider_outage import (
    PROVIDER_OUTAGE_MODES,
    ProviderHealthEvent,
    ProviderHealthTracker,
)
from shellsmith.episode.failure import (
    outcome_failure_fields,
    serialize_failure,
)
from shellsmith.campaign.rollout_campaign import (
    MAX_PARALLELISM,
    RolloutJob,
    RolloutRunner,
    _annotate_outcome,
    not_started_outcome,
    outcome_is_settled,
    run_live_job,
    summarize_campaign,
    write_report,
)


CAMPAIGN_STREAM_SCHEMA = "shellsmith.campaign_stream.v1"
CAMPAIGN_MODES = ("inference", "tinker_train", "mlx_train", "studio_train")


def validate_parallelism_caps(*, parallelism: int, parallelism_per_provider: int) -> None:
    """Fail loud when total / per-provider caps are inconsistent."""
    if not 1 <= parallelism <= MAX_PARALLELISM:
        raise RuntimeError(
            f"parallelism must be between 1 and {MAX_PARALLELISM}."
        )
    if not 1 <= parallelism_per_provider <= parallelism:
        raise RuntimeError(
            "parallelism_per_provider must be between 1 and parallelism "
            f"(got per_provider={parallelism_per_provider}, total={parallelism})."
        )


@dataclass
class DualCapPool:
    """Async slot pool: global total + per gm_runtime provider caps.

    Main callers:
        ``run_campaign_wave`` and inference ``run_campaign`` job wrappers.

    Why this exists:
        A single semaphore cannot protect Claude and Codex rate limits separately
        while still bounding overall host load. Dual caps do both.

    Acquire waits until *both* total and that runtime have free capacity before
    taking either slot (no silent steal; no holding total while blocked on
    provider).
    """

    total: int
    per_provider: int
    _in_flight_total: int = field(default=0, init=False, repr=False)
    _in_flight_provider: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )
    _condition: asyncio.Condition = field(
        default_factory=asyncio.Condition, init=False, repr=False
    )

    def __post_init__(self) -> None:
        validate_parallelism_caps(
            parallelism=self.total, parallelism_per_provider=self.per_provider
        )

    @staticmethod
    def _runtime_key(gm_runtime: str) -> str:
        key = gm_runtime.strip().lower()
        if not key:
            raise RuntimeError("gm_runtime is required for DualCapPool slots.")
        return key

    def has_capacity(self, gm_runtime: str) -> bool:
        """Non-blocking peek: True when both caps have room for one more episode."""
        key = self._runtime_key(gm_runtime)
        return (
            self._in_flight_total < self.total
            and self._in_flight_provider.get(key, 0) < self.per_provider
        )

    @asynccontextmanager
    async def slot(self, gm_runtime: str) -> AsyncIterator[None]:
        """Hold one episode slot under both caps until the block exits."""
        key = self._runtime_key(gm_runtime)
        async with self._condition:
            while True:
                in_provider = self._in_flight_provider.get(key, 0)
                if (
                    self._in_flight_total < self.total
                    and in_provider < self.per_provider
                ):
                    self._in_flight_total += 1
                    self._in_flight_provider[key] = in_provider + 1
                    break
                await self._condition.wait()
        try:
            yield
        finally:
            async with self._condition:
                self._in_flight_total -= 1
                self._in_flight_provider[key] = (
                    self._in_flight_provider.get(key, 0) - 1
                )
                if self._in_flight_provider[key] <= 0:
                    del self._in_flight_provider[key]
                self._condition.notify_all()


@dataclass
class OpenCampaignSpec:
    """Identity for one campaign enclosure (tryout or hero stream)."""

    campaign_id: str
    mode: str = "inference"
    parallelism: int = 8
    parallelism_per_provider: int = 4
    provider_outage: str = "halt"
    timeout_seconds: float = 7_200.0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in CAMPAIGN_MODES:
            raise RuntimeError(
                f"Unsupported campaign mode {self.mode!r}; "
                f"expected one of {CAMPAIGN_MODES}."
            )
        if self.provider_outage not in PROVIDER_OUTAGE_MODES:
            raise RuntimeError(
                f"Unsupported provider_outage {self.provider_outage!r}."
            )
        validate_parallelism_caps(
            parallelism=self.parallelism,
            parallelism_per_provider=self.parallelism_per_provider,
        )


@dataclass
class CampaignHandle:
    """Live handle for an opened campaign root."""

    root: Path
    campaign_id: str
    mode: str
    parallelism: int
    parallelism_per_provider: int
    provider_outage: str
    pool: DualCapPool
    health: ProviderHealthTracker = field(default_factory=ProviderHealthTracker)
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    status: str = "running"
    wave_count: int = 0
    provider_health_events: list[dict[str, Any]] = field(default_factory=list)


def open_campaign(root: Path, spec: OpenCampaignSpec) -> CampaignHandle:
    """Create a new campaign directory and write the stream manifest.

    An empty existing directory is allowed (caller may have created ``--run-dir``).
    A non-empty directory raises — no silent reuse of a prior campaign.
    """
    root = root.expanduser().resolve()
    if root.exists():
        if any(root.iterdir()):
            raise RuntimeError(f"Campaign output directory already exists: {root}.")
    else:
        root.mkdir(parents=True)
    manifest = {
        "schema": CAMPAIGN_STREAM_SCHEMA,
        "campaign_id": spec.campaign_id,
        "mode": spec.mode,
        "parallelism": spec.parallelism,
        "parallelism_per_provider": spec.parallelism_per_provider,
        "provider_outage": spec.provider_outage,
        "timeout_seconds": spec.timeout_seconds,
        "opened_at": datetime.now(UTC).isoformat(),
        "extra": spec.extra,
    }
    root.joinpath("campaign_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    handle = CampaignHandle(
        root=root,
        campaign_id=spec.campaign_id,
        mode=spec.mode,
        parallelism=spec.parallelism,
        parallelism_per_provider=spec.parallelism_per_provider,
        provider_outage=spec.provider_outage,
        pool=DualCapPool(
            total=spec.parallelism,
            per_provider=spec.parallelism_per_provider,
        ),
    )
    _checkpoint(handle)
    return handle


def load_campaign(root: Path) -> CampaignHandle:
    """Resume bookkeeping from an existing campaign root (manifest required)."""
    root = root.expanduser().resolve()
    manifest_path = root.joinpath("campaign_manifest.json")
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing campaign_manifest.json under {root}.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_path = root.joinpath("campaign_report.json")
    outcomes: list[dict[str, Any]] = []
    status = "running"
    wave_count = 0
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rollouts = report.get("rollouts")
        if isinstance(rollouts, list):
            outcomes = [item for item in rollouts if isinstance(item, dict)]
        status = str(report.get("status") or status)
        wave_count = int(report.get("wave_count") or 0)
    parallelism = int(manifest.get("parallelism") or 8)
    parallelism_per_provider = int(
        manifest.get("parallelism_per_provider")
        or min(4, parallelism)
    )
    validate_parallelism_caps(
        parallelism=parallelism,
        parallelism_per_provider=parallelism_per_provider,
    )
    return CampaignHandle(
        root=root,
        campaign_id=str(manifest["campaign_id"]),
        mode=str(manifest.get("mode") or "inference"),
        parallelism=parallelism,
        parallelism_per_provider=parallelism_per_provider,
        provider_outage=str(manifest.get("provider_outage") or "halt"),
        pool=DualCapPool(
            total=parallelism, per_provider=parallelism_per_provider
        ),
        outcomes=outcomes,
        status=status,
        wave_count=wave_count,
    )


def _checkpoint(handle: CampaignHandle) -> dict[str, Any]:
    report = summarize_campaign(
        campaign_id=handle.campaign_id,
        outcomes=handle.outcomes,
        parallelism=handle.parallelism,
        status=handle.status,
    )
    report["mode"] = handle.mode
    report["wave_count"] = handle.wave_count
    report["parallelism_per_provider"] = handle.parallelism_per_provider
    report["provider_health"] = handle.health.snapshot()
    write_report(handle.root, report)
    return report


async def _record_health(
    handle: CampaignHandle, outcome: dict[str, Any]
) -> dict[str, Any]:
    """Observe one outcome on the campaign health tracker and stamp the row."""
    result = await handle.health.observe_outcome_async(outcome)
    annotated = dict(outcome)
    annotated["provider_health_event"] = result.event.value
    annotated["provider_health_status"] = result.status.value
    annotated["provider_consecutive_failures"] = result.consecutive_failures
    annotated["provider_trip_count"] = result.trip_count
    if result.event in {
        ProviderHealthEvent.ENTERED_COOLDOWN,
        ProviderHealthEvent.MARKED_DOWN,
    }:
        handle.provider_health_events.append(
            {
                "event": result.event.value,
                "runtime": result.runtime,
                "status": result.status.value,
                "trip_count": result.trip_count,
                "episode_id": outcome.get("episode_id"),
                "error": outcome.get("error"),
                "cooldown_remaining_seconds": result.cooldown_remaining_seconds,
            }
        )
    return annotated


async def _run_wave_jobs(jobs: list[Any]) -> list[dict[str, Any]]:
    """Run one wave's job coroutines, releasing every slot before propagating.

    Main callers:
        ``run_campaign_wave``.

    Why this exists:
        A bare ``asyncio.gather`` abandons its siblings the moment one job
        raises: the exception surfaces immediately while the other episodes keep
        running as orphan tasks that still hold DualCapPool slots and still
        write into the campaign root. Provider capacity would then be consumed
        by work nobody collects. Cancelling and awaiting the rest first makes
        slot accounting exact even on a failed wave, and the original error
        still propagates unchanged.
    """
    tasks = [asyncio.ensure_future(job) for job in jobs]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def discard_campaign_outcomes(
    handle: CampaignHandle,
    outcomes: list[dict[str, Any]],
    *,
    reason: str,
) -> None:
    """Withdraw the training claim of a discarded group, everywhere it is stored.

    Main callers:
        ``stream_engine.collect_adaptive_world`` on DISCARD_AND_REDO_BACKUP.

    Steps, in order:
        1. rewrite each episode's ``logs/durable_outcome.json`` so a resume
           replays it as *not* accepted;
        2. patch the matching campaign scoreboard rows in memory, so the running
           process agrees with what a resume would read;
        3. re-checkpoint ``campaign_report.json``.

    Why this exists:
        A provider fallback drops the group's captures and ``capture_index``
        rows. The episodes themselves stay on disk and on the scoreboard as
        evidence, and they must: they really ran. But if they keep claiming
        ``accepted_count: 1`` with no capture behind them, the next resume banks
        an untrainable group and ``captures_for_outcomes`` wedges collection.
    """
    from shellsmith.train.durable_capture import (
        discarded_outcome_fields,
        mark_durable_outcome_discarded,
    )

    if not outcomes:
        return
    fields = discarded_outcome_fields(reason)
    discarded_ids: set[str] = set()
    for outcome in outcomes:
        episode_id = str(outcome.get("episode_id") or "")
        if not episode_id:
            raise RuntimeError("Cannot discard a campaign outcome with no episode_id.")
        discarded_ids.add(episode_id)
        rollout_dir = outcome.get("rollout_dir")
        if isinstance(rollout_dir, str) and rollout_dir:
            mark_durable_outcome_discarded(Path(rollout_dir), reason=reason)
        outcome.update(fields)
    for row in handle.outcomes:
        if (
            isinstance(row, dict)
            and str(row.get("episode_id") or "") in discarded_ids
        ):
            row.update(fields)
    _checkpoint(handle)


def _drop_unsettled_rows(handle: CampaignHandle, episode_ids: set[str]) -> None:
    """Remove ``episode_not_started`` placeholders for episodes about to run.

    Main callers:
        ``run_campaign_wave``, right after it decides which jobs to launch.

    Why this exists:
        A provider-down placeholder is evidence that an attempt slot was blocked,
        and it stays on the scoreboard for as long as that remains true. Once the
        attempt is actually launched, the placeholder would become a duplicate
        row for the same ``episode_id`` — inflating ``rollout_count`` and
        ``infrastructure_failure_count`` with an episode that did run. The real
        outcome replaces it.
    """
    if not episode_ids:
        return
    handle.outcomes[:] = [
        item
        for item in handle.outcomes
        if not (
            isinstance(item, dict)
            and str(item.get("episode_id") or "") in episode_ids
            and not outcome_is_settled(item)
        )
    ]


async def run_campaign_wave(
    handle: CampaignHandle,
    jobs: list[RolloutJob],
    *,
    args: argparse.Namespace,
    rollout_runner: RolloutRunner = run_live_job,
) -> list[dict[str, Any]]:
    """Run one wave of jobs into an open campaign and refresh the scoreboard.

    Each job waits out a provider cooldown (if any) before taking a DualCapPool
    slot. Outcomes update the shared ProviderHealthTracker; a second failure
    streak marks the runtime DOWN for halt/fallback.

    Resume behavior:
        Episode IDs already present in ``handle.outcomes`` are skipped, *unless*
        that row is ``episode_not_started`` (a provider-down placeholder for an
        episode that never launched). Those rows are dropped and the attempt is
        run for real, so one transient outage does not burn its ``rollout_NN``
        slots for the lifetime of the campaign.
        Jobs with ``logs/durable_outcome.json`` are replayed without re-running.
        Returns the wave's outcome rows (not the full campaign report).
    """
    from shellsmith.train.durable_capture import (
        durable_outcome_path_for_rollout,
        load_durable_outcome,
        write_durable_outcome,
    )

    if handle.status not in {"running", "collecting"}:
        raise RuntimeError(
            f"Cannot run a wave on campaign status {handle.status!r}."
        )
    if not jobs:
        _checkpoint(handle)
        return []

    settled_by_episode = {
        str(item.get("episode_id")): item
        for item in handle.outcomes
        if isinstance(item, dict)
        and item.get("episode_id")
        and outcome_is_settled(item)
    }
    already_done = set(settled_by_episode)
    pending_jobs: list[RolloutJob] = []
    wave_outcomes: list[dict[str, Any]] = []
    for job in jobs:
        if job.episode_id in already_done:
            # Prior crash already recorded this episode on the campaign scoreboard.
            wave_outcomes.append(dict(settled_by_episode[job.episode_id]))
            continue
        durable = load_durable_outcome(job.output_dir)
        if durable is not None:
            annotated = _annotate_outcome(job, durable)
            wave_outcomes.append(await _record_health(handle, annotated))
            continue
        pending_jobs.append(job)
    # An unsettled placeholder must not survive alongside the real attempt.
    _drop_unsettled_rows(handle, {job.episode_id for job in pending_jobs})

    async def bounded(job: RolloutJob) -> dict[str, Any]:
        if handle.health.is_down(job.gm_runtime):
            # No durable outcome: the episode never ran, so a later resume must
            # be free to try this attempt slot again once the provider is back.
            return await _record_health(handle, not_started_outcome(job))
        try:
            await handle.health.wait_until_available(job.gm_runtime)
        except RuntimeError:
            # The runtime was marked DOWN by a sibling job while this one waited
            # out a cooldown. That is the same "never started" situation as the
            # check above, not a wave failure: no pool slot was taken and no
            # episode ran. Letting it escape would abort the whole wave and
            # strand every sibling episode still holding a provider slot.
            if not handle.health.is_down(job.gm_runtime):
                raise
            return await _record_health(handle, not_started_outcome(job))
        async with handle.pool.slot(job.gm_runtime):
            try:
                outcome = await rollout_runner(job, args)
            except Exception as caught:
                failure_detail = serialize_failure(
                    caught,
                    stage="rollout_runner",
                    component="campaign_runner",
                    # Preserve legacy custom-runner behavior; first-party local
                    # boundaries carry an explicit False annotation.
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
            if not durable_outcome_path_for_rollout(job.output_dir).is_file():
                write_durable_outcome(job.output_dir, annotated)
            return await _record_health(handle, annotated)

    if pending_jobs:
        wave_outcomes.extend(
            await _run_wave_jobs([bounded(job) for job in pending_jobs])
        )
    # Only append outcomes that were not already on the scoreboard.
    for outcome in wave_outcomes:
        episode_id = str(outcome.get("episode_id") or "")
        if episode_id and episode_id not in already_done:
            handle.outcomes.append(outcome)
    handle.wave_count += 1
    handle.status = "running"
    _checkpoint(handle)
    return wave_outcomes


def finalize_campaign(
    handle: CampaignHandle,
    *,
    status: str,
    provider_outage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mark the campaign terminal and write the final scoreboard."""
    handle.status = status
    report = summarize_campaign(
        campaign_id=handle.campaign_id,
        outcomes=handle.outcomes,
        parallelism=handle.parallelism,
        status=status,
        provider_outage=provider_outage,
    )
    report["mode"] = handle.mode
    report["wave_count"] = handle.wave_count
    report["parallelism_per_provider"] = handle.parallelism_per_provider
    report["provider_health"] = handle.health.snapshot()
    if provider_outage is not None:
        report["provider_outage"] = provider_outage
    elif handle.provider_health_events:
        report["provider_outage"] = {
            "health_events": list(handle.provider_health_events),
            "provider_health": handle.health.snapshot(),
        }
    write_report(handle.root, report)
    return report


def build_task_attempt_jobs(
    *,
    campaign_id: str,
    campaign_root: Path,
    world_file: Path,
    world_id: str,
    flavor: str,
    attempt_start: int,
    attempt_count: int,
    gm_runtime: str,
    gm_model: str,
    gm_provider: str,
    gm_reasoning_effort: str,
    gm_fallback_applied: bool = False,
    original_gm_model: str | None = None,
) -> list[RolloutJob]:
    """Build consecutive attempt jobs for one world file inside an open campaign."""
    if attempt_count < 1:
        raise RuntimeError("attempt_count must be at least one.")
    if attempt_start < 1:
        raise RuntimeError("attempt_start must be at least one.")
    jobs: list[RolloutJob] = []
    for offset in range(attempt_count):
        attempt = attempt_start + offset
        jobs.append(
            RolloutJob(
                world_file=world_file,
                world_id=world_id,
                flavor=flavor,
                attempt=attempt,
                episode_id=f"{campaign_id}-{world_id}-{attempt:02d}",
                output_dir=campaign_root.joinpath(
                    world_id, f"rollout_{attempt:02d}"
                ),
                gm_runtime=gm_runtime,
                gm_model=gm_model,
                gm_provider=gm_provider,
                gm_reasoning_effort=gm_reasoning_effort,
                gm_fallback_applied=gm_fallback_applied,
                original_gm_model=original_gm_model,
            )
        )
    return jobs


def manifest_jsonable(spec: OpenCampaignSpec) -> dict[str, Any]:
    return asdict(spec)
