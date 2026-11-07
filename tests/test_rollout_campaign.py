"""Focused tests for global cross-task shellsmith rollout scheduling."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pytest

from shellsmith.campaign import rollout_campaign
from shellsmith.campaign.rollout_campaign import (
    RolloutJob,
    _annotate_outcome,
    build_jobs,
    fallback_output_dir,
    run_campaign,
    select_worlds,
)


def _task(path: Path, title: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# {title}

## World
Private coherent world facts.

## Player
Investigate with available tools.

## Reward
Reward evidence and task_outcomes.example.

## Termination conditions
Successful after proof. Failed for harm. Neutral for a blocker.
""",
        encoding="utf-8",
    )
    return path


def _campaign_args(tmp_path: Path, *tasks: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = dict(
        world_file=list(tasks),
        from_draft_catalog=False,
        draft_root=tmp_path,
        flavor=None,
        exclude_task=[],
        limit_tasks=None,
        rollouts_per_task=2,
        parallelism=2,
        shuffle_seed=0,
        output_dir=tmp_path / "campaign",
        campaign_id="test",
        gm_runtime="codex",
        gm_model="gpt-6.1-sol",
        gm_provider="codex-cli",
        gm_api_key_env="OPENROUTER_API_KEY",
        gm_reasoning_effort="low",
        max_world_steps=100,
        timeout_seconds=7_200.0,
        provider_outage="halt",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_build_jobs_round_robins_tasks_before_attempts(tmp_path: Path) -> None:
    a = _task(tmp_path / "a.md", "A")
    b = _task(tmp_path / "b.md", "B")

    jobs = build_jobs(
        campaign_id="c",
        output_dir=tmp_path / "out",
        selected=[(a, "agentic"), (b, "paper")],
        rollouts_per_task=2,
        shuffle_seed=0,
        gm_runtime="codex",
        gm_model="gpt-6.1-sol",
        gm_provider="codex-cli",
        gm_reasoning_effort="low",
    )

    assert [(job.world_id, job.attempt) for job in jobs] == [
        ("a", 1),
        ("b", 1),
        ("a", 2),
        ("b", 2),
    ]
    assert all(job.gm_model == "gpt-6.1-sol" for job in jobs)


def test_campaign_default_has_separate_invocation_and_api_timeouts() -> None:
    args = rollout_campaign._parser().parse_args([])

    assert args.timeout_seconds == 900.0
    assert args.gm_api_idle_timeout_seconds == 180.0
    assert args.provider_outage == "halt"


def test_provider_fallback_uses_a_new_sibling_output_directory(tmp_path: Path) -> None:
    """A cancelled primary run owns its directory; fallback must not overwrite it."""
    task = _task(tmp_path / "task.md", "Task")
    job = build_jobs(
        campaign_id="campaign",
        output_dir=tmp_path / "campaign",
        selected=[(task, "agentic")],
        rollouts_per_task=1,
        shuffle_seed=0,
        gm_runtime="codex",
        gm_model="gpt-6.1-sol",
        gm_provider="openrouter",
        gm_reasoning_effort="low",
    )[0]

    fallback_dir = fallback_output_dir(job, backup_runtime="claude")

    assert fallback_dir == job.output_dir.with_name("rollout_01__fallback_claude")
    assert fallback_dir != job.output_dir


def test_campaign_uses_one_global_concurrency_limit_and_aggregates(
    tmp_path: Path,
) -> None:
    a = _task(tmp_path / "a.md", "A")
    b = _task(tmp_path / "b.md", "B")
    active = 0
    peak = 0

    async def fake_runner(job, _args):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        reward = 0.2 if job.attempt == 1 else 0.8
        return {
            "episode_id": job.episode_id,
            "world_id": job.world_id,
            "flavor": job.flavor,
            "attempt": job.attempt,
            "rollout_dir": str(job.output_dir),
            "error": None,
            "audit": {
                "accepted_for_training": True,
                "final_task": {"reward": reward},
                "reasoning": {"mean_quality": 0.7, "mean_efficiency": 0.8},
            },
        }

    args = _campaign_args(tmp_path, a, b)
    _, report = asyncio.run(run_campaign(args, rollout_runner=fake_runner))

    assert peak == 2
    assert report["status"] == "completed"
    assert report["task_count"] == 2
    assert report["rollout_count"] == 4
    assert report["accepted_count"] == 4
    assert [task["reward"]["population_stddev"] for task in report["tasks"]] == [
        0.3,
        0.3,
    ]


def test_selection_rejects_unknown_exclusions_and_duplicate_task_ids(
    tmp_path: Path,
) -> None:
    first = _task(tmp_path / "one" / "same.md", "First")
    second = _task(tmp_path / "two" / "same.md", "Second")

    base = argparse.Namespace(
        world_file=[first],
        from_draft_catalog=False,
        draft_root=tmp_path,
        flavor=None,
        exclude_task=["misspelled"],
        limit_tasks=None,
    )
    with pytest.raises(RuntimeError, match="were not selected"):
        select_worlds(base)

    base.exclude_task = []
    base.world_file = [first, second]
    with pytest.raises(RuntimeError, match="duplicate IDs"):
        select_worlds(base)


def test_campaign_isolates_runner_failures_and_checkpoints_each_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = _task(tmp_path / "a.md", "A")
    b = _task(tmp_path / "b.md", "B")
    checkpoint_counts: list[int] = []
    real_write_report = rollout_campaign.write_report

    def recording_write_report(output_dir, report):
        checkpoint_counts.append(report["rollout_count"])
        real_write_report(output_dir, report)

    monkeypatch.setattr(rollout_campaign, "write_report", recording_write_report)

    async def partly_failing_runner(job, _args):
        if job.world_id == "a":
            raise RuntimeError("isolated failure")
        await asyncio.sleep(0.05)
        return {
            "episode_id": job.episode_id,
            "world_id": job.world_id,
            "flavor": job.flavor,
            "attempt": job.attempt,
            "rollout_dir": str(job.output_dir),
            "error": None,
            "audit": {
                "accepted_for_training": True,
                "final_task": {"reward": 0.5},
                "reasoning": {"mean_quality": 0.6, "mean_efficiency": 0.7},
            },
        }

    args = _campaign_args(
        tmp_path,
        a,
        b,
        rollouts_per_task=1,
        parallelism=1,
        parallelism_per_provider=1,
        output_dir=tmp_path / "campaign",
    )
    _, report = asyncio.run(run_campaign(args, rollout_runner=partly_failing_runner))

    assert checkpoint_counts[-1] == 2
    assert report["rollout_count"] == 2
    assert report["accepted_count"] == 1
    assert report["infrastructure_failure_count"] == 1
    assert (args.output_dir / "campaign_report.json").is_file()


def test_campaign_halts_remaining_jobs_on_provider_outage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from shellsmith.campaign.provider_outage import (
        ProviderHealthTracker,
    )

    monkeypatch.setattr(
        rollout_campaign,
        "ProviderHealthTracker",
        lambda: ProviderHealthTracker(
            consecutive_failure_threshold=1,
            cooldown_seconds=0.0,
        ),
    )
    a = _task(tmp_path / "a.md", "A")
    b = _task(tmp_path / "b.md", "B")
    c = _task(tmp_path / "c.md", "C")
    started: list[str] = []
    hang = asyncio.Event()

    async def failing_runner(job, _args):
        # Keep the third job pending so DOWN still has something to cancel.
        if job.world_id == "c":
            await hang.wait()
        started.append(job.episode_id)
        await asyncio.sleep(0.01)
        return {
            "episode_id": job.episode_id,
            "world_id": job.world_id,
            "flavor": job.flavor,
            "attempt": job.attempt,
            "rollout_dir": str(job.output_dir),
            "error": "RuntimeError: rate limit exceeded",
            "audit": None,
        }

    args = _campaign_args(
        tmp_path,
        a,
        b,
        c,
        rollouts_per_task=1,
        parallelism=1,
        parallelism_per_provider=1,
        provider_outage="halt",
        output_dir=tmp_path / "halt-campaign",
    )
    _, report = asyncio.run(run_campaign(args, rollout_runner=failing_runner))

    assert report["status"] == "halted_provider_unavailable"
    assert report["provider_outage"]["action"] == "halt"
    assert report["rollout_count"] == 2
    assert started == ["test-a-01", "test-b-01"]


def test_campaign_fallback_remaps_remaining_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from shellsmith.campaign.provider_outage import (
        ProviderHealthTracker,
    )

    monkeypatch.setattr(
        rollout_campaign,
        "ProviderHealthTracker",
        lambda: ProviderHealthTracker(
            consecutive_failure_threshold=1,
            cooldown_seconds=0.0,
        ),
    )
    a = _task(tmp_path / "a.md", "A")
    b = _task(tmp_path / "b.md", "B")
    c = _task(tmp_path / "c.md", "C")
    models_seen: list[tuple[str, str]] = []
    hang = asyncio.Event()

    async def runner(job, _args):
        # Park the third primary attempt until remap; remapped retry must proceed.
        if job.world_id == "c" and not job.gm_fallback_applied:
            await hang.wait()
        models_seen.append((job.episode_id, job.gm_model))
        await asyncio.sleep(0.01)
        if not job.gm_fallback_applied:
            return {
                "episode_id": job.episode_id,
                "world_id": job.world_id,
                "flavor": job.flavor,
                "attempt": job.attempt,
                "rollout_dir": str(job.output_dir),
                "error": "RuntimeError: service unavailable overloaded",
                "audit": None,
            }
        return {
            "episode_id": job.episode_id,
            "world_id": job.world_id,
            "flavor": job.flavor,
            "attempt": job.attempt,
            "rollout_dir": str(job.output_dir),
            "error": None,
            "audit": {
                "accepted_for_training": True,
                "final_task": {"reward": 0.4},
                "reasoning": {"mean_quality": 0.5, "mean_efficiency": 0.5},
            },
        }

    args = _campaign_args(
        tmp_path,
        a,
        b,
        c,
        rollouts_per_task=1,
        parallelism=1,
        parallelism_per_provider=1,
        provider_outage="fallback",
        output_dir=tmp_path / "fallback-campaign",
    )
    _, report = asyncio.run(run_campaign(args, rollout_runner=runner))

    assert report["status"] == "completed_with_provider_fallback"
    assert report["provider_outage"]["action"] == "remapped_remaining"
    assert any(model == "opus" for _, model in models_seen)
    remapped = report["provider_outage"]["remapped_episode_ids"]
    assert "test-c-01" in remapped
    assert report["accepted_count"] == 1


def _job(tmp_path: Path) -> RolloutJob:
    return RolloutJob(
        world_file=tmp_path / "task.md",
        world_id="task",
        flavor="agentic",
        attempt=1,
        episode_id="ep-1",
        output_dir=tmp_path / "out",
        gm_runtime="codex",
        gm_model="gpt-5.6-terra",
        gm_provider="codex-cli",
        gm_reasoning_effort="low",
    )


def test_annotate_outcome_preserves_runner_accepted_count_zero(
    tmp_path: Path,
) -> None:
    """Tinker capture failure must not be overwritten by a green audit."""
    annotated = _annotate_outcome(
        _job(tmp_path),
        {
            "episode_id": "ep-1",
            "world_id": "task",
            "flavor": "agentic",
            "attempt": 1,
            "rollout_dir": str(tmp_path / "out"),
            "error": (
                "RuntimeError: Tinker record count does not match sampled "
                "Pi blocks: records=23 blocks=24."
            ),
            "audit": {
                "accepted_for_training": True,
                "final_task": {"reward": 0.58},
            },
            "accepted_count": 0,
        },
    )
    assert annotated["accepted_count"] == 0
    assert annotated["gm_model"] == "gpt-5.6-terra"


def test_annotate_outcome_derives_accepted_count_when_absent(
    tmp_path: Path,
) -> None:
    annotated = _annotate_outcome(
        _job(tmp_path),
        {
            "episode_id": "ep-1",
            "world_id": "task",
            "flavor": "agentic",
            "attempt": 1,
            "rollout_dir": str(tmp_path / "out"),
            "error": None,
            "audit": {
                "accepted_for_training": True,
                "final_task": {"reward": 0.4},
            },
        },
    )
    assert annotated["accepted_count"] == 1
