"""Unit tests for shellsmith campaign stream adaptive controls.

No live Claude/Tinker required — fake runners + decision-table assertions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from shellsmith.campaign.rollout_campaign import (
    RolloutJob,
)
from shellsmith.campaign.stream_campaign import (
    DualCapPool,
    OpenCampaignSpec,
    build_task_attempt_jobs,
    finalize_campaign,
    open_campaign,
    run_campaign_wave,
)
from shellsmith.campaign.provider_outage import (
    ProviderHealthTracker,
)
from shellsmith.train.stream_engine import (
    UpdateWorldResult,
    WorldCollectResult,
    collect_adaptive_world,
    collect_update_worlds,
)
from shellsmith.train.stream_policy import (
    AdaptivePolicy,
    StreamPolicyConfig,
    WorldAction,
    WorldDecision,
    additional_attempts,
    evaluate_world_outcomes,
)
from shellsmith.train.capture_store import (
    CaptureStore,
)
from shellsmith.credit.to_tinker_bridge import (
    CapturedRollout,
)


def _accepted(reward: float, *, episode_id: str = "ep") -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "accepted_count": 1,
        "provider_infrastructure_failure": False,
        "audit": {
            "accepted_for_training": True,
            "final_task": {"reward": reward},
        },
    }


def _rejected(*, infra: bool = False, episode_id: str = "ep") -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "accepted_count": 0,
        "provider_infrastructure_failure": infra,
        "error": (
            "ProviderHTTPError: service unavailable"
            if infra
            else "task failed"
        ),
        "audit": None,
    }


def test_additional_attempts_trainable_fill() -> None:
    policy = AdaptivePolicy()
    assert (
        additional_attempts(
            {"attempt_count": 4, "accepted_count": 2, "reward": {"values": [0.1, 0.2]}},
            policy,
        )
        == 2
    )


def test_additional_attempts_spread_top_up() -> None:
    policy = AdaptivePolicy()
    assert (
        additional_attempts(
            {
                "attempt_count": 4,
                "accepted_count": 4,
                "reward": {"values": [0.5, 0.5, 0.5, 0.5]},
            },
            policy,
        )
        == 4
    )


def test_additional_attempts_ready_when_spread_ok() -> None:
    policy = AdaptivePolicy()
    assert (
        additional_attempts(
            {
                "attempt_count": 4,
                "accepted_count": 4,
                "reward": {"values": [0.1, 0.3, 0.5, 0.7]},
            },
            policy,
        )
        == 0
    )


def test_additional_attempts_cap_at_four_and_max_eight() -> None:
    policy = AdaptivePolicy()
    assert (
        additional_attempts(
            {"attempt_count": 5, "accepted_count": 0, "reward": {"values": []}},
            policy,
        )
        == 3
    )
    assert (
        additional_attempts(
            {"attempt_count": 8, "accepted_count": 2, "reward": {"values": []}},
            policy,
        )
        == 0
    )


def test_maximum_attempts_allows_twelve_for_retry() -> None:
    AdaptivePolicy(maximum_attempts=12).validate()
    with pytest.raises(ValueError, match="cannot exceed"):
        AdaptivePolicy(maximum_attempts=13).validate()


def test_evaluate_top_up_after_eight_when_max_is_twelve() -> None:
    config = StreamPolicyConfig(maximum_attempts=12)
    outcomes = [_rejected(episode_id=f"r{i}") for i in range(6)]
    outcomes.append(_accepted(0.2, episode_id="a6"))
    outcomes.append(_accepted(0.3, episode_id="a7"))
    assert len(outcomes) == 8
    decision = evaluate_world_outcomes(
        outcomes,
        config=config,
        gm_runtime="codex",
        gm_model="gpt-6.1-sol",
        gm_provider="openrouter",
        gm_fallback_applied=False,
    )
    assert decision.action == WorldAction.TOP_UP
    assert decision.top_up_attempts == 2


def test_build_insufficient_groups_retry_plan_keeps_ready_and_elevates() -> None:
    from shellsmith.train.stream_policy import (
        build_insufficient_groups_retry_plan,
        elevate_maximum_attempts_for_retry,
        pinned_policy_from_outcomes,
    )

    assert elevate_maximum_attempts_for_retry(base_maximum_attempts=8) == 12
    state = {
        "status": "insufficient_groups",
        "next_world_offset": 4,
        "steps": [
            {"step": 0, "accepted_group_count": 2},
            {
                "step": 1,
                "skip_reason": "insufficient_remaining_complete_groups",
                "tasks": [
                    {
                        "decision": "group_ready",
                        "attempts": [
                            {
                                "accepted_count": 1,
                                "policy_kind": "sampler_checkpoint",
                                "policy_ref": "tinker://ckpt",
                                "policy_rft_updates_completed": 1,
                            }
                        ],
                    },
                    {
                        "decision": "skip_insufficient",
                        "rollout_world_id": "skip_task",
                        "gm_runtime": "codex",
                        "gm_model": "gpt-5.6-terra",
                        "gm_provider": "codex-cli",
                        "gm_fallback_applied": False,
                        "original_gm_model": None,
                        "attempts": [
                            {
                                "accepted_count": 0,
                                "policy_kind": "sampler_checkpoint",
                                "policy_ref": "tinker://ckpt",
                                "policy_rft_updates_completed": 1,
                            }
                        ]
                        * 8,
                    },
                ],
            },
        ],
    }
    plan = build_insufficient_groups_retry_plan(
        state, base_maximum_attempts=8, extra_attempts=4
    )
    assert plan.start_world_offset == 2
    assert plan.end_world_offset == 4
    assert plan.ready_task_indexes == (0,)
    assert plan.retry_task_indexes == (1,)
    assert plan.elevated_maximum_attempts == 12
    kind, ref, rft = pinned_policy_from_outcomes(
        state["steps"][1]["tasks"][1]["attempts"]
    )
    assert kind == "sampler_checkpoint"
    assert ref == "tinker://ckpt"
    assert rft == 1


def test_pinned_policy_from_outcomes_rejects_mix() -> None:
    from shellsmith.train.stream_policy import (
        pinned_policy_from_outcomes,
    )

    with pytest.raises(RuntimeError, match="mixed Player policies"):
        pinned_policy_from_outcomes(
            [
                {
                    "policy_kind": "base",
                    "policy_ref": None,
                    "policy_rft_updates_completed": 0,
                },
                {
                    "policy_kind": "sampler_checkpoint",
                    "policy_ref": "tinker://ckpt",
                    "policy_rft_updates_completed": 1,
                },
            ]
        )


def test_evaluate_group_ready() -> None:
    config = StreamPolicyConfig()
    outcomes = [_accepted(0.1, episode_id=f"a{i}") for i in range(4)]
    outcomes[3] = _accepted(0.5, episode_id="a3")
    decision = evaluate_world_outcomes(
        outcomes,
        config=config,
        gm_runtime="codex",
        gm_model="gpt-6.1-sol",
        gm_provider="openrouter",
        gm_fallback_applied=False,
    )
    assert decision.action == WorldAction.GROUP_READY


def test_evaluate_skip_insufficient_at_max() -> None:
    config = StreamPolicyConfig()
    outcomes = [_rejected(episode_id=f"r{i}") for i in range(8)]
    decision = evaluate_world_outcomes(
        outcomes,
        config=config,
        gm_runtime="codex",
        gm_model="gpt-6.1-sol",
        gm_provider="openrouter",
        gm_fallback_applied=False,
    )
    assert decision.action == WorldAction.SKIP_INSUFFICIENT


def test_evaluate_circuit_halt() -> None:
    config = StreamPolicyConfig(provider_outage="halt")
    outcomes = [_rejected(episode_id="i1"), _rejected(episode_id="i2")]
    decision = evaluate_world_outcomes(
        outcomes,
        config=config,
        gm_runtime="codex",
        gm_model="gpt-6.1-sol",
        gm_provider="openrouter",
        gm_fallback_applied=False,
        provider_down=True,
    )
    assert decision.action == WorldAction.HALT_STREAM


def test_evaluate_circuit_fallback_discard_and_redo() -> None:
    config = StreamPolicyConfig(provider_outage="fallback")
    outcomes = [_rejected(episode_id="i1"), _rejected(episode_id="i2")]
    decision = evaluate_world_outcomes(
        outcomes,
        config=config,
        gm_runtime="codex",
        gm_model="gpt-6.1-sol",
        gm_provider="openrouter",
        gm_fallback_applied=False,
        provider_down=True,
    )
    assert decision.action == WorldAction.DISCARD_AND_REDO_BACKUP
    assert decision.backup_model == "opus"


def test_evaluate_circuit_fallback_already_applied_halts() -> None:
    config = StreamPolicyConfig(provider_outage="fallback")
    outcomes = [_rejected(episode_id="i1"), _rejected(episode_id="i2")]
    decision = evaluate_world_outcomes(
        outcomes,
        config=config,
        gm_runtime="claude",
        gm_model="opus",
        gm_provider="anthropic",
        gm_fallback_applied=True,
        provider_down=True,
    )
    assert decision.action == WorldAction.HALT_STREAM


def test_capture_store_rejects_mixed_identity() -> None:
    store = CaptureStore()
    with pytest.raises(RuntimeError, match="mixed identity"):
        store.require_single_identity(
            (
                CapturedRollout(
                    rollout_root=Path("/tmp/a"),
                    group_identity_key="key-a",
                    captures=(),
                ),
                CapturedRollout(
                    rollout_root=Path("/tmp/b"),
                    group_identity_key="key-b",
                    captures=(),
                ),
            )
        )


def test_task_stats_ignore_audit_accept_without_capture_count() -> None:
    from shellsmith.train.stream_policy import (
        task_stats_from_outcomes,
    )

    # Audit green but capture build failed (accepted_count stayed 0).
    stats = task_stats_from_outcomes(
        [
            {
                "episode_id": "ep-fail-capture",
                "accepted_count": 0,
                "error": "RuntimeError: Tinker record count does not match",
                "audit": {
                    "accepted_for_training": True,
                    "final_task": {"reward": 0.58},
                },
            },
            _accepted(0.4, episode_id="ep-ok"),
        ]
    )
    assert stats["accepted_count"] == 1
    assert stats["reward"]["values"] == [0.4]


def test_captures_for_outcomes_skips_audit_only_accepts() -> None:
    store = CaptureStore()
    capture = CapturedRollout(
        rollout_root=Path("/tmp/ok"),
        group_identity_key="key-a",
        captures=(),
    )
    store.put(capture, episode_id="ep-ok")
    outcomes = [
        {
            "episode_id": "ep-ghost",
            "accepted_count": 0,
            "audit": {"accepted_for_training": True},
            "error": "RuntimeError: Tinker record count mismatch",
        },
        {
            "episode_id": "ep-ok",
            "accepted_count": 1,
            "audit": {"accepted_for_training": True},
        },
    ]
    got = store.captures_for_outcomes(outcomes)
    assert got == (capture,)


def test_one_shot_stream_api_writes_report(tmp_path: Path) -> None:
    async def _run() -> None:
        handle = open_campaign(
            tmp_path.joinpath("camp"),
            OpenCampaignSpec(campaign_id="camp-test", mode="inference"),
        )
        world_file = tmp_path.joinpath("task.md")
        world_file.write_text("# task\n", encoding="utf-8")

        async def fake_runner(
            job: RolloutJob, args: argparse.Namespace
        ) -> dict[str, Any]:
            job.output_dir.mkdir(parents=True, exist_ok=True)
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
                },
            }

        jobs = build_task_attempt_jobs(
            campaign_id=handle.campaign_id,
            campaign_root=handle.root,
            world_file=world_file,
            world_id="demo_task",
            flavor="agentic",
            attempt_start=1,
            attempt_count=1,
            gm_runtime="codex",
            gm_model="gpt-6.1-sol",
            gm_provider="openrouter",
            gm_reasoning_effort="low",
        )
        args = argparse.Namespace()
        await run_campaign_wave(
            handle, jobs, args=args, rollout_runner=fake_runner
        )
        report = finalize_campaign(handle, status="complete")
        assert report["status"] == "complete"
        assert handle.root.joinpath("campaign_report.json").is_file()
        assert handle.root.joinpath("campaign_manifest.json").is_file()

    asyncio.run(_run())


def test_adaptive_lifecycle_top_up_then_ready(tmp_path: Path) -> None:
    async def _run() -> None:
        handle = open_campaign(
            tmp_path.joinpath("camp"),
            OpenCampaignSpec(campaign_id="adapt", mode="inference", parallelism=4),
        )
        world_file = tmp_path.joinpath("task.md")
        world_file.write_text("# task\n", encoding="utf-8")
        call_count = {"n": 0}

        async def fake_runner(
            job: RolloutJob, args: argparse.Namespace
        ) -> dict[str, Any]:
            call_count["n"] += 1
            job.output_dir.mkdir(parents=True, exist_ok=True)
            # First four: two accepted with no spread → needs top-up.
            # Later attempts: fill to four accepted with spread.
            if job.attempt <= 2:
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
                    },
                }
            if job.attempt <= 4:
                return {
                    "episode_id": job.episode_id,
                    "world_id": job.world_id,
                    "flavor": job.flavor,
                    "attempt": job.attempt,
                    "rollout_dir": str(job.output_dir),
                    "error": "fail",
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
                    "final_task": {"reward": 0.1 + 0.2 * job.attempt},
                },
            }

        config = StreamPolicyConfig(
            initial_attempts=4,
            maximum_attempts=8,
            minimum_trainable=4,
            minimum_reward_range=0.15,
            low_spread_top_up=4,
            campaign_mode="inference",
        )
        result = await collect_adaptive_world(
            handle=handle,
            args=argparse.Namespace(),
            config=config,
            world_file=world_file,
            world_id="demo_task",
            flavor="agentic",
            gm_runtime="codex",
            gm_model="gpt-6.1-sol",
            gm_provider="openrouter",
            gm_reasoning_effort="low",
            rollout_runner=fake_runner,
        )
        assert result.decision.action == WorldAction.GROUP_READY
        assert call_count["n"] > 4
        assert any(
            handle.root.joinpath("demo_task", f"rollout_{i:02d}").is_dir()
            for i in range(5, 9)
        )

    asyncio.run(_run())


def test_stream_train_cadence_banks_two_groups(tmp_path: Path) -> None:
    """Two GROUP_READY worlds with groups_per_update=2 → one optim bank fill."""

    async def _run() -> None:
        handle = open_campaign(
            tmp_path.joinpath("hero"),
            OpenCampaignSpec(campaign_id="hero", mode="tinker_train"),
        )
        world_file = tmp_path.joinpath("task.md")
        world_file.write_text("# task\n", encoding="utf-8")
        optim_calls: list[int] = []

        async def always_ready(
            job: RolloutJob, args: argparse.Namespace
        ) -> dict[str, Any]:
            job.output_dir.mkdir(parents=True, exist_ok=True)
            return {
                "episode_id": job.episode_id,
                "world_id": job.world_id,
                "flavor": job.flavor,
                "attempt": job.attempt,
                "rollout_dir": str(job.output_dir),
                "error": None,
                "audit": {
                    "accepted_for_training": True,
                    "final_task": {"reward": 0.2 * job.attempt},
                },
            }

        config = StreamPolicyConfig(
            initial_attempts=4,
            minimum_trainable=4,
            groups_per_update=2,
            campaign_mode="tinker_train",
        )
        banked = 0
        for task_id in ("world_a", "world_b"):
            result = await collect_adaptive_world(
                handle=handle,
                args=argparse.Namespace(),
                config=config,
                world_file=world_file,
                world_id=task_id,
                flavor="agentic",
                gm_runtime="codex",
                gm_model="gpt-6.1-sol",
                gm_provider="openrouter",
                gm_reasoning_effort="low",
                rollout_runner=always_ready,
            )
            assert result.decision.action == WorldAction.GROUP_READY
            banked += 1
            if banked >= config.groups_per_update:
                optim_calls.append(banked)
                banked = 0
        assert optim_calls == [2]

    asyncio.run(_run())


def test_fallback_discards_and_redoes_on_backup(tmp_path: Path) -> None:
    async def _run() -> None:
        handle = open_campaign(
            tmp_path.joinpath("fb"),
            OpenCampaignSpec(
                campaign_id="fb",
                mode="inference",
                provider_outage="fallback",
            ),
        )
        # Fast health trips for the test (2 fails → cooldown 0 → 2 fails → down).
        handle.health = ProviderHealthTracker(
            consecutive_failure_threshold=2,
            cooldown_seconds=0.0,
        )
        world_file = tmp_path.joinpath("task.md")
        world_file.write_text("# task\n", encoding="utf-8")
        models_seen: list[str] = []

        async def runner(
            job: RolloutJob, args: argparse.Namespace
        ) -> dict[str, Any]:
            models_seen.append(job.gm_model)
            job.output_dir.mkdir(parents=True, exist_ok=True)
            if job.gm_model == "gpt-6.1-sol":
                return {
                    "episode_id": job.episode_id,
                    "world_id": job.world_id,
                    "flavor": job.flavor,
                    "attempt": job.attempt,
                    "rollout_dir": str(job.output_dir),
                    "error": "ProviderHTTPError: service unavailable (503)",
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
                    "final_task": {"reward": 0.2 * job.attempt},
                },
            }

        config = StreamPolicyConfig(
            initial_attempts=2,
            maximum_attempts=8,
            minimum_trainable=2,
            provider_outage="fallback",
            campaign_mode="inference",
        )
        result = await collect_adaptive_world(
            handle=handle,
            args=argparse.Namespace(),
            config=config,
            world_file=world_file,
            world_id="demo_task",
            flavor="agentic",
            gm_runtime="codex",
            gm_model="gpt-6.1-sol",
            gm_provider="openrouter",
            gm_reasoning_effort="low",
            rollout_runner=runner,
        )
        assert result.decision.action == WorldAction.GROUP_READY
        assert result.gm_model == "opus"
        assert result.gm_fallback_applied is True
        assert result.discarded_outcomes
        assert "gpt-6.1-sol" in models_seen
        assert "opus" in models_seen
        assert handle.health.is_down("codex") is True

    asyncio.run(_run())


def test_dual_cap_total_binds_same_runtime() -> None:
    """All jobs share one runtime → total cap binds (never > total in-flight)."""

    async def _run() -> None:
        pool = DualCapPool(total=2, per_provider=2)
        in_flight = 0
        peak = 0
        lock = asyncio.Lock()

        async def worker() -> None:
            nonlocal in_flight, peak
            async with pool.slot("codex"):
                async with lock:
                    in_flight += 1
                    peak = max(peak, in_flight)
                await asyncio.sleep(0.05)
                async with lock:
                    in_flight -= 1

        await asyncio.gather(*(worker() for _ in range(6)))
        assert peak == 2
        assert pool.has_capacity("codex")

    asyncio.run(_run())


def test_dual_cap_per_provider_binds_mixed_runtimes() -> None:
    """total=4, per=2 with claude+codex → at most 2+2, never 4 of one."""

    async def _run() -> None:
        pool = DualCapPool(total=4, per_provider=2)
        in_flight: dict[str, int] = {"claude": 0, "codex": 0}
        peaks: dict[str, int] = {"claude": 0, "codex": 0}
        peak_total = 0
        lock = asyncio.Lock()

        async def worker(runtime: str) -> None:
            nonlocal peak_total
            async with pool.slot(runtime):
                async with lock:
                    in_flight[runtime] += 1
                    peaks[runtime] = max(peaks[runtime], in_flight[runtime])
                    peak_total = max(peak_total, sum(in_flight.values()))
                await asyncio.sleep(0.05)
                async with lock:
                    in_flight[runtime] -= 1

        await asyncio.gather(
            *(worker("claude") for _ in range(4)),
            *(worker("codex") for _ in range(4)),
        )
        assert peaks["claude"] == 2
        assert peaks["codex"] == 2
        assert peak_total == 4

    asyncio.run(_run())


def test_update_boundary_second_update_waits(tmp_path: Path) -> None:
    """Second update does not start until the first update banks its groups."""

    async def _run() -> None:
        events: list[str] = []
        gate = asyncio.Event()

        async def collect_one(
            world_file: Path,
            task_args: argparse.Namespace,
            world_config: StreamPolicyConfig,
            flavor: str,
        ) -> WorldCollectResult:
            world_id = world_file.stem
            events.append(f"start:{world_id}")
            if world_id in {"w0", "w1"}:
                await gate.wait()
            events.append(f"done:{world_id}")
            return WorldCollectResult(
                world_id=world_id,
                decision=WorldDecision(
                    action=WorldAction.GROUP_READY,
                    reason="test",
                ),
                gm_runtime="codex",
                gm_model="gpt-6.1-sol",
                gm_provider="openrouter",
            )

        task_runs: list[tuple[Path, argparse.Namespace]] = []
        for name in ("w0", "w1", "w2", "w3"):
            path = tmp_path.joinpath(f"{name}.md")
            path.write_text("# t\n", encoding="utf-8")
            task_runs.append(
                (
                    path,
                    argparse.Namespace(
                        initial_attempts=4,
                        maximum_attempts=8,
                        minimum_trainable=4,
                        minimum_reward_range=0.15,
                        low_spread_top_up=4,
                    ),
                )
            )

        config = StreamPolicyConfig(
            groups_per_update=2,
            campaign_mode="tinker_train",
            parallelism=8,
            parallelism_per_provider=4,
        )

        async def first_update() -> None:
            result = await collect_update_worlds(
                task_runs=task_runs,
                start_offset=0,
                groups_per_update=2,
                base_config=config,
                classify_flavor=lambda _path: "agentic",
                collect_one=collect_one,
            )
            events.append("update1_done")
            assert len(result.worlds) == 2
            assert result.next_world_offset == 2

        async def release_then_second() -> None:
            for _ in range(50):
                if "start:w0" in events and "start:w1" in events:
                    break
                await asyncio.sleep(0.01)
            assert "start:w0" in events and "start:w1" in events
            assert "start:w2" not in events
            gate.set()
            for _ in range(50):
                if "update1_done" in events:
                    break
                await asyncio.sleep(0.01)
            assert "update1_done" in events
            result = await collect_update_worlds(
                task_runs=task_runs,
                start_offset=2,
                groups_per_update=2,
                base_config=config,
                classify_flavor=lambda _path: "agentic",
                collect_one=collect_one,
            )
            events.append("update2_done")
            assert [item.result.world_id for item in result.worlds] == [
                "w2",
                "w3",
            ]

        await asyncio.gather(first_update(), release_then_second())
        assert events.index("update1_done") < events.index("start:w2")
        assert "update2_done" in events

    asyncio.run(_run())


def test_update_boundary_retry_keeps_ready_and_recollects_skip(
    tmp_path: Path,
) -> None:
    """``--retry`` preload keeps GROUP_READY stragglers and re-opens skips first."""

    async def _run() -> None:
        collected: list[str] = []

        async def collect_one(
            world_file: Path,
            task_args: argparse.Namespace,
            world_config: StreamPolicyConfig,
            flavor: str,
        ) -> WorldCollectResult:
            collected.append(world_file.stem)
            assert world_config.maximum_attempts == 12
            return WorldCollectResult(
                world_id=world_file.stem,
                decision=WorldDecision(
                    action=WorldAction.GROUP_READY,
                    reason="retry filled",
                ),
                gm_runtime="codex",
                gm_model="gpt-5.6-terra",
                gm_provider="codex-cli",
            )

        task_runs: list[tuple[Path, argparse.Namespace]] = []
        for name in ("ready", "skip", "extra"):
            path = tmp_path.joinpath(f"{name}.md")
            path.write_text("# t\n", encoding="utf-8")
            task_runs.append(
                (
                    path,
                    argparse.Namespace(
                        initial_attempts=4,
                        maximum_attempts=12,
                        minimum_trainable=4,
                        minimum_reward_range=0.15,
                        low_spread_top_up=4,
                    ),
                )
            )

        ready = UpdateWorldResult(
            stream_offset=0,
            world_file=task_runs[0][0],
            task_args=task_runs[0][1],
            result=WorldCollectResult(
                world_id="ready",
                decision=WorldDecision(
                    action=WorldAction.GROUP_READY,
                    reason="already ready",
                ),
                gm_runtime="codex",
                gm_model="gpt-5.6-terra",
                gm_provider="codex-cli",
            ),
        )
        config = StreamPolicyConfig(
            groups_per_update=2,
            campaign_mode="tinker_train",
            parallelism=8,
            parallelism_per_provider=4,
            maximum_attempts=12,
        )
        result = await collect_update_worlds(
            task_runs=task_runs,
            start_offset=0,
            groups_per_update=2,
            base_config=config,
            classify_flavor=lambda _path: "agentic",
            collect_one=collect_one,
            preloaded_worlds=[ready],
            priority_offsets={1},
        )
        assert collected == ["skip"]
        assert [item.result.world_id for item in result.worlds] == [
            "ready",
            "skip",
        ]
        assert result.next_world_offset == 2

    asyncio.run(_run())


def test_inference_run_campaign_uses_dual_cap_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inference run_campaign schedules jobs through DualCapPool slots."""

    async def _run() -> None:
        from shellsmith.campaign import (
            rollout_campaign as rc,
        )

        slots_seen: list[str] = []
        original_slot = DualCapPool.slot

        @asynccontextmanager
        async def tracking_slot(self: DualCapPool, gm_runtime: str):
            slots_seen.append(gm_runtime)
            async with original_slot(self, gm_runtime):
                yield

        monkeypatch.setattr(DualCapPool, "slot", tracking_slot)

        world_file = tmp_path.joinpath("dual_cap_task.md")
        world_file.write_text("# task\n", encoding="utf-8")
        monkeypatch.setattr(
            rc,
            "select_worlds",
            lambda _args: [(world_file, "agentic")],
        )

        async def fake_runner(
            job: RolloutJob, args: argparse.Namespace
        ) -> dict[str, Any]:
            job.output_dir.mkdir(parents=True, exist_ok=True)
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
                },
            }

        args = rc._parser().parse_args(
            [
                "--world-file",
                str(world_file),
                "--output-dir",
                str(tmp_path.joinpath("out")),
                "--campaign-id",
                "dual-cap-inf",
                "--rollouts-per-task",
                "2",
                "--parallelism",
                "2",
                "--parallelism-per-provider",
                "1",
                "--gm-runtime",
                "codex",
                "--gm-model",
                "gpt-6.1-sol",
            ]
        )
        _output_dir, report = await rc.run_campaign(
            args, rollout_runner=fake_runner
        )
        assert report["parallelism"] == 2
        assert report["parallelism_per_provider"] == 1
        assert slots_seen == ["codex", "codex"]
        assert report["status"] == "completed"

    asyncio.run(_run())


def test_validate_parallelism_caps_rejects_per_above_total() -> None:
    from shellsmith.campaign.stream_campaign import (
        validate_parallelism_caps,
    )

    with pytest.raises(RuntimeError, match="parallelism_per_provider"):
        validate_parallelism_caps(parallelism=4, parallelism_per_provider=8)

# ---------------------------------------------------------------------------
# Provider fallback must stay internally consistent across a --retry resume.
#
# The bug these cover: after DISCARD_AND_REDO_BACKUP the group's rollouts live
# under "{task}__backup_{model}" and were produced by the backup GM, but
# the settled result advertised only the primary task id. A later --retry then
# re-derived the world from the CLI namespace (primary) and from world_file.stem
# (primary folder), so top-ups ran on the wrong provider, into the wrong
# directory, and the GRPO group ended up mixing two GM models.
# ---------------------------------------------------------------------------


def test_fallback_reports_backup_storage_identity(tmp_path: Path) -> None:
    """After a fallback, the settled result names the backup folder + model."""

    async def _run() -> None:
        handle = open_campaign(
            tmp_path.joinpath("fb2"),
            OpenCampaignSpec(
                campaign_id="fb2",
                mode="inference",
                provider_outage="fallback",
            ),
        )
        handle.health = ProviderHealthTracker(
            consecutive_failure_threshold=2,
            cooldown_seconds=0.0,
        )
        world_file = tmp_path.joinpath("task.md")
        world_file.write_text("# task\n", encoding="utf-8")

        async def runner(
            job: RolloutJob, args: argparse.Namespace
        ) -> dict[str, Any]:
            job.output_dir.mkdir(parents=True, exist_ok=True)
            if job.gm_model == "gpt-6.1-sol":
                return {
                    "episode_id": job.episode_id,
                    "world_id": job.world_id,
                    "flavor": job.flavor,
                    "attempt": job.attempt,
                    "rollout_dir": str(job.output_dir),
                    "error": "ProviderHTTPError: service unavailable (503)",
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
                    "final_task": {"reward": 0.2 * job.attempt},
                },
            }

        config = StreamPolicyConfig(
            initial_attempts=2,
            maximum_attempts=8,
            minimum_trainable=2,
            provider_outage="fallback",
            campaign_mode="inference",
        )
        result = await collect_adaptive_world(
            handle=handle,
            args=argparse.Namespace(),
            config=config,
            world_file=world_file,
            world_id="demo_task",
            flavor="agentic",
            gm_runtime="codex",
            gm_model="gpt-6.1-sol",
            gm_provider="openrouter",
            gm_reasoning_effort="low",
            rollout_runner=runner,
        )
        # Stream identity stays the slate task; storage identity moves.
        assert result.world_id == "demo_task"
        assert result.rollout_world_id == "demo_task__backup_opus"
        assert result.original_gm_model == "gpt-6.1-sol"
        assert result.gm_model == "opus"
        assert result.gm_fallback_applied is True
        # The advertised storage identity is the folder that really holds them.
        assert handle.root.joinpath(
            "demo_task__backup_opus", "rollout_01"
        ).is_dir()

    asyncio.run(_run())


def test_seeded_retry_tops_up_inside_the_backup_group(tmp_path: Path) -> None:
    """Seeded top-ups keep the backup folder, model, runtime, and provider."""

    async def _run() -> None:
        handle = open_campaign(
            tmp_path.joinpath("fb3"),
            OpenCampaignSpec(campaign_id="fb3", mode="inference"),
        )
        world_file = tmp_path.joinpath("task.md")
        world_file.write_text("# task\n", encoding="utf-8")
        jobs_seen: list[RolloutJob] = []

        async def runner(
            job: RolloutJob, args: argparse.Namespace
        ) -> dict[str, Any]:
            jobs_seen.append(job)
            job.output_dir.mkdir(parents=True, exist_ok=True)
            return {
                "episode_id": job.episode_id,
                "world_id": job.world_id,
                "flavor": job.flavor,
                "attempt": job.attempt,
                "rollout_dir": str(job.output_dir),
                "error": None,
                "audit": {
                    "accepted_for_training": True,
                    "final_task": {"reward": 0.1 * job.attempt},
                },
            }

        config = StreamPolicyConfig(
            initial_attempts=4,
            maximum_attempts=12,
            minimum_trainable=4,
            provider_outage="halt",
            campaign_mode="tinker_train",
        )
        seed = [_rejected(episode_id=f"seed-{index}") for index in range(4)]
        result = await collect_adaptive_world(
            handle=handle,
            args=argparse.Namespace(),
            config=config,
            world_file=world_file,
            world_id="demo_task__backup_opus",
            flavor="agentic",
            gm_runtime="claude",
            gm_model="opus",
            gm_provider="claude-cli",
            gm_reasoning_effort="low",
            gm_fallback_applied=True,
            original_gm_model="gpt-6.1-sol",
            rollout_runner=runner,
            seed_outcomes=seed,
        )
        assert jobs_seen, "retry must schedule at least one top-up attempt"
        # Every new attempt continues the backup group, never the primary.
        assert {job.gm_model for job in jobs_seen} == {"opus"}
        assert {job.gm_runtime for job in jobs_seen} == {"claude"}
        assert {job.gm_provider for job in jobs_seen} == {"claude-cli"}
        assert {job.gm_fallback_applied for job in jobs_seen} == {True}
        assert {job.original_gm_model for job in jobs_seen} == {"gpt-6.1-sol"}
        assert {job.world_id for job in jobs_seen} == {
            "demo_task__backup_opus"
        }
        assert {job.output_dir.parent.name for job in jobs_seen} == {
            "demo_task__backup_opus"
        }
        # Seeded attempts are not re-run; numbering continues after them.
        assert min(job.attempt for job in jobs_seen) == 5
        assert result.rollout_world_id == "demo_task__backup_opus"
        assert result.gm_fallback_applied is True

    asyncio.run(_run())


def test_retry_world_identity_requires_a_recorded_storage_identity() -> None:
    from shellsmith.train.stream_policy import (
        retry_world_identity_from_task_row,
    )

    complete = {
        "rollout_world_id": "demo_task__backup_opus",
        "gm_runtime": "claude",
        "gm_model": "opus",
        "gm_provider": "claude-cli",
        "gm_fallback_applied": True,
        "original_gm_model": "gpt-6.1-sol",
    }
    identity = retry_world_identity_from_task_row(complete)
    assert identity.rollout_world_id == "demo_task__backup_opus"
    assert identity.gm_model == "opus"
    assert identity.original_gm_model == "gpt-6.1-sol"

    with pytest.raises(RuntimeError, match="rollout_world_id"):
        retry_world_identity_from_task_row(
            {key: value for key, value in complete.items() if key != "rollout_world_id"}
        )
    with pytest.raises(RuntimeError, match="original_gm_model"):
        retry_world_identity_from_task_row(
            {**complete, "original_gm_model": None}
        )
    with pytest.raises(RuntimeError, match="gm_fallback_applied"):
        retry_world_identity_from_task_row(
            {**complete, "gm_fallback_applied": None}
        )
    with pytest.raises(RuntimeError, match="gm_provider"):
        retry_world_identity_from_task_row(
            {**complete, "gm_provider": ""}
        )


def test_retry_plan_rejects_a_skip_row_without_storage_identity() -> None:
    from shellsmith.train.stream_policy import (
        build_insufficient_groups_retry_plan,
    )

    state = {
        "status": "insufficient_groups",
        "next_world_offset": 1,
        "steps": [
            {
                "step": 0,
                "start_world_offset": 0,
                "next_world_offset": 1,
                "skip_reason": "insufficient_remaining_complete_groups",
                "tasks": [
                    {
                        "decision": "skip_insufficient",
                        "gm_runtime": "codex",
                        "gm_model": "gpt-6.1-sol",
                        "gm_provider": "codex-cli",
                        "gm_fallback_applied": False,
                        "attempts": [
                            {
                                "accepted_count": 0,
                                "policy_kind": "base",
                                "policy_ref": None,
                                "policy_rft_updates_completed": 0,
                            }
                        ],
                    }
                ],
            }
        ],
    }
    with pytest.raises(RuntimeError, match="rollout_world_id"):
        build_insufficient_groups_retry_plan(state, base_maximum_attempts=8)


# ---------------------------------------------------------------------------
# A provider-down attempt slot must stay retryable across a resume.
#
# The bug these cover: when a GM runtime was marked DOWN, the wave wrote a
# synthetic "episode not started" row to logs/durable_outcome.json and onto the
# campaign scoreboard. Both are consulted on resume, so those rollout_NN slots
# were permanently burned: every later resume replayed the same synthetic
# failure, the group could never reach minimum_trainable, and the world skipped
# deterministically forever.
# ---------------------------------------------------------------------------


def _campaign_with_down_provider(root: Path):
    handle = open_campaign(
        root,
        OpenCampaignSpec(campaign_id=root.name, mode="tinker_train"),
    )
    handle.health = ProviderHealthTracker(
        consecutive_failure_threshold=1,
        cooldown_seconds=0.0,
    )
    for index in range(4):
        handle.health.observe(
            "codex", success=False, error=f"RuntimeError: outage {index}"
        )
    assert handle.health.is_down("codex") is True
    return handle


def test_provider_down_attempt_slots_are_retryable_after_resume(
    tmp_path: Path,
) -> None:
    """Down-provider placeholders never settle; a resume runs them for real."""

    async def _run() -> None:
        from shellsmith.campaign.stream_campaign import (
            load_campaign,
        )
        from shellsmith.train.durable_capture import (
            durable_outcome_path_for_rollout,
        )

        root = tmp_path.joinpath("camp")
        handle = _campaign_with_down_provider(root)
        world_file = tmp_path.joinpath("task.md")
        world_file.write_text("# task\n", encoding="utf-8")
        ran: list[str] = []

        async def runner(
            job: RolloutJob, args: argparse.Namespace
        ) -> dict[str, Any]:
            ran.append(job.episode_id)
            job.output_dir.mkdir(parents=True, exist_ok=True)
            return {
                "episode_id": job.episode_id,
                "world_id": job.world_id,
                "flavor": job.flavor,
                "attempt": job.attempt,
                "rollout_dir": str(job.output_dir),
                "error": None,
                "audit": {
                    "accepted_for_training": True,
                    "final_task": {"reward": 0.3},
                },
            }

        def build(count: int) -> list[RolloutJob]:
            return build_task_attempt_jobs(
                campaign_id=handle.campaign_id,
                campaign_root=root,
                world_file=world_file,
                world_id="demo_task",
                flavor="agentic",
                attempt_start=1,
                attempt_count=count,
                gm_runtime="codex",
                gm_model="gpt-6.1-sol",
                gm_provider="codex-cli",
                gm_reasoning_effort="low",
            )

        blocked = await run_campaign_wave(
            handle, build(2), args=argparse.Namespace(), rollout_runner=runner
        )
        assert ran == [], "a DOWN provider must not start episodes"
        assert all(row["episode_not_started"] is True for row in blocked)
        # Evidence stays on the scoreboard...
        assert len(handle.outcomes) == 2
        # ...but nothing durable settles those attempt slots.
        for job in build(2):
            assert not durable_outcome_path_for_rollout(job.output_dir).is_file()

        # Resume the same campaign root with a healthy provider.
        resumed = load_campaign(root)
        assert resumed.health.is_down("codex") is False
        replayed = await run_campaign_wave(
            resumed, build(2), args=argparse.Namespace(), rollout_runner=runner
        )
        assert len(ran) == 2, "resume must actually run the blocked attempts"
        assert all(row["accepted_count"] == 1 for row in replayed)
        # Exactly one row per episode: the placeholder was replaced, not doubled.
        episode_ids = [str(row["episode_id"]) for row in resumed.outcomes]
        assert sorted(episode_ids) == sorted(set(episode_ids))
        assert len(episode_ids) == 2

    asyncio.run(_run())


def test_settled_durable_outcomes_are_replayed_not_rerun(tmp_path: Path) -> None:
    """A finished episode is replayed from disk; its accept still counts."""

    async def _run() -> None:
        from shellsmith.campaign.stream_campaign import (
            load_campaign,
        )

        root = tmp_path.joinpath("camp2")
        handle = open_campaign(
            root, OpenCampaignSpec(campaign_id="camp2", mode="tinker_train")
        )
        world_file = tmp_path.joinpath("task.md")
        world_file.write_text("# task\n", encoding="utf-8")
        run_count = {"n": 0}

        async def runner(
            job: RolloutJob, args: argparse.Namespace
        ) -> dict[str, Any]:
            run_count["n"] += 1
            job.output_dir.mkdir(parents=True, exist_ok=True)
            return {
                "episode_id": job.episode_id,
                "world_id": job.world_id,
                "flavor": job.flavor,
                "attempt": job.attempt,
                "rollout_dir": str(job.output_dir),
                "error": None,
                "audit": {
                    "accepted_for_training": True,
                    "final_task": {"reward": 0.42},
                },
            }

        def build() -> list[RolloutJob]:
            return build_task_attempt_jobs(
                campaign_id=handle.campaign_id,
                campaign_root=root,
                world_file=world_file,
                world_id="demo_task",
                flavor="agentic",
                attempt_start=1,
                attempt_count=2,
                gm_runtime="codex",
                gm_model="gpt-6.1-sol",
                gm_provider="codex-cli",
                gm_reasoning_effort="low",
            )

        await run_campaign_wave(
            handle, build(), args=argparse.Namespace(), rollout_runner=runner
        )
        assert run_count["n"] == 2

        resumed = load_campaign(root)
        replayed = await run_campaign_wave(
            resumed, build(), args=argparse.Namespace(), rollout_runner=runner
        )
        assert run_count["n"] == 2, "settled episodes must not be re-run"
        assert [row["accepted_count"] for row in replayed] == [1, 1]
        assert len(resumed.outcomes) == 2

    asyncio.run(_run())


def test_write_durable_outcome_refuses_a_never_started_episode(
    tmp_path: Path,
) -> None:
    from shellsmith.train.durable_capture import (
        write_durable_outcome,
    )

    with pytest.raises(RuntimeError, match="never started"):
        write_durable_outcome(
            tmp_path,
            {"episode_id": "ep-1", "episode_not_started": True},
        )


# ---------------------------------------------------------------------------
# HALT_STREAM must not take healthy siblings down with it.
#
# The bug: collect_update_worlds walked the asyncio.wait() batch and `break`ed
# on the first HALT_STREAM result. Every other task in the SAME batch had also
# already produced a real result, but it was still sitting in `pending`, so the
# subsequent cancel-and-clear threw those results away. The trainer's own loop
# had the mirror-image bug (`break` on the first halted row). Either way a world
# that finished normally vanished from the halted step, while the stream cursor
# advanced past it — so nothing ever revisited it.
# ---------------------------------------------------------------------------


def _world_task_runs(tmp_path: Path, names: tuple[str, ...]):
    task_runs: list[tuple[Path, argparse.Namespace]] = []
    for name in names:
        path = tmp_path.joinpath(f"{name}.md")
        path.write_text("# t\n", encoding="utf-8")
        task_runs.append(
            (
                path,
                argparse.Namespace(
                    initial_attempts=4,
                    maximum_attempts=8,
                    minimum_trainable=4,
                    minimum_reward_range=0.15,
                    low_spread_top_up=4,
                ),
            )
        )
    return task_runs


def test_halt_keeps_siblings_that_settled_in_the_same_batch(
    tmp_path: Path,
) -> None:
    """A world that finished alongside the halting world is still banked."""

    async def _run() -> None:
        release = asyncio.Event()

        async def collect_one(
            world_file: Path,
            task_args: argparse.Namespace,
            world_config: StreamPolicyConfig,
            flavor: str,
        ) -> WorldCollectResult:
            world_id = world_file.stem
            # Both worlds settle in the same asyncio.wait() batch.
            await release.wait()
            if world_id == "halting":
                return WorldCollectResult(
                    world_id=world_id,
                    rollout_world_id=world_id,
                    decision=WorldDecision(
                        action=WorldAction.HALT_STREAM,
                        reason="provider down",
                    ),
                    outcomes=[_rejected(infra=True, episode_id="halt-1")],
                    gm_runtime="codex",
                    gm_model="gpt-6.1-sol",
                    gm_provider="codex-cli",
                )
            return WorldCollectResult(
                world_id=world_id,
                rollout_world_id=world_id,
                decision=WorldDecision(
                    action=WorldAction.GROUP_READY, reason="banked"
                ),
                outcomes=[
                    _accepted(0.1 * index, episode_id=f"{world_id}-{index}")
                    for index in range(4)
                ],
                gm_runtime="claude",
                gm_model="sonnet",
                gm_provider="claude-cli",
            )

        task_runs = _world_task_runs(tmp_path, ("halting", "healthy"))
        config = StreamPolicyConfig(
            groups_per_update=2,
            campaign_mode="tinker_train",
            parallelism=8,
            parallelism_per_provider=4,
        )

        async def release_soon() -> None:
            await asyncio.sleep(0.02)
            release.set()

        update, _ = await asyncio.gather(
            collect_update_worlds(
                task_runs=task_runs,
                start_offset=0,
                groups_per_update=2,
                base_config=config,
                classify_flavor=lambda _path: "agentic",
                collect_one=collect_one,
            ),
            release_soon(),
        )

        assert update.halted is True
        banked = {item.result.world_id: item.result for item in update.worlds}
        assert set(banked) == {"halting", "healthy"}, (
            "the healthy sibling settled in the same batch and must survive"
        )
        assert banked["healthy"].decision.action == WorldAction.GROUP_READY
        assert len(banked["healthy"].outcomes) == 4

    asyncio.run(_run())


def test_halt_banks_a_peer_that_settled_while_being_cancelled(
    tmp_path: Path,
) -> None:
    """A slower peer that finished before the cancel landed is not discarded."""

    async def _run() -> None:
        halt_first = asyncio.Event()

        async def collect_one(
            world_file: Path,
            task_args: argparse.Namespace,
            world_config: StreamPolicyConfig,
            flavor: str,
        ) -> WorldCollectResult:
            world_id = world_file.stem
            if world_id == "halting":
                return WorldCollectResult(
                    world_id=world_id,
                    rollout_world_id=world_id,
                    decision=WorldDecision(
                        action=WorldAction.HALT_STREAM, reason="provider down"
                    ),
                    outcomes=[_rejected(infra=True, episode_id="halt-1")],
                    gm_runtime="codex",
                    gm_model="gpt-6.1-sol",
                    gm_provider="codex-cli",
                )
            # Settles on the very next event-loop pass: already done by the
            # time the halt path tries to cancel it.
            await asyncio.sleep(0)
            halt_first.set()
            return WorldCollectResult(
                world_id=world_id,
                rollout_world_id=world_id,
                decision=WorldDecision(
                    action=WorldAction.GROUP_READY, reason="banked"
                ),
                outcomes=[
                    _accepted(0.1 * index, episode_id=f"{world_id}-{index}")
                    for index in range(4)
                ],
                gm_runtime="claude",
                gm_model="sonnet",
                gm_provider="claude-cli",
            )

        task_runs = _world_task_runs(tmp_path, ("halting", "healthy"))
        config = StreamPolicyConfig(
            groups_per_update=2,
            campaign_mode="tinker_train",
            parallelism=8,
            parallelism_per_provider=4,
        )
        update = await collect_update_worlds(
            task_runs=task_runs,
            start_offset=0,
            groups_per_update=2,
            base_config=config,
            classify_flavor=lambda _path: "agentic",
            collect_one=collect_one,
        )
        assert update.halted is True
        ids = {item.result.world_id for item in update.worlds}
        assert "halting" in ids
        assert "healthy" in ids, (
            "a peer that already settled must be banked, not cancelled away"
        )

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Provider-slot accounting: a failed or blocked attempt must free its capacity.
#
# Two bugs covered:
#   1. `wait_until_available` raises when the runtime is marked DOWN *while a
#      job waits out a cooldown*. That escaped `bounded`, so the whole wave
#      raised instead of recording a "never started" attempt.
#   2. The wave fanned out with a bare asyncio.gather(). One raising job made
#      the exception surface immediately while every sibling kept running as an
#      orphan task, still holding its DualCapPool slot.
# ---------------------------------------------------------------------------


def test_runtime_marked_down_during_cooldown_does_not_abort_the_wave(
    tmp_path: Path,
) -> None:
    """A job blocked behind a cooldown that ends in DOWN becomes not-started."""

    async def _run() -> None:
        root = tmp_path.joinpath("slots")
        handle = open_campaign(
            root,
            OpenCampaignSpec(
                campaign_id="slots",
                mode="tinker_train",
                parallelism=2,
                parallelism_per_provider=1,
            ),
        )
        tracker = ProviderHealthTracker(
            consecutive_failure_threshold=1,
            cooldown_seconds=60.0,
        )

        async def sleep_then_trip(_seconds: float) -> None:
            """Stand in for "a sibling job finished while we napped"."""
            state = tracker.state("codex")
            state.cooldown_until_monotonic = tracker.time_fn()
            tracker.observe("codex", success=False, error="RuntimeError: 503")

        tracker.sleep_fn = sleep_then_trip
        # First trip → COOLDOWN. The wave's up-front is_down() check passes.
        tracker.observe("codex", success=False, error="RuntimeError: 503")
        assert tracker.is_in_cooldown("codex") is True
        assert tracker.is_down("codex") is False
        handle.health = tracker

        world_file = tmp_path.joinpath("task.md")
        world_file.write_text("# task\n", encoding="utf-8")
        started: list[str] = []

        async def runner(
            job: RolloutJob, args: argparse.Namespace
        ) -> dict[str, Any]:
            started.append(job.episode_id)
            job.output_dir.mkdir(parents=True, exist_ok=True)
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
                },
            }

        jobs = build_task_attempt_jobs(
            campaign_id=handle.campaign_id,
            campaign_root=root,
            world_file=world_file,
            world_id="demo_task",
            flavor="agentic",
            attempt_start=1,
            attempt_count=2,
            gm_runtime="codex",
            gm_model="gpt-6.1-sol",
            gm_provider="codex-cli",
            gm_reasoning_effort="low",
        )
        # The wave must settle, not raise out of wait_until_available.
        outcomes = await run_campaign_wave(
            handle, jobs, args=argparse.Namespace(), rollout_runner=runner
        )
        assert tracker.is_down("codex") is True
        assert started == []
        assert all(row["episode_not_started"] is True for row in outcomes)
        # Nothing is holding a provider slot afterwards.
        assert handle.pool._in_flight_total == 0
        assert handle.pool.has_capacity("codex") is True

    asyncio.run(_run())


def test_failed_wave_releases_every_sibling_provider_slot(tmp_path: Path) -> None:
    """One raising job cancels the rest instead of orphaning their slots."""

    async def _run() -> None:
        root = tmp_path.joinpath("slots2")
        handle = open_campaign(
            root,
            OpenCampaignSpec(
                campaign_id="slots2",
                mode="tinker_train",
                parallelism=4,
                parallelism_per_provider=4,
            ),
        )
        world_file = tmp_path.joinpath("task.md")
        world_file.write_text("# task\n", encoding="utf-8")
        cancelled: list[str] = []

        async def runner(
            job: RolloutJob, args: argparse.Namespace
        ) -> dict[str, Any]:
            job.output_dir.mkdir(parents=True, exist_ok=True)
            if job.attempt == 1:
                # accepted_count outside {0, 1} makes _annotate_outcome raise,
                # which is *outside* the wave's per-job error handling.
                return {
                    "episode_id": job.episode_id,
                    "world_id": job.world_id,
                    "flavor": job.flavor,
                    "attempt": job.attempt,
                    "rollout_dir": str(job.output_dir),
                    "error": None,
                    "audit": None,
                    "accepted_count": 5,
                }
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                cancelled.append(job.episode_id)
                raise
            return {}

        jobs = build_task_attempt_jobs(
            campaign_id=handle.campaign_id,
            campaign_root=root,
            world_file=world_file,
            world_id="demo_task",
            flavor="agentic",
            attempt_start=1,
            attempt_count=3,
            gm_runtime="codex",
            gm_model="gpt-6.1-sol",
            gm_provider="codex-cli",
            gm_reasoning_effort="low",
        )
        with pytest.raises(RuntimeError, match="accepted_count"):
            await run_campaign_wave(
                handle, jobs, args=argparse.Namespace(), rollout_runner=runner
            )
        # Siblings were cancelled, not abandoned, and their slots came back.
        assert len(cancelled) == 2
        assert handle.pool._in_flight_total == 0
        assert handle.pool._in_flight_provider == {}

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Discard-and-redo must withdraw the discarded group's training claim.
#
# The bug: the fallback path dropped the in-memory captures and the
# capture_index rows, but logs/durable_outcome.json and the campaign scoreboard
# still said accepted_count: 1. On resume the wave replayed those rows, the
# adaptive policy banked the group, and captures_for_outcomes died with
# "Accepted outcome ... has no capture in the store" — collection wedged.
# ---------------------------------------------------------------------------


def test_fallback_discard_withdraws_the_training_claim(tmp_path: Path) -> None:
    """Discarded episodes survive as evidence but stop claiming acceptance."""

    async def _run() -> None:
        from shellsmith.campaign.stream_campaign import (
            load_campaign,
        )
        from shellsmith.train.durable_capture import (
            durable_outcome_path_for_rollout,
        )

        root = tmp_path.joinpath("fb4")
        handle = open_campaign(
            root,
            OpenCampaignSpec(
                campaign_id="fb4",
                mode="tinker_train",
                provider_outage="fallback",
            ),
        )
        handle.health = ProviderHealthTracker(
            consecutive_failure_threshold=2,
            cooldown_seconds=0.0,
        )
        world_file = tmp_path.joinpath("task.md")
        world_file.write_text("# task\n", encoding="utf-8")
        store = CaptureStore(_run_dir=root)

        async def runner(
            job: RolloutJob, args: argparse.Namespace
        ) -> dict[str, Any]:
            job.output_dir.mkdir(parents=True, exist_ok=True)
            if job.gm_model == "gpt-6.1-sol":
                # Accepted by the audit AND infrastructure-failing: exactly the
                # shape that used to survive the discard with a live capture
                # claim and no capture.
                outcome = {
                    "episode_id": job.episode_id,
                    "world_id": job.world_id,
                    "flavor": job.flavor,
                    "attempt": job.attempt,
                    "rollout_dir": str(job.output_dir),
                    "error": "ProviderHTTPError: service unavailable (503)",
                    "accepted_count": 1,
                    "audit": {
                        "accepted_for_training": True,
                        "final_task": {"reward": 0.2 * job.attempt},
                    },
                }
                store.put(
                    CapturedRollout(
                        rollout_root=job.output_dir,
                        group_identity_key="primary-key",
                        captures=(),
                    ),
                    episode_id=job.episode_id,
                )
                return outcome
            return {
                "episode_id": job.episode_id,
                "world_id": job.world_id,
                "flavor": job.flavor,
                "attempt": job.attempt,
                "rollout_dir": str(job.output_dir),
                "error": None,
                "accepted_count": 1,
                "audit": {
                    "accepted_for_training": True,
                    "final_task": {"reward": 0.2 * job.attempt},
                },
            }

        config = StreamPolicyConfig(
            initial_attempts=2,
            maximum_attempts=8,
            minimum_trainable=4,
            provider_outage="fallback",
            campaign_mode="tinker_train",
        )
        result = await collect_adaptive_world(
            handle=handle,
            args=argparse.Namespace(),
            config=config,
            world_file=world_file,
            world_id="demo_task",
            flavor="agentic",
            gm_runtime="codex",
            gm_model="gpt-6.1-sol",
            gm_provider="openrouter",
            gm_reasoning_effort="low",
            rollout_runner=runner,
            capture_store=store,
        )
        assert result.gm_fallback_applied is True
        assert result.discarded_outcomes

        discarded_ids = {
            str(row["episode_id"]) for row in result.discarded_outcomes
        }
        assert discarded_ids
        for episode_id in discarded_ids:
            assert store.get(episode_id) is None

        # The durable files stay, but no longer claim training acceptance.
        for row in result.discarded_outcomes:
            assert row["accepted_count"] == 0
            assert row["discarded_by_provider_fallback"] is True
            path = durable_outcome_path_for_rollout(Path(row["rollout_dir"]))
            assert path.is_file(), "evidence must be preserved, not deleted"
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["outcome"]["accepted_count"] == 0
            assert payload["outcome"]["discarded_by_provider_fallback"] is True
            # The audit is untouched: the episode really did run.
            assert payload["outcome"]["audit"]["accepted_for_training"] is True

        # A resume reads the scoreboard: no discarded row claims acceptance,
        # so nothing can bank a group whose captures are gone.
        resumed = load_campaign(root)
        for row in resumed.outcomes:
            if str(row.get("episode_id")) in discarded_ids:
                assert row["accepted_count"] == 0
        empty_store = CaptureStore(_run_dir=root)
        replayed = [
            row
            for row in resumed.outcomes
            if str(row.get("episode_id")) in discarded_ids
        ]
        # Would previously raise "has no capture in the store".
        assert empty_store.captures_for_outcomes(replayed) == ()

    asyncio.run(_run())
