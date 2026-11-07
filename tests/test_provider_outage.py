"""Unit tests for campaign-owned provider health + backup remap."""

from __future__ import annotations

import asyncio

import pytest

from shellsmith.campaign.provider_outage import (
    ProviderHealthEvent,
    ProviderHealthStatus,
    ProviderHealthTracker,
    apply_backup_to_job_fields,
    episode_counts_as_provider_failure,
    provider_infrastructure_failure,
    resolve_backup_world,
)


def test_bidirectional_sol_opus_and_terra_sonnet() -> None:
    assert resolve_backup_world(
        runtime="codex", model="gpt-6.1-sol", provider="codex-cli"
    ) == resolve_backup_world(runtime="x", model="sol", provider="y")
    sol_backup = resolve_backup_world(
        runtime="codex", model="gpt-6.1-sol", provider="codex-cli"
    )
    assert (sol_backup.runtime, sol_backup.model, sol_backup.provider) == (
        "claude",
        "opus",
        "claude-cli",
    )
    opus_backup = resolve_backup_world(
        runtime="claude", model="opus", provider="claude-cli"
    )
    assert (opus_backup.runtime, opus_backup.model, opus_backup.provider) == (
        "codex",
        "gpt-6.1-sol",
        "codex-cli",
    )
    terra_backup = resolve_backup_world(
        runtime="codex", model="gpt-5.6-terra", provider="codex-cli"
    )
    assert (terra_backup.runtime, terra_backup.model, terra_backup.provider) == (
        "claude",
        "sonnet",
        "claude-cli",
    )
    sonnet_backup = resolve_backup_world(
        runtime="claude", model="sonnet", provider="claude-cli"
    )
    assert (sonnet_backup.runtime, sonnet_backup.model, sonnet_backup.provider) == (
        "codex",
        "gpt-5.6-terra",
        "codex-cli",
    )


def test_unknown_model_raises() -> None:
    with pytest.raises(RuntimeError, match="No provider-outage backup"):
        resolve_backup_world(runtime="pi", model="meta/muse-spark-1.1", provider="x")


def test_hard_error_counts_regardless_of_message() -> None:
    assert provider_infrastructure_failure(
        "You've hit your monthly spend limit"
    ) is True
    assert provider_infrastructure_failure("totally novel boom") is True
    assert provider_infrastructure_failure(None) is False
    assert provider_infrastructure_failure("") is False


def test_streak_enters_cooldown_then_down_with_fake_clock() -> None:
    now = {"t": 0.0}

    def time_fn() -> float:
        return now["t"]

    tracker = ProviderHealthTracker(
        consecutive_failure_threshold=5,
        cooldown_seconds=300.0,
        time_fn=time_fn,
    )
    for index in range(4):
        result = tracker.observe(
            "claude", success=False, error=f"fail-{index}"
        )
        assert result.event == ProviderHealthEvent.FAILURE_COUNTED
        assert tracker.is_in_cooldown("claude") is False
        assert tracker.is_down("claude") is False

    trip = tracker.observe("claude", success=False, error="fail-4")
    assert trip.event == ProviderHealthEvent.ENTERED_COOLDOWN
    assert tracker.is_in_cooldown("claude") is True
    assert tracker.state("claude").trip_count == 1

    # In-flight failures during cooldown do not start a second trip.
    ignored = tracker.observe("claude", success=False, error="during-cool")
    assert ignored.event == ProviderHealthEvent.IGNORED_DURING_COOLDOWN
    assert tracker.is_down("claude") is False

    now["t"] = 301.0
    assert tracker.is_in_cooldown("claude") is False

    for index in range(4):
        tracker.observe("claude", success=False, error=f"again-{index}")
    down = tracker.observe("claude", success=False, error="again-4")
    assert down.event == ProviderHealthEvent.MARKED_DOWN
    assert tracker.is_down("claude") is True


def test_non_provider_failure_is_neutral_and_does_not_reset_streak() -> None:
    tracker = ProviderHealthTracker(consecutive_failure_threshold=5)
    tracker.observe("claude", success=False, error="provider fail")

    result = tracker.observe_outcome(
        {
            "gm_runtime": "claude",
            "error": "ExecutionTimeoutError: embedded prompt exceeded deadline",
            "provider_attributable": False,
        }
    )

    assert result.event == ProviderHealthEvent.IGNORED_NON_PROVIDER_FAILURE
    assert tracker.state("claude").consecutive_failures == 1
    assert episode_counts_as_provider_failure(
        {
            "error": "ExecutionTimeoutError: world turn exceeded deadline",
            "provider_attributable": True,
        }
    ) is True
    assert episode_counts_as_provider_failure(
        {"error": "TimeoutError: ", "provider_attributable": False}
    ) is False


def test_success_resets_streak() -> None:
    tracker = ProviderHealthTracker(consecutive_failure_threshold=5)
    for _ in range(4):
        tracker.observe("codex", success=False, error="x")
    ok = tracker.observe("codex", success=True)
    assert ok.event == ProviderHealthEvent.OK
    assert tracker.state("codex").consecutive_failures == 0
    tracker.observe("codex", success=False, error="y")
    assert tracker.state("codex").consecutive_failures == 1
    assert tracker.state("codex").status == ProviderHealthStatus.HEALTHY


def test_runtimes_are_isolated() -> None:
    tracker = ProviderHealthTracker(consecutive_failure_threshold=2)
    tracker.observe("claude", success=False, error="a")
    tracker.observe("claude", success=False, error="b")
    assert tracker.is_in_cooldown("claude") is True
    assert tracker.is_in_cooldown("codex") is False
    tracker.observe("codex", success=False, error="c")
    assert tracker.state("codex").consecutive_failures == 1


def test_wait_until_available_sleeps_through_cooldown() -> None:
    now = {"t": 0.0}
    slept: list[float] = []

    def time_fn() -> float:
        return now["t"]

    async def sleep_fn(seconds: float) -> None:
        slept.append(seconds)
        now["t"] += seconds

    tracker = ProviderHealthTracker(
        consecutive_failure_threshold=2,
        cooldown_seconds=10.0,
        time_fn=time_fn,
        sleep_fn=sleep_fn,
    )
    tracker.observe("claude", success=False, error="1")
    tracker.observe("claude", success=False, error="2")

    async def _run() -> None:
        await tracker.wait_until_available("claude")

    asyncio.run(_run())
    assert slept == [10.0]
    assert tracker.is_in_cooldown("claude") is False


def test_apply_backup_refuses_second_hop() -> None:
    backup, original = apply_backup_to_job_fields(
        gm_runtime="codex",
        gm_model="gpt-6.1-sol",
        gm_provider="codex-cli",
        gm_fallback_applied=False,
    )
    assert original == "gpt-6.1-sol"
    assert backup.model == "opus"
    with pytest.raises(RuntimeError, match="already applied"):
        apply_backup_to_job_fields(
            gm_runtime=backup.runtime,
            gm_model=backup.model,
            gm_provider=backup.provider,
            gm_fallback_applied=True,
        )
