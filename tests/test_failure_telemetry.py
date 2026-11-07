"""Failure telemetry must preserve the boundary that actually timed out."""

from __future__ import annotations

import json

from shellsmith.episode.failure import (
    ExecutionTimeoutError,
    outcome_failure_fields,
    read_episode_failure_detail,
    serialize_failure,
)


def test_annotated_timeout_preserves_cause_stage_and_attribution(tmp_path) -> None:
    try:
        try:
            raise TimeoutError()
        except TimeoutError as cause:
            raise ExecutionTimeoutError(
                component="player",
                stage="player_prompt",
                timeout_seconds=900.0,
                provider_attributable=False,
            ) from cause
    except ExecutionTimeoutError as error:
        detail = serialize_failure(error, stage="campaign_job")

    assert detail["error_type"] == "ExecutionTimeoutError"
    assert detail["failure_stage"] == "player_prompt"
    assert detail["failure_component"] == "player"
    assert detail["timeout_seconds"] == 900.0
    assert detail["provider_attributable"] is False
    assert detail["cause_type"] == "TimeoutError"
    assert "raise ExecutionTimeoutError" in detail["traceback"]

    episode_path = tmp_path / "episode.json"
    episode_path.write_text(
        json.dumps({"failure_detail": detail}), encoding="utf-8"
    )
    restored = read_episode_failure_detail(episode_path)
    assert restored == detail
    assert outcome_failure_fields(restored)["failure_stage"] == "player_prompt"


def test_inner_world_timeout_cannot_be_relabelled_by_outer_vat_timeout() -> None:
    try:
        try:
            raise ExecutionTimeoutError(
                component="gm",
                stage="world_turn",
                timeout_seconds=120.0,
                provider_attributable=True,
            )
        except ExecutionTimeoutError as world_error:
            raise ExecutionTimeoutError(
                component="player",
                stage="player_prompt",
                timeout_seconds=900.0,
                provider_attributable=False,
            ) from world_error
    except ExecutionTimeoutError as error:
        detail = serialize_failure(error, stage="player")

    assert detail["failure_stage"] == "world_turn"
    assert detail["failure_component"] == "gm"
    assert detail["timeout_seconds"] == 120.0
    assert detail["provider_attributable"] is True


def test_world_timeout_is_explicitly_provider_attributable() -> None:
    error = ExecutionTimeoutError(
        component="gm",
        stage="world_turn",
        timeout_seconds=120.0,
        provider_attributable=True,
    )

    detail = serialize_failure(error, stage="player")

    assert detail["failure_stage"] == "world_turn"
    assert detail["failure_component"] == "gm"
    assert detail["provider_attributable"] is True
