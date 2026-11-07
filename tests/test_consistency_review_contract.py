"""Contract tests for the required per-turn consistency-review receipt.

Architecture map:
    world response JSON
      -> schema requires one complete consistency_review object
      -> artifact parser rejects missing or malformed receipts
      -> validated immutable receipt remains private beside the observation

Why this exists:
    Semantic consistency enforcement is implemented elsewhere. This file pins the
    hard protocol boundary so neither old responses nor partial review claims can
    silently enter that enforcement layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shellsmith.episode.schemas import (
    CONSISTENCY_REVIEW_SCHEMA,
    MAX_TOOL_RESPONSE_CHARS,
    TOOL_RESPONSE_SCHEMA,
    TURN_RESPONSE_SCHEMA,
)
from shellsmith.episode.world_artifacts import (
    load_turn_response,
)


def _response() -> dict:
    return {
        "action_id": "call-1",
        "tool_response": {"content": "42 rows\n", "is_error": False},
        "turn_reward": 0.4,
        "difficulty_rating": 0.5,
        "completion_percentage": 25,
        "commentary": "The query established a stable count.",
        "reasoning_assessment": None,
        "consistency_review": {
            "reviewed_revision": 3,
            "observation_claims": ["The bounded query returned 42 rows."],
            "new_commitments": ["The affected-row count is 42."],
            "conflicts": [],
            "executed_cross_checks": ["Compared the query count with the manifest."],
        },
        "trainable": True,
        "integrity_issues": [],
        "terminate_episode": False,
        "termination_reason": None,
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_turn_schema_requires_complete_consistency_review() -> None:
    assert "consistency_review" in TURN_RESPONSE_SCHEMA["required"]
    assert (
        TURN_RESPONSE_SCHEMA["properties"]["consistency_review"]
        is CONSISTENCY_REVIEW_SCHEMA
    )
    assert CONSISTENCY_REVIEW_SCHEMA["required"] == [
        "reviewed_revision",
        "observation_claims",
        "new_commitments",
        "conflicts",
        "executed_cross_checks",
    ]
    assert CONSISTENCY_REVIEW_SCHEMA["additionalProperties"] is False
    assert TOOL_RESPONSE_SCHEMA["properties"]["content"]["maxLength"] == 48_000


def test_loader_rejects_tool_response_too_large_for_pi_transport(
    tmp_path: Path,
) -> None:
    payload = _response()
    payload["tool_response"]["content"] = "x" * (MAX_TOOL_RESPONSE_CHARS + 1)
    path = tmp_path / "response.json"
    _write(path, payload)

    with pytest.raises(RuntimeError, match="transport ceiling"):
        load_turn_response(path, expected_action_id="call-1")


def test_loader_preserves_complete_consistency_review(tmp_path: Path) -> None:
    path = tmp_path / "response.json"
    _write(path, _response())

    artifact = load_turn_response(path, expected_action_id="call-1")

    assert artifact.consistency_review.reviewed_revision == 3
    assert artifact.consistency_review.observation_claims == (
        "The bounded query returned 42 rows.",
    )
    assert artifact.consistency_review.new_commitments == (
        "The affected-row count is 42.",
    )
    assert artifact.consistency_review.conflicts == ()
    assert artifact.consistency_review.executed_cross_checks == (
        "Compared the query count with the manifest.",
    )
    assert "consistency_review" not in artifact.metadata


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("consistency_review"), "must be a JSON object"),
        (
            lambda payload: payload["consistency_review"].pop("conflicts"),
            "conflicts must be a string array",
        ),
        (
            lambda payload: payload["consistency_review"].update(
                {"reviewed_revision": -1}
            ),
            "reviewed_revision must be a non-negative integer",
        ),
        (
            lambda payload: payload["consistency_review"].update(
                {"reviewed_revision": 1.5}
            ),
            "reviewed_revision must be a non-negative integer",
        ),
        (
            lambda payload: payload["consistency_review"].update(
                {"executed_cross_checks": ["ran", 7]}
            ),
            "executed_cross_checks must be a string array",
        ),
    ],
)
def test_loader_rejects_missing_or_malformed_consistency_review(
    tmp_path: Path, mutation, message: str
) -> None:
    payload = _response()
    mutation(payload)
    path = tmp_path / "response.json"
    _write(path, payload)

    with pytest.raises(RuntimeError, match=message):
        load_turn_response(path, expected_action_id="call-1")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["tool_response"].update(
                {"content": "", "is_error": True}
            ),
            "must describe the in-world failure",
        ),
        (
            lambda payload: payload["consistency_review"].update(
                {"conflicts": ["Two authoritative counts disagree."]}
            ),
            "trainable must be false",
        ),
        (
            lambda payload: payload.update(
                {
                    "trainable": False,
                    "integrity_issues": ["World state became inconsistent."],
                }
            ),
            "terminate_episode must be true",
        ),
        (
            lambda payload: payload.update(
                {
                    "trainable": False,
                    "terminate_episode": True,
                    "termination_reason": "World state became inconsistent.",
                    "integrity_issues": [],
                }
            ),
            "integrity_issues must explain",
        ),
    ],
)
def test_loader_rejects_internally_contradictory_turns(
    tmp_path: Path, mutation, message: str
) -> None:
    payload = _response()
    mutation(payload)
    path = tmp_path / "response.json"
    _write(path, payload)

    with pytest.raises(RuntimeError, match=message):
        load_turn_response(path, expected_action_id="call-1")
