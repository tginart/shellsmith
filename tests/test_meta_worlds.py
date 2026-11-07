"""Purpose: Protect the boundary between an inner simulator and its outer episode.
Input: Representative nested simulator artifacts.
Output: Contract checks for artifact attribution and outer integrity.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shellsmith.episode.world_artifacts import (
    load_final_response,
    load_turn_response,
)


@pytest.mark.parametrize(
    ("inner_trainable", "turn_reward"),
    [(False, 0.5), (True, -0.8)],
)
def test_inner_integrity_decision_does_not_control_outer_trainability(
    tmp_path: Path,
    inner_trainable: bool,
    turn_reward: float,
) -> None:
    """Purpose: Preserve inner judgments as task data under the outer wire contract.
    Input: A correct or incorrect inner judgment and its outer task reward.
    Output: A trainable outer artifact whose inner judgment remains unchanged.
    """
    inner = {
        "trainable": inner_trainable,
        "consistency_review": {"conflicts": ["Inner counts disagree."]},
    }
    payload = _outer_turn(inner=inner, turn_reward=turn_reward)
    path = tmp_path / "response.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    artifact = load_turn_response(path, expected_action_id="outer-read-1")

    assert artifact.trainable is True
    assert artifact.terminate_episode is False
    assert artifact.consistency_review.conflicts == ()
    assert artifact.turn_reward == turn_reward
    assert json.loads(artifact.tool_response.content) == inner


def test_real_outer_conflict_still_requires_outer_integrity_failure(
    tmp_path: Path,
) -> None:
    payload = _outer_turn(inner={"trainable": True}, turn_reward=0.0)
    payload["consistency_review"]["conflicts"] = ["Outer state changed silently."]
    path = tmp_path / "response.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="trainable must be false"):
        load_turn_response(path, expected_action_id="outer-read-1")


def test_outer_final_preserves_a_nested_inner_integrity_failure(
    tmp_path: Path,
) -> None:
    """Purpose: Keep an inner final judgment separate from outer final integrity.
    Input: A coherent outer final with a nested inner integrity failure.
    Output: Independent outer reward and trainability plus the unchanged inner data.
    """
    inner = {
        "final_reward": 0.1,
        "trainable": False,
        "factual_consistency": {
            "passed": False,
            "contradictions": ["Inner counts disagree."],
        },
    }
    payload = {
        "final_reward": 0.8,
        "completion_percentage": 100,
        "trainable": True,
        "criteria": [],
        "process_deductions": [],
        "commentary": "The operator correctly identified the inner failure.",
        "integrity_issues": [],
        "factual_consistency": {"passed": True, "contradictions": []},
        "task_outcomes": {"inner_simulator": inner},
    }
    path = tmp_path / "response.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    artifact = load_final_response(path)

    assert artifact.final_reward == 0.8
    assert artifact.trainable is True
    assert artifact.factual_consistency_passed is True
    assert artifact.factual_contradictions == ()
    assert artifact.task_outcomes["inner_simulator"] == inner


def _outer_turn(*, inner: dict, turn_reward: float) -> dict:
    """Purpose: Build one coherent outer read of a Player-authored inner artifact.
    Input: The inner artifact and the outer reward for its author.
    Output: A complete outer response with no outer consistency conflict.
    """
    return {
        "action_id": "outer-read-1",
        "tool_response": {"content": json.dumps(inner), "is_error": False},
        "turn_reward": turn_reward,
        "difficulty_rating": 0.5,
        "completion_percentage": 50,
        "commentary": "The read preserves the submitted inner artifact.",
        "reasoning_assessment": None,
        "consistency_review": {
            "reviewed_revision": 0,
            "observation_claims": ["The artifact contains the submitted bytes."],
            "new_commitments": [],
            "conflicts": [],
            "executed_cross_checks": [],
        },
        "trainable": True,
        "integrity_issues": [],
        "terminate_episode": False,
        "termination_reason": None,
    }
