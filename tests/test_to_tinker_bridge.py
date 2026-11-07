"""Focused tests for live shellsmith to Tinker datum conversion."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shellsmith.credit.to_tinker_bridge import (
    CapturedRollout,
    CapturedModelTurn,
    build_tinker_group,
    captures_from_vat_log,
    feedback_for_capture,
)


class FakeTensorData:
    @staticmethod
    def from_torch(value: object) -> object:
        return value


class FakeTorch:
    @staticmethod
    def tensor(value: object) -> object:
        return value


class FakePrompt:
    def __init__(self, tokens: list[int]) -> None:
        self.tokens = tokens

    @property
    def length(self) -> int:
        return len(self.tokens)

    def append(self, chunk: object) -> "FakePrompt":
        return FakePrompt(self.tokens + list(getattr(chunk, "tokens")))


@dataclass
class FakeEncodedTextChunk:
    tokens: list[int]


@dataclass
class FakeDatum:
    model_input: object
    loss_fn_inputs: dict[str, object]


@dataclass
class FakeRecord:
    request_id: str
    prompt: FakePrompt
    sampled_tokens: list[int]
    sampled_logprobs: list[float]
    prompt_tokens: int = 3
    created_at: float = 1.0
    content: str = "action"
    finish_reason: str = "tool_calls"
    payload: dict[str, object] | None = None
    hydrations: tuple[object, ...] = ()
    forced_close_token_span: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.payload is None:
            self.payload = {}


DEPS = SimpleNamespace(
    torch=FakeTorch(),
    TensorData=FakeTensorData,
    types=SimpleNamespace(EncodedTextChunk=FakeEncodedTextChunk, Datum=FakeDatum),
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def rollout(
    tmp_path: Path,
    *,
    index: int,
    turn_reward: float,
    group_key: str = "exact-group",
    with_reasoning: bool = False,
) -> CapturedRollout:
    root = tmp_path / f"rollout-{index}"
    write_json(
        root / "acceptance_audit.json",
        {
            "accepted_for_training": True,
            "world_id": "soap",
            "final_task": {"reward": 0.5},
        },
    )
    response: dict[str, object] = {
        "turn_reward": turn_reward,
        "reasoning_assessment": None,
    }
    if with_reasoning:
        response["reasoning_assessment"] = {
            "quality": 0.9,
            "efficiency": 0.8,
            "commentary": "focused",
        }
    write_json(root / "world/turns/000000/response.json", response)
    record = FakeRecord(
        request_id=f"request-{index}",
        prompt=FakePrompt([1, 2, 3]),
        sampled_tokens=[10 + index, 20 + index],
        sampled_logprobs=[-0.2, -0.3],
    )
    return CapturedRollout(
        rollout_root=root,
        group_identity_key=group_key,
        captures=(CapturedModelTurn(record, "world_action", (0,)),),
    )


def test_turn_reward_passes_through_and_missing_reasoning_is_neutral(
    tmp_path: Path,
) -> None:
    capture = rollout(tmp_path, index=0, turn_reward=0.6)
    feedback = feedback_for_capture(capture)

    assert feedback.turns[0].reward_delta == 0.6
    assert feedback.turns[0].reasoning_score is None


def test_unscored_final_model_turn_gets_zero_delta(tmp_path: Path) -> None:
    capture = rollout(tmp_path, index=0, turn_reward=0.6)
    final_record = FakeRecord(
        request_id="final",
        prompt=FakePrompt([1, 2, 3]),
        sampled_tokens=[30, 31],
        sampled_logprobs=[-0.1, -0.1],
    )
    capture = CapturedRollout(
        capture.rollout_root,
        capture.group_identity_key,
        capture.captures
        + (CapturedModelTurn(final_record, "unscored_model_turn"),),
    )

    feedback = feedback_for_capture(capture)
    assert [turn.reward_delta for turn in feedback.turns] == [0.6, 0.0]


def test_identical_terminal_rewards_still_train_from_turn_credit(
    tmp_path: Path,
) -> None:
    captures = tuple(
        rollout(
            tmp_path,
            index=index,
            turn_reward=reward,
            with_reasoning=index < 2,
        )
        for index, reward in enumerate((0.8, -0.8, 0.4, -0.4))
    )

    result = build_tinker_group(
        deps=DEPS,
        group_id="group",
        step=0,
        rollouts=captures,
    )

    assert result.rollout_advantages == pytest.approx((0.0, 0.0, 0.0, 0.0))
    assert len(result.datums) == 4
    assert any(item.turn_advantage.final_advantage > 0 for item in result.datums)
    assert any(item.turn_advantage.final_advantage < 0 for item in result.datums)
    assert all("mask" in item.datum.loss_fn_inputs for item in result.datums)


def test_bridge_rejects_mixed_trainer_group_keys(tmp_path: Path) -> None:
    captures = [
        rollout(tmp_path, index=index, turn_reward=0.2)
        for index in range(4)
    ]
    captures[3] = rollout(
        tmp_path,
        index=3,
        turn_reward=0.2,
        group_key="different-simulator",
    )

    with pytest.raises(RuntimeError, match="cannot mix trainer rollout-group"):
        build_tinker_group(
            deps=DEPS,
            group_id="group",
            step=0,
            rollouts=tuple(captures),
        )


def test_bridge_rejects_missing_live_behavior_logprobs(tmp_path: Path) -> None:
    capture = rollout(tmp_path, index=0, turn_reward=0.2)
    record = capture.captures[0].record
    record.sampled_logprobs = []

    with pytest.raises(RuntimeError, match="missing sampled tokens/logprobs"):
        feedback_for_capture(capture)


def test_bridge_zeroes_forced_close_span_on_parsed_records(tmp_path: Path) -> None:
    """The GW bridge must keep the live </think> span after validate().

    TrainableRolloutRecordSpec used to drop forced_close_token_span, so
    getattr on the parsed record was always None and the injected tokens
    received episode advantage.
    """

    first = rollout(tmp_path, index=0, turn_reward=0.8, with_reasoning=True)
    second = rollout(tmp_path, index=1, turn_reward=-0.8, with_reasoning=True)
    first.captures[0].record.forced_close_token_span = (0, 1)

    result = build_tinker_group(
        deps=DEPS,
        group_id="group",
        step=0,
        rollouts=(first, second),
    )
    first_datum = next(item for item in result.datums if item.request_id == "request-0")
    assert first_datum.turn_advantage.final_advantage != 0
    assert first_datum.datum.loss_fn_inputs["mask"] == [0.0, 0.0, 0.0, 1.0]
    advantages = first_datum.datum.loss_fn_inputs["advantages"]
    assert advantages[2] == 0.0
    assert advantages[3] == pytest.approx(first_datum.turn_advantage.final_advantage)


def test_bridge_rejects_nonfinite_sampled_logprobs(tmp_path: Path) -> None:
    capture = rollout(tmp_path, index=0, turn_reward=0.2)
    capture.captures[0].record.sampled_logprobs = [-0.2, float("nan")]
    other = rollout(tmp_path, index=1, turn_reward=-0.2)
    with pytest.raises(RuntimeError, match="must be finite"):
        build_tinker_group(
            deps=DEPS,
            group_id="group",
            step=0,
            rollouts=(capture, other),
        )


def test_vat_log_maps_parallel_tool_calls_to_one_tinker_record(tmp_path: Path) -> None:
    root = tmp_path / "rollout"
    for index, action_id in enumerate(("call-a", "call-b")):
        write_json(
            root / f"world/turns/{index:06d}/request.json",
            {"action": {"id": action_id}},
        )
    events = [
        {
            "direction": "runner_to_host",
            "message": {
                "type": "agent_event",
                "event_type": "message_end",
                "payload": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "toolCall", "id": "call-a"},
                            {"type": "toolCall", "id": "call-b"},
                        ],
                    }
                },
            },
        }
    ]
    vat = root / "logs/vat.jsonl"
    vat.parent.mkdir(parents=True)
    vat.write_text("".join(json.dumps(event) + "\n" for event in events))
    record = FakeRecord(
        request_id="parallel",
        prompt=FakePrompt([1, 2, 3]),
        sampled_tokens=[10, 11],
        sampled_logprobs=[-0.1, -0.2],
    )

    captures = captures_from_vat_log(records=[record], rollout_root=root)
    assert captures[0].world_turn_indices == (0, 1)


def test_vat_log_collapses_same_turn_parallel_tools_to_one_index(
    tmp_path: Path,
) -> None:
    """Multiple tools in one reply that share one GM move → one index."""
    root = tmp_path / "rollout"
    write_json(
        root / "world/turns/000001/request.json",
        {
            "actions": [
                {"id": "call-a"},
                {"id": "call-b"},
                {"id": "call-c"},
            ]
        },
    )
    events = [
        {
            "direction": "runner_to_host",
            "message": {
                "type": "agent_event",
                "event_type": "message_end",
                "payload": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "toolCall", "id": "call-a"},
                            {"type": "toolCall", "id": "call-b"},
                            {"type": "toolCall", "id": "call-c"},
                        ],
                    }
                },
            },
        }
    ]
    vat = root / "logs/vat.jsonl"
    vat.parent.mkdir(parents=True)
    vat.write_text("".join(json.dumps(event) + "\n" for event in events))
    record = FakeRecord(
        request_id="same-turn-parallel",
        prompt=FakePrompt([1, 2, 3]),
        sampled_tokens=[10, 11],
        sampled_logprobs=[-0.1, -0.2],
    )

    captures = captures_from_vat_log(records=[record], rollout_root=root)
    assert captures[0].world_turn_indices == (1,)


def test_vat_log_consumes_compaction_summary_record_at_compaction_end(
    tmp_path: Path,
) -> None:
    """Pi's summary generation samples through the proxy WITHOUT a message_end.

    The compaction_generation plan must be consumed at the compaction_end event
    itself; deferring it to the next assistant message would shift every later
    plan by one and fail the records/blocks reconciliation.
    """
    root = tmp_path / "rollout"
    for index, action_id in enumerate(("call-a", "call-b")):
        write_json(
            root / f"world/turns/{index:06d}/request.json",
            {"action": {"id": action_id}},
        )

    def assistant_event(action_id: str) -> dict:
        return {
            "direction": "runner_to_host",
            "message": {
                "type": "agent_event",
                "event_type": "message_end",
                "payload": {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "toolCall", "id": action_id}],
                    }
                },
            },
        }

    events = [
        assistant_event("call-a"),
        {
            "direction": "runner_to_host",
            "message": {
                "type": "agent_event",
                "event_type": "compaction_end",
                "payload": {
                    "aborted": False,
                    "compaction": {"summary": "structured checkpoint summary"},
                },
            },
        },
        assistant_event("call-b"),
    ]
    vat = root / "logs/vat.jsonl"
    vat.parent.mkdir(parents=True)
    vat.write_text("".join(json.dumps(event) + "\n" for event in events))

    records = [
        FakeRecord(
            request_id=name,
            prompt=FakePrompt([1, 2, 3]),
            sampled_tokens=[10, 11],
            sampled_logprobs=[-0.1, -0.2],
        )
        for name in ("turn-a", "summary-generation", "turn-b")
    ]

    captures = captures_from_vat_log(records=records, rollout_root=root)
    assert [capture.kind for capture in captures] == [
        "world_action",
        "compaction_generation",
        "world_action",
    ]
    assert captures[0].world_turn_indices == (0,)
    assert captures[1].world_turn_indices == ()
    assert captures[2].world_turn_indices == (1,)


def test_vat_log_drops_blank_assistant_message_end_shells(tmp_path: Path) -> None:
    """Blank Pi assistant message_end shells are not Tinker samples.

    Live crash shape: empty content=[] shells counted as final_text plans made
    records=N and blocks=N+1 (or N+2). Dropping them restores alignment.
    """
    root = tmp_path / "rollout"
    write_json(
        root / "world/turns/000000/request.json",
        {"action": {"id": "call-a"}},
    )

    def blank_assistant() -> dict:
        return {
            "direction": "runner_to_host",
            "message": {
                "type": "agent_event",
                "event_type": "message_end",
                "payload": {
                    "message": {
                        "role": "assistant",
                        "content": [],
                    }
                },
            },
        }

    events = [
        blank_assistant(),
        {
            "direction": "runner_to_host",
            "message": {
                "type": "agent_event",
                "event_type": "message_end",
                "payload": {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "toolCall", "id": "call-a"}],
                    }
                },
            },
        },
        blank_assistant(),
        {
            "direction": "runner_to_host",
            "message": {
                "type": "agent_event",
                "event_type": "message_end",
                "payload": {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "done"}],
                    }
                },
            },
        },
    ]
    vat = root / "logs/vat.jsonl"
    vat.parent.mkdir(parents=True)
    vat.write_text("".join(json.dumps(event) + "\n" for event in events))

    records = [
        FakeRecord(
            request_id=name,
            prompt=FakePrompt([1, 2, 3]),
            sampled_tokens=[10, 11],
            sampled_logprobs=[-0.1, -0.2],
        )
        for name in ("tool-turn", "final-text")
    ]

    captures = captures_from_vat_log(records=records, rollout_root=root)
    assert [capture.kind for capture in captures] == ["world_action", "final_text"]
    assert captures[0].world_turn_indices == (0,)
    assert captures[1].world_turn_indices == ()


# ---------------------------------------------------------------------------
# A replayed compaction summary is kept in the batch, trained on nothing.
#
# The bug: the bridge forced compaction_replay advantages to 0.0 and *then*
# applied the "skip anything at zero advantage" rule, so those captures were
# dropped outright and the masking branch below it was unreachable. The stated
# contract (turn_credit.py header, smoke checklist item 5) is that the replay is
# included with zero sampled-token contribution.
# ---------------------------------------------------------------------------


def test_compaction_replay_is_masked_not_dropped(tmp_path: Path) -> None:
    captures = []
    for index, reward in enumerate((0.9, -0.9)):
        base = rollout(tmp_path, index=index, turn_reward=reward)
        replay_record = FakeRecord(
            request_id=f"replay-{index}",
            prompt=FakePrompt([1, 2, 3]),
            sampled_tokens=[40 + index, 41 + index],
            sampled_logprobs=[-0.4, -0.5],
        )
        captures.append(
            CapturedRollout(
                base.rollout_root,
                base.group_identity_key,
                base.captures
                + (CapturedModelTurn(replay_record, "compaction_replay"),),
            )
        )

    result = build_tinker_group(
        deps=DEPS, group_id="group", step=0, rollouts=tuple(captures)
    )

    replay_ids = {f"replay-{index}" for index in range(2)}
    emitted = {item.request_id for item in result.datums}
    assert replay_ids <= emitted, "compaction replays must reach the batch"
    assert set(result.masked_compaction_replay_requests) == replay_ids
    assert not (replay_ids & set(result.skipped_zero_advantage_requests))

    for item in result.datums:
        if item.request_id not in replay_ids:
            continue
        inputs = item.datum.loss_fn_inputs
        # Nothing in a replayed summary may reach the optimizer.
        assert all(value == 0.0 for value in inputs["advantages"])
        assert all(value == 0.0 for value in inputs["mask"])
        # ...but the sequence itself is intact.
        assert len(inputs["target_tokens"]) == len(inputs["advantages"])
        assert len(inputs["target_tokens"]) > 0

    # The scored turns are unaffected.
    scored = [item for item in result.datums if item.request_id not in replay_ids]
    assert scored
    assert any(
        any(value != 0.0 for value in item.datum.loss_fn_inputs["advantages"])
        for item in scored
    )
    assert "masked_compaction_replay_requests" in result.audit_jsonable()


def test_zero_advantage_records_are_still_skipped(tmp_path: Path) -> None:
    """The replay exemption must not resurrect genuinely-empty datums."""
    captures = tuple(
        rollout(tmp_path, index=index, turn_reward=0.5) for index in range(2)
    )
    result = build_tinker_group(
        deps=DEPS, group_id="group", step=0, rollouts=captures
    )
    # Identical rewards and identical turn rewards → no signal anywhere.
    assert result.datums == ()
    assert set(result.skipped_zero_advantage_requests) == {
        "request-0",
        "request-1",
    }
    assert result.masked_compaction_replay_requests == ()


def test_cispo_normalizes_all_sequences_back_to_one_rollout_weight(
    tmp_path: Path,
) -> None:
    """Splitting one rollout into requests must not multiply its CISPO weight."""
    captures = []
    for index, reward in enumerate((0.9, -0.9)):
        base = rollout(tmp_path, index=index, turn_reward=0.0)
        write_json(
            base.rollout_root / "acceptance_audit.json",
            {
                "accepted_for_training": True,
                "world_id": "soap",
                "final_task": {"reward": reward},
            },
        )
        summary = FakeRecord(
            request_id=f"summary-{index}",
            prompt=FakePrompt([1, 2, 3]),
            sampled_tokens=[50, 51],
            sampled_logprobs=[-0.2, -0.2],
            content="checkpoint summary",
        )
        continuation = FakeRecord(
            request_id=f"continuation-{index}",
            prompt=FakePrompt([1, 2, 3]),
            sampled_tokens=[60, 61],
            sampled_logprobs=[-0.2, -0.2],
        )
        captures.append(
            CapturedRollout(
                base.rollout_root,
                base.group_identity_key,
                base.captures
                + (
                    CapturedModelTurn(summary, "compaction_generation"),
                    CapturedModelTurn(continuation, "final_text"),
                ),
            )
        )

    result = build_tinker_group(
        deps=DEPS, group_id="normalized", step=0, rollouts=tuple(captures)
    )

    assert result.rollout_trainable_token_counts == (6, 6)
    assert result.rollout_sequence_counts == (2, 2)
    assert result.rollout_compaction_counts == (1, 1)
    first = [item for item in result.datums if item.rollout_index == 0]
    assert first
    assert all(item.rollout_normalization_scale == pytest.approx(1 / 6) for item in first)
    assert sum(
        sum(abs(value) for value in item.datum.loss_fn_inputs["advantages"])
        for item in first
    ) == pytest.approx(abs(first[0].turn_advantage.final_advantage))


def test_normalization_excludes_replay_and_forced_close_tokens(tmp_path: Path) -> None:
    captures = []
    for index, reward in enumerate((0.9, -0.9)):
        base = rollout(tmp_path, index=index, turn_reward=reward)
        base.captures[0].record.forced_close_token_span = (0, 1)
        replay = FakeRecord(
            request_id=f"replay-normalization-{index}",
            prompt=FakePrompt([1, 2, 3]),
            sampled_tokens=[70, 71, 72],
            sampled_logprobs=[-0.2, -0.2, -0.2],
        )
        captures.append(
            CapturedRollout(
                base.rollout_root,
                base.group_identity_key,
                base.captures + (CapturedModelTurn(replay, "compaction_replay"),),
            )
        )
    result = build_tinker_group(
        deps=DEPS, group_id="masked", step=0, rollouts=tuple(captures)
    )
    assert result.rollout_trainable_token_counts == (1, 1)
