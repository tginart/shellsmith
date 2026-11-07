"""Harvest durable Pi evidence and endpoint records through the real audit gate."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from shellsmith.credit.to_tinker_bridge import build_tinker_group, feedback_for_capture
from shellsmith.credit.trajectory_pi import PiCompactionAdapter, PiSampleEvent, align_pi_samples
from shellsmith.episode.acceptance_audit import write_audit
from shellsmith.episode.consistency_ledger import append_entry
from shellsmith.policy.harvest import harvest_episode, records_consumed_by_pi
from shellsmith.policy.records import TinkerRolloutRecord
from shellsmith.policy.sampler import protocol_row
from shellsmith.train.durable_capture import load_capture, serialize_capture
from test_acceptance_audit import _rollout
from test_player_endpoint import _installed_pi_compaction_cases, _pi_summary_payload, _sampler_fixture
from test_to_tinker_bridge import DEPS, FakePrompt, write_json

GROUP_KEY = "shellsmith:checkout_idempotency_001:codex:gpt-6.1-sol:Qwen/Qwen3.8-27B:peft:262144"
COMMAND = "cd /virtual/repo && python -m pytest -q"


def test_rejected_generation_diagnostics_survive_consumed_record_filtering(tmp_path, monkeypatch):
    from shellsmith.policy import harvest

    root = tmp_path / "episode"
    rejected = SimpleNamespace(protocol_row={"request_id": "rejected", "protocol_error": "incomplete tool call",
        "generation_notices": [{"code": "response_limit_exceeded", "message": "Response reached its limit."}],
        "reasoning_effort": "medium", "response_limit": 4096})
    monkeypatch.setattr(harvest, "records_consumed_by_pi", lambda records, **kwargs: [])
    monkeypatch.setattr(harvest, "persist_tinker_records_for_protocol", lambda records, path, **kwargs: None)
    monkeypatch.setattr(harvest, "_ensure_acceptance_audit", lambda root: None)
    monkeypatch.setattr(harvest, "_accepted_for_training", lambda root: False)
    assert harvest.harvest_episode([rejected], episode_root=root, group_identity_key="group") is None
    diagnostics = json.loads((root / "logs/generation_diagnostics.json").read_text())
    assert diagnostics["requests"][0]["request_id"] == "rejected"
    assert diagnostics["requests"][0]["protocol_error"] == "incomplete tool call"


def _record(
    request_id: str,
    *,
    call_id: str | None = "call-1",
    content: str = "",
    compaction: bool = False,
    compaction_kind: str = "history",
    tokens: tuple[int, ...] = (10, 20),
) -> TinkerRolloutRecord:
    """Use the endpoint's record type and protocol-row serializer, not harvest doubles."""
    record = TinkerRolloutRecord(
        request_id=request_id,
        created_at=1.0,
        prompt=FakePrompt([1, 2, 3]),
        prompt_tokens=3,
        sampled_tokens=list(tokens),
        sampled_logprobs=[-0.2] * len(tokens),
        content=content,
        finish_reason="tool_calls" if call_id else "stop",
        payload=_pi_summary_payload(compaction_kind) if compaction else {
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Run the tests."},
            ],
        },
        reasoning_content="Run the regression suite." if call_id else "",
        session_id="harvest-fixture",
    )
    assistant: dict[str, Any] = {"role": "assistant", "content": content}
    if call_id:
        assistant["tool_calls"] = [{
            "id": call_id,
            "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": COMMAND})},
        }]
    sample = {
        "renderer_name": "qwen3_5",
        "thinking": not compaction,
        "tool_schema_sha256": "a" * 64,
        "parse_termination": "clean",
        "thinking_tokens": 0,
        "response_tokens": len(tokens),
    }
    return replace(record, protocol_row=protocol_row(record, sample, assistant))


def _episode(
    tmp_path: Path,
    episode_id: str,
    *,
    turn_reward: float = 0.3,
    accepted: bool = True,
    write_initial_audit: bool = True,
) -> tuple[Path, TinkerRolloutRecord]:
    # Reuse the complete world/ledger/vat fixture already exercised by the audit
    # tests. Acceptance decisions are computed from these artifacts, never seeded.
    root = _rollout(tmp_path / episode_id)
    response_path = root / "world/turns/000000/response.json"
    response = json.loads(response_path.read_text())
    response.update(turn_reward=turn_reward, reasoning_assessment=None)
    write_json(response_path, response)
    if not accepted:
        final_path = root / "world/final/response.json"
        final = json.loads(final_path.read_text())
        final["trainable"] = False
        write_json(final_path, final)
    if write_initial_audit:
        assert write_audit(root)["accepted_for_training"] is accepted
    return root, _record(f"request-{episode_id}")


def _append_events(root: Path, *events: dict[str, Any]) -> None:
    with (root / "logs/vat.jsonl").open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


def _text_event(text: str) -> dict[str, Any]:
    return {
        "direction": "runner_to_host",
        "message": {
            "type": "agent_event",
            "event_type": "message_end",
            "payload": {"message": {"role": "assistant", "content": [{"type": "text", "text": text}]}},
        },
    }


def _compaction_event(summary: str, *, decorated: bool = False, aborted: bool = False) -> dict[str, Any]:
    details = {"readFiles": [], "modifiedFiles": []}
    if decorated:
        # Pinned Pi appends these file-operation sections after the sampled text.
        details = {"readFiles": ["/virtual/repo/tests.py"], "modifiedFiles": ["/virtual/repo/app.py"]}
        summary += (
            "\n\n<read-files>\n/virtual/repo/tests.py\n</read-files>"
            "\n\n<modified-files>\n/virtual/repo/app.py\n</modified-files>"
        )
    return {
        "direction": "runner_to_host",
        "message": {
            "type": "agent_event",
            "event_type": "compaction_end",
            "payload": {
                "reason": "threshold",
                "aborted": aborted,
                "will_retry": aborted,
                "compaction": {
                    "summary": summary,
                    "first_kept_entry_id": "kept-entry",
                    "tokens_before": 160000,
                    "details": details,
                },
            },
        },
    }


def _audit(root: Path) -> dict[str, Any]:
    return json.loads((root / "acceptance_audit.json").read_text())


def _persisted_ids(root: Path) -> list[str]:
    protocol = json.loads((root / "scenario/model_protocol.json").read_text())
    rows = [json.loads(line) for line in (root / "logs/tinker_rollout_records.jsonl").read_text().splitlines()]
    ids = [row["request_id"] for row in rows]
    assert [request["request_id"] for request in protocol["requests"]] == ids
    return ids


def test_harvested_episodes_feed_turn_cispo(tmp_path: Path) -> None:
    captures = []
    for episode_id, turn_reward in (("t0", 0.8), ("t1", -0.8)):
        root, record = _episode(tmp_path, episode_id, turn_reward=turn_reward)
        captures.append(harvest_episode([record], episode_root=root, group_identity_key=GROUP_KEY))
    assert all(capture is not None for capture in captures)
    result = build_tinker_group(deps=DEPS, group_id="group", step=0, rollouts=tuple(captures))
    assert result.rollout_advantages == pytest.approx((0.0, 0.0))
    assert len(result.datums) == 2
    assert any(item.turn_advantage.final_advantage > 0 for item in result.datums)
    assert any(item.turn_advantage.final_advantage < 0 for item in result.datums)


def test_rejected_audit_returns_none(tmp_path: Path) -> None:
    root, record = _episode(tmp_path, "nope", accepted=False)
    assert harvest_episode([record], episode_root=root, group_identity_key=GROUP_KEY) is None
    assert _audit(root)["accepted_for_training"] is False


def test_missing_audit_is_written_after_protocol_rows(tmp_path: Path) -> None:
    root, record = _episode(tmp_path, "need-audit", write_initial_audit=False)
    protocol_path = root / "scenario/model_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["requests"] = []
    write_json(protocol_path, protocol)

    capture = harvest_episode([record], episode_root=root, group_identity_key=GROUP_KEY)

    assert capture is not None
    assert capture.captures[0].record is record
    assert _persisted_ids(root) == [record.request_id]
    assert _audit(root)["accepted_for_training"] is True
    assert _audit(root)["model_protocol_integrity"]["request_count"] == 1


def test_zero_records_is_loud(tmp_path: Path) -> None:
    root, _ = _episode(tmp_path, "empty", write_initial_audit=False)
    with pytest.raises(RuntimeError, match="zero Tinker records"):
        harvest_episode([], episode_root=root, group_identity_key=GROUP_KEY)


@pytest.mark.parametrize("content", [
    "Explain the literal <summary_request> tag, then run the regression suite.",
    [{"type": "text", "text": "Explain <summary_request> and </summary_request>, then run tests."}],
])
def test_sampler_user_summary_tag_remains_a_harvested_world_action(tmp_path: Path, content) -> None:
    root, _ = _episode(tmp_path, "summary-tag-user", write_initial_audit=False)
    sampler, _, _ = _sampler_fixture(
        thinking=True, content=[{"type": "thinking", "thinking": "Run the regression suite."}],
        tool_calls=[{"id": "call-1", "type": "function", "function": {
            "name": "bash", "arguments": json.dumps({"command": COMMAND}),
        }}],
    )
    record, _ = sampler.sample(
        {"messages": [{"role": "user", "content": content}], "reasoning_effort": "high"},
        request_id="ordinary-action", session_id=root.name,
    )
    assert record.protocol_row["compaction_summary"] is False

    capture = harvest_episode([record], episode_root=root, group_identity_key=GROUP_KEY)

    assert capture is not None
    assert capture.captures[0].record is record
    assert capture.captures[0].kind == "world_action"
    assert _persisted_ids(root) == [record.request_id]
    assert _audit(root)["accepted_for_training"] is True
    trajectory = PiCompactionAdapter().adapt(rollout_id=root.name, records=[record], evidence=root)
    assert trajectory.turns[0].world_turn_indices == (0,)
    assert not trajectory.boundaries


def test_installed_pi_compactions_flow_through_sampler_and_harvest(tmp_path: Path) -> None:
    # Actual installed Pi compact()/wire conversion and Shellsmith sampler,
    # harvest/audit/alignment. Only external model/SDK interfaces are fixtures.
    cases = _installed_pi_compaction_cases()
    assert [case["name"] for case in cases] == ["history", "update", "split", "prefix-only"]
    for case in cases:
        root, action = _episode(tmp_path, case["name"], write_initial_audit=False)
        records = [action]
        assert len(case["requests"]) == (2 if case["name"] == "split" else 1)
        for index, request in enumerate(case["requests"]):
            sampler, _, _ = _sampler_fixture(content=request["text"])
            record, _ = sampler.sample(request["payload"], request_id=f"summary-{index}", session_id=root.name)
            assert record.protocol_row["compaction_summary"] is True
            assert record.content == request["text"]
            records.append(record)
        compaction = case["compaction"]
        event = _compaction_event(compaction["summary"])
        event["message"]["payload"]["compaction"].update(
            first_kept_entry_id=compaction["firstKeptEntryId"], tokens_before=compaction["tokensBefore"],
            details=compaction["details"],
        )
        _append_events(root, event)

        capture = harvest_episode(records, episode_root=root, group_identity_key=GROUP_KEY)

        assert capture is not None
        assert [item.record for item in capture.captures] == records
        assert [item.kind for item in capture.captures] == ["world_action"] + ["compaction_generation"] * (len(records) - 1)
        assert _persisted_ids(root) == [record.request_id for record in records]
        assert _audit(root)["accepted_for_training"] is True
        assert _audit(root)["model_protocol_integrity"]["assistant_sample_count"] == 1
        trajectory = PiCompactionAdapter().adapt(rollout_id=root.name, records=records, evidence=root)
        assert len(trajectory.boundaries) == 1
        boundary = trajectory.boundaries[0]
        assert boundary.generation_request_id == records[-1].request_id
        assert all(turn.boundary_id == boundary.boundary_id for turn in trajectory.turns[1:])
        assert all(turn.record is record for turn, record in zip(trajectory.turns, records, strict=True))


def test_unconsumed_retry_is_dropped(tmp_path: Path) -> None:
    root, consumed = _episode(tmp_path, "retry", write_initial_audit=False)
    dropped = _record("request-dropped", call_id="call-never-consumed", tokens=(30,))
    assert records_consumed_by_pi([dropped, consumed], episode_root=root, episode_id="retry") == [consumed]


def test_missing_consumed_response_is_loud(tmp_path: Path) -> None:
    root, _ = _episode(tmp_path, "missing", write_initial_audit=False)
    wrong = _record("wrong", call_id="call-other", tokens=(30,))
    with pytest.raises(RuntimeError, match="missing a proxy response consumed by Pi"):
        records_consumed_by_pi([wrong], episode_root=root, episode_id="missing")


def test_ambiguous_retry_with_same_visible_content_is_loud(tmp_path: Path) -> None:
    root, consumed = _episode(tmp_path, "ambiguous", write_initial_audit=False)
    dropped = _record("request-dropped", tokens=(99,))
    with pytest.raises(RuntimeError, match="ambiguous proxy responses"):
        records_consumed_by_pi([dropped, consumed], episode_root=root, episode_id="ambiguous")


@pytest.mark.parametrize("summaries", [("First checkpoint", "Second checkpoint"), ("Same checkpoint", "Same checkpoint")])
@pytest.mark.parametrize("decorated", [False, True])
def test_harvest_repeated_compactions_preserves_consumed_order(
    tmp_path: Path, summaries: tuple[str, str], decorated: bool,
) -> None:
    root, action = _episode(tmp_path, "compactions", write_initial_audit=False)
    _append_events(
        root,
        _compaction_event(summaries[0], decorated=decorated),
        _text_event("Continue checking the fix."),
        _compaction_event(summaries[1], decorated=decorated),
        _text_event("Finished verification."),
    )
    records = [
        action,
        _record("summary-1", call_id=None, content=summaries[0], compaction=True, tokens=(31,)),
        _record("continuation", call_id=None, content="Continue checking the fix.", tokens=(40, 41)),
        _record("summary-2", call_id=None, content=summaries[1], compaction=True, tokens=(51, 52)),
        _record("final", call_id=None, content="Finished verification.", tokens=(60,)),
    ]

    capture = harvest_episode(records, episode_root=root, group_identity_key=GROUP_KEY)

    assert capture is not None
    assert [item.record for item in capture.captures] == records
    assert [item.kind for item in capture.captures] == [
        "world_action", "compaction_generation", "final_text", "compaction_generation", "final_text",
    ]
    assert _persisted_ids(root) == [record.request_id for record in records]
    audit = _audit(root)
    assert audit["accepted_for_training"] is True
    assert audit["model_protocol_integrity"]["request_count"] == 5
    assert audit["model_protocol_integrity"]["assistant_sample_count"] == 3


@pytest.mark.parametrize("retry_first", [False, True])
def test_compaction_retry_is_selected_by_summary_not_position(tmp_path: Path, retry_first: bool) -> None:
    root, action = _episode(tmp_path, "summary-retry", write_initial_audit=False)
    _append_events(
        root,
        _compaction_event("Checkpoint", aborted=True),
        _compaction_event("Checkpoint with verified details", decorated=True),
    )
    consumed = _record("consumed-summary", call_id=None, content="Checkpoint with verified details", compaction=True)
    retry = _record("unconsumed-summary", call_id=None, content="Checkpoint", compaction=True, tokens=(99,))
    records = [action, retry, consumed] if retry_first else [action, consumed, retry]

    capture = harvest_episode(records, episode_root=root, group_identity_key=GROUP_KEY)

    assert capture is not None
    assert [item.record for item in capture.captures] == [action, consumed]
    assert _persisted_ids(root) == [action.request_id, consumed.request_id]
    assert _audit(root)["accepted_for_training"] is True


def test_repeated_compactions_with_indistinguishable_extra_retry_are_ambiguous(tmp_path: Path) -> None:
    root, action = _episode(tmp_path, "ambiguous-compactions", write_initial_audit=False)
    _append_events(root, _compaction_event("Same checkpoint"), _compaction_event("Same checkpoint"))
    summaries = [
        _record(f"summary-{index}", call_id=None, content="Same checkpoint", compaction=True, tokens=(30 + index,))
        for index in range(3)
    ]
    with pytest.raises(RuntimeError, match="ambiguous proxy responses"):
        harvest_episode([action, *summaries], episode_root=root, group_identity_key=GROUP_KEY)
    assert not (root / "acceptance_audit.json").exists()


def test_missing_repeated_compaction_is_not_filled_by_reusing_a_record(tmp_path: Path) -> None:
    root, action = _episode(tmp_path, "missing-compaction", write_initial_audit=False)
    _append_events(root, _compaction_event("Same checkpoint"), _compaction_event("Same checkpoint"))
    summary = _record("summary", call_id=None, content="Same checkpoint", compaction=True)
    with pytest.raises(RuntimeError, match="missing a proxy response consumed by Pi"):
        records_consumed_by_pi([action, summary], episode_root=root, episode_id="missing-compaction")


def test_compaction_records_cannot_be_reordered_to_fit_the_journal(tmp_path: Path) -> None:
    root, action = _episode(tmp_path, "out-of-order", write_initial_audit=False)
    _append_events(root, _compaction_event("First checkpoint"), _compaction_event("Second checkpoint"))
    first = _record("summary-1", call_id=None, content="First checkpoint", compaction=True)
    second = _record("summary-2", call_id=None, content="Second checkpoint", compaction=True)
    with pytest.raises(RuntimeError, match="missing a proxy response consumed by Pi"):
        records_consumed_by_pi([action, second, first], episode_root=root, episode_id="out-of-order")


def test_duplicate_record_identity_does_not_create_an_extra_turn(tmp_path: Path) -> None:
    root, action = _episode(tmp_path, "duplicate", write_initial_audit=False)
    _append_events(root, _compaction_event("Checkpoint"))
    summary = _record("summary", call_id=None, content="Checkpoint", compaction=True)
    duplicate = replace(summary, protocol_row=dict(summary.protocol_row))
    capture = harvest_episode([action, summary, duplicate], episode_root=root, group_identity_key=GROUP_KEY)
    assert capture is not None
    assert [item.record for item in capture.captures] == [action, summary]
    assert _persisted_ids(root) == [action.request_id, summary.request_id]


def test_existing_rejected_audit_is_recomputed_after_attaching_requests(tmp_path: Path) -> None:
    root, record = _episode(tmp_path, "empty-protocol", write_initial_audit=False)
    protocol_path = root / "scenario/model_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["requests"] = []
    write_json(protocol_path, protocol)
    stale = write_audit(root)
    assert stale["accepted_for_training"] is False
    assert any("requests empty" in problem for problem in stale["model_protocol_integrity"]["problems"])

    capture = harvest_episode([record], episode_root=root, group_identity_key=GROUP_KEY)

    assert capture is not None
    assert capture.captures[0].record is record
    assert _persisted_ids(root) == [record.request_id]
    refreshed = _audit(root)
    assert refreshed["accepted_for_training"] is True
    assert refreshed["model_protocol_integrity"]["problems"] == []


@pytest.mark.parametrize("corruption", ["renderer", "tool_arguments"])
def test_existing_accepted_audit_is_recomputed_after_replacing_requests(tmp_path: Path, corruption: str) -> None:
    root, record = _episode(tmp_path, "replaced-protocol")
    assert _audit(root)["accepted_for_training"] is True
    changed_row = json.loads(json.dumps(record.protocol_row))
    if corruption == "renderer":
        changed_row["renderer_name"] = "wrong-renderer"
        problem = "renderer mismatch"
    else:
        changed_row["tool_calls"][0]["arguments"] = {"command": "echo not-the-journaled-command"}
        problem = "tool arguments mismatch"
    changed = replace(record, protocol_row=changed_row)

    assert harvest_episode([changed], episode_root=root, group_identity_key=GROUP_KEY) is None

    assert _persisted_ids(root) == [record.request_id]
    rejected = _audit(root)
    assert rejected["accepted_for_training"] is False
    assert any(problem in item for item in rejected["model_protocol_integrity"]["problems"])
    # The rejection must not stick when a later harvest restores valid evidence.
    assert harvest_episode([record], episode_root=root, group_identity_key=GROUP_KEY) is not None
    assert _audit(root)["accepted_for_training"] is True


# This literal is intentionally independent of production parsing. It is the
# exact pinned Pi 0.84.1 compact() wrapper, not a renderer or model output.
SPLIT_SEPARATOR = "\n\n---\n\n**Turn Context (split turn):**\n\n"


def _append_world_turn(root: Path, index: int, action_id: str, reward: float) -> None:
    request = json.loads((root / "world/turns/000000/request.json").read_text())
    request["step_index"] = index
    request["consistency_ledger"]["expected_revision"] = index
    request["action"]["id"] = action_id
    response = json.loads((root / "world/turns/000000/response.json").read_text())
    response["action_id"] = action_id
    response["turn_reward"] = reward
    review = response["consistency_review"]
    review["reviewed_revision"] = index
    write_json(root / f"world/turns/{index:06d}/request.json", request)
    write_json(root / f"world/turns/{index:06d}/response.json", response)
    append_entry(
        root / "world/consistency_ledger.json", expected_revision=index,
        turn_index=index, action_id=action_id,
        observation=response["tool_response"]["content"],
        claims=review["observation_claims"], new_commitments=review["new_commitments"],
        cross_checks=review["executed_cross_checks"],
    )
    _append_events(root, {
        "direction": "runner_to_host",
        "message": {
            "type": "agent_event", "event_type": "message_end",
            "payload": {"message": {"role": "assistant", "content": [{
                "type": "toolCall", **request["action"],
            }]}},
        },
    }, {
        "direction": "runner_to_host",
        "message": {"type": "tool_request", "tool_call_id": action_id,
                    "tool_name": "bash", "arguments": {"command": COMMAND}, "harness_rejected": False},
    }, {
        "direction": "host_to_runner",
        "message": {"type": "tool_response", "tool_call_id": action_id, **response["tool_response"]},
    })


def _split_episode(
    tmp_path: Path, episode_id: str, parts: tuple[tuple[str | None, str], ...], *,
    decorated: bool = True, terminal_reward: float = 0.9,
) -> tuple[Path, list[TinkerRolloutRecord], list[list[TinkerRolloutRecord]]]:
    root, action = _episode(tmp_path, episode_id, write_initial_audit=False)
    final_path = root / "world/final/response.json"
    final = json.loads(final_path.read_text())
    final["final_reward"] = terminal_reward
    write_json(final_path, final)
    records = [action]
    generations = []
    for index, (history, prefix) in enumerate(parts):
        summary = ("No prior history." if history is None else history) + SPLIT_SEPARATOR + prefix
        event = _compaction_event(summary, decorated=decorated)
        event["message"]["payload"]["compaction"]["first_kept_entry_id"] = f"kept-{index}"
        _append_events(root, event)
        generated = []
        for part_index, text in enumerate((prefix,) if history is None else (history, prefix)):
            record = _record(
                f"{episode_id}-summary-{index}-{part_index}", call_id=None,
                content=text, compaction=True, tokens=(100 + 10 * index + part_index, 200 + part_index),
                compaction_kind="prefix" if history is None or part_index == 1 else "history",
            )
            # Each summary call retains its own renderer prompt and behavior logprobs.
            record = replace(record, prompt=FakePrompt([1, 3 + index, 5 + part_index]),
                             sampled_logprobs=[-0.1 - part_index / 10, -0.4])
            record.protocol_row["sampled_logprobs"] = record.sampled_logprobs
            generated.append(record)
        records.extend(generated)
        generations.append(generated)
        action_id = f"call-{index + 2}"
        _append_world_turn(root, index + 1, action_id, reward=0.4 + index / 10)
        records.append(_record(f"{episode_id}-after-{index}", call_id=action_id, tokens=(300 + index,)))
    return root, records, generations


@pytest.mark.parametrize("parts", [
    ((None, "Current turn prefix"),),
    (("Earlier conversation", "Current turn prefix"),),
    (("Identical parts", "Identical parts"),),
    ((None, "First prefix"), ("Updated history", "Second prefix")),
    (("Repeated history", "Repeated prefix"), ("Repeated history", "Repeated prefix")),
])
@pytest.mark.parametrize("decorated", [False, True])
def test_split_turn_harvest_adapter_and_datums_preserve_each_sample(
    tmp_path: Path, parts: tuple[tuple[str | None, str], ...], decorated: bool,
) -> None:
    captures = []
    originals = []
    for rollout_index, reward in enumerate((0.9, 0.1)):
        root, records, generations = _split_episode(
            tmp_path, f"split-{rollout_index}", parts, decorated=decorated, terminal_reward=reward,
        )
        capture = harvest_episode(records, episode_root=root, group_identity_key=GROUP_KEY)
        assert capture is not None
        assert all(item.record is record for item, record in zip(capture.captures, records, strict=True))
        assert _persisted_ids(root) == [record.request_id for record in records]
        assert _audit(root)["accepted_for_training"] is True
        trajectory = PiCompactionAdapter().adapt(rollout_id=root.name, records=records, evidence=root)
        assert len(trajectory.boundaries) == len(parts)
        assert trajectory.sequence_count == len(parts) + 1
        for boundary_index, (boundary, generated) in enumerate(zip(trajectory.boundaries, generations, strict=True)):
            turns = [turn for turn in trajectory.turns if turn.record in generated]
            assert [turn.record for turn in turns] == generated
            assert all(turn.kind == "compaction_generation" and turn.world_turn_indices == () for turn in turns)
            assert {turn.boundary_id for turn in turns} == {boundary.boundary_id}
            assert {turn.sequence_index for turn in turns} == {boundary_index}
            assert boundary.generation_request_id == generated[-1].request_id
            assert boundary.next_request_id == f"split-{rollout_index}-after-{boundary_index}"
            assert (boundary.sequence_before, boundary.sequence_after) == (boundary_index, boundary_index + 1)
        world_turns = [turn for turn in trajectory.turns if turn.kind == "world_action"]
        assert [turn.world_turn_indices for turn in world_turns] == [(index,) for index in range(len(parts) + 1)]
        assert [turn.sequence_index for turn in world_turns] == list(range(len(parts) + 1))
        feedback = {turn.request_id: turn.reward_delta for turn in feedback_for_capture(capture).turns}
        assert feedback[records[0].request_id] == 0.3
        for index, generated in enumerate(generations):
            assert all(feedback[record.request_id] == 0.0 for record in generated)
            assert feedback[f"split-{rollout_index}-after-{index}"] == pytest.approx(0.4 + index / 10)
        captures.append(capture)
        originals.append(records)

    result = build_tinker_group(deps=DEPS, group_id="split-group", step=0, rollouts=tuple(captures))
    assert result.rollout_compaction_counts == (len(parts), len(parts))
    assert result.rollout_sequence_counts == (len(parts) + 1, len(parts) + 1)
    assert result.rollout_trainable_token_counts == tuple(sum(len(record.sampled_tokens) for record in rows) for rows in originals)
    assert len(result.datums) == sum(len(rows) for rows in originals)
    for datum in result.datums:
        record = originals[datum.rollout_index][datum.record_index]
        assert datum.request_id == record.request_id
        assert datum.datum.model_input.tokens == record.prompt.tokens + record.sampled_tokens[:-1]
        inputs = datum.datum.loss_fn_inputs
        padding = record.prompt.length - 1
        assert inputs["target_tokens"] == [0] * padding + record.sampled_tokens
        assert inputs["logprobs"] == [0.0] * padding + record.sampled_logprobs
        assert inputs["mask"] == [0.0] * padding + [1.0] * len(record.sampled_tokens)
        assert any(value != 0.0 for value in inputs["advantages"][padding:])


def _insert_failed_compaction(root: Path, *, aborted: bool) -> None:
    vat_path = root / "logs/vat.jsonl"
    events = [json.loads(line) for line in vat_path.read_text().splitlines()]
    index = next(index for index, event in enumerate(events) if event["message"].get("event_type") == "compaction_end")
    events.insert(index, {"direction": "runner_to_host", "message": {
        "type": "agent_event", "event_type": "compaction_end",
        "payload": {"reason": "threshold", "aborted": aborted, "will_retry": False},
    }})
    vat_path.write_text("".join(json.dumps(event) + "\n" for event in events))


@pytest.mark.parametrize("aborted", [True, False])
def test_split_turn_failed_attempt_records_do_not_become_generations(tmp_path: Path, aborted: bool) -> None:
    root, records, _ = _split_episode(tmp_path, "abort-split", (("History", "Prefix"),))
    _insert_failed_compaction(root, aborted=aborted)
    abandoned = [_record(f"aborted-{index}", call_id=None, content=text, compaction=True, tokens=(990 + index,))
                 for index, text in enumerate(("Abandoned history", "Abandoned prefix"))]
    capture = harvest_episode([records[0], *abandoned, *records[1:]], episode_root=root, group_identity_key=GROUP_KEY)
    assert capture is not None
    assert [item.record for item in capture.captures] == records
    assert _persisted_ids(root) == [record.request_id for record in records]
    trajectory = PiCompactionAdapter().adapt(rollout_id=root.name, records=records, evidence=root)
    assert len(trajectory.boundaries) == 1
    assert trajectory.sequence_count == 2


@pytest.mark.parametrize("duplicated_part", ["History", "Prefix"])
def test_split_turn_indistinguishable_retry_is_rejected(tmp_path: Path, duplicated_part: str) -> None:
    root, records, _ = _split_episode(tmp_path, "ambiguous-split", (("History", "Prefix"),))
    _insert_failed_compaction(root, aborted=True)
    retry = _record("retry", call_id=None, content=duplicated_part, compaction=True, tokens=(999,))
    index = next(index for index, record in enumerate(records) if record.content == duplicated_part)
    with pytest.raises(RuntimeError, match="ambiguous proxy responses"):
        harvest_episode([*records[:index], retry, *records[index:]], episode_root=root, group_identity_key=GROUP_KEY)


def test_split_turn_literal_history_cannot_hide_a_possibly_sampled_history(tmp_path: Path) -> None:
    root, records, _ = _split_episode(tmp_path, "literal-history", ((None, "Prefix"),))
    possible_history = _record("possible-history", call_id=None, content="No prior history.", compaction=True)
    with pytest.raises(RuntimeError, match="ambiguous proxy responses"):
        harvest_episode([records[0], possible_history, *records[1:]], episode_root=root, group_identity_key=GROUP_KEY)


@pytest.mark.parametrize("missing_part", ["History", "Prefix"])
def test_split_turn_does_not_fabricate_missing_sample_tokens(tmp_path: Path, missing_part: str) -> None:
    root, records, _ = _split_episode(tmp_path, "missing-split", (("History", "Prefix"),))
    with pytest.raises(RuntimeError, match="missing a proxy response consumed by Pi"):
        harvest_episode([record for record in records if record.content != missing_part], episode_root=root, group_identity_key=GROUP_KEY)


@pytest.mark.parametrize("missing_provenance", ["details", "first_kept_entry_id", "tokens_before"])
def test_split_turn_requires_pinned_event_provenance(tmp_path: Path, missing_provenance: str) -> None:
    root, records, _ = _split_episode(tmp_path, "unproven-split", ((None, "Prefix"),), decorated=False)
    vat_path = root / "logs/vat.jsonl"
    events = [json.loads(line) for line in vat_path.read_text().splitlines()]
    for event in events:
        if event["message"].get("event_type") == "compaction_end":
            del event["message"]["payload"]["compaction"][missing_provenance]
    vat_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    with pytest.raises(RuntimeError, match="missing a proxy response consumed by Pi"):
        harvest_episode(records, episode_root=root, group_identity_key=GROUP_KEY)


@pytest.mark.parametrize("history", [None, "Earlier history"])
def test_split_turn_terminal_boundary_has_no_fabricated_continuation(tmp_path: Path, history: str | None) -> None:
    root, action = _episode(tmp_path, "terminal-split", write_initial_audit=False)
    prefix = "Terminal prefix"
    summary = ("No prior history." if history is None else history) + SPLIT_SEPARATOR + prefix
    _append_events(root, _compaction_event(summary, decorated=True))
    records = [action, *[
        _record(f"summary-{index}", call_id=None, content=text, compaction=True)
        for index, text in enumerate((prefix,) if history is None else (history, prefix))
    ]]
    assert harvest_episode(records, episode_root=root, group_identity_key=GROUP_KEY) is not None
    trajectory = PiCompactionAdapter().adapt(rollout_id=root.name, records=records, evidence=root)
    assert len(trajectory.boundaries) == 1
    assert trajectory.sequence_count == 1
    assert trajectory.boundaries[0].next_request_id is None
    assert trajectory.boundaries[0].sequence_after is None
    assert trajectory.boundaries[0].generation_request_id == records[-1].request_id


def test_split_turn_wrapper_replay_is_masked_without_masking_generators(tmp_path: Path) -> None:
    captures = []
    replay_ids = set()
    for index, reward in enumerate((0.9, 0.1)):
        root, records, generated = _split_episode(tmp_path, f"replay-{index}", (("History", "Prefix"),), terminal_reward=reward)
        events = [json.loads(line) for line in (root / "logs/vat.jsonl").read_text().splitlines()]
        summary = next(event["message"]["payload"]["compaction"]["summary"] for event in events
                       if event["message"].get("event_type") == "compaction_end")
        _append_events(root, _text_event(summary))
        replay = _record(f"replayed-{index}", call_id=None, content=summary, tokens=(800, 801))
        replay_ids.add(replay.request_id)
        capture = harvest_episode([*records, replay], episode_root=root, group_identity_key=GROUP_KEY)
        assert capture is not None
        assert capture.captures[-1].kind == "compaction_replay"
        assert all(item.kind == "compaction_generation" for item in capture.captures if item.record in generated[0])
        captures.append(capture)
    result = build_tinker_group(deps=DEPS, group_id="split-replay", step=0, rollouts=tuple(captures))
    assert set(result.masked_compaction_replay_requests) == replay_ids
    assert result.rollout_compaction_counts == (1, 1)
    for datum in result.datums:
        mask = datum.datum.loss_fn_inputs["mask"]
        assert (not any(mask)) is (datum.request_id in replay_ids)


class _DurablePrompt(FakePrompt):
    def to_ints(self) -> list[int]:
        return list(self.tokens)

    @classmethod
    def from_ints(cls, tokens: list[int]) -> "_DurablePrompt":
        return cls(tokens)


def test_split_turn_capture_resume_reconstructs_the_same_boundaries_and_datums(tmp_path: Path) -> None:
    deps = SimpleNamespace(**vars(DEPS), tinker=SimpleNamespace(types=SimpleNamespace(ModelInput=_DurablePrompt)))
    fresh = []
    restored = []
    for index, reward in enumerate((0.9, 0.1)):
        root, records, _ = _split_episode(tmp_path, f"resume-{index}", ((None, "Prefix"), ("History", "Next prefix")), terminal_reward=reward)
        records = [replace(record, prompt=_DurablePrompt(record.prompt.tokens)) for record in records]
        capture = harvest_episode(records, episode_root=root, group_identity_key=GROUP_KEY)
        assert capture is not None
        path = root / "capture.json"
        write_json(path, serialize_capture(capture, episode_id=f"resume-{index}"))
        fresh.append(capture)
        restored.append(load_capture(path, deps=deps))
    before = build_tinker_group(deps=deps, group_id="resume-split", step=0, rollouts=tuple(fresh))
    after = build_tinker_group(deps=deps, group_id="resume-split", step=0, rollouts=tuple(restored))
    assert after.rollout_compaction_counts == before.rollout_compaction_counts == (2, 2)
    assert after.rollout_sequence_counts == before.rollout_sequence_counts == (3, 3)
    assert after.rollout_trainable_token_counts == before.rollout_trainable_token_counts
    assert [item.request_id for item in after.datums] == [item.request_id for item in before.datums]
    assert [item.datum.loss_fn_inputs for item in after.datums] == [item.datum.loss_fn_inputs for item in before.datums]
    assert [item.datum.model_input.tokens for item in after.datums] == [item.datum.model_input.tokens for item in before.datums]


@pytest.mark.parametrize("resume", [False, True])
def test_split_turn_missing_journal_does_not_guess_separate_boundaries(tmp_path: Path, resume: bool) -> None:
    deps = SimpleNamespace(**vars(DEPS), tinker=SimpleNamespace(types=SimpleNamespace(ModelInput=_DurablePrompt)))
    captures = []
    for index in range(2):
        root, records, _ = _split_episode(tmp_path, f"lost-journal-{index}", (("History", "Prefix"),))
        records = [replace(record, prompt=_DurablePrompt(record.prompt.tokens)) for record in records]
        capture = harvest_episode(records, episode_root=root, group_identity_key=GROUP_KEY)
        assert capture is not None
        if resume:
            path = root / "capture.json"
            write_json(path, serialize_capture(capture, episode_id=f"lost-journal-{index}"))
            capture = load_capture(path, deps=deps)
        captures.append(capture)
        (root / "logs/vat.jsonl").unlink()
    with pytest.raises(RuntimeError, match="require the durable vat journal"):
        build_tinker_group(deps=deps, group_id="missing-journal", step=0, rollouts=tuple(captures))


def test_split_turn_marker_in_assistant_text_is_not_a_compaction(tmp_path: Path) -> None:
    root, action = _episode(tmp_path, "quoted-wrapper", write_initial_audit=False)
    text = "No prior history." + SPLIT_SEPARATOR + "Quoted documentation."
    _append_events(root, _text_event(text))
    response = _record("quoted", call_id=None, content=text)
    capture = harvest_episode([action, response], episode_root=root, group_identity_key=GROUP_KEY)
    assert capture is not None
    assert [item.kind for item in capture.captures] == ["world_action", "final_text"]
    trajectory = PiCompactionAdapter().adapt(rollout_id=root.name, records=[action, response], evidence=root)
    assert trajectory.boundaries == ()


def test_split_turn_multiple_possible_wrappers_are_not_guessed(tmp_path: Path) -> None:
    root, records, _ = _split_episode(tmp_path, "nested-wrapper", (("History" + SPLIT_SEPARATOR + "Quoted prefix", "Prefix"),))
    with pytest.raises(RuntimeError, match="Ambiguous Pi split-turn wrapper"):
        harvest_episode(records, episode_root=root, group_identity_key=GROUP_KEY)


def test_pi_alignment_matches_exhaustive_small_ordered_cases() -> None:
    # Independent enumeration checks the production reachability algorithm,
    # including optional literal history and competing retry assignments.
    for allow_unconsumed in (False, True):
        for record_count in range(4):
            for texts in product("ab", repeat=record_count):
                records = [{"request_id": f"r{index}", "sampled_tokens": [index], "sampled_logprobs": [-0.2],
                            "content": text, "compaction_summary": True} for index, text in enumerate(texts)]
                for event_count in range(4):
                    for specs in product((('a', False), ('b', False), ('a', True)), repeat=event_count):
                        events = [PiSampleEvent(("compaction", text), "compaction_generation", 0, optional=optional)
                                  for text, optional in specs]

                        def enumerate_paths(event_index: int, cursor: int) -> set[tuple[tuple[int, int], ...]]:
                            if event_index == event_count:
                                return {()} if allow_unconsumed or cursor == record_count else set()
                            text, optional = specs[event_index]
                            paths = enumerate_paths(event_index + 1, cursor) if optional else set()
                            candidates = range(cursor, record_count) if allow_unconsumed else range(cursor, min(cursor + 1, record_count))
                            for index in candidates:
                                if texts[index] == text:
                                    paths.update(((event_index, index),) + tail for tail in enumerate_paths(event_index + 1, index + 1))
                            return paths

                        paths = enumerate_paths(0, 0)
                        if len(paths) == 1:
                            assert tuple(align_pi_samples(records, events, episode_id="enumerated", allow_unconsumed=allow_unconsumed)) == next(iter(paths))
                        else:
                            problem = "ambiguous proxy responses" if paths else (
                                "missing a proxy response consumed by Pi" if allow_unconsumed else "Tinker record count does not match"
                            )
                            with pytest.raises(RuntimeError, match=problem):
                                align_pi_samples(records, events, episode_id="enumerated", allow_unconsumed=allow_unconsumed)
