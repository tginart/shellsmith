"""Pi journal/sample alignment, including multi-request compaction boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from .compaction import CompactionAwareTrajectory, CompactionBoundary, CompactionTurn, RequestKind


# Pi 0.84.1 core/compaction/compaction.js compact(), not model-authored framing.
_SPLIT_TURN_SEPARATOR = "\n\n---\n\n**Turn Context (split turn):**\n\n"
_NO_PRIOR_HISTORY = "No prior history."


@dataclass(frozen=True)
class PiSampleEvent:
    """One sampled part of a journal event; several parts can share a boundary."""

    signature: tuple[str, Any]
    kind: RequestKind
    sequence_index: int
    boundary_index: int | None = None
    summary: str | None = None
    compaction_part: str | None = None
    optional: bool = False


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Required Pi rollout artifact is absent: {path}.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Pi rollout artifact must contain an object: {path}.")
    return value


def _compaction_parts(envelope: dict[str, Any], compaction: dict[str, Any]) -> list[tuple[str, str, bool]]:
    """Return (sample text, part name, optional) using only pinned Pi framing.

    The sentinel history may be literal OR a sampled identical response. Keep
    both possibilities until ordered record alignment proves one; never invent
    a record for the literal or silently drop a possibly sampled history.
    """
    summary = compaction["summary"].strip()
    details = compaction.get("details")
    file_details = isinstance(details, dict) and all(
        isinstance(details.get(key), list) and all(isinstance(path, str) for path in details[key])
        for key in ("readFiles", "modifiedFiles")
    )
    if file_details:
        sections = []
        for key, tag in (("readFiles", "read-files"), ("modifiedFiles", "modified-files")):
            if details[key]:
                sections.append(f"<{tag}>\n" + "\n".join(details[key]) + f"\n</{tag}>")
        if sections:
            suffix = "\n\n" + "\n\n".join(sections)
            if not summary.endswith(suffix):
                raise RuntimeError("Pi compaction file-operation details do not match its summary.")
            summary = summary[:-len(suffix)].strip()

    payload = envelope["message"]["payload"]
    tokens_before = compaction.get("tokens_before")
    pinned_event = (
        envelope.get("direction") == "runner_to_host"
        and payload.get("aborted") is False
        and payload.get("reason") in {"threshold", "overflow", "manual"}
        and isinstance(compaction.get("first_kept_entry_id"), str)
        and bool(compaction["first_kept_entry_id"].strip())
        and isinstance(tokens_before, int) and not isinstance(tokens_before, bool) and tokens_before >= 0
        and file_details
    )
    if not pinned_event or _SPLIT_TURN_SEPARATOR not in summary:
        return [(summary, "summary", False)]
    if summary.count(_SPLIT_TURN_SEPARATOR) != 1:
        raise RuntimeError("Ambiguous Pi split-turn wrapper: multiple history/prefix separators.")
    history, prefix = (part.strip() for part in summary.split(_SPLIT_TURN_SEPARATOR))
    if not history or not prefix:
        raise RuntimeError("Pi split-turn compaction has an empty history or prefix summary.")
    return [(history, "history", history == _NO_PRIOR_HISTORY), (prefix, "turn_prefix", False)]


def pi_sample_events(episode_root: Path) -> list[PiSampleEvent]:
    """Interpret the durable journal once for both harvest and credit alignment."""
    path = episode_root / "logs/vat.jsonl"
    if not path.is_file():
        raise RuntimeError(f"Vat audit log is absent: {path}.")
    events: list[PiSampleEvent] = []
    known_summaries: dict[str, int] = {}
    compaction_count = 0
    pending_boundary: int | None = None
    sequence_index = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            envelope = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Malformed vat JSONL at {path}:{line_number}.") from error
        message = envelope.get("message") if isinstance(envelope, dict) else None
        if not isinstance(message, dict) or message.get("type") != "agent_event":
            continue
        event_type = message.get("event_type")
        payload = message.get("payload")
        if event_type == "compaction_end":
            if not isinstance(payload, dict) or payload.get("aborted") is True:
                continue
            compaction = payload.get("compaction")
            if compaction is None:
                # Pinned Pi also emits non-aborted failures with no result.
                continue
            summary = compaction.get("summary") if isinstance(compaction, dict) else None
            if not isinstance(summary, str) or not summary.strip():
                raise RuntimeError("Successful Pi compaction is missing its summary.")
            summary = summary.strip()
            boundary_index = compaction_count
            compaction_count += 1
            known_summaries[summary] = boundary_index
            for text, part, optional in _compaction_parts(envelope, compaction):
                events.append(PiSampleEvent(
                    signature=("compaction", text), kind="compaction_generation",
                    sequence_index=sequence_index, boundary_index=boundary_index,
                    summary=summary, compaction_part=part, optional=optional,
                ))
            pending_boundary = boundary_index
            sequence_index += 1
            continue
        if event_type != "message_end" or not isinstance(payload, dict):
            continue
        projected = payload.get("message")
        if not isinstance(projected, dict) or projected.get("role") != "assistant":
            continue
        content = projected.get("content")
        if not isinstance(content, list):
            raise RuntimeError("Projected assistant message content must be a list.")
        tool_ids = tuple(
            str(block["id"]) for block in content
            if isinstance(block, dict) and block.get("type") == "toolCall" and block.get("id")
        )
        if tool_ids:
            events.append(PiSampleEvent(
                signature=("tools", tool_ids), kind="world_action",
                sequence_index=sequence_index, boundary_index=pending_boundary,
            ))
            pending_boundary = None
            continue
        text = "\n".join(
            str(block.get("text", "")) for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        if not text and not any(
            isinstance(block, dict) and block.get("type") in {"thinking", "toolCall"} for block in content
        ):
            continue
        replay_boundary = known_summaries.get(text)
        events.append(PiSampleEvent(
            signature=("text", text),
            kind="compaction_replay" if replay_boundary is not None else "final_text",
            sequence_index=sequence_index,
            boundary_index=replay_boundary if replay_boundary is not None else pending_boundary,
            summary=text if replay_boundary is not None else None,
        ))
        pending_boundary = None
    return events


def _record_signature(record: Any) -> tuple[str, Any] | None:
    row = record if isinstance(record, dict) else getattr(record, "protocol_row", None)
    if not isinstance(row, dict) or not row:
        return None
    text = str(row.get("content") or row.get("assistant_text") or "").strip()
    if row.get("compaction_summary") is True:
        return ("compaction", text)
    tool_ids = tuple(
        str(item["id"]) for item in row.get("tool_calls") or ()
        if isinstance(item, dict) and item.get("id")
    )
    return ("tools", tool_ids) if tool_ids else ("text", text)


def _record_training_identity(record: Any) -> tuple[Any, ...]:
    row = record if isinstance(record, dict) else getattr(record, "protocol_row", {})
    row = row if isinstance(row, dict) else {}
    tokens = getattr(record, "sampled_tokens", None)
    logprobs = getattr(record, "sampled_logprobs", None)
    return (
        getattr(record, "request_id", None) or row.get("request_id"),
        tuple(tokens if tokens is not None else row.get("sampled_tokens") or ()),
        tuple(logprobs if logprobs is not None else row.get("sampled_logprobs") or ()),
    )


def align_pi_samples(
    records: Sequence[Any], events: Sequence[PiSampleEvent], *, episode_id: str, allow_unconsumed: bool,
) -> list[tuple[int, int]]:
    """Return (event, record) indices only when every complete alignment agrees.

    A suffix reachability table reserves later genuine requests and permits Pi's
    optional literal history. Comparing all viable transitions rejects retries
    with different token identities and sampled-versus-literal ambiguities.
    Runtime/memory are bounded by journal parts times records, not retry paths.
    """
    signatures = [_record_signature(record) for record in records]
    identities = [_record_training_identity(record) for record in records]
    event_count, record_count = len(events), len(records)

    def matches(event_index: int, record_index: int) -> bool:
        event = events[event_index]
        signature = signatures[record_index]
        if signature is not None:
            return signature == event.signature
        if allow_unconsumed:
            return False
        # Legacy/durable captures lack protocol rows. Their ordered count remains
        # strict; split summaries additionally need exact persisted sample text.
        if event.compaction_part in {"history", "turn_prefix"}:
            return str(getattr(records[record_index], "content", "")).strip() == event.signature[1]
        return True

    suffix = [bytearray(record_count + 1) for _ in range(event_count + 1)]
    suffix[event_count][record_count] = 1
    if allow_unconsumed:
        suffix[event_count][:] = b"\x01" * (record_count + 1)
    for event_index in range(event_count - 1, -1, -1):
        optional = events[event_index].optional
        suffix[event_index][record_count] = optional and suffix[event_index + 1][record_count]
        for record_index in range(record_count - 1, -1, -1):
            suffix[event_index][record_index] = (
                (optional and suffix[event_index + 1][record_index])
                or (matches(event_index, record_index) and suffix[event_index + 1][record_index + 1])
                or (allow_unconsumed and suffix[event_index][record_index + 1])
            )
    if not suffix[0][0]:
        if allow_unconsumed:
            raise RuntimeError(
                f"Episode {episode_id} is missing a proxy response consumed by Pi: no complete ordered alignment"
            )
        raise RuntimeError(
            "Tinker record count does not match sampled Pi blocks, or response content differs: "
            f"records={record_count} blocks={event_count}."
        )

    reachable = bytearray(record_count + 1)
    reachable[0] = 1
    for event_index, event in enumerate(events):
        following = bytearray(record_count + 1)
        choices: set[tuple[Any, ...] | None] = set()
        for record_index in range(record_count + 1):
            if not reachable[record_index] or not suffix[event_index][record_index]:
                continue
            if event.optional and suffix[event_index + 1][record_index]:
                choices.add(None)
                following[record_index] = 1
            if record_index < record_count:
                if allow_unconsumed and suffix[event_index][record_index + 1]:
                    reachable[record_index + 1] = 1
                if matches(event_index, record_index) and suffix[event_index + 1][record_index + 1]:
                    choices.add(identities[record_index])
                    following[record_index + 1] = 1
            if len(choices) > 1:
                raise RuntimeError(
                    f"Episode {episode_id} has ambiguous proxy responses for a "
                    f"Pi-consumed {event.signature[0]} at event index {event_index}; "
                    "CISPO cannot choose which sampled tokens Pi actually consumed"
                )
        reachable = following

    selected: list[tuple[int, int]] = []
    event_index = record_index = 0
    while event_index < event_count:
        if events[event_index].optional and suffix[event_index + 1][record_index]:
            event_index += 1
        elif record_index < record_count and matches(event_index, record_index) and suffix[event_index + 1][record_index + 1]:
            selected.append((event_index, record_index))
            event_index += 1
            record_index += 1
        else:
            record_index += 1
    return selected


class PiCompactionAdapter:
    """Align consumed requests while preserving one seam per compaction event."""

    harness_id = "pi"

    def adapt(
        self, *, rollout_id: str, records: Sequence[object],
        evidence: Path | Sequence[dict[str, Any]] | None = None,
    ) -> CompactionAwareTrajectory:
        if not isinstance(evidence, Path):
            raise RuntimeError("Pi compaction adaptation requires the rollout root path.")
        action_to_turn: dict[str, int] = {}
        for request_path in sorted(evidence.joinpath("world", "turns").glob("*/request.json")):
            request = _load_object(request_path)
            action_ids = [
                str(item["id"]) for item in request.get("actions") or ()
                if isinstance(item, dict) and item.get("id")
            ]
            if not action_ids:
                action = request.get("action")
                if isinstance(action, dict) and action.get("id"):
                    action_ids = [str(action["id"])]
            if not action_ids:
                raise RuntimeError(f"World request is missing action.id: {request_path}.")
            for action_id in dict.fromkeys(action_ids):
                if action_id in action_to_turn:
                    raise RuntimeError(f"World action ID appears twice: {action_id!r}.")
                action_to_turn[action_id] = int(request_path.parent.name)

        events = pi_sample_events(evidence)
        alignment = align_pi_samples(records, events, episode_id=rollout_id, allow_unconsumed=False)
        turns: list[CompactionTurn] = []
        generations: dict[str, list[CompactionTurn]] = {}
        for event_index, record_index in alignment:
            event = events[event_index]
            indices: tuple[int, ...] = ()
            if event.kind == "world_action":
                tool_ids = event.signature[1]
                missing = [value for value in tool_ids if value not in action_to_turn]
                if missing:
                    raise RuntimeError(f"Assistant tool calls are absent from world requests: {missing!r}.")
                indices = tuple(dict.fromkeys(action_to_turn[value] for value in tool_ids))
            boundary_id = (
                f"{rollout_id}:compaction:{event.boundary_index}" if event.boundary_index is not None else None
            )
            turn = CompactionTurn(
                record=records[record_index], kind=event.kind, world_turn_indices=indices,
                sequence_index=event.sequence_index, boundary_id=boundary_id, summary=event.summary,
                retained_context_evidence=(
                    ("pi_compaction_end", f"summary_part:{event.compaction_part}") if event.compaction_part else ()
                ),
            )
            turns.append(turn)
            if turn.kind == "compaction_generation" and boundary_id is not None:
                generations.setdefault(boundary_id, []).append(turn)

        boundaries: list[CompactionBoundary] = []
        for boundary_id, generated in generations.items():
            # The final generator commits this seam; all generators retain the
            # same boundary ID and pre-compaction sequence, each training once.
            last = generated[-1]
            continuation = next(
                (candidate for candidate in turns if candidate.sequence_index == last.sequence_index + 1), None,
            )
            boundaries.append(CompactionBoundary(
                boundary_id=boundary_id, generation_request_id=last.request_id,
                next_request_id=continuation.request_id if continuation is not None else None,
                sequence_before=last.sequence_index,
                sequence_after=last.sequence_index + 1 if continuation is not None else None,
                summary=last.summary or "",
                retained_context_evidence=(
                    "pi_compaction_end",
                    *(f"generation_request:{turn.request_id}" for turn in generated),
                    "continuation" if continuation is not None else "terminal_compaction",
                ),
            ))
        return CompactionAwareTrajectory(
            rollout_id=rollout_id, harness_id=self.harness_id, turns=tuple(turns), boundaries=tuple(boundaries),
        ).validate()
