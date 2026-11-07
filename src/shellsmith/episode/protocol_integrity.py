"""Recompute Pi/Tinker/world protocol integrity from durable rollout artifacts.

Architecture map:
    scenario/model_protocol.json + logs/vat.jsonl + world turn artifacts
        -> per-request renderer/thinking/parse checks
        -> assistant tool-call / world-request / Pi-result correlation
        -> legacy adapter marker scan
        -> model_protocol_integrity report

Main callers:
    ``acceptance_audit.audit_rollout`` as an independent AND gate.

Why this exists:
    The world model's ``trainable`` flag describes simulator confidence, not wire
    correctness. Stack/protocol corruption must never be overridden by a generous
    world judgment. Harness-invalid tool calls that receive an ordinary Pi error
  and a world no-op remain trainable; only correlation bugs fail this gate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shellsmith.episode.world_artifacts import (
    load_turn_response,
)
from shellsmith.policy.reasoning import thinking_mode_for_effort


SCHEMA = "shellsmith.model_protocol_integrity.v1"
LEGACY_HISTORY_MARKER = re.compile(
    r"Previous tool-call history\s*\(not valid output\)",
    re.IGNORECASE,
)
UNPARSED_NATIVE_BLOCK = re.compile(
    r"<\|im_start\|>tool_call|<tool_call\b|unparsed_tool_calls",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ToolCallRecord:
    """One assistant tool call at the Pi projection boundary."""

    call_id: str
    name: str
    arguments: dict[str, Any]
    harness_rejected: bool = False


@dataclass(frozen=True)
class ModelRequestRecord:
    """One sampled Player request and its protocol evidence."""

    request_id: str
    renderer_name: str | None
    thinking_mode: str | None
    parse_termination: str | None
    unparsed_tool_call_count: int
    tool_calls: tuple[ToolCallRecord, ...]
    assistant_text: str
    assistant_content_types: tuple[str, ...]
    sampled_token_count: int | None = None
    # Pi context-compaction summary generation: sampled through the proxy but
    # never surfaced as an assistant message_end event, so it is excluded from
    # assistant-sample reconciliation and per-call protocol checks.
    compaction_summary: bool = False


def _load_object(path: Path, *, context: str) -> dict[str, Any]:
    if path.is_symlink():
        raise RuntimeError(f"{context} is a forbidden symlink: {path}.")
    if not path.is_file():
        raise RuntimeError(f"{context} is missing: {path}.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must contain a JSON object: {path}.")
    return value


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _tool_call_from_block(block: dict[str, Any]) -> ToolCallRecord | None:
    if block.get("type") != "toolCall":
        return None
    call_id = block.get("id")
    name = block.get("name")
    arguments = block.get("arguments")
    if not isinstance(call_id, str) or not call_id:
        return None
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(arguments, dict):
        arguments = {}
    return ToolCallRecord(
        call_id=call_id,
        name=name,
        arguments={str(key): value for key, value in arguments.items()},
    )


def _assistant_message_from_payload(payload: dict[str, Any]) -> tuple[str, tuple[str, ...], tuple[ToolCallRecord, ...]]:
    message = payload.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return "", (), ()
    content = message.get("content")
    if not isinstance(content, list):
        return "", (), ()
    text_parts: list[str] = []
    content_types: list[str] = []
    tool_calls: list[ToolCallRecord] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", "unknown"))
        content_types.append(block_type)
        if block_type == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        parsed = _tool_call_from_block(block)
        if parsed is not None:
            tool_calls.append(parsed)
    return "\n".join(text_parts), tuple(content_types), tuple(tool_calls)


def _assistant_payload_is_substantive(payload: dict[str, Any]) -> bool:
    """Ignore empty Pi message_end shells when counting Player samples."""
    text, content_types, tool_calls = _assistant_message_from_payload(payload)
    if tool_calls:
        return True
    if text.strip():
        return True
    return any(item in {"thinking", "toolCall"} for item in content_types)


def _iter_vat_envelopes(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink():
        raise RuntimeError(f"Vat journal is a forbidden symlink: {path}.")
    if not path.is_file():
        return []
    envelopes: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Malformed vat JSONL at {path}:{line_number}."
            ) from error
        if isinstance(raw, dict):
            envelopes.append(raw)
    return envelopes


def _harness_rejected_tool_ids(envelopes: list[dict[str, Any]]) -> set[str]:
    """Return tool-call IDs the harness actually rejected before execution.

    The only authoritative signal is the tool bridge's ``tool_request`` message
    with ``harness_rejected: true`` — the bridge emits exactly one per sibling.
    ``tool_execution_end`` with ``isError`` is deliberately NOT used here: a
    world-simulated ``is_error`` observation produces that same event for a
    perfectly valid call, and classifying it as rejected both false-failed the
    audit and exempted the call from the argument-match/exactly-once checks.
    """
    rejected: set[str] = set()
    for envelope in envelopes:
        message = envelope.get("message")
        if not isinstance(message, dict) or message.get("type") != "tool_request":
            continue
        if message.get("harness_rejected") is True:
            tool_call_id = message.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id:
                rejected.add(tool_call_id)
    return rejected


def _world_turn_maps(root: Path) -> tuple[dict[str, Path], dict[str, dict[str, Any]], list[str]]:
    """Map every request action ID (primary + siblings) to turn dirs and payloads."""
    action_to_turn_dir: dict[str, Path] = {}
    requests_by_action: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    turns_root: Path = root.joinpath("world", "turns")
    if not turns_root.is_dir():
        return action_to_turn_dir, requests_by_action, ["world/turns directory is missing"]
    for directory in sorted(turns_root.iterdir()):
        if directory.is_symlink():
            problems.append(f"{directory.name} is a forbidden symlinked turn directory")
            continue
        if not directory.is_dir():
            continue
        if not directory.name.isdigit():
            problems.append(f"invalid world turn directory {directory.name!r}")
            continue
        request_path: Path = directory.joinpath("request.json")
        response_path: Path = directory.joinpath("response.json")
        if not request_path.is_file() or not response_path.is_file():
            problems.append(f"{directory.name} lacks a complete request/response pair")
            continue
        request = _load_object(request_path, context=f"{directory.name} request")
        actions = request.get("actions", [request.get("action")])
        if not isinstance(actions, list) or not actions:
            problems.append(f"{directory.name} has no valid world action list")
            continue
        ids = []
        for action in actions:
            if (not isinstance(action, dict)
                    or not isinstance(action.get("id"), str) or not action["id"]
                    or not isinstance(action.get("name"), str) or not action["name"]
                    or not isinstance(action.get("arguments"), dict)):
                problems.append(f"{directory.name} contains a malformed world action")
                continue
            ids.append(action["id"])
        if len(ids) != len(set(ids)):
            problems.append(f"{directory.name} contains duplicate world action IDs")
        if "actions" in request and "action" in request and request["action"] != actions[0]:
            problems.append(f"{directory.name} primary action disagrees with first sibling")
        action_ids = _action_ids_from_request(request)
        if not action_ids:
            problems.append(f"{directory.name} request has no valid action ID")
            continue
        for action_id in action_ids:
            if action_id in action_to_turn_dir:
                problems.append(f"duplicate world action ID {action_id!r}")
                continue
            action_to_turn_dir[action_id] = directory
            requests_by_action[action_id] = request
    return action_to_turn_dir, requests_by_action, problems


def _action_ids_from_request(request: dict[str, Any]) -> list[str]:
    """Return sibling action ids in request order, falling back to primary action.id."""
    ids: list[str] = []
    actions = request.get("actions")
    if isinstance(actions, list):
        for item in actions:
            if not isinstance(item, dict):
                continue
            action_id = item.get("id")
            if isinstance(action_id, str) and action_id and action_id not in ids:
                ids.append(action_id)
    if ids:
        return ids
    action = request.get("action")
    if isinstance(action, dict):
        action_id = action.get("id")
        if isinstance(action_id, str) and action_id:
            return [action_id]
    return []


def _action_object_for_id(request: dict[str, Any], action_id: str) -> dict[str, Any] | None:
    actions = request.get("actions")
    if isinstance(actions, list):
        for item in actions:
            if isinstance(item, dict) and item.get("id") == action_id:
                return item
    action = request.get("action")
    if isinstance(action, dict) and action.get("id") == action_id:
        return action
    return None


def model_protocol_requests_from_rollout_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build model_protocol.requests[] rows from proxy rollout_records.jsonl objects.

    Main callers:
        ``populate_model_protocol_requests``, trainer after draining the hook proxy.
    """
    malformed = _request_metadata_problems({"requests": rows})
    if malformed:
        raise RuntimeError("Malformed rollout protocol evidence: " + "; ".join(malformed))
    requests: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            continue
        tool_calls_raw = row.get("tool_calls")
        tool_calls: list[dict[str, Any]] = []
        if isinstance(tool_calls_raw, list):
            for item in tool_calls_raw:
                if not isinstance(item, dict):
                    continue
                call_id = item.get("id")
                name = item.get("name")
                if not isinstance(call_id, str) or not isinstance(name, str):
                    continue
                arguments = item.get("arguments")
                if not isinstance(arguments, dict):
                    arguments = {}
                tool_calls.append(
                    {
                        "id": call_id,
                        "name": name,
                        "arguments": {
                            str(key): value for key, value in arguments.items()
                        },
                        "harness_rejected": item.get("harness_rejected") is True,
                    }
                )
        token_count = row.get("sampled_token_count")
        if token_count is None and isinstance(row.get("sampled_tokens"), list):
            token_count = len(row["sampled_tokens"])
        sampled_token_count = (
            int(token_count)
            if isinstance(token_count, int) and not isinstance(token_count, bool)
            else None
        )
        unparsed = row.get("unparsed_tool_call_count", 0)
        unparsed_count = (
            int(unparsed) if isinstance(unparsed, int) and not isinstance(unparsed, bool) else 0
        )
        content_types = row.get("assistant_content_types")
        if not isinstance(content_types, list):
            content_types = []
        requests.append(
            {
                "request_id": request_id,
                "renderer_name": (
                    str(row["renderer_name"])
                    if isinstance(row.get("renderer_name"), str)
                    else None
                ),
                "thinking_mode": (
                    str(row["thinking_mode"])
                    if isinstance(row.get("thinking_mode"), str)
                    else None
                ),
                "parse_termination": (
                    str(row["parse_termination"])
                    if isinstance(row.get("parse_termination"), str)
                    else None
                ),
                "unparsed_tool_call_count": unparsed_count,
                "tool_calls": tool_calls,
                "assistant_text": str(row.get("content") or row.get("assistant_text") or ""),
                "assistant_content_types": [
                    str(item) for item in content_types if isinstance(item, str)
                ],
                "sampled_token_count": sampled_token_count,
                "reasoning_content_present": (
                    row.get("reasoning_content_present") is True
                    or bool(str(row.get("reasoning_content") or "").strip())
                ),
                "thinking_forced_close": row.get("thinking_forced_close") is True,
                "thinking_tokens": row.get("thinking_tokens"),
                "response_tokens": row.get("response_tokens"),
                "compaction_summary": row.get("compaction_summary") is True,
            }
        )
    return requests


def populate_model_protocol_requests(
    root: Path,
    requests: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge sampled request rows into scenario/model_protocol.json.

    Keeps the existing config stub fields and replaces ``requests``.
    """
    root = root.expanduser().resolve()
    path: Path = root.joinpath("scenario", "model_protocol.json")
    if path.is_file():
        protocol = _load_object(path, context="model protocol")
    else:
        protocol = {
            "schema": "shellsmith.model_protocol.v1",
        }
    if protocol.get("schema") != "shellsmith.model_protocol.v1":
        raise RuntimeError(
            f"Cannot populate unsupported model_protocol schema at {path}."
        )
    protocol["requests"] = list(requests)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return protocol


def load_rollout_record_rows(root: Path) -> list[dict[str, Any]]:
    """Load proxy sample rows copied into the episode for protocol population."""
    candidates = (
        root.joinpath("logs", "policy_rollout_records.jsonl"),
        root.joinpath("logs", "tinker_rollout_records.jsonl"),
        root.joinpath("logs", "rollout_records.jsonl"),
    )
    found = None
    for path in candidates:
        if path.is_symlink():
            raise RuntimeError(f"Rollout record file is a forbidden symlink: {path}.")
        if not path.is_file():
            continue
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Malformed rollout record JSONL at {path}:{line_number}."
                ) from error
            if not isinstance(value, dict):
                raise RuntimeError(f"Rollout record must be an object at {path}:{line_number}.")
            rows.append(value)
        if found is not None and rows != found:
            raise RuntimeError("Conflicting rollout record files; cannot choose the sampled evidence.")
        found = rows
    return found if found is not None else []


def ensure_model_protocol_requests(root: Path) -> list[str]:
    """Populate empty model_protocol.requests from episode rollout records if present.

    Returns loud problems when the stub stayed empty despite available sample rows.
    """
    problems: list[str] = []
    protocol, protocol_problems = _load_model_protocol(root)
    problems.extend(protocol_problems)
    if protocol is None:
        return problems
    existing = protocol.get("requests")
    if isinstance(existing, list) and existing:
        return problems
    rows = load_rollout_record_rows(root)
    if rows:
        populate_model_protocol_requests(
            root,
            model_protocol_requests_from_rollout_rows(rows),
        )
        return problems
    return problems


def _load_model_protocol(root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    path: Path = root.joinpath("scenario", "model_protocol.json")
    if not path.is_file():
        return None, [f"scenario/model_protocol.json is missing: {path}."]
    payload = _load_object(path, context="model protocol")
    if payload.get("schema") != "shellsmith.model_protocol.v1":
        return payload, ["unsupported model_protocol schema"]
    return payload, []


def _request_records_from_protocol(protocol: dict[str, Any]) -> list[ModelRequestRecord]:
    records: list[ModelRequestRecord] = []
    raw_requests = protocol.get("requests")
    if not isinstance(raw_requests, list):
        return records
    for index, raw in enumerate(raw_requests):
        if not isinstance(raw, dict):
            continue
        request_id = raw.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            continue
        tool_calls: list[ToolCallRecord] = []
        raw_tools = raw.get("tool_calls")
        for item in raw_tools if isinstance(raw_tools, list) else []:
            if not isinstance(item, dict):
                continue
            call_id = item.get("id")
            name = item.get("name")
            arguments = item.get("arguments")
            if not isinstance(call_id, str) or not isinstance(name, str):
                continue
            if not isinstance(arguments, dict):
                arguments = {}
            tool_calls.append(
                ToolCallRecord(
                    call_id=call_id,
                    name=name,
                    arguments={str(key): value for key, value in arguments.items()},
                    harness_rejected=item.get("harness_rejected") is True,
                )
            )
        unparsed = raw.get("unparsed_tool_call_count", 0)
        unparsed_count = int(unparsed) if isinstance(unparsed, int) and not isinstance(unparsed, bool) else 0
        token_count = raw.get("sampled_token_count")
        sampled_token_count = (
            int(token_count)
            if isinstance(token_count, int) and not isinstance(token_count, bool)
            else None
        )
        records.append(
            ModelRequestRecord(
                request_id=request_id,
                renderer_name=(
                    str(raw["renderer_name"])
                    if isinstance(raw.get("renderer_name"), str)
                    else None
                ),
                thinking_mode=(
                    str(raw["thinking_mode"])
                    if isinstance(raw.get("thinking_mode"), str)
                    else None
                ),
                parse_termination=(
                    str(raw["parse_termination"])
                    if isinstance(raw.get("parse_termination"), str)
                    else None
                ),
                unparsed_tool_call_count=unparsed_count,
                tool_calls=tuple(tool_calls),
                assistant_text=str(raw.get("assistant_text") or ""),
                assistant_content_types=tuple(
                    str(item)
                    for item in (raw.get("assistant_content_types")
                                 if isinstance(raw.get("assistant_content_types"), list) else [])
                    if isinstance(item, str)
                ),
                sampled_token_count=sampled_token_count,
                compaction_summary=raw.get("compaction_summary") is True,
            )
        )
    return records


def _request_metadata_problems(protocol: dict[str, Any]) -> list[str]:
    """Reject malformed evidence instead of silently dropping or coercing it."""
    problems = []
    rows = protocol.get("requests")
    if not isinstance(rows, list):
        return ["model_protocol.requests must be a list"]
    seen = set()
    for index, row in enumerate(rows):
        label = f"model_protocol.requests[{index}]"
        if not isinstance(row, dict):
            problems.append(f"{label} must be an object")
            continue
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            problems.append(f"{label} lacks a valid request_id")
        elif request_id in seen:
            problems.append(f"duplicate model request ID {request_id!r}")
        else:
            seen.add(request_id)
        for key in ("renderer_name", "thinking_mode", "parse_termination", "assistant_text"):
            if key in row and not isinstance(row[key], str):
                problems.append(f"{label}.{key} must be a string")
        if "thinking_mode" in row and row["thinking_mode"] not in ("thinking", "disable_thinking"):
            problems.append(f"{label}.thinking_mode is not a supported wire mode")
        for key in ("unparsed_tool_call_count", "sampled_token_count"):
            if key == "sampled_token_count" and row.get(key) is None:
                continue  # Legacy captures do not always record token counts.
            if key in row and (type(row[key]) is not int or row[key] < 0):
                problems.append(f"{label}.{key} must be a nonnegative integer")
        if "compaction_summary" in row and type(row["compaction_summary"]) is not bool:
            problems.append(f"{label}.compaction_summary must be a boolean")
        types = row.get("assistant_content_types", [])
        if not isinstance(types, list) or any(not isinstance(item, str) for item in types):
            problems.append(f"{label}.assistant_content_types must be a string list")
        calls = row.get("tool_calls", [])
        if not isinstance(calls, list):
            problems.append(f"{label}.tool_calls must be a list")
            continue
        if row.get("compaction_summary") is True and calls:
            problems.append(f"{label} compaction summary contains tool calls")
        for call in calls:
            if (not isinstance(call, dict)
                    or not isinstance(call.get("id"), str) or not call["id"].strip()
                    or not isinstance(call.get("name"), str) or not call["name"].strip()
                    or not isinstance(call.get("arguments"), dict)):
                problems.append(f"{label} contains malformed tool-call evidence")
            elif "harness_rejected" in call and type(call["harness_rejected"]) is not bool:
                problems.append(f"{label} tool-call harness_rejected must be a boolean")
    return problems


def _request_records_from_vat(
    envelopes: list[dict[str, Any]],
) -> list[ModelRequestRecord]:
    """Derive assistant sample records from Pi vat events when protocol file is absent."""
    records: list[ModelRequestRecord] = []
    request_index = 0
    for envelope in envelopes:
        message = envelope.get("message")
        if not isinstance(message, dict):
            continue
        if message.get("type") != "agent_event":
            continue
        if message.get("event_type") != "message_end":
            continue
        payload = message.get("payload")
        if not isinstance(payload, dict):
            continue
        text, content_types, tool_calls = _assistant_message_from_payload(payload)
        projected = payload.get("message")
        if not isinstance(projected, dict) or projected.get("role") != "assistant":
            continue
        if not _assistant_payload_is_substantive(payload):
            continue
        request_index += 1
        records.append(
            ModelRequestRecord(
                request_id=f"vat-derived-{request_index:04d}",
                renderer_name=None,
                thinking_mode=None,
                parse_termination=None,
                unparsed_tool_call_count=0,
                tool_calls=tool_calls,
                assistant_text=text,
                assistant_content_types=content_types,
            )
        )
    return records


def _arguments_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True, ensure_ascii=False) == json.dumps(
        right, sort_keys=True, ensure_ascii=False
    )


def _vat_path_rewrites(envelopes: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """Return (workspace_host_path, vat_root_host_path) from the initialize envelope."""
    for envelope in envelopes:
        message = envelope.get("message")
        if not isinstance(message, dict) or message.get("type") != "initialize":
            continue
        workspace_path = message.get("workspace_path")
        if not isinstance(workspace_path, str) or not workspace_path:
            continue
        # Do NOT resolve() here. On macOS /tmp is a symlink to /private/tmp, so
        # resolving rewrites the workspace path to a form the recorded payloads
        # never contain — they keep the unresolved path the orchestrator saw —
        # making every host-path virtualization miss and spuriously fail
        # protocol integrity (a real acceptance gate) on local macOS audit hosts.
        # ``Path.parent`` is purely lexical and does not touch the filesystem, so
        # this mirrors the orchestrator, which virtualizes workspace_path verbatim.
        workspace = workspace_path
        vat_root = str(Path(workspace_path).parent)
        return workspace, vat_root
    return None, None


def _rewrite_host_paths_to_virtual(
    value: Any,
    *,
    workspace_host: str | None,
    vat_root_host: str | None,
) -> Any:
    """Mirror orchestrator virtualization: host vat paths become /workspace and /vat."""
    if isinstance(value, dict):
        return {
            str(key): _rewrite_host_paths_to_virtual(
                item,
                workspace_host=workspace_host,
                vat_root_host=vat_root_host,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_host_paths_to_virtual(
                item,
                workspace_host=workspace_host,
                vat_root_host=vat_root_host,
            )
            for item in value
        ]
    if not isinstance(value, str):
        return value
    rewritten = value
    if workspace_host:
        rewritten = rewritten.replace(workspace_host, "/workspace")
    if vat_root_host:
        rewritten = rewritten.replace(vat_root_host, "/vat")
    return rewritten


def _parse_termination_is_clean(value: str | None) -> bool:
    """Accept native stop and Tinker's clean stop_sequence/eos labels."""
    if value is None:
        return True
    normalized = str(value).strip().lower()
    return normalized in {"clean", "stop", "stop_sequence", "eos"}


def _scan_model_visible_text(root: Path, envelopes: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    texts: list[str] = []
    for envelope in envelopes:
        message = envelope.get("message")
        if not isinstance(message, dict):
            continue
        if message.get("type") != "agent_event" or message.get("event_type") != "message_end":
            continue
        payload = message.get("payload")
        if not isinstance(payload, dict):
            continue
        text, _, _ = _assistant_message_from_payload(payload)
        if text:
            texts.append(text)
    final_path: Path = root.joinpath("world", "final", "request.json")
    if final_path.is_file():
        final_request = _load_object(final_path, context="final request")
        final_text = final_request.get("player_final_response")
        if isinstance(final_text, str):
            texts.append(final_text)
    for index, text in enumerate(texts):
        if LEGACY_HISTORY_MARKER.search(text):
            problems.append(
                f"model-visible text contains legacy history marker at index {index}"
            )
        if UNPARSED_NATIVE_BLOCK.search(text):
            problems.append(
                f"model-visible text contains unparsed native tool-call residue at index {index}"
            )
    return problems


def audit_protocol_integrity(root: Path) -> dict[str, Any]:
    """Recompute protocol integrity for one rollout directory.

    Main callers:
        ``acceptance_audit.audit_rollout``.
    """
    root = root.expanduser().resolve()
    problems: list[str] = []
    # Model-authored anomalies (malformed tool syntax, truncation) that stay
    # trainable: reported for observability, but they never fail the gate.
    warnings: list[str] = []
    problems.extend(ensure_model_protocol_requests(root))
    run_config_path: Path = root.joinpath("run_config.json")
    run_config = _load_object(run_config_path, context="run_config") if run_config_path.is_file() else {}

    protocol, protocol_problems = _load_model_protocol(root)
    problems.extend(protocol_problems)

    vat_path: Path = root.joinpath("logs", "vat.jsonl")
    envelopes = _iter_vat_envelopes(vat_path)
    harness_rejected = _harness_rejected_tool_ids(envelopes)
    workspace_host, vat_root_host = _vat_path_rewrites(envelopes)

    expected_renderer = None
    expected_thinking = None
    if protocol is not None:
        for field in ("expected_renderer", "expected_thinking_mode"):
            if field in protocol and (not isinstance(protocol[field], str) or not protocol[field].strip()):
                problems.append(f"model_protocol.{field} must be a nonempty string")
        if isinstance(protocol.get("expected_renderer"), str):
            expected_renderer = protocol["expected_renderer"]
        if isinstance(protocol.get("expected_thinking_mode"), str):
            expected_thinking = protocol["expected_thinking_mode"]
    if expected_renderer is None and isinstance(run_config.get("player_renderer"), str):
        expected_renderer = run_config["player_renderer"]
    configured_renderer = run_config.get("player_renderer")
    if (isinstance(configured_renderer, str) and expected_renderer is not None
            and expected_renderer != configured_renderer):
        problems.append("model_protocol expected_renderer conflicts with run_config.player_renderer")
    if expected_thinking is not None and expected_thinking not in {"thinking", "disable_thinking"}:
        problems.append("model_protocol.expected_thinking_mode is not a supported wire mode")
    if "player_thinking" in run_config:
        try:
            configured_mode = thinking_mode_for_effort(run_config["player_thinking"])
        except ValueError as error:
            problems.append(f"run_config.player_thinking is invalid: {error}")
        else:
            if expected_thinking is None:
                expected_thinking = configured_mode
            elif expected_thinking != configured_mode:
                problems.append("model_protocol expected_thinking_mode conflicts with run_config.player_thinking")

    projected_records = _request_records_from_vat(envelopes)
    assistant_sample_count = len(projected_records)

    request_records: list[ModelRequestRecord] = []
    request_problems = _request_metadata_problems(protocol) if protocol is not None else []
    problems.extend(request_problems)
    if protocol is not None:
        request_records = _request_records_from_protocol(protocol)
    if not request_records and assistant_sample_count > 0:
        problems.append(
            "model_protocol.requests empty while vat has "
            f"{assistant_sample_count} assistant samples"
        )
        request_records = _request_records_from_vat(envelopes)
    elif protocol is None:
        request_records = _request_records_from_vat(envelopes)
    if not request_records:
        problems.append("no sampled assistant evidence is available")

    action_to_turn_dir, requests_by_action, turn_problems = _world_turn_maps(root)
    problems.extend(turn_problems)

    if protocol is not None and request_records and not any(
        "model_protocol.requests empty" in item for item in problems
    ):
        # Compaction-summary generations are sampled through the proxy but Pi
        # never emits an assistant message_end for them — only conversational
        # samples reconcile against vat events.
        protocol_count = sum(
            1 for record in request_records if not record.compaction_summary
        )
        if protocol_count != assistant_sample_count:
            problems.append(
                "model_protocol request count does not match completed assistant "
                f"samples: protocol={protocol_count} vat={assistant_sample_count}"
            )
        else:
            # Equal counts alone do not prove the same responses reached Pi.
            # Compare the ordered, conversational projections as well as the
            # proxy-to-world correlation below. Compaction has its own journal
            # alignment in harvest and is excluded from this projection.
            conversational = [record for record in request_records if not record.compaction_summary]
            for record, actual in zip(conversational, projected_records, strict=True):
                declared_calls = [(call.call_id, call.name, call.arguments) for call in record.tool_calls]
                actual_calls = [(call.call_id, call.name, call.arguments) for call in actual.tool_calls]
                if declared_calls != actual_calls:
                    problems.append(f"request {record.request_id} tool calls disagree with Pi assistant projection")
                if record.assistant_text.strip() != actual.assistant_text.strip():
                    problems.append(f"request {record.request_id} text disagrees with Pi assistant projection")
        if not request_problems:
            from shellsmith.credit.trajectory_pi import align_pi_samples, pi_sample_events

            try:
                align_pi_samples(protocol["requests"], pi_sample_events(root),
                                 episode_id=str(root.name), allow_unconsumed=False)
            except RuntimeError as error:
                problems.append(f"sample/compaction journal alignment failed: {error}")

    seen_call_ids: set[str] = set()
    pi_results_for_call: dict[str, int] = {}
    harness_requests: dict[str, list[dict[str, Any]]] = {}
    for envelope in envelopes:
        message = envelope.get("message")
        if not isinstance(message, dict) or message.get("type") != "tool_request":
            continue
        call_id = message.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            problems.append("tool_request message is missing tool_call_id")
            continue
        harness_requests.setdefault(call_id, []).append(message)

    for record in request_records:
        if record.compaction_summary:
            # Internal Pi summary generation: no tool calls, no world turn,
            # and Pi may legitimately request it without thinking. Nothing to
            # correlate.
            continue
        if expected_renderer is not None and record.renderer_name != expected_renderer:
            problems.append(
                f"request {record.request_id} renderer mismatch: "
                f"expected {expected_renderer!r}, observed {record.renderer_name!r}"
            )
        if expected_thinking is not None and record.thinking_mode != expected_thinking:
            problems.append(
                f"request {record.request_id} thinking-mode mismatch: "
                f"expected {expected_thinking!r}, observed {record.thinking_mode!r}"
            )
        if not _parse_termination_is_clean(record.parse_termination):
            # Model-authored malformed/truncated output: trainable negative
            # behavior per the world-turn contract ("harness-invalid Player
            # tool schemas alone do not make the rollout untrainable").
            warnings.append(
                f"request {record.request_id} parse_termination is not clean: "
                f"{record.parse_termination!r}"
            )
        if record.unparsed_tool_call_count != 0:
            warnings.append(
                f"request {record.request_id} has unparsed_tool_call_count="
                f"{record.unparsed_tool_call_count}"
            )
        if LEGACY_HISTORY_MARKER.search(record.assistant_text):
            problems.append(
                f"request {record.request_id} contains legacy history-marker text"
            )

        for call in record.tool_calls:
            if call.call_id in seen_call_ids:
                problems.append(f"duplicate assistant tool-call ID {call.call_id!r}")
            seen_call_ids.add(call.call_id)
            # Legacy imported journals can omit tool_request envelopes. When
            # present, correlate the bridge input as well as assistant/world.
            if harness_requests:
                requests = harness_requests.get(call.call_id, [])
                if len(requests) != 1:
                    problems.append(f"tool call {call.call_id!r} has {len(requests)} harness requests; expected 1")
                elif (requests[0].get("tool_name") != call.name
                      or requests[0].get("arguments") != call.arguments):
                    problems.append(f"tool call {call.call_id!r} disagrees with harness request")

            rejected = call.harness_rejected or call.call_id in harness_rejected
            has_world_request = call.call_id in action_to_turn_dir
            if call.harness_rejected and call.call_id not in harness_rejected:
                problems.append(f"tool call {call.call_id!r} rejection lacks a harness event")

            if rejected and has_world_request:
                # A rejected sibling legitimately shares its turn artifact with
                # valid siblings, so load with the full expected id set (the
                # top-level action_id is the primary/first sibling, not this
                # call). The defect is only a rejected slot that still carries
                # a successful Player-facing observation.
                request = requests_by_action[call.call_id]
                expected_ids = _action_ids_from_request(request)
                response_path: Path = action_to_turn_dir[call.call_id].joinpath(
                    "response.json"
                )
                try:
                    response = load_turn_response(
                        response_path,
                        expected_action_id=(
                            expected_ids[0] if expected_ids else call.call_id
                        ),
                        expected_action_ids=(
                            tuple(expected_ids) if expected_ids else None
                        ),
                    )
                except RuntimeError as error:
                    problems.append(
                        f"Pi-rejected tool call {call.call_id!r} shares a world "
                        f"response artifact that failed validation: {error}"
                    )
                else:
                    slot = next(
                        (
                            item
                            for item in response.tool_slots
                            if item.action_id == call.call_id
                        ),
                        None,
                    )
                    if slot is None:
                        # Legacy single-tool artifact: the primary response IS
                        # this call's observation.
                        if not response.tool_response.is_error:
                            problems.append(
                                "Pi-rejected tool call has a successful world "
                                f"response: {call.call_id!r}"
                            )
                    elif (
                        slot.tool_response is not None
                        and not slot.harness_rejected
                        and not slot.tool_response.is_error
                    ):
                        problems.append(
                            "Pi-rejected tool call has a successful world "
                            f"response: {call.call_id!r}"
                        )
                continue

            if rejected:
                continue

            if not has_world_request:
                problems.append(
                    f"validated tool call {call.call_id!r} has no world request"
                )
                continue

            request = requests_by_action[call.call_id]
            action = _action_object_for_id(request, call.call_id)
            if not isinstance(action, dict):
                problems.append(f"world request for {call.call_id!r} lacks action object")
                continue
            if action.get("name") != call.name:
                problems.append(
                    f"tool name mismatch for {call.call_id!r}: "
                    f"assistant={call.name!r} world={action.get('name')!r}"
                )
            world_args = action.get("arguments")
            if not isinstance(world_args, dict):
                world_args = {}
            assistant_args = _rewrite_host_paths_to_virtual(
                call.arguments,
                workspace_host=workspace_host,
                vat_root_host=vat_root_host,
            )
            if not isinstance(assistant_args, dict):
                assistant_args = {}
            if not _arguments_match(assistant_args, world_args):
                problems.append(f"tool arguments mismatch for {call.call_id!r}")

    for action_id in action_to_turn_dir:
        if action_id in harness_rejected:
            continue
        if not request_records and assistant_sample_count == 0:
            continue
        if action_id not in seen_call_ids:
            problems.append(
                f"world request {action_id!r} has no matching assistant tool call"
            )

    for envelope in envelopes:
        message = envelope.get("message")
        if not isinstance(message, dict):
            continue
        if message.get("type") != "tool_response":
            continue
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            problems.append("tool_response message is missing tool_call_id")
            continue
        pi_results_for_call[tool_call_id] = pi_results_for_call.get(tool_call_id, 0) + 1

    for tool_call_id in pi_results_for_call:
        if tool_call_id not in seen_call_ids:
            problems.append(f"Pi tool result {tool_call_id!r} has no matching assistant tool call")
    for tool_call_id in harness_requests:
        if tool_call_id not in seen_call_ids:
            problems.append(f"harness tool request {tool_call_id!r} has no matching assistant tool call")

    for action_id in seen_call_ids | set(action_to_turn_dir):
        count = pi_results_for_call.get(action_id, 0)
        if count != 1:
            problems.append(
                f"tool result for {action_id!r} was returned to Pi {count} times; expected 1"
            )

    for finding in _scan_model_visible_text(root, envelopes):
        # Unparsed tool-call residue is the MODEL emitting broken tool syntax
        # (which then rides visible text) — on-policy behavior the reward
        # should punish, not harness corruption. Keep it visible as a warning.
        if "unparsed native tool-call residue" in finding:
            warnings.append(finding)
        else:
            problems.append(finding)

    # Sibling order stability: world turn indices should increase with first appearance.
    ordered_calls: list[str] = []
    for record in request_records:
        for call in record.tool_calls:
            if call.call_id not in ordered_calls:
                ordered_calls.append(call.call_id)
    turn_indices = [
        int(action_to_turn_dir[call_id].name)
        for call_id in ordered_calls
        if call_id in action_to_turn_dir
    ]
    if turn_indices != sorted(turn_indices):
        problems.append("world turn order does not preserve assistant sibling order")

    passed = not problems
    return {
        "schema": SCHEMA,
        "passed": passed,
        "warnings": warnings,
        "expected_renderer": expected_renderer,
        "expected_thinking_mode": expected_thinking,
        "request_count": len(request_records),
        "assistant_sample_count": assistant_sample_count,
        "world_turn_count": len(action_to_turn_dir),
        "harness_rejected_tool_count": len(harness_rejected),
        "problems": problems,
        "model_protocol_present": protocol is not None,
    }
