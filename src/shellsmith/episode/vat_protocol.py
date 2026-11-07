"""Strict JSONL protocol shared by Python and the Node Pi-SDK vat runner.

Architecture map:
    Python -> Node:
        initialize -> prompt -> zero or more tool_response messages -> shutdown

    Node -> Python:
        initialized -> agent_event/tool_request* -> prompt_complete ->
        shutdown_complete

    A tool request deliberately suspends its JavaScript `execute()` Promise.
    Python resumes that Promise only after the Game Master has produced and
    persisted the corresponding private turn result.

Important invariants:
    - One JSON object occupies exactly one LF-terminated line.
    - Every message carries the exact protocol version.
    - `tool_call_id` is the correlation key for the only data path back into Pi.
    - Unknown message types and malformed fields fail the episode explicitly.

Why this exists:
    Pi session events are an implementation detail of the Node process. The
    Python orchestrator needs a much smaller stable protocol that cannot silently
    execute a local tool or confuse an old response with a new tool call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from shellsmith.types import JsonValue


VAT_PROTOCOL_VERSION = 1
DEFAULT_VIRTUAL_TOOLS = ("read", "bash", "edit", "write", "grep", "find", "ls")


def _required_string(raw: dict[str, Any], key: str, *, context: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context}.{key} must be a non-empty string.")
    return value


def _required_bool(raw: dict[str, Any], key: str, *, context: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise RuntimeError(f"{context}.{key} must be a boolean.")
    return value


def _required_object(
    raw: dict[str, Any], key: str, *, context: str
) -> dict[str, JsonValue]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"{context}.{key} must be a JSON object.")
    return {str(child_key): child_value for child_key, child_value in value.items()}


def _validate_envelope(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RuntimeError("Vat protocol message must be a JSON object.")
    if raw.get("protocol_version") != VAT_PROTOCOL_VERSION:
        raise RuntimeError(
            "Vat protocol version mismatch: "
            f"expected {VAT_PROTOCOL_VERSION}, got {raw.get('protocol_version')!r}."
        )
    _required_string(raw, "type", context="vat message")
    return raw


@dataclass(frozen=True)
class VatInitialized:
    """The Node runner has constructed exactly one in-memory Pi session."""

    request_id: str
    session_id: str
    enabled_tools: tuple[str, ...]


@dataclass(frozen=True)
class VatToolRequest:
    """One Pi tool executor waiting for a world-simulator observation."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, JsonValue]
    harness_rejected: bool = False
    harness_reject_reason: str | None = None


@dataclass(frozen=True)
class VatAgentEvent:
    """A normalized non-authoritative Pi lifecycle event for audit."""

    event_type: str
    payload: dict[str, JsonValue]


@dataclass(frozen=True)
class VatPromptComplete:
    """The Pi agent loop settled after a prompt and all of its tool calls."""

    request_id: str
    final_text: str
    message_count: int
    terminated: bool


@dataclass(frozen=True)
class VatShutdownComplete:
    """The Node runner disposed its in-memory Pi session cleanly."""

    request_id: str


@dataclass(frozen=True)
class VatRunnerError:
    """A fatal or command-scoped failure reported by the Node process."""

    message: str
    request_id: str | None
    fatal: bool


VatRunnerMessage: TypeAlias = (
    VatInitialized
    | VatToolRequest
    | VatAgentEvent
    | VatPromptComplete
    | VatShutdownComplete
    | VatRunnerError
)


@dataclass(frozen=True)
class VatToolResponse:
    """The world result used to resolve one suspended Pi tool Promise.

    `content` may be a plain string or Pi-compatible text/image blocks. The Node
    boundary normalizes a string into one text block. `details` remains private
    harness metadata and must not contain reward or simulator commentary.
    """

    tool_call_id: str
    content: JsonValue
    is_error: bool
    details: dict[str, JsonValue] | None = None
    terminate: bool = False
    omit_player_facing: bool = False

    def to_wire(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "protocol_version": VAT_PROTOCOL_VERSION,
            "type": "tool_response",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
            "is_error": self.is_error,
            "terminate": self.terminate,
            "omit_player_facing": self.omit_player_facing,
        }
        if self.details is not None:
            payload["details"] = self.details
        return payload


def initialize_command(
    *,
    request_id: str,
    episode_id: str,
    workspace_path: str,
    agent_home_path: str,
    model_provider: str,
    model_id: str,
    model_base_url: str,
    model_api: str,
    model_context_window: int,
    model_max_tokens: int,
    model_reasoning: bool,
    thinking_level: str,
    enabled_tools: tuple[str, ...] = DEFAULT_VIRTUAL_TOOLS,
    api_key_env: str | None = None,
    pi_sdk_module_path: str | None = None,
    compaction_reserve_tokens: int = 65536,
    compaction_keep_recent_tokens: int = 20000,
    retry: dict | None = None,
) -> dict[str, JsonValue]:
    """Build the only command allowed to construct the Pi session."""
    model: dict[str, JsonValue] = {
        "provider": model_provider,
        "id": model_id,
        "base_url": model_base_url,
        "api": model_api,
        "context_window": model_context_window,
        "max_tokens": model_max_tokens,
        "reasoning": model_reasoning,
        "thinking_level": thinking_level,
    }
    if api_key_env is not None:
        model["api_key_env"] = api_key_env
    if (compaction_reserve_tokens, compaction_keep_recent_tokens) != (65536, 20000):
        model["compaction"] = {"reserve_tokens": compaction_reserve_tokens,
                               "keep_recent_tokens": compaction_keep_recent_tokens}
    command: dict[str, JsonValue] = {
        "protocol_version": VAT_PROTOCOL_VERSION,
        "type": "initialize",
        "request_id": request_id,
        "episode_id": episode_id,
        "workspace_path": workspace_path,
        "agent_home_path": agent_home_path,
        "model": model,
        "enabled_tools": list(enabled_tools),
    }
    if pi_sdk_module_path is not None:
        command["pi_sdk_module_path"] = pi_sdk_module_path
    if retry is not None:
        command["retry"] = retry
    return command


def prompt_command(*, request_id: str, text: str) -> dict[str, JsonValue]:
    """Build one prompt command for the already initialized Pi session."""
    if not text.strip():
        raise RuntimeError("Vat prompt text must not be empty.")
    return {
        "protocol_version": VAT_PROTOCOL_VERSION,
        "type": "prompt",
        "request_id": request_id,
        "text": text,
    }


def shutdown_command(*, request_id: str) -> dict[str, JsonValue]:
    """Build the graceful process shutdown command."""
    return {
        "protocol_version": VAT_PROTOCOL_VERSION,
        "type": "shutdown",
        "request_id": request_id,
    }


def encode_jsonl(message: dict[str, JsonValue]) -> bytes:
    """Serialize one compact LF-delimited wire record."""
    return (
        json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def decode_runner_message(line: bytes | str) -> VatRunnerMessage:
    """Parse and strictly type one Node-to-Python protocol record."""
    try:
        text = line.decode("utf-8") if isinstance(line, bytes) else line
        raw = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Vat runner emitted malformed JSONL: {error}.") from error
    message = _validate_envelope(raw)
    message_type = message["type"]

    if message_type == "initialized":
        tools = message.get("enabled_tools")
        if not isinstance(tools, list) or not all(
            isinstance(tool, str) and tool for tool in tools
        ):
            raise RuntimeError("initialized.enabled_tools must be a string array.")
        return VatInitialized(
            request_id=_required_string(message, "request_id", context="initialized"),
            session_id=_required_string(message, "session_id", context="initialized"),
            enabled_tools=tuple(tools),
        )

    if message_type == "tool_request":
        harness_rejected = message.get("harness_rejected", False)
        if not isinstance(harness_rejected, bool):
            raise RuntimeError("tool_request.harness_rejected must be a boolean.")
        harness_reject_reason = message.get("harness_reject_reason")
        if harness_reject_reason is not None and not isinstance(harness_reject_reason, str):
            raise RuntimeError(
                "tool_request.harness_reject_reason must be a string or null."
            )
        return VatToolRequest(
            tool_call_id=_required_string(
                message, "tool_call_id", context="tool_request"
            ),
            tool_name=_required_string(message, "tool_name", context="tool_request"),
            arguments=_required_object(
                message, "arguments", context="tool_request"
            ),
            harness_rejected=harness_rejected,
            harness_reject_reason=harness_reject_reason,
        )

    if message_type == "agent_event":
        return VatAgentEvent(
            event_type=_required_string(
                message, "event_type", context="agent_event"
            ),
            payload=_required_object(message, "payload", context="agent_event"),
        )

    if message_type == "prompt_complete":
        message_count = message.get("message_count")
        if isinstance(message_count, bool) or not isinstance(message_count, int):
            raise RuntimeError("prompt_complete.message_count must be an integer.")
        final_text = message.get("final_text")
        if not isinstance(final_text, str):
            raise RuntimeError("prompt_complete.final_text must be a string.")
        return VatPromptComplete(
            request_id=_required_string(
                message, "request_id", context="prompt_complete"
            ),
            final_text=final_text,
            message_count=message_count,
            terminated=_required_bool(
                message, "terminated", context="prompt_complete"
            ),
        )

    if message_type == "shutdown_complete":
        return VatShutdownComplete(
            request_id=_required_string(
                message, "request_id", context="shutdown_complete"
            )
        )

    if message_type == "error":
        request_id = message.get("request_id")
        if request_id is not None and not isinstance(request_id, str):
            raise RuntimeError("error.request_id must be a string or null.")
        return VatRunnerError(
            message=_required_string(message, "message", context="error"),
            request_id=request_id,
            fatal=_required_bool(message, "fatal", context="error"),
        )

    raise RuntimeError(f"Unknown vat runner message type: {message_type!r}.")


def runner_message_kind(message: VatRunnerMessage) -> Literal[
    "initialized",
    "tool_request",
    "agent_event",
    "prompt_complete",
    "shutdown_complete",
    "error",
]:
    """Return the stable wire name for logging without inspecting dataclass names."""
    if isinstance(message, VatInitialized):
        return "initialized"
    if isinstance(message, VatToolRequest):
        return "tool_request"
    if isinstance(message, VatAgentEvent):
        return "agent_event"
    if isinstance(message, VatPromptComplete):
        return "prompt_complete"
    if isinstance(message, VatShutdownComplete):
        return "shutdown_complete"
    return "error"
