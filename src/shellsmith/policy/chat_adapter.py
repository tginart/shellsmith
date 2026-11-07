"""Native OpenAI-chat ↔ Tinker Qwen message/tool adapter.

Architecture map:
    Pi OpenAI payload (messages + tools)
      -> openai_tools_to_tinker_specs / openai_messages_to_tinker_messages
      -> renderer.create_conversation_prefix_with_tools + conversation
      -> sample / parse_response
      -> tinker_message_to_openai_assistant (ThinkingPart + ToolCall)

Why this exists:
    The Player must not see an invented plaintext tool language. Structured
    tool schemas, tool calls, tool results, and thinking stay structured across
    the Pi <-> Tinker boundary. Ported from the original
    monorepo's Tinker chat adapter without image support (the
    Player is text-only).

Main callers:
    ``policy.sampler.TinkerPlayerSampler.sample``.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Mapping


class TinkerNativeProtocolError(RuntimeError):
    """Raised when native tool/thinking conversion or parse integrity fails."""


def openai_tools_to_tinker_specs(tools: Any) -> tuple[list[dict[str, Any]], str]:
    """Convert Pi OpenAI tool objects into Cookbook ToolSpec dicts.

    Returns:
        (tool_specs, sha256_hex of normalized tool list)
    """

    if tools is None:
        return [], _tool_schema_sha256([])
    if not isinstance(tools, list):
        raise TinkerNativeProtocolError("OpenAI tools must be a list.")

    specs: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise TinkerNativeProtocolError(f"Tool at index {index} is not an object.")
        tool_type = tool.get("type") or "function"
        if tool_type != "function":
            raise TinkerNativeProtocolError(
                f"Only function tools are supported; got type={tool_type!r}."
            )
        function = tool.get("function")
        if not isinstance(function, dict):
            # Some callers already pass ToolSpec-shaped objects.
            name = str(tool.get("name") or "").strip()
            description = str(tool.get("description") or "")
            parameters = tool.get("parameters")
            if not name or not isinstance(parameters, dict):
                raise TinkerNativeProtocolError(
                    f"Tool at index {index} is missing function.name/parameters."
                )
        else:
            name = str(function.get("name") or "").strip()
            description = str(function.get("description") or "")
            parameters = function.get("parameters")
            if not name:
                raise TinkerNativeProtocolError(f"Tool at index {index} has an empty name.")
            if not isinstance(parameters, dict):
                raise TinkerNativeProtocolError(
                    f"Tool {name!r} parameters must be a JSON Schema object."
                )
        if name in seen_names:
            raise TinkerNativeProtocolError(f"Duplicate tool name: {name!r}.")
        seen_names.add(name)
        specs.append(
            {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        )
    return specs, _tool_schema_sha256(specs)


def openai_messages_to_tinker_messages(
    payload: Mapping[str, Any],
    *,
    deps: Any,
) -> tuple[str, list[Any]]:
    """Convert OpenAI chat messages into Cookbook Message objects.

    Returns:
        (system_prompt_text, non-system messages)
    """

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise TinkerNativeProtocolError("Chat payload requires a non-empty messages list.")

    system_prompt = ""
    conversation: list[Any] = []
    seen_assistant_call_ids: set[str] = set()

    for message_index, raw in enumerate(raw_messages):
        if not isinstance(raw, dict):
            raise TinkerNativeProtocolError(f"Message {message_index} is not an object.")
        role = str(raw.get("role") or "").strip().lower()
        if role in {"system", "developer"}:
            if conversation:
                raise TinkerNativeProtocolError(
                    "Extra system/developer messages after conversation start are rejected."
                )
            if system_prompt:
                raise TinkerNativeProtocolError(
                    "Exactly one leading system message is required; found another."
                )
            system_prompt = _content_to_text(raw.get("content"))
            continue

        if role == "user":
            parts = _user_content_parts(raw.get("content"), deps=deps)
            conversation.append(
                deps.Message(role="user", content=_normalize_content(parts, deps))
            )
            continue

        if role == "assistant":
            content_parts = _assistant_content_parts(raw, deps=deps)
            tool_calls = _assistant_tool_calls(raw, deps=deps)
            for call in tool_calls:
                call_id = getattr(call, "id", None) or ""
                if call_id:
                    seen_assistant_call_ids.add(str(call_id))
            message_kwargs: dict[str, Any] = {
                "role": "assistant",
                "content": content_parts if content_parts else "",
            }
            if tool_calls:
                message_kwargs["tool_calls"] = tool_calls
            conversation.append(deps.Message(**message_kwargs))
            continue

        if role == "tool":
            tool_call_id = str(raw.get("tool_call_id") or "").strip()
            name = str(raw.get("name") or "").strip()
            if not tool_call_id:
                raise TinkerNativeProtocolError("Tool result is missing tool_call_id.")
            if tool_call_id not in seen_assistant_call_ids:
                raise TinkerNativeProtocolError(
                    f"Tool result {tool_call_id!r} has no matching earlier assistant call."
                )
            tool_parts = [
                deps.TextPart(type="text", text=_content_to_text(raw.get("content")))
            ]
            conversation.append(
                deps.Message(
                    role="tool",
                    content=_normalize_content(tool_parts, deps),
                    tool_call_id=tool_call_id,
                    name=name or None,
                )
            )
            continue

        raise TinkerNativeProtocolError(f"Unsupported message role: {role!r}.")

    return system_prompt, conversation


def build_native_generation_messages(
    *,
    renderer: Any,
    tool_specs: list[dict[str, Any]],
    system_prompt: str,
    conversation: list[Any],
) -> list[Any]:
    """Prefix tool specs via the renderer, then append the conversation."""

    if not hasattr(renderer, "create_conversation_prefix_with_tools"):
        raise TinkerNativeProtocolError(
            "Renderer does not support create_conversation_prefix_with_tools."
        )
    prefix = renderer.create_conversation_prefix_with_tools(tool_specs, system_prompt or "")
    return list(prefix) + list(conversation)


def tinker_message_to_openai_assistant(
    parsed_message: Any,
    *,
    deps: Any,
    assign_missing_call_ids: bool = True,
) -> dict[str, Any]:
    """Convert a parsed Cookbook assistant Message into an OpenAI assistant dict."""

    if isinstance(parsed_message, dict):
        role = parsed_message.get("role", "assistant")
        content = parsed_message.get("content")
        tool_calls = parsed_message.get("tool_calls")
        unparsed = parsed_message.get("unparsed_tool_calls") or []
    else:
        role = getattr(parsed_message, "role", "assistant")
        content = getattr(parsed_message, "content", "")
        tool_calls = getattr(parsed_message, "tool_calls", None)
        unparsed = getattr(parsed_message, "unparsed_tool_calls", None) or []

    if unparsed:
        raise TinkerNativeProtocolError(
            f"Native renderer left {len(unparsed)} unparsed_tool_calls; refusing text recovery."
        )

    text, reasoning = _split_text_and_thinking(content, deps=deps)
    openai_message: dict[str, Any] = {
        "role": role,
        "content": text if text else None,
    }
    if reasoning:
        openai_message["reasoning_content"] = reasoning

    openai_tool_calls: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls or []):
        if isinstance(tool_call, dict):
            function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
            call_id = str(tool_call.get("id") or "")
            name = str(function.get("name") or "")
            arguments = function.get("arguments")
        else:
            function = getattr(tool_call, "function", None)
            call_id = str(getattr(tool_call, "id", None) or "")
            name = str(getattr(function, "name", "") or "")
            arguments = getattr(function, "arguments", "{}")
        if assign_missing_call_ids and not call_id:
            call_id = f"call_{uuid.uuid4().hex[:12]}_{index}"
        openai_tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name or "unknown",
                    "arguments": _stringify_arguments(arguments),
                },
            }
        )
    if openai_tool_calls:
        openai_message["tool_calls"] = openai_tool_calls
    return openai_message


def finish_reason_for_assistant(
    openai_message: Mapping[str, Any],
    *,
    termination: Any,
) -> str:
    if openai_message.get("tool_calls"):
        return "tool_calls"
    if getattr(termination, "is_stop_sequence", False) or getattr(termination, "is_clean", False):
        return "stop"
    return "length"


def extract_thinking_text(parsed_message: Any) -> str:
    _, reasoning = _split_text_and_thinking(
        parsed_message.get("content") if isinstance(parsed_message, dict) else getattr(parsed_message, "content", ""),
        deps=None,
    )
    return reasoning


def _tool_schema_sha256(specs: list[dict[str, Any]]) -> str:
    blob = json.dumps(specs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, dict) and item.get("type") in {"thinking", "reasoning"}:
                continue
            else:
                raise TinkerNativeProtocolError(
                    f"Unsupported content part in text extraction: {item!r}."
                )
        return "\n".join(part for part in parts if part)
    raise TinkerNativeProtocolError("Message content must be a string or content-part list.")


def _user_content_parts(content: Any, *, deps: Any) -> list[Any]:
    if content is None or content == "":
        return [deps.TextPart(type="text", text="")]
    if isinstance(content, str):
        return [deps.TextPart(type="text", text=content)]
    if not isinstance(content, list):
        raise TinkerNativeProtocolError("User content must be a string or list.")
    parts: list[Any] = []
    for item in content:
        if isinstance(item, str):
            parts.append(deps.TextPart(type="text", text=item))
        elif isinstance(item, dict) and item.get("type") == "text":
            parts.append(deps.TextPart(type="text", text=str(item.get("text") or "")))
        elif isinstance(item, dict) and item.get("type") == "image_url":
            # The Player is text-only; image content is a protocol error.
            raise TinkerNativeProtocolError(
                "image_url content parts are not supported: the Player is text-only."
            )
        else:
            raise TinkerNativeProtocolError(f"Unsupported user content part: {item!r}.")
    return parts or [deps.TextPart(type="text", text="")]


def _thinking_text(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def dedupe_assistant_thinking(raw: Mapping[str, Any]) -> tuple[str, Any]:
    """Return (reasoning, content) with thinking present only once.

    Prefer ``reasoning_content`` / ``reasoning``. Drop identical thinking
    parts from ``content`` so Prime / Tinker traces do not show the same
    think-block twice.
    """

    reasoning = _thinking_text(raw.get("reasoning_content") or raw.get("reasoning"))
    content = raw.get("content")
    if not isinstance(content, list):
        return reasoning, content
    kept: list[Any] = []
    collected: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") in {"thinking", "reasoning"}:
            text = _thinking_text(item.get("thinking") or item.get("text") or "")
            if not text:
                continue
            if reasoning and text == reasoning:
                continue
            if any(text == existing for existing in collected):
                continue
            collected.append(text)
            kept.append(item)
            continue
        kept.append(item)
    if not reasoning and collected:
        reasoning = "\n\n".join(collected)
        kept = [
            item
            for item in kept
            if not (
                isinstance(item, dict) and item.get("type") in {"thinking", "reasoning"}
            )
        ]
    return reasoning, kept


def _assistant_content_parts(raw: Mapping[str, Any], *, deps: Any) -> list[Any]:
    parts: list[Any] = []
    reasoning, content = dedupe_assistant_thinking(raw)
    if reasoning:
        parts.append(deps.ThinkingPart(type="thinking", thinking=reasoning))
    if isinstance(content, str) and content:
        parts.append(deps.TextPart(type="text", text=content))
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"thinking", "reasoning"}:
                text = _thinking_text(item.get("thinking") or item.get("text") or "")
                if text:
                    parts.append(deps.ThinkingPart(type="thinking", thinking=text))
            elif isinstance(item, dict) and item.get("type") == "text":
                text = str(item.get("text") or "")
                if text:
                    parts.append(deps.TextPart(type="text", text=text))
            elif isinstance(item, str) and item:
                parts.append(deps.TextPart(type="text", text=item))
            else:
                raise TinkerNativeProtocolError(
                    f"Unsupported assistant content part: {item!r}."
                )
    return parts


def _assistant_tool_calls(raw: Mapping[str, Any], *, deps: Any) -> list[Any]:
    raw_calls = raw.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise TinkerNativeProtocolError("assistant.tool_calls must be a list.")
    calls: list[Any] = []
    for index, call in enumerate(raw_calls):
        if not isinstance(call, dict):
            raise TinkerNativeProtocolError(f"tool_calls[{index}] is not an object.")
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(function.get("name") or "").strip()
        if not name:
            raise TinkerNativeProtocolError(f"tool_calls[{index}] is missing function.name.")
        arguments = _stringify_arguments(function.get("arguments"))
        call_id = str(call.get("id") or f"call_{uuid.uuid4().hex[:12]}_{index}")
        calls.append(
            deps.ToolCall(
                type="function",
                id=call_id,
                function=deps.ToolCall.FunctionBody(name=name, arguments=arguments),
            )
        )
    return calls


def _normalize_content(parts: list[Any], deps: Any) -> Any:
    if len(parts) == 1 and getattr(parts[0], "type", None) == "text":
        return getattr(parts[0], "text", "")
    return parts


def _split_text_and_thinking(content: Any, *, deps: Any | None) -> tuple[str, str]:
    if content is None:
        return "", ""
    if isinstance(content, str):
        return content, ""
    if not isinstance(content, list):
        return str(content), ""
    texts: list[str] = []
    thinking: list[str] = []
    for part in content:
        if isinstance(part, dict):
            part_type = part.get("type")
            if part_type == "thinking":
                thinking.append(str(part.get("thinking") or ""))
            elif part_type == "text":
                texts.append(str(part.get("text") or ""))
            else:
                raise TinkerNativeProtocolError(f"Unsupported parsed content part: {part!r}.")
            continue
        part_type = getattr(part, "type", None)
        if part_type == "thinking":
            thinking.append(str(getattr(part, "thinking", "") or ""))
        elif part_type == "text":
            texts.append(str(getattr(part, "text", "") or ""))
        else:
            # TypedDict-like MappingProxy or other structures
            if hasattr(part, "get"):
                if part.get("type") == "thinking":
                    thinking.append(str(part.get("thinking") or ""))
                elif part.get("type") == "text":
                    texts.append(str(part.get("text") or ""))
                else:
                    raise TinkerNativeProtocolError(f"Unsupported parsed content part: {part!r}.")
            else:
                raise TinkerNativeProtocolError(f"Unsupported parsed content part: {part!r}.")
    return "\n".join(t for t in texts if t), "\n".join(t for t in thinking if t)


def _stringify_arguments(arguments: Any) -> str:
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, sort_keys=True)
