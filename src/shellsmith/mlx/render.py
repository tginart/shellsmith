"""Token-exact native Qwen3.8 text/tool rendering and strict response parsing."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from shellsmith.policy.chat_adapter import TinkerNativeProtocolError, _content_to_text, openai_tools_to_tinker_specs
from .generation import DEFAULT_REASONING_EFFORT, FORCED_CLOSE_TOKENS, REASONING_EFFORTS

NativeProtocolError = TinkerNativeProtocolError
_FUNCTION = re.compile(r"\s*<function=([^<>\s]+)>\s*(.*?)\s*</function>\s*", re.DOTALL)
_PARAMETER = re.compile(r"\s*<parameter=([^<>\s]+)>(.*?)</parameter>\s*", re.DOTALL)
_TOOL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def normalize_messages(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Keep tools/reasoning structured; refuse media, unmatched results or roles."""
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise NativeProtocolError("MLX chat requires a nonempty messages list.")
    messages, seen_calls, pending_calls = [], set(), set()
    for index, raw in enumerate(raw_messages):
        if not isinstance(raw, dict):
            raise NativeProtocolError(f"Message {index} is not an object.")
        role = raw.get("role")
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant", "tool"} or (role == "system" and index != 0):
            raise NativeProtocolError(f"Unsupported message role/position: {role!r}.")
        if pending_calls and role != "tool":
            raise NativeProtocolError("Native conversation advanced before pending tool results were returned.")
        message = {"role": role, "content": _content_to_text(raw.get("content"))}
        if role == "assistant":
            reasoning = raw.get("reasoning_content")
            if reasoning is not None and not isinstance(reasoning, str):
                raise NativeProtocolError("Assistant reasoning_content must be text.")
            if reasoning is not None:
                message["reasoning_content"] = reasoning
            calls = raw.get("tool_calls") or []
            if not isinstance(calls, list):
                raise NativeProtocolError("Assistant tool_calls must be a list.")
            normalized = []
            for call in calls:
                if not isinstance(call, dict) or call.get("type", "function") != "function":
                    raise NativeProtocolError("Only native function calls are supported.")
                function = call.get("function")
                identifier = call.get("id")
                if not isinstance(function, dict) or not isinstance(identifier, str) or not identifier or identifier in seen_calls:
                    raise NativeProtocolError("Assistant tool call requires a unique ID and function.")
                name = function.get("name")
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except ValueError as exc:
                        raise NativeProtocolError("Assistant tool arguments are not valid JSON.") from exc
                if not isinstance(name, str) or not name or not isinstance(arguments, dict):
                    raise NativeProtocolError("Assistant tool call requires a name and object arguments.")
                seen_calls.add(identifier)
                pending_calls.add(identifier)
                normalized.append({"id": identifier, "type": "function", "function": {"name": name, "arguments": arguments}})
            if normalized:
                message["tool_calls"] = normalized
        elif role == "tool":
            identifier = raw.get("tool_call_id")
            if not isinstance(identifier, str) or identifier not in pending_calls:
                raise NativeProtocolError("Tool result has no matching earlier assistant call.")
            pending_calls.remove(identifier)
            message["tool_call_id"] = identifier
        messages.append(message)
    if pending_calls:
        raise NativeProtocolError("Native generation cannot start with outstanding tool calls.")
    return messages


def _valid_type(value, declared):
    kinds = declared if isinstance(declared, list) else [declared]
    return any({"string": isinstance(value, str), "object": isinstance(value, dict), "array": isinstance(value, list),
                "boolean": isinstance(value, bool), "integer": type(value) is int,
                "number": type(value) in (int, float), "null": value is None}.get(kind, False) for kind in kinds)


def _parse_arguments(body: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        raise NativeProtocolError("Native tool schema properties must be an object.")
    cursor = 0
    while cursor < len(body):
        if not body[cursor:].strip():
            break
        match = _PARAMETER.match(body, cursor)
        if match is None:
            raise NativeProtocolError("Native tool call contains malformed parameter blocks.")
        name, text = match.groups()
        if name in arguments or (name not in properties and schema.get("additionalProperties") is False):
            raise NativeProtocolError(f"Native tool parameter is duplicate or undeclared: {name}")
        parameter_schema = properties.get(name) or {}
        declared = parameter_schema.get("type")
        if declared == "string":
            # The native template wraps values in exactly one LF on each
            # side. Preserve the argument's own whitespace and newlines.
            value = text[1:] if text.startswith("\n") else text
            value = value[:-1] if value.endswith("\n") else value
        else:
            try:
                value = json.loads(text)
            except ValueError as exc:
                if declared is not None:
                    raise NativeProtocolError(f"Native tool parameter {name} is not valid {declared} JSON.") from exc
                value = text.strip()
        if declared is not None and not _valid_type(value, declared):
            raise NativeProtocolError(f"Native tool parameter {name} has an invalid type.")
        def finite_json(item):
            if isinstance(item, float):
                return math.isfinite(item)
            if isinstance(item, dict):
                return all(finite_json(child) for child in item.values())
            if isinstance(item, list):
                return all(finite_json(child) for child in item)
            return True
        if not finite_json(value):
            raise NativeProtocolError(f"Native tool parameter {name} contains nonfinite JSON numbers.")
        if "enum" in parameter_schema and value not in parameter_schema["enum"]:
            raise NativeProtocolError(f"Native tool parameter {name} is outside its enum.")
        arguments[name] = value
        cursor = match.end()
    if any(name not in arguments for name in schema.get("required", [])):
        raise NativeProtocolError("Native tool call omitted required parameters.")
    return arguments


def parse_native_response(text: str, *, thinking: bool, tool_specs: list[dict[str, Any]],
                          request_id: str, clean_stop: bool) -> tuple[dict[str, Any], str]:
    """Parse the artifact's exact XML function-call grammar; no text recovery."""
    reasoning = ""
    if thinking:
        if "</think>" not in text:
            raise NativeProtocolError("Thinking response did not include its recorded close boundary.")
        reasoning, text = text.split("</think>", 1)
        reasoning = reasoning.removeprefix("<think>").strip()
        if "<think>" in reasoning or "</think>" in text:
            raise NativeProtocolError("Thinking response contains duplicate control boundaries.")
    # Remove only an actual final turn-control token from the display surface.
    text = re.sub(r"(?:<\|im_end\|>|<\|endoftext\|>)\s*$", "", text).strip()
    tools = {spec["name"]: spec for spec in tool_specs}
    matches = list(_TOOL.finditer(text))
    assistant: dict[str, Any] = {"role": "assistant", "content": None}
    if reasoning:
        assistant["reasoning_content"] = reasoning
    if not matches:
        if any(marker in text for marker in ("<tool_call", "</tool_call", "<function=", "<parameter=")):
            raise NativeProtocolError("Native renderer left an incomplete tool call; refusing text recovery.")
        assistant["content"] = text or None
        return assistant, "stop" if clean_stop else "length"
    content = text[:matches[0].start()].strip()
    cursor, calls = matches[0].start(), []
    for index, match in enumerate(matches):
        if text[cursor:match.start()].strip():
            raise NativeProtocolError("Native function calls contain unexpected interleaved or trailing text.")
        function = _FUNCTION.fullmatch(match.group(1))
        if function is None:
            raise NativeProtocolError("Native tool call does not contain exactly one function block.")
        name, body = function.groups()
        if name not in tools:
            raise NativeProtocolError(f"Native tool call references an unavailable function: {name}")
        arguments = _parse_arguments(body, tools[name]["parameters"])
        calls.append({"id": f"call_{request_id}_{index}", "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), allow_nan=False)}})
        cursor = match.end()
    if text[cursor:].strip():
        raise NativeProtocolError("Native tool calls must have no suffix.")
    assistant["content"], assistant["tool_calls"] = content or None, calls
    return assistant, "tool_calls"


class Renderer:
    def __init__(self, model_path: str | Path, *, tokenizer=None,
                 reasoning_effort=DEFAULT_REASONING_EFFORT):
        if reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"Native Qwen reasoning_effort must be one of {REASONING_EFFORTS}.")
        self.reasoning_effort = reasoning_effort
        model_path = Path(model_path).expanduser()
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
        self.tokenizer = tokenizer
        self.template = (Path(model_path) / "chat_template.jinja").read_text()
        self.close_ids = tuple(self.tokenizer.encode("</think>", add_special_tokens=False))
        self.forced_close_ids = tuple(self.tokenizer.encode("\n</think>\n\n", add_special_tokens=False))
        if len(self.close_ids) != 1 or not self.forced_close_ids:
            raise ValueError("The admitted Qwen tokenizer must expose an exact thinking boundary.")
        if len(self.forced_close_ids) != FORCED_CLOSE_TOKENS:
            raise ValueError("Qwen thinking boundary differs from the advertised completion reserve.")
        self.eos_ids = set()
        for text in ("<|im_end|>", "<|endoftext|>"):
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError("The admitted Qwen tokenizer lacks native turn-control tokens.")
            self.eos_ids.update(ids)

    def render(self, payload: Mapping[str, Any], *, thinking: bool):
        from shellsmith.train.backend import TokenPrompt

        messages = normalize_messages(payload)
        specs, tool_sha = openai_tools_to_tinker_specs(payload.get("tools") or [])
        # The supplied native template accepts complete OpenAI tool dictionaries.
        tools = [{"type": "function", "function": copy.deepcopy(spec)} for spec in specs]
        text = self.tokenizer.apply_chat_template(messages, chat_template=self.template, tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=thinking,
                                                  reasoning_effort=self.reasoning_effort, tools=tools or None)
        ids = tuple(self.tokenizer.encode(text, add_special_tokens=False))
        if not ids:
            raise NativeProtocolError("Native Qwen template rendered an empty generation prompt.")
        return TokenPrompt(ids), specs, tool_sha
