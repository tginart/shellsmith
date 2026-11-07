"""Sample one Player action from Tinker and record it token-exactly.

Architecture map
----------------
    TinkerPlayerSampler.sample(payload, request_id, session_id)
        OpenAI chat payload (messages + tools) from Pi
          -> chat_adapter: tool specs + Cookbook messages (native tool format)
          -> renderer.build_generation_prompt          prompt (tinker ModelInput)
          -> budgets: thinking 16,384 + response 12,288, bounded by config,
                      request and context (including any injected </think>)
          -> thinking on:  sample_thinking_two_pass (think pass, optional forced
                           </think>, response pass)
             thinking off: one sample
          -> renderer.parse_response -> OpenAI assistant message (tool_calls,
             reasoning_content)
          -> TinkerRolloutRecord (+ protocol_row for the acceptance audit)

Invariants
----------
- Every returned record carries the exact prompt, sampled tokens and
  behavior-policy logprobs; a missing or mismatched logprob list is an error.
- Injected ``</think>`` tokens are marked in ``forced_close_token_span`` so the
  datum builder never trains them as policy actions.
- Unparsed tool calls are an error, never recovered from text.

Why this exists
---------------
Ported from the Tinker path of the original monorepo's LLM proxy
(``TinkerSampler.sample`` and its helpers), without the image, Verifiers and
FCP branches. ``policy.server`` puts it behind an OpenAI-compatible endpoint.

Main callers:
    ``policy.server.PlayerEndpoint`` for each ``/chat/completions`` request.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
import threading
import time
from typing import Any, Mapping

from shellsmith.policy.chat_adapter import (
    TinkerNativeProtocolError,
    build_native_generation_messages,
    finish_reason_for_assistant,
    openai_messages_to_tinker_messages,
    openai_tools_to_tinker_specs,
    tinker_message_to_openai_assistant,
)
from shellsmith.policy.records import TinkerRolloutRecord

# Shellsmith's working window advertised to Pi; the Tinker model must support
# at least this many tokens (``Qwen/Qwen3.8-27B`` serves 262,144 with LoRA).
CONTEXT_WINDOW_TOKENS = 224 * 1024
MIN_COMPLETION_TOKENS = 1024
THINKING_TOKEN_BUDGET = 16_384
RESPONSE_TOKEN_BUDGET = 12_288
THINK_ANOMALY_RESAMPLE_LIMIT = 3
THINK_CLOSE_TEXT = "\n</think>\n\n"
THINK_CLOSE_MARKER = "</think>"
# Exact prompts from lockfile-pinned Pi 0.84.1 core/compaction/{utils,compaction}.js.
# Keep all three forms: split-turn compaction samples history and/or a turn prefix.
PI_COMPACTION_SUMMARY_SYSTEM_PROMPT = """You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary."""
PI_COMPACTION_SUMMARY_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""
PI_COMPACTION_UPDATE_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""
PI_COMPACTION_TURN_PREFIX_PROMPT = """This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix."""
_PI_COMPACTION_USER_FRAMING = re.compile(
    r"<conversation>\n.*\n</conversation>\n\n(?:"
    r"(?:" + re.escape(PI_COMPACTION_SUMMARY_PROMPT)
    + r"|<previous-summary>\n.+\n</previous-summary>\n\n" + re.escape(PI_COMPACTION_UPDATE_PROMPT)
    + r")(?:\n\nAdditional focus: .+)?"
    + r"|" + re.escape(PI_COMPACTION_TURN_PREFIX_PROMPT) + r")",
    re.DOTALL,
)


class PlayerRequestError(RuntimeError):
    """A Player request the endpoint must refuse, with the HTTP status to return."""

    def __init__(self, message: str, status: int = 400, *, record: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.record = record


@dataclass(frozen=True)
class PlayerSamplingConfig:
    """Sampling knobs for the Player.

    ``thinking_renderer`` is used when Pi asks for reasoning; Qwen3.8's chat
    template offers xhigh (its default), medium and low reasoning, and no
    "high". ``plain_renderer`` serves requests with reasoning turned off.

    ``max_tokens`` is a positive-integer ceiling on the entire recorded
    completion: thinking + injected close + visible response. Request caps
    can only lower it. Without request caps, plain sampling still defaults to
    ``RESPONSE_TOKEN_BUDGET``; thinking retains the ordinary per-pass budgets.
    """

    base_model: str
    thinking_renderer: str = "qwen3_8_xhigh_reasoning"
    plain_renderer: str = "qwen3_8_disable_thinking"
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = THINKING_TOKEN_BUDGET + RESPONSE_TOKEN_BUDGET
    thinking_anomaly_attempts: int = THINK_ANOMALY_RESAMPLE_LIMIT


def thinking_requested(payload: Mapping[str, Any]) -> bool:
    """Resolve explicit effort with the same mapping as the episode audit."""
    from shellsmith.policy.reasoning import thinking_mode_for_effort

    value = None
    for key in ("reasoning_effort", "reasoningEffort"):
        if key in payload and payload[key] is not None:
            value = payload[key]
            break
    else:
        reasoning = payload.get("reasoning")
        if isinstance(reasoning, dict):
            value = reasoning.get("effort")
    try:
        return thinking_mode_for_effort(value, default="disable_thinking") == "thinking"
    except ValueError as error:
        raise PlayerRequestError(str(error)) from error



def _positive_token_cap(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PlayerRequestError(f"{name} must be a positive integer.")
    return value


def _completion_token_budget(payload: Mapping[str, Any], *, configured: int, thinking: bool) -> int:
    """Intersect the configured ceiling with every non-null request cap.

    Null request caps mean omitted; neither alias takes precedence over the
    other. Reject malformed caps before loading a renderer or calling Tinker.
    """
    caps = [_positive_token_cap(configured, name="config.max_tokens")]
    for name in ("max_tokens", "max_completion_tokens"):
        if payload.get(name) is not None:
            caps.append(_positive_token_cap(payload[name], name=name))
    if not thinking and len(caps) == 1:
        caps.append(RESPONSE_TOKEN_BUDGET)
    return min(caps)


def clamp_completion_budget(*, prompt_tokens: int, max_tokens: int) -> int:
    """Shrink the completion budget to fit the context window, or refuse.

    Late-episode prompts grow; real endpoints clamp the completion budget
    rather than fail while thousands of tokens remain. If clamping would leave
    fewer than ``MIN_COMPLETION_TOKENS``, return 422 so Pi compacts instead of
    recording a phantom empty turn. An explicitly smaller budget that already
    fits is allowed.
    """
    max_tokens = _positive_token_cap(max_tokens, name="max_tokens")
    if prompt_tokens + max_tokens <= CONTEXT_WINDOW_TOKENS:
        return max_tokens
    available = CONTEXT_WINDOW_TOKENS - prompt_tokens
    if available < MIN_COMPLETION_TOKENS:
        raise PlayerRequestError(
            f"Context budget exceeded: prompt_tokens={prompt_tokens}, "
            f"window={CONTEXT_WINDOW_TOKENS}, available={max(0, available)}.",
            status=422,
        )
    return available


def _append_token_ids(prompt: Any, token_ids: list[int], *, deps: Any) -> Any:
    """Return ``prompt`` with raw token ids appended (the original is unchanged)."""
    if not token_ids:
        return prompt
    return prompt.append(deps.types.EncodedTextChunk(tokens=list(token_ids)))


def _encode(tokenizer: Any, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def _checked_sequence(sampled: Any, *, stage: str, max_tokens: int) -> tuple[Any, list[int], list[float]]:
    sequence = sampled.sequences[0]
    if sequence.logprobs is None:
        raise PlayerRequestError(f"Tinker {stage} sample has no logprobs (required for RL).", status=500)
    tokens, logprobs = list(sequence.tokens), list(sequence.logprobs)
    if len(tokens) != len(logprobs):
        raise PlayerRequestError(
            f"Tinker {stage} token/logprob length mismatch: {len(tokens)} != {len(logprobs)}.", status=500
        )
    if len(tokens) > max_tokens:
        raise PlayerRequestError(
            f"Tinker {stage} exceeded its token budget: {len(tokens)} > {max_tokens}.", status=500
        )
    return sequence, tokens, logprobs


def sample_thinking_two_pass(
    *,
    sampling_client: Any,
    prompt: Any,
    tokenizer: Any,
    deps: Any,
    params: Any,
    thinking_cap: int,
    response_cap: int,
    base_stop: list[Any],
    total_cap: int | None = None,
    anomaly_attempts: int = THINK_ANOMALY_RESAMPLE_LIMIT,
) -> dict[str, Any]:
    """Sample a budgeted think pass, then the visible response.

    ``total_cap`` bounds the concatenated completion, including injected tokens,
    and defaults to the sum of the phase caps for existing callers. Anomalous
    retries are separate candidates, not part of the recorded completion.
    Reserve enough room for a forced close and at least one response token,
    but only charge the close when injected: a natural close must not reduce
    the response allowance or be truncated to make room for an unused close.

    Steps:
        1. Think pass: stop on ``</think>`` or at ``thinking_cap`` tokens. An
           end-of-turn inside open thinking is resampled (up to 3 tries), then
           refused; inventing a close would fabricate policy tokens.
        2. If the think pass ran out of budget still inside thinking, append
           ``\\n</think>\\n\\n`` and score those tokens with ``compute_logprobs``
           (they are reported as a forced-close span, never trained).
        3. Response pass from prompt + think tokens (+ close) under ``response_cap``.
        4. Return the concatenated tokens/logprobs; the original prompt is the
           training prompt.

    Main callers:
        ``TinkerPlayerSampler.sample`` when reasoning is requested.
    """
    thinking_cap = _positive_token_cap(thinking_cap, name="thinking_cap")
    response_cap = _positive_token_cap(response_cap, name="response_cap")
    total = clamp_completion_budget(
        prompt_tokens=int(prompt.length), max_tokens=thinking_cap + response_cap if total_cap is None else total_cap,
    )
    forced_close_ids = _encode(tokenizer, THINK_CLOSE_TEXT)
    if not forced_close_ids:
        raise PlayerRequestError("Tokenizer produced an empty </think> sequence.", status=500)
    if total < len(forced_close_ids) + 2:
        raise PlayerRequestError(
            f"Thinking completion budget {total} cannot fit a thinking token, "
            f"{len(forced_close_ids)} forced </think> tokens and a response token."
        )
    thinking_cap = min(thinking_cap, total - len(forced_close_ids) - 1)
    close_ids = _encode(tokenizer, THINK_CLOSE_MARKER)
    think_stop = list(base_stop)
    if think_stop and all(isinstance(item, int) for item in think_stop):
        think_stop += [token for token in close_ids if token not in think_stop]
    elif THINK_CLOSE_MARKER not in think_stop:
        if any(isinstance(item, int) for item in think_stop):
            raise PlayerRequestError(f"Stop sequences mix ints and strings: {think_stop!r}.", status=500)
        think_stop.append(THINK_CLOSE_MARKER)

    anomalies: list[str] = []
    if type(anomaly_attempts) is not int or anomaly_attempts < 1:
        raise PlayerRequestError("Thinking anomaly attempts must be positive.")
    for attempt in range(1, anomaly_attempts + 1):
        sampled_think = sampling_client.sample(
            prompt=prompt, num_samples=1, sampling_params=params(max_tokens=thinking_cap, stop=think_stop)
        ).result()
        seq_think, think_tokens, think_logprobs = _checked_sequence(
            sampled_think, stage="think-pass", max_tokens=thinking_cap,
        )
        has_close = bool(think_tokens) and THINK_CLOSE_MARKER in tokenizer.decode(think_tokens)
        stop_reason = getattr(seq_think, "stop_reason", None)
        if has_close or stop_reason == "length":
            break
        anomalies.append(f"attempt {attempt}: stop_reason={stop_reason!r} after {len(think_tokens)} think tokens")
    else:
        raise PlayerRequestError(
            "Think pass repeatedly ended the turn inside open thinking: " + "; ".join(anomalies), status=500
        )

    forced_ids: list[int] = []
    forced_logprobs: list[float] = []
    if not has_close:
        forced_ids = forced_close_ids
        scored = sampling_client.compute_logprobs(
            _append_token_ids(prompt, think_tokens + forced_ids, deps=deps)
        ).result()
        tail = list(scored)[-len(forced_ids):] if isinstance(scored, list) else []
        if len(tail) != len(forced_ids) or any(value is None for value in tail):
            raise PlayerRequestError("compute_logprobs did not score every forced </think> token.", status=500)
        forced_logprobs = [float(value) for value in tail]

    response_allowance = min(response_cap, total - len(think_tokens) - len(forced_ids))
    sampled_response = sampling_client.sample(
        prompt=_append_token_ids(prompt, think_tokens + forced_ids, deps=deps),
        num_samples=1,
        sampling_params=params(max_tokens=response_allowance, stop=list(base_stop)),
    ).result()
    seq_response, response_tokens, response_logprobs = _checked_sequence(
        sampled_response, stage="response-pass", max_tokens=response_allowance,
    )
    return {
        "sampled_tokens": think_tokens + forced_ids + response_tokens,
        "sampled_logprobs": think_logprobs + forced_logprobs + response_logprobs,
        "thinking_tokens": len(think_tokens),
        "response_tokens": len(response_tokens),
        "forced_close_token_span": (
            (len(think_tokens), len(think_tokens) + len(forced_ids)) if forced_ids else None
        ),
        "think_stop_reason": stop_reason,
        "response_stop_reason": getattr(seq_response, "stop_reason", None),
    }


def payload_is_compaction_summary(payload: Mapping[str, Any]) -> bool:
    """Match pinned Pi's standalone, tool-free compaction request framing.

    A user quoting summary tags (or an earlier summary in ordinary history) is
    not provenance. Pi supplies its full system prompt and one serialized user
    request; its OpenAI transport may use the developer role for reasoning models.
    """
    messages = payload.get("messages")
    if (
        payload.get("tools")
        or not isinstance(messages, list)
        or len(messages) != 2
        or not all(isinstance(message, dict) for message in messages)
        or messages[0].get("role") not in ("system", "developer")
        or messages[1].get("role") != "user"
    ):
        return False

    def text_of(message: dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, list) and all(
            isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
            for part in content
        ):
            content = "".join(part["text"] for part in content)
        return content if isinstance(content, str) else ""

    return (
        text_of(messages[0]) == PI_COMPACTION_SUMMARY_SYSTEM_PROMPT
        and _PI_COMPACTION_USER_FRAMING.fullmatch(text_of(messages[1])) is not None
    )


def protocol_tool_calls(assistant_message: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize OpenAI tool_calls into ``model_protocol.requests[].tool_calls`` rows."""
    rows: list[dict[str, Any]] = []
    for item in assistant_message.get("tool_calls") or []:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict) or not item.get("id") or not function.get("name"):
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        rows.append({"id": item["id"], "name": function["name"], "arguments": {str(k): v for k, v in arguments.items()}})
    return rows


def protocol_row(record: TinkerRolloutRecord, sample: Mapping[str, Any], assistant_message: Mapping[str, Any]) -> dict[str, Any]:
    """The JSON object the acceptance audit reads as one ``model_protocol`` request."""
    content_types = []
    if record.reasoning_content:
        content_types.append("reasoning")
    if isinstance(assistant_message.get("content"), str) and assistant_message["content"].strip():
        content_types.append("text")
    if assistant_message.get("tool_calls"):
        content_types.append("tool_call")
    return {
        "request_id": record.request_id,
        "created_at": record.created_at,
        "session_id": record.session_id,
        "prompt_tokens": record.prompt_tokens,
        "sampled_tokens": record.sampled_tokens,
        "sampled_logprobs": record.sampled_logprobs,
        "content": record.content,
        "reasoning_content": record.reasoning_content,
        "finish_reason": record.finish_reason,
        "api_shape": "native_tools",
        "renderer_name": sample["renderer_name"],
        "thinking_mode": "thinking" if sample["thinking"] else "disable_thinking",
        "tool_schema_sha256": sample["tool_schema_sha256"],
        "unparsed_tool_call_count": 0,
        "parse_termination": sample["parse_termination"],
        "thinking_forced_close": record.forced_close_token_span is not None,
        "thinking_tokens": sample["thinking_tokens"],
        "response_tokens": sample["response_tokens"],
        "tool_calls": protocol_tool_calls(assistant_message),
        "assistant_content_types": content_types,
        "reasoning_content_present": bool(record.reasoning_content),
        "compaction_summary": payload_is_compaction_summary(record.payload),
        "forced_close_token_span": (
            list(record.forced_close_token_span) if record.forced_close_token_span is not None else None
        ),
    }


class TinkerPlayerSampler:
    """Samples Player actions from the current Tinker ``SamplingClient``.

    ``set_sampling_client`` swaps in new weights between training steps; one
    request always uses one client from start to finish.
    """

    def __init__(self, *, config: PlayerSamplingConfig, deps: Any, sampling_client: Any) -> None:
        self.config = config
        self.deps = deps
        self._lock = threading.Lock()
        self._sampling_client = sampling_client
        self._session_clients: dict[str, Any] = {}
        self._renderers: dict[str, Any] = {}

    def set_sampling_client(self, sampling_client: Any) -> None:
        with self._lock:
            self._sampling_client = sampling_client

    def pin(self,session_id: str) -> Any:
        with self._lock:
            return self._session_clients.setdefault(session_id,self._sampling_client)

    def release(self,session_id: str) -> None:
        with self._lock:
            self._session_clients.pop(session_id,None)

    def _renderer(self, name: str) -> Any:
        with self._lock:
            if name not in self._renderers:
                tokenizer = self.deps.get_tokenizer(self.config.base_model)
                self._renderers[name] = self.deps.renderers.get_renderer(name, tokenizer)
            return self._renderers[name]

    def validate_thinking_budget(self) -> None:
        """Reject unusable configured reasoning caps before collecting any world."""
        total = _positive_token_cap(self.config.max_tokens, name="config.max_tokens")
        tokenizer = self._renderer(self.config.thinking_renderer).tokenizer
        close = _encode(tokenizer, THINK_CLOSE_TEXT)
        if not close:
            raise PlayerRequestError("Tokenizer produced an empty </think> sequence.", status=500)
        minimum = len(close) + 2
        if total < minimum:
            raise PlayerRequestError(
                f"Thinking completion budget {total} requires at least {minimum} tokens "
                "for thinking, forced </think> and a response."
            )

    def sample(self, payload: dict[str, Any], *, request_id: str, session_id: str) -> tuple[TinkerRolloutRecord, dict[str, Any]]:
        """Sample one assistant turn; return (record, OpenAI assistant message).

        Main callers:
            ``policy.server`` request handler.
        """
        thinking = thinking_requested(payload)
        requested = _completion_token_budget(payload, configured=self.config.max_tokens, thinking=thinking)
        sampling_client = self.pin(session_id) if session_id else self._sampling_client
        renderer_name = self.config.thinking_renderer if thinking else self.config.plain_renderer
        renderer = self._renderer(renderer_name)
        try:
            tool_specs, tool_schema_sha256 = openai_tools_to_tinker_specs(payload.get("tools") or [])
            system_prompt, conversation = openai_messages_to_tinker_messages(payload, deps=self.deps)
            messages = build_native_generation_messages(
                renderer=renderer, tool_specs=tool_specs, system_prompt=system_prompt, conversation=conversation
            )
        except TinkerNativeProtocolError as error:
            raise PlayerRequestError(str(error), status=422) from error
        prompt = renderer.build_generation_prompt(messages)
        stop = payload.get("stop") or renderer.get_stop_sequences()
        base_stop = [stop] if isinstance(stop, str) else list(stop)
        temperature = float(payload.get("temperature") if payload.get("temperature") is not None else self.config.temperature)
        top_p = float(payload.get("top_p") if payload.get("top_p") is not None else self.config.top_p)

        def params(*, max_tokens: int, stop: list[Any]) -> Any:
            return self.deps.tinker.types.SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop)

        total = clamp_completion_budget(prompt_tokens=int(prompt.length), max_tokens=requested)
        if thinking:
            thinking_cap, response_cap = THINKING_TOKEN_BUDGET, RESPONSE_TOKEN_BUDGET
            if thinking_cap + response_cap > total:
                # Shrink thinking first, retaining the ordinary response cap
                # where possible. The two-pass helper reserves the injected close.
                thinking_floor = min(256, max(1, total // 4))
                thinking_cap = max(total - response_cap, thinking_floor)
                response_cap = max(1, total - thinking_cap)
            result = sample_thinking_two_pass(
                sampling_client=sampling_client, prompt=prompt, tokenizer=renderer.tokenizer, deps=self.deps,
                params=params, thinking_cap=thinking_cap, response_cap=response_cap, base_stop=base_stop,
                total_cap=total,
                anomaly_attempts=self.config.thinking_anomaly_attempts,
            )
        else:
            sampled = sampling_client.sample(prompt=prompt, num_samples=1, sampling_params=params(max_tokens=total, stop=base_stop)).result()
            _, tokens, logprobs = _checked_sequence(sampled, stage="single-pass", max_tokens=total)
            result = {"sampled_tokens": tokens, "sampled_logprobs": logprobs, "thinking_tokens": 0,
                      "response_tokens": len(tokens), "forced_close_token_span": None}

        parsed, termination = renderer.parse_response(result["sampled_tokens"])
        try:
            assistant = tinker_message_to_openai_assistant(parsed, deps=self.deps)
        except TinkerNativeProtocolError as error:
            raise PlayerRequestError(str(error), status=500) from error
        record = TinkerRolloutRecord(
            request_id=request_id,
            created_at=time.time(),
            prompt=prompt,
            prompt_tokens=int(prompt.length),
            sampled_tokens=list(result["sampled_tokens"]),
            sampled_logprobs=list(result["sampled_logprobs"]),
            content=str(assistant.get("content") or ""),
            finish_reason=finish_reason_for_assistant(assistant, termination=termination),
            payload=payload,
            reasoning_content=str(assistant.get("reasoning_content") or ""),
            session_id=session_id,
            forced_close_token_span=result["forced_close_token_span"],
        )
        sample_info = {
            "renderer_name": renderer_name,
            "thinking": thinking,
            "tool_schema_sha256": tool_schema_sha256,
            "parse_termination": str(getattr(termination, "value", getattr(termination, "name", termination))),
            "thinking_tokens": result["thinking_tokens"],
            "response_tokens": result["response_tokens"],
        }
        return replace(record, protocol_row=protocol_row(record, sample_info, assistant)), assistant
