"""Serialized native MLX two-pass Player sampling with exact token records."""

from __future__ import annotations

from dataclasses import replace
import logging
import math
import time
from typing import Any

from shellsmith.policy.records import PolicyRolloutRecord
from shellsmith.policy.sampler import (
    PlayerRequestError, PlayerSamplingConfig, THINK_ANOMALY_RESAMPLE_LIMIT,
    _positive_token_cap, payload_is_compaction_summary, protocol_row, thinking_requested,
)

from .render import NativeProtocolError, parse_native_response
from .generation import completion_budget, plain_budget, reasoning_budget, reasoning_effort

LOGGER = logging.getLogger(__name__)


class MLXPlayerSampler:
    def __init__(self, session: Any):
        self.session = session
        self.settings = session.settings
        self.effort = reasoning_effort(self.settings)
        self.reasoning_tokens = reasoning_budget(self.settings)
        if getattr(session.renderer, "reasoning_effort", self.effort) != self.effort:
            raise ValueError("Native renderer effort differs from the sampler profile.")
        self.config = PlayerSamplingConfig(base_model=self.settings.model_id,
            max_tokens=completion_budget(self.settings),
            thinking_renderer=f"qwen3_8_{self.effort}_reasoning")

    def validate_thinking_budget(self):
        if self.config.max_tokens < len(self.session.renderer.forced_close_ids) + 2:
            raise PlayerRequestError("MLX thinking budget cannot fit a close boundary and a response.")

    def _budget(self, payload, *, thinking, summary, prompt_length):
        if summary:
            total = self.settings.compaction_tokens
        elif thinking:
            total = (self.reasoning_tokens + len(self.session.renderer.forced_close_ids)
                     + self.settings.response_tokens)
        else:
            total = plain_budget(self.settings)
        # Preserve historical explicit ceilings; new defaults derive the total.
        if self.settings.max_tokens:
            total = min(total, self.settings.max_tokens)
        for name in ("max_tokens", "max_completion_tokens"):
            if payload.get(name) is not None:
                total = min(total, _positive_token_cap(payload[name], name=name))
        available = self.settings.context_window - prompt_length
        if total > available:
            raise PlayerRequestError(f"context_length_exceeded: MLX context budget exceeded: prompt_tokens={prompt_length}, "
                                     f"window={self.settings.context_window}, requested_completion_tokens={total}, "
                                     f"available={max(0, available)}. Compact the conversation before generation.", status=422)
        if total < 1:
            raise PlayerRequestError("MLX generation has no admitted completion tokens.", status=422)
        return total

    @staticmethod
    def _validate_sampling(payload):
        for name in ("temperature", "top_p"):
            value = payload.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value != 1.0):
                raise PlayerRequestError(f"MLX v1 requires {name}=1.0 for exact behavior probabilities.")
        if payload.get("n", 1) != 1:
            raise PlayerRequestError("MLX v1 supports one completion per request.")
        for name in ("presence_penalty", "frequency_penalty", "repetition_penalty", "logit_bias", "top_k", "min_p"):
            value = payload.get(name)
            if value not in (None, 0, {}, []):
                raise PlayerRequestError(f"MLX v1 does not support request-level {name}.")
        if payload.get("stop") not in (None, [], "<|im_end|>", "<|endoftext|>"):
            raise PlayerRequestError("MLX v1 uses only the native Qwen turn-control stop tokens.")

    def sample(self, payload: dict[str, Any], *, request_id: str, session_id: str):
        if not request_id:
            raise PlayerRequestError("MLX sampling requires a request ID.")
        self._validate_sampling(payload)
        summary = payload_is_compaction_summary(payload)
        thinking = thinking_requested(payload) and not summary
        with self.session.lock:
            self.session.require_published()
            try:
                prompt, tools, tool_sha = self.session.renderer.render(payload, thinking=thinking)
            except NativeProtocolError as exc:
                raise PlayerRequestError(str(exc), status=422) from exc
            total = self._budget(payload, thinking=thinking, summary=summary, prompt_length=prompt.length)
            forced_span = None
            think_count = 0
            attempts = []
            notices = []
            if thinking:
                close = self.session.renderer.forced_close_ids
                if total < len(close) + 2:
                    raise PlayerRequestError("MLX completion cap cannot fit thinking, its close boundary, and a response.")
                think_cap = min(self.reasoning_tokens, total - len(close) - 1)
                for attempt in range(getattr(self.session, "thinking_anomaly_attempts",
                                             getattr(self.settings, "thinking_anomaly_attempts", THINK_ANOMALY_RESAMPLE_LIMIT))):
                    cursor = self.session.prepare_prompt(prompt.to_ints())
                    tokens, probabilities, reason, cursor = self.session.draw_pass(
                        cursor, maximum=think_cap, stop_ids=self.session.renderer.eos_ids | set(self.session.renderer.close_ids))
                    naturally_closed = tokens[-1:] == list(self.session.renderer.close_ids)
                    attempts.append({"attempt": attempt + 1, "sampled_tokens": list(tokens),
                                     "sampled_logprobs": list(probabilities), "parse_termination": reason,
                                     "disposition": "accepted" if naturally_closed or reason == "length"
                                                    else "rejected_eos_inside_think"})
                    if naturally_closed or reason == "length":
                        break
                else:
                    message = "MLX think pass repeatedly ended the turn inside open thinking."
                    record = PolicyRolloutRecord(request_id=request_id, created_at=time.time(), prompt=prompt,
                        prompt_tokens=prompt.length, sampled_tokens=list(tokens), sampled_logprobs=list(probabilities),
                        content="", finish_reason="stop", payload=payload, session_id=session_id,
                        backend="mlx", policy_ref=self.session.policy_ref, policy_identity=self.session.policy_identity,
                        diagnostic_generation_attempts=tuple(attempts), protocol_row={
                            "renderer_name": self.config.thinking_renderer, "thinking": True,
                            "tool_schema_sha256": tool_sha, "parse_termination": reason, "protocol_error": message,
                            "backend": "mlx", "policy_ref": self.session.policy_ref,
                            "policy_identity_sha256": self.session.policy_identity["identity_sha256"],
                            "prompt_token_ids": prompt.to_ints()})
                    del cursor
                    raise PlayerRequestError(message, status=500, record=record)
                think_count = len(tokens)
                if not naturally_closed:
                    notices.append({"code": "reasoning_limit_exceeded",
                        "message": f"Reasoning reached its {think_cap}-token limit; thinking was closed and response generation continued.",
                        "limit_tokens": think_cap, "configured_limit_tokens": self.reasoning_tokens})
                    forced_span = (len(tokens), len(tokens) + len(close))
                    forced, cursor = self.session.force_tokens(cursor, close)
                    tokens.extend(close)
                    probabilities.extend(forced)
                response_cap = min(self.settings.response_tokens, total - len(tokens))
                tail, tail_probabilities, reason, cursor = self.session.draw_pass(
                    cursor, maximum=response_cap, stop_ids=self.session.renderer.eos_ids)
                tokens.extend(tail)
                probabilities.extend(tail_probabilities)
                response_count = len(tail)
            else:
                cursor = self.session.prepare_prompt(prompt.to_ints())
                tokens, probabilities, reason, cursor = self.session.draw_pass(
                    cursor, maximum=total, stop_ids=self.session.renderer.eos_ids)
                response_count = len(tokens)
                response_cap = total
            del cursor
            if reason == "length":
                notices.append({"code": "response_limit_exceeded",
                    "message": f"Response reached its {response_cap}-token limit.",
                    "limit_tokens": response_cap})
            for notice in notices:
                LOGGER.warning("%s request=%s: %s", notice["code"], request_id, notice["message"])
            text = self.session.renderer.tokenizer.decode(tokens, skip_special_tokens=False)
            record = PolicyRolloutRecord(request_id=request_id, created_at=time.time(), prompt=prompt,
                prompt_tokens=prompt.length, sampled_tokens=tokens, sampled_logprobs=probabilities,
                content="", finish_reason="length" if reason == "length" else "stop", payload=payload,
                reasoning_content="", session_id=session_id, forced_close_token_span=forced_span,
                backend="mlx", policy_ref=self.session.policy_ref, policy_identity=self.session.policy_identity,
                diagnostic_generation_attempts=tuple(attempts))
            sample_info = {"renderer_name": self.config.thinking_renderer if thinking else self.config.plain_renderer,
                           "thinking": thinking, "tool_schema_sha256": tool_sha, "parse_termination": reason,
                           "thinking_tokens": think_count, "response_tokens": response_count,
                           "reasoning_effort": self.effort if thinking else "off",
                           "reasoning_limit": self.reasoning_tokens if thinking else 0,
                           "response_limit": response_cap,
                           "generation_notices": notices}
            try:
                assistant, finish = parse_native_response(text, thinking=thinking, tool_specs=tools,
                                                          request_id=request_id, clean_stop=reason == "stop")
            except NativeProtocolError as exc:
                row = {**sample_info, "backend": "mlx", "policy_ref": self.session.policy_ref,
                       "policy_identity_sha256": self.session.policy_identity["identity_sha256"],
                       "prompt_token_ids": prompt.to_ints(), "protocol_error": str(exc)}
                failure = PlayerRequestError(str(exc), status=500)
                failure.record = replace(record, protocol_row=row)
                raise failure from exc
            record = replace(record, content=str(assistant.get("content") or ""), finish_reason=finish,
                             reasoning_content=str(assistant.get("reasoning_content") or ""))
            row = protocol_row(record, sample_info, assistant)
            row.update({key: sample_info[key] for key in
                        ("reasoning_effort", "reasoning_limit", "response_limit", "generation_notices")})
            row.update({"backend": "mlx", "policy_ref": self.session.policy_ref,
                        "policy_identity_sha256": self.session.policy_identity["identity_sha256"]})
            return replace(record, protocol_row=row), assistant
