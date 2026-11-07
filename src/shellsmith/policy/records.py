"""The token-exact record of one Player action sampled from Tinker.

Architecture map
----------------
    Pi chat request (OpenAI-compatible)
        -> renderer prompt                      (``prompt``: tinker.ModelInput)
        -> Tinker sample                        (``sampled_tokens``, ``sampled_logprobs``)
        -> assistant message returned to Pi     (``content``, ``reasoning_content``)
    TinkerRolloutRecord  ->  CapturedModelTurn (credit.to_tinker_bridge)  ->  CISPO datum

Why this exists
---------------
CISPO cannot train from the final assistant text alone. It needs the exact
prompt object, the sampled tokens and the behavior-policy logprobs from the
rollout. This record is that handoff from the policy endpoint to the trainer.
Ported from the original monorepo's LLM proxy without the image
``hydrations`` field: the Player is text-only.

Main callers:
    The Shellsmith policy endpoint (writes); ``train.durable_capture`` (rehydrates
    on resume); ``credit.to_tinker_bridge`` (reads).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PolicyRolloutRecord:
    """One Player action sampled through the policy endpoint."""

    request_id: str
    created_at: float
    prompt: Any
    prompt_tokens: int
    sampled_tokens: list[int]
    sampled_logprobs: list[float]
    content: str
    finish_reason: str
    payload: dict[str, Any]
    # Separate from visible ``content`` so audits and the Game Master can see the
    # Player's pre-action thinking (via Pi ThinkingPart).
    reasoning_content: str = ""
    # Which episode this sample belongs to when several episodes share one
    # endpoint. Empty when the caller runs a single episode per endpoint.
    session_id: str = ""
    # Half-open [start, end) span inside ``sampled_tokens`` holding injected
    # ``</think>`` close tokens from a forced thinking close. These tokens were
    # never drawn from the policy, so the datum builder zeroes their loss mask.
    forced_close_token_span: tuple[int, int] | None = None
    # JSONL-shaped protocol ledger row for this sample (tool_calls, renderer,
    # thinking). The acceptance audit reads these as model_protocol requests.
    protocol_row: dict[str, Any] = field(default_factory=dict)
    backend: str = "tinker"
    policy_ref: str | None = None
    policy_identity: dict[str, Any] = field(default_factory=dict)
    # Rejected thinking draws remain diagnostic evidence; only sampled_tokens
    # participates in credit and optimizer replay.
    diagnostic_generation_attempts: tuple[dict[str, Any], ...] = ()


# Public compatibility name used by existing integrations and saved captures.
TinkerRolloutRecord = PolicyRolloutRecord
