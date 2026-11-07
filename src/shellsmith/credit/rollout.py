"""Rollout and grouped-rollout contracts.

Architecture map:
    1. A task/env spec produces one or more rollout JSON records.
    2. A GRPO-style training step groups rollouts from the same task/env seed.
    3. `RolloutGroupSpec` stores the rewards used to compute advantages before
       trainable rollout records become optimizer datums.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from shellsmith.types import (
    AdvantageBaseline,
    AdvantageNormalization,
    JsonValue,
    require_finite_float,
)


def _forced_close_token_span(
    raw: object, *, sampled_len: int
) -> tuple[int, int] | None:
    """Parse the harness-injected ``</think>`` span, or None if absent.

    Main callers:
        ``TrainableRolloutRecordSpec.from_record``

    Why this exists:
        The live ``TinkerRolloutRecord`` carries this span. The parsed
        contract used to drop it, so the shellsmith Tinker bridge
        trained on tokens the policy never sampled. Bounds-check here so
        every later datum builder sees the same validated pair.
    """

    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise RuntimeError(
            f"forced_close_token_span must be a [start, end) pair, got {raw!r}."
        )
    start = int(raw[0])
    end = int(raw[1])
    if not (0 <= start <= end <= sampled_len):
        raise RuntimeError(
            "Forced-close token span is out of bounds: "
            f"{(start, end)!r} for {sampled_len} sampled tokens."
        )
    return (start, end)


@dataclass(frozen=True)
class TrainableRolloutRecordSpec:
    request_id: str
    created_at: float
    prompt: object
    prompt_tokens: int
    sampled_tokens: tuple[int, ...]
    sampled_logprobs: tuple[float, ...]
    content: str
    finish_reason: str
    payload: dict[str, JsonValue]
    hydrations: tuple[dict[str, JsonValue], ...] = ()
    forced_close_token_span: tuple[int, int] | None = None

    @staticmethod
    def from_record(record: object) -> "TrainableRolloutRecordSpec":
        sampled_tokens = tuple(int(item) for item in (getattr(record, "sampled_tokens", []) or []))
        sampled_logprobs = tuple(
            require_finite_float(item, context="sampled_logprobs")
            for item in (getattr(record, "sampled_logprobs", []) or [])
        )
        prompt = getattr(record, "prompt", None)
        if prompt is None:
            raise RuntimeError("Trainable rollout record is missing its live Tinker prompt object.")
        if not sampled_tokens or not sampled_logprobs:
            raise RuntimeError("Trainable rollout record is missing sampled tokens/logprobs.")
        if len(sampled_tokens) != len(sampled_logprobs):
            raise RuntimeError(
                "Trainable rollout record has mismatched sampled token/logprob lengths: "
                f"{len(sampled_tokens)} vs {len(sampled_logprobs)}."
            )
        hydrations = tuple(
            hydration.to_jsonable() if hasattr(hydration, "to_jsonable") else dict(hydration)
            for hydration in (getattr(record, "hydrations", []) or [])
        )
        return TrainableRolloutRecordSpec(
            request_id=str(getattr(record, "request_id", "")),
            created_at=float(getattr(record, "created_at", 0.0) or 0.0),
            prompt=prompt,
            prompt_tokens=int(getattr(record, "prompt_tokens", 0) or 0),
            sampled_tokens=sampled_tokens,
            sampled_logprobs=sampled_logprobs,
            content=str(getattr(record, "content", "")),
            finish_reason=str(getattr(record, "finish_reason", "")),
            payload=dict(getattr(record, "payload", {}) or {}),
            hydrations=hydrations,
            forced_close_token_span=_forced_close_token_span(
                getattr(record, "forced_close_token_span", None),
                sampled_len=len(sampled_tokens),
            ),
        )


@dataclass(frozen=True)
class RolloutOutcomeSpec:
    rollout_index: int
    run_id: str
    rollout_path: str
    reward: float
    trainability: str
    trainable_for_update: bool
    accepted_record_indices: tuple[int, ...]

    @staticmethod
    def from_outcome(outcome: dict[str, JsonValue]) -> "RolloutOutcomeSpec":
        return RolloutOutcomeSpec(
            rollout_index=int(outcome["rollout_index"]),
            run_id=str(outcome.get("run_id") or ""),
            rollout_path=str(outcome.get("rollout_path") or ""),
            reward=require_finite_float(outcome["reward"], context="outcome.reward"),
            trainability=str(outcome.get("trainability") or ""),
            trainable_for_update=bool(outcome.get("trainable_for_update")),
            accepted_record_indices=tuple(range(int(outcome.get("n_records") or 0)))
            if bool(outcome.get("trainable_for_update"))
            else (),
        )


@dataclass(frozen=True)
class RolloutGroupSpec:
    schema: Literal["shellsmith.rollout_group.v0"]
    group_id: str
    step: int
    world_spec_id: str
    advantage_baseline: AdvantageBaseline
    outcomes: tuple[RolloutOutcomeSpec, ...]
    advantage_normalization: AdvantageNormalization = "none"
    advantage_std_floor: float = 0.25
    advantage_cap: float = 3.0

    def advantages(self) -> tuple[float, ...]:
        """Mean-center (and optionally cap-standardize) finite rewards.

        A non-finite reward or advantage is a group-level failure. Python
        ``min``/``max`` treat NaN comparisons as false, so a NaN centered
        value would otherwise clamp to ``+advantage_cap`` and become a
        max-positive update. Do not train that group.
        """

        for outcome in self.outcomes:
            if not math.isfinite(outcome.reward):
                raise RuntimeError(
                    f"Rollout group {self.group_id!r} has a non-finite reward "
                    f"{outcome.reward!r} on rollout {outcome.rollout_index}."
                )
        trainable_rewards = [outcome.reward for outcome in self.outcomes if outcome.trainable_for_update]
        if self.advantage_baseline == "zero":
            center = 0.0
        elif trainable_rewards:
            center = sum(trainable_rewards) / len(trainable_rewards)
        else:
            center = 0.0
        if not math.isfinite(center):
            raise RuntimeError(
                f"Rollout group {self.group_id!r} produced a non-finite reward center."
            )
        centered = [
            outcome.reward - center if outcome.trainable_for_update else 0.0
            for outcome in self.outcomes
        ]
        if self.advantage_normalization == "none" or not trainable_rewards:
            values = tuple(centered)
        else:
            variance = sum((reward - center) ** 2 for reward in trainable_rewards) / len(
                trainable_rewards
            )
            scale = max(variance**0.5, self.advantage_std_floor)
            if not math.isfinite(scale) or scale <= 0:
                raise RuntimeError(
                    f"Rollout group {self.group_id!r} produced a non-finite advantage scale."
                )
            values = tuple(
                max(-self.advantage_cap, min(self.advantage_cap, value / scale))
                if outcome.trainable_for_update
                else 0.0
                for outcome, value in zip(self.outcomes, centered)
            )
        for index, value in enumerate(values):
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Rollout group {self.group_id!r} produced a non-finite advantage "
                    f"{value!r} on rollout {index}."
                )
        return values

    def to_jsonable(self) -> dict[str, JsonValue]:
        return {
            "schema": self.schema,
            "group_id": self.group_id,
            "step": self.step,
            "world_spec_id": self.world_spec_id,
            "advantage_baseline": self.advantage_baseline,
            "advantage_normalization": self.advantage_normalization,
            "advantage_std_floor": self.advantage_std_floor,
            "advantage_cap": self.advantage_cap,
            "outcomes": [outcome.__dict__.copy() for outcome in self.outcomes],
            "advantages": list(self.advantages()),
        }
