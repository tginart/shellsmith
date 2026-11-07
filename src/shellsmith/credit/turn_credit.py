"""Turn-level reward feedback and final advantage construction.

Architecture map:
    Judge feedback for each model request -> signed reward deltas are compared
    only across world-scored turns in the same rollout group -> reasoning
    quality times efficiency uses the same token-weighted zero-sum residual ->
    both corrections adjust the already-computed rollout advantage -> one final
    scalar advantage is assigned to the generated tokens for that request.

Important invariants:
    - A feedback block, when present, covers every captured model request in its
      rollout exactly once. Partial or ambiguous credit assignment fails loudly.
    - Signed reward deltas are never mean-centered globally; zero remains neutral.
    - Reasoning scores are comparative within scored turns only. Missing pairs are
      excluded from the reasoning mean and contribute zero reasoning residual.
    - Unscored final text, compaction generation, and compaction replay receive
      rollout advantage only with zero turn-level residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Literal

from shellsmith.types import JsonValue, require_dict


WORLD_SCORED_KIND = "world_action"


@dataclass(frozen=True)
class TurnFeedbackSpec:
    """Judge feedback for one captured model completion/tool-call block."""

    request_id: str
    reward_delta: float
    reasoning_quality: float | None = None
    reasoning_efficiency: float | None = None

    @staticmethod
    def from_dict(raw: Any) -> "TurnFeedbackSpec":
        payload = require_dict(raw, context="turn feedback")
        request_id = str(payload.get("request_id") or "").strip()
        if not request_id:
            raise RuntimeError("Turn feedback requires a non-empty request_id.")
        reward_delta = bounded_float(payload.get("reward_delta"), context="turn feedback reward_delta", low=-1.0, high=1.0)
        quality_raw = payload.get("reasoning_quality")
        efficiency_raw = payload.get("reasoning_efficiency")
        if (quality_raw is None) != (efficiency_raw is None):
            raise RuntimeError(
                "Turn feedback reasoning_quality and reasoning_efficiency must either both be present or both be absent."
            )
        quality = (
            None
            if quality_raw is None
            else bounded_float(quality_raw, context="turn feedback reasoning_quality", low=0.0, high=1.0)
        )
        efficiency = (
            None
            if efficiency_raw is None
            else bounded_float(efficiency_raw, context="turn feedback reasoning_efficiency", low=0.0, high=1.0)
        )
        return TurnFeedbackSpec(
            request_id=request_id,
            reward_delta=reward_delta,
            reasoning_quality=quality,
            reasoning_efficiency=efficiency,
        )

    @property
    def reasoning_score(self) -> float | None:
        if self.reasoning_quality is None or self.reasoning_efficiency is None:
            return None
        return self.reasoning_quality * self.reasoning_efficiency

    def to_jsonable(self) -> dict[str, JsonValue]:
        return {
            "request_id": self.request_id,
            "reward_delta": self.reward_delta,
            "reasoning_quality": self.reasoning_quality,
            "reasoning_efficiency": self.reasoning_efficiency,
            "reasoning_score": self.reasoning_score,
        }


@dataclass(frozen=True)
class RolloutTurnFeedbackSpec:
    """Complete turn-feedback payload for one rollout."""

    schema: Literal["shellsmith.rollout_turn_feedback.v0"]
    turns: tuple[TurnFeedbackSpec, ...]

    @staticmethod
    def from_dict(raw: Any) -> "RolloutTurnFeedbackSpec":
        payload = require_dict(raw, context="rollout turn feedback")
        if payload.get("schema") != "shellsmith.rollout_turn_feedback.v0":
            raise RuntimeError("Unsupported rollout turn feedback schema.")
        turns_raw = payload.get("turns")
        if not isinstance(turns_raw, list) or not turns_raw:
            raise RuntimeError("Rollout turn feedback requires a non-empty turns list.")
        turns = tuple(TurnFeedbackSpec.from_dict(item) for item in turns_raw)
        request_ids = [turn.request_id for turn in turns]
        if len(set(request_ids)) != len(request_ids):
            raise RuntimeError("Rollout turn feedback request_id values must be unique.")
        return RolloutTurnFeedbackSpec(schema="shellsmith.rollout_turn_feedback.v0", turns=turns)

    def require_exact_request_ids(self, request_ids: list[str]) -> None:
        """Fail unless judge feedback maps one-to-one to captured model records.

        Main callers:
            The Tinker supervisor immediately after one simulator rollout.
        """
        if len(set(request_ids)) != len(request_ids):
            raise RuntimeError("Captured rollout request_id values must be unique for turn credit assignment.")
        expected = set(request_ids)
        actual = {turn.request_id for turn in self.turns}
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise RuntimeError(
                "Rollout turn feedback must cover captured requests exactly: "
                f"missing={missing} extra={extra}."
            )

    def to_jsonable(self) -> dict[str, JsonValue]:
        return {
            "schema": self.schema,
            "turns": [turn.to_jsonable() for turn in self.turns],
        }


@dataclass(frozen=True)
class TurnAdvantageConfig:
    """Bounded coefficients for token-weighted zero-sum turn residuals.

    ``residual_cap`` is the absolute ceiling on the combined turn correction.
    ``residual_cap_floor`` is the *lower* bound on the per-turn cap that
    ``resolve_residual_cap`` derives from the rollout advantage — see that
    function for why a floor is required.
    """

    credit_coefficient: float = 0.5
    reasoning_coefficient: float = 0.25
    reward_std_floor: float = 0.1
    reasoning_std_floor: float = 0.1
    residual_cap: float = 2.0
    residual_cap_floor: float = 0.25
    final_advantage_cap: float = 4.0


def resolve_residual_cap(
    rollout_advantage: float, *, config: TurnAdvantageConfig
) -> float:
    """Return the per-turn correction ceiling for one rollout advantage.

    The rule, in words:
        Start from "turn credit may move the advantage by at most half of its
        own magnitude" — that is what keeps turn credit from flipping the sign
        of a rollout the group clearly ranked. Then clamp that value into
        ``[residual_cap_floor, residual_cap]``.

    Why the floor exists:
        Without it the cap is exactly ``0.5 * |A_roll|``, which collapses toward
        zero precisely where turn credit matters most. A group whose rewards
        barely differ lands near the GRPO advantage floor — say
        ``A_roll = 0.03`` — and the cap becomes ``0.015``, so every per-turn
        correction is clamped into noise and the update carries almost no
        signal. Sign preservation is only meaningful once the rollout advantage
        is itself meaningful; below ``2 * residual_cap_floor`` there is no sign
        worth protecting and turn credit is allowed to decide.

    Why this is a function:
        ``compute_group_turn_advantages`` and ``neutral_turn_advantage`` both
        publish ``residual_cap`` in their audit rows. They used to compute it
        with two different expressions that disagreed at ``A_roll == 0`` (2.0
        versus 0.0). One definition now, one call site each.

    Main callers:
        ``compute_group_turn_advantages``, ``neutral_turn_advantage``.
    """
    sign_preserving = 0.5 * abs(rollout_advantage)
    return min(config.residual_cap, max(config.residual_cap_floor, sign_preserving))


@dataclass(frozen=True)
class TurnAdvantageSpec:
    """Auditable decomposition of the advantage assigned to one model turn."""

    request_id: str
    rollout_advantage: float
    reward_delta: float
    request_kind: str
    sampled_token_count: int
    reward_group_mean: float
    reward_residual: float
    reward_group_std: float
    reward_scale: float
    normalized_turn_credit: float
    reasoning_quality: float | None
    reasoning_efficiency: float | None
    reasoning_score: float | None
    reasoning_group_mean: float
    reasoning_residual: float
    reasoning_group_std: float
    reasoning_scale: float
    reasoning_z_score: float
    residual_cap: float
    credit_correction: float
    reasoning_correction: float
    final_advantage: float

    def to_jsonable(self) -> dict[str, JsonValue]:
        return self.__dict__.copy()


def compute_group_turn_advantages(
    *,
    rollout_advantages: list[float],
    feedback_by_rollout: list[RolloutTurnFeedbackSpec | None],
    request_ids_by_rollout: list[list[str]],
    sampled_token_counts_by_rollout: list[list[int]],
    request_kinds_by_rollout: list[list[str]] | None = None,
    config: TurnAdvantageConfig,
) -> list[list[TurnAdvantageSpec]]:
    """Compute final per-request advantages for a complete rollout group.

    Main callers:
        ``to_tinker_bridge.build_tinker_group`` and the Tinker supervisor.

    Why this exists:
        Turn rewards and reasoning scores are compared only within world-scored
        requests using token-weighted zero-sum residuals. Unscored captures keep
        rollout advantage only. If fewer than two scored turns have variance,
        every request in the group keeps rollout advantage unchanged.
    """
    if not (
        len(rollout_advantages)
        == len(feedback_by_rollout)
        == len(request_ids_by_rollout)
        == len(sampled_token_counts_by_rollout)
    ):
        raise RuntimeError("Turn advantage inputs must have one entry per rollout.")
    if request_kinds_by_rollout is not None and len(request_kinds_by_rollout) != len(
        rollout_advantages
    ):
        raise RuntimeError("request_kinds_by_rollout must align with rollout count.")
    validate_turn_advantage_config(config)

    scored_reward_rows: list[tuple[float, int]] = []
    scored_reasoning_rows: list[tuple[float, int]] = []
    for rollout_index, feedback in enumerate(feedback_by_rollout):
        if feedback is None:
            continue
        kinds = (
            request_kinds_by_rollout[rollout_index]
            if request_kinds_by_rollout is not None
            else [WORLD_SCORED_KIND] * len(feedback.turns)
        )
        token_counts = sampled_token_counts_by_rollout[rollout_index]
        turn_by_id = {turn.request_id: turn for turn in feedback.turns}
        for request_id, kind, token_count in zip(
            request_ids_by_rollout[rollout_index], kinds, token_counts, strict=True
        ):
            turn = turn_by_id.get(request_id)
            if turn is None or kind != WORLD_SCORED_KIND:
                continue
            weight = max(int(token_count), 1)
            scored_reward_rows.append((turn.reward_delta, weight))
            if turn.reasoning_score is not None:
                scored_reasoning_rows.append((turn.reasoning_score, weight))

    reward_residuals_enabled = _residuals_enabled(scored_reward_rows)
    reasoning_residuals_enabled = _residuals_enabled(scored_reasoning_rows)
    reward_mean = _weighted_mean(scored_reward_rows)
    reward_std = _weighted_std(scored_reward_rows, center=reward_mean)
    reward_scale = max(reward_std, config.reward_std_floor)
    reasoning_mean = _weighted_mean(scored_reasoning_rows)
    reasoning_std = _weighted_std(scored_reasoning_rows, center=reasoning_mean)
    reasoning_scale = max(reasoning_std, config.reasoning_std_floor)

    group_results: list[list[TurnAdvantageSpec]] = []
    for rollout_index, (rollout_advantage, feedback, request_ids) in enumerate(
        zip(rollout_advantages, feedback_by_rollout, request_ids_by_rollout, strict=True)
    ):
        token_counts = sampled_token_counts_by_rollout[rollout_index]
        kinds = (
            request_kinds_by_rollout[rollout_index]
            if request_kinds_by_rollout is not None
            else [WORLD_SCORED_KIND] * len(request_ids)
        )
        if feedback is None:
            group_results.append(
                [
                    neutral_turn_advantage(
                        request_id,
                        rollout_advantage,
                        request_kind=kind,
                        sampled_token_count=token_count,
                        config=config,
                    )
                    for request_id, kind, token_count in zip(
                        request_ids, kinds, token_counts, strict=True
                    )
                ]
            )
            continue
        feedback.require_exact_request_ids(request_ids)
        turn_by_request_id = {turn.request_id: turn for turn in feedback.turns}
        rollout_results: list[TurnAdvantageSpec] = []
        for request_id, kind, token_count in zip(
            request_ids, kinds, token_counts, strict=True
        ):
            turn = turn_by_request_id[request_id]
            residual_cap = resolve_residual_cap(rollout_advantage, config=config)
            if kind != WORLD_SCORED_KIND or not reward_residuals_enabled:
                reward_residual = 0.0
                normalized_credit = 0.0
            else:
                reward_residual = turn.reward_delta - reward_mean
                normalized_credit = clamp(
                    config.credit_coefficient * reward_residual / reward_scale,
                    residual_cap,
                )
            if (
                kind != WORLD_SCORED_KIND
                or turn.reasoning_score is None
                or not reasoning_residuals_enabled
            ):
                reasoning_residual = 0.0
                reasoning_z_score = 0.0
                reasoning_correction = 0.0
            else:
                reasoning_residual = turn.reasoning_score - reasoning_mean
                reasoning_z_score = reasoning_residual / reasoning_scale
                reasoning_correction = clamp(
                    config.reasoning_coefficient * reasoning_z_score,
                    residual_cap,
                )
            credit_correction = normalized_credit
            # Cap the combined residual so turn credit cannot flip A_roll's sign.
            total_correction = clamp(
                credit_correction + reasoning_correction,
                residual_cap,
            )
            if total_correction != credit_correction + reasoning_correction:
                # Preserve audit fields but keep the applied total within ρ.
                scale = (
                    abs(total_correction)
                    / abs(credit_correction + reasoning_correction)
                    if (credit_correction + reasoning_correction) != 0
                    else 1.0
                )
                credit_correction = credit_correction * scale
                reasoning_correction = reasoning_correction * scale
            final_advantage = clamp(
                rollout_advantage + total_correction,
                config.final_advantage_cap,
            )
            rollout_results.append(
                TurnAdvantageSpec(
                    request_id=request_id,
                    rollout_advantage=rollout_advantage,
                    reward_delta=turn.reward_delta,
                    request_kind=kind,
                    sampled_token_count=max(int(token_count), 0),
                    reward_group_mean=reward_mean,
                    reward_residual=reward_residual,
                    reward_group_std=reward_std,
                    reward_scale=reward_scale,
                    normalized_turn_credit=normalized_credit,
                    reasoning_quality=turn.reasoning_quality,
                    reasoning_efficiency=turn.reasoning_efficiency,
                    reasoning_score=turn.reasoning_score,
                    reasoning_group_mean=reasoning_mean,
                    reasoning_residual=reasoning_residual,
                    reasoning_group_std=reasoning_std,
                    reasoning_scale=reasoning_scale,
                    reasoning_z_score=reasoning_z_score,
                    residual_cap=residual_cap,
                    credit_correction=credit_correction,
                    reasoning_correction=reasoning_correction,
                    final_advantage=final_advantage,
                )
            )
        group_results.append(rollout_results)
    return group_results


def neutral_turn_advantage(
    request_id: str,
    rollout_advantage: float,
    *,
    request_kind: str = "unscored_model_turn",
    sampled_token_count: int = 0,
    config: TurnAdvantageConfig,
) -> TurnAdvantageSpec:
    return TurnAdvantageSpec(
        request_id=request_id,
        rollout_advantage=rollout_advantage,
        reward_delta=0.0,
        request_kind=request_kind,
        sampled_token_count=sampled_token_count,
        reward_group_mean=0.0,
        reward_residual=0.0,
        reward_group_std=0.0,
        reward_scale=config.reward_std_floor,
        normalized_turn_credit=0.0,
        reasoning_quality=None,
        reasoning_efficiency=None,
        reasoning_score=None,
        reasoning_group_mean=0.0,
        reasoning_residual=0.0,
        reasoning_group_std=0.0,
        reasoning_scale=config.reasoning_std_floor,
        reasoning_z_score=0.0,
        residual_cap=resolve_residual_cap(rollout_advantage, config=config),
        credit_correction=0.0,
        reasoning_correction=0.0,
        final_advantage=rollout_advantage,
    )


def validate_turn_advantage_config(config: TurnAdvantageConfig) -> None:
    positive_fields = {
        "reward_std_floor": config.reward_std_floor,
        "reasoning_std_floor": config.reasoning_std_floor,
        "residual_cap": config.residual_cap,
        "residual_cap_floor": config.residual_cap_floor,
        "final_advantage_cap": config.final_advantage_cap,
    }
    for name, value in positive_fields.items():
        if value <= 0:
            raise RuntimeError(f"Turn advantage {name} must be greater than 0.")
    if config.residual_cap_floor > config.residual_cap:
        raise RuntimeError(
            "Turn advantage residual_cap_floor cannot exceed residual_cap "
            f"(floor={config.residual_cap_floor}, cap={config.residual_cap})."
        )
    if config.credit_coefficient < 0 or config.reasoning_coefficient < 0:
        raise RuntimeError("Turn advantage correction coefficients must be non-negative.")


def bounded_float(value: Any, *, context: str, low: float, high: float) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{context} must be a number between {low} and {high}.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} must be a number between {low} and {high}.") from exc
    if not low <= number <= high:
        raise RuntimeError(f"{context} must be between {low} and {high}.")
    return number


def clamp(value: float, cap: float) -> float:
    return max(-cap, min(cap, value))


def _weighted_mean(rows: list[tuple[float, int]]) -> float:
    if not rows:
        return 0.0
    total_weight = sum(weight for _, weight in rows)
    if total_weight <= 0:
        return 0.0
    return sum(value * weight for value, weight in rows) / total_weight


def _weighted_std(rows: list[tuple[float, int]], *, center: float) -> float:
    if not rows:
        return 0.0
    total_weight = sum(weight for _, weight in rows)
    if total_weight <= 0:
        return 0.0
    variance = sum(weight * (value - center) ** 2 for value, weight in rows) / total_weight
    return sqrt(variance)


def _residuals_enabled(rows: list[tuple[float, int]]) -> bool:
    if len(rows) < 2:
        return False
    values = [value for value, _ in rows]
    return len(set(round(value, 12) for value in values)) > 1
