"""Build Tinker datums from any validated compaction-aware trajectory.

Architecture map:
    CompactionAwareTrajectory + per-request raw advantages
        -> rollout-wide trainable-token denominator
        -> normalized Tinker datums and explicit skipped/masked audit rows

This module deliberately knows nothing about Shellsmith rewards. Pi,
BladeRunner, and future harnesses arrive here only after their adapters have
made compaction boundaries explicit and their task set has assigned credit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from shellsmith.credit.grpo import GRPOTrainableDatumSpec
from shellsmith.credit.rollout import TrainableRolloutRecordSpec
from shellsmith.credit.compaction import CompactionAwareTrajectory


@dataclass(frozen=True)
class NormalizedTinkerDatum:
    request_id: str
    record_index: int
    raw_advantage: float
    normalized_advantage: float
    rollout_trainable_token_count: int
    rollout_normalization_scale: float
    masked_replay: bool
    datum: Any


@dataclass(frozen=True)
class NormalizedTinkerTrajectory:
    datums: tuple[NormalizedTinkerDatum, ...]
    skipped_zero_advantage_requests: tuple[str, ...]
    masked_compaction_replay_requests: tuple[str, ...]
    trainable_token_count: int
    segments: tuple[Any, ...] = ()
    history_rewrites: tuple[dict, ...] = ()


def build_normalized_tinker_trajectory(
    *,
    deps: Any,
    trajectory: CompactionAwareTrajectory,
    advantages_by_request_id: Mapping[str, float],
    group_id: str,
    step: int,
    rollout_index: int,
    reward: float,
    datum_index_start: int = 0,
    include_action_mask: bool = True,
) -> NormalizedTinkerTrajectory:
    """Normalize one logical rollout regardless of its number of sequences."""
    trajectory.validate()
    expected = {turn.request_id for turn in trajectory.turns}
    actual = set(advantages_by_request_id)
    if actual != expected:
        raise RuntimeError(
            "Tinker trajectory advantages must cover requests exactly: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}."
        )
    token_count = trajectory.trainable_sampled_token_count
    scale = 1.0 / token_count
    datums: list[NormalizedTinkerDatum] = []
    skipped: list[str] = []
    masked: list[str] = []
    response_sequences = []
    for record_index, turn in enumerate(trajectory.turns):
        record = TrainableRolloutRecordSpec.from_record(turn.record)
        raw_advantage = float(advantages_by_request_id[turn.request_id])
        is_replay = turn.kind == "compaction_replay"
        zero_credit = not is_replay and abs(raw_advantage) < 1e-12
        normalized = 0.0 if is_replay or zero_credit else raw_advantage * scale
        if is_replay:
            masked.append(turn.request_id)
        spec = GRPOTrainableDatumSpec(
            schema="shellsmith.grpo_trainable_datum.v0",
            datum_index=datum_index_start + len(datums),
            group_id=group_id,
            step=step,
            rollout_index=rollout_index,
            record_index=record_index,
            request_id=record.request_id,
            prompt=record.prompt,
            sampled_tokens=record.sampled_tokens,
            sampled_logprobs=record.sampled_logprobs,
            reward=reward,
            advantage=normalized,
            forced_close_token_span=record.forced_close_token_span,
            train_sampled_tokens=not is_replay and not zero_credit,
        )
        response_sequences.append((turn.sequence_index, spec.to_token_sequence()))
        if zero_credit:
            skipped.append(turn.request_id)
            continue
        datums.append(
            NormalizedTinkerDatum(
                request_id=turn.request_id,
                record_index=record_index,
                raw_advantage=raw_advantage,
                normalized_advantage=normalized,
                rollout_trainable_token_count=token_count,
                rollout_normalization_scale=scale,
                masked_replay=is_replay,
                datum=spec.to_token_sequence() if deps is None else spec.to_tinker_datum(
                    deps, include_action_mask=include_action_mask
                ),
            )
        )
    from shellsmith.train.segments import pack_rollout_segments
    packed = pack_rollout_segments(response_sequences)
    return NormalizedTinkerTrajectory(
        segments=packed.sequences, history_rewrites=packed.rewrites,
        datums=tuple(datums),
        skipped_zero_advantage_requests=tuple(skipped),
        masked_compaction_replay_requests=tuple(masked),
        trainable_token_count=token_count,
    )
