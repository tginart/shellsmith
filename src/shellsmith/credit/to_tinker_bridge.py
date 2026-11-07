"""Convert accepted shellsmith captures into live Tinker RFT datums.

Architecture map:
    live Tinker proxy records ─┐
                              ├─> exact record↔world-turn mapping
    accepted rollout artifacts┘             │
                                            v
                              rollout-relative advantages
                              + signed per-turn reward
                              + optional reasoning correction
                                            │
                                            v
                              GRPOTrainableDatumSpec
                                            │
                                            v
                                  tinker.types.Datum

Why this is an in-memory bridge:
    A Tinker datum needs the renderer-specific prompt object and the exact
    behavior-policy sampled tokens/logprobs. Those objects exist in the live
    ``TinkerRolloutRecord`` but are intentionally not reconstructed from text
    logs. Completed OpenAI-compatible (non-Tinker) calibration rollouts therefore cannot be made
    trainable after the fact; production collection must sample through the
    Tinker proxy and call this bridge before releasing the live records.

Important invariants:
    - Every captured model request is represented exactly once.
    - A request either maps to one scored world action or is explicitly marked
      unscored (for example a final answer or compaction summary).
    - ``turn_reward`` is already a signed value in [-1, 1] and is passed through
      unchanged as PR 2444's ``reward_delta``.
    - Missing quality/efficiency pairs are neutral, not zero-quality examples.
    - Only acceptance-audited rollouts from one exact trainer group may mix.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

from shellsmith.credit.rollout import (
    RolloutGroupSpec,
    RolloutOutcomeSpec,
    TrainableRolloutRecordSpec,
)
from shellsmith.credit.turn_credit import (
    RolloutTurnFeedbackSpec,
    TurnAdvantageConfig,
    TurnAdvantageSpec,
    WORLD_SCORED_KIND,
    compute_group_turn_advantages,
)
from shellsmith.credit.compaction import (
    CompactionAwareTrajectory,
    CompactionTurn,
)
from shellsmith.credit.trajectory_pi import PiCompactionAdapter
from shellsmith.credit.compaction_datums import (
    build_normalized_tinker_trajectory,
)


RecordKind = Literal[
    "world_action",
    "unscored_model_turn",
    "compaction_generation",
    "compaction_replay",
    "final_text",
]


@dataclass(frozen=True)
class CapturedModelTurn:
    """One live Tinker record with an explicit reward-attribution boundary."""

    record: object
    kind: RecordKind
    world_turn_indices: tuple[int, ...] = ()

    def validate(self) -> TrainableRolloutRecordSpec:
        parsed = TrainableRolloutRecordSpec.from_record(self.record)
        if not parsed.request_id:
            raise RuntimeError("Captured Tinker record requires a non-empty request_id.")
        if self.kind == WORLD_SCORED_KIND:
            if not self.world_turn_indices or any(
                index < 0 for index in self.world_turn_indices
            ):
                raise RuntimeError(
                    "world_action capture requires non-negative turn indices."
                )
            if len(set(self.world_turn_indices)) != len(self.world_turn_indices):
                raise RuntimeError("world_action capture repeats a turn index.")
        elif self.kind in {
            "unscored_model_turn",
            "compaction_generation",
            "compaction_replay",
            "final_text",
        }:
            if self.world_turn_indices:
                raise RuntimeError(f"{self.kind} cannot carry world turn indices.")
        else:
            raise RuntimeError(f"Unsupported captured model turn kind: {self.kind!r}.")
        return parsed


@dataclass(frozen=True)
class CapturedRollout:
    """One accepted rollout plus all live behavior-policy request records."""

    rollout_root: Path
    group_identity_key: str
    captures: tuple[CapturedModelTurn, ...]


@dataclass(frozen=True)
class BridgedDatum:
    """One auditable live Tinker datum and its advantage decomposition."""

    rollout_index: int
    record_index: int
    request_id: str
    reward: float
    rollout_advantage: float
    turn_advantage: TurnAdvantageSpec
    normalized_advantage: float
    rollout_trainable_token_count: int
    rollout_normalization_scale: float
    datum: Any


@dataclass(frozen=True)
class TinkerBridgeResult:
    """Complete optimizer-ready result for one exact rollout group."""

    group_id: str
    world_id: str
    group_identity_key: str
    rollout_rewards: tuple[float, ...]
    rollout_advantages: tuple[float, ...]
    datums: tuple[BridgedDatum, ...]
    skipped_zero_advantage_requests: tuple[str, ...]
    # Replayed compaction summaries: present in the batch, trained on nothing.
    masked_compaction_replay_requests: tuple[str, ...] = ()
    rollout_trainable_token_counts: tuple[int, ...] = ()
    rollout_sequence_counts: tuple[int, ...] = ()
    rollout_compaction_counts: tuple[int, ...] = ()
    segments: tuple[Any, ...] = ()
    history_rewrites: tuple[dict, ...] = ()

    def audit_jsonable(self) -> dict[str, Any]:
        return {
            "schema": "shellsmith.tinker_bridge.v1",
            "group_id": self.group_id,
            "world_id": self.world_id,
            "group_identity_key": self.group_identity_key,
            "rollout_rewards": list(self.rollout_rewards),
            "rollout_advantages": list(self.rollout_advantages),
            "datum_count": len(self.datums),
            "segment_count": len(self.segments),
            "history_rewrites": list(self.history_rewrites),
            "skipped_zero_advantage_requests": list(
                self.skipped_zero_advantage_requests
            ),
            "masked_compaction_replay_requests": list(
                self.masked_compaction_replay_requests
            ),
            "rollout_trainable_token_counts": list(self.rollout_trainable_token_counts),
            "rollout_sequence_counts": list(self.rollout_sequence_counts),
            "rollout_compaction_counts": list(self.rollout_compaction_counts),
            "datums": [
                {
                    "rollout_index": item.rollout_index,
                    "record_index": item.record_index,
                    "request_id": item.request_id,
                    "reward": item.reward,
                    "rollout_advantage": item.rollout_advantage,
                    "raw_final_turn_advantage": item.turn_advantage.final_advantage,
                    "rollout_trainable_token_count": item.rollout_trainable_token_count,
                    "rollout_normalization_scale": item.rollout_normalization_scale,
                    "normalized_advantage": item.normalized_advantage,
                    "turn_advantage": item.turn_advantage.to_jsonable(),
                }
                for item in self.datums
            ],
        }


def _load_json(path: Path, *, context: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{context} is absent: {path}.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must contain a JSON object: {path}.")
    return value


def accepted_rollout_metadata(root: Path) -> tuple[str, float]:
    """Load the deterministic acceptance decision and terminal reward."""
    audit = _load_json(
        root.joinpath("acceptance_audit.json"), context="acceptance audit"
    )
    if audit.get("accepted_for_training") is not True:
        raise RuntimeError(f"Rollout is not accepted for training: {root}.")
    task_id = audit.get("world_id")
    if not isinstance(task_id, str) or not task_id:
        episode = _load_json(root.joinpath("episode.json"), context="episode state")
        task_id = episode.get("world_id")
    if not isinstance(task_id, str) or not task_id:
        raise RuntimeError(f"Accepted rollout is missing world_id: {root}.")
    final_task = audit.get("final_task")
    reward = final_task.get("reward") if isinstance(final_task, dict) else None
    if not isinstance(reward, (int, float)) or isinstance(reward, bool):
        raise RuntimeError(f"Accepted rollout has no numeric terminal reward: {root}.")
    return task_id, float(reward)


def feedback_for_capture(capture: CapturedRollout) -> RolloutTurnFeedbackSpec:
    """Build exact PR 2444 feedback for every captured Tinker request."""
    turns: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    scored_indices: set[int] = set()
    for item in capture.captures:
        record = item.validate()
        if record.request_id in request_ids:
            raise RuntimeError(
                f"Captured request_id appears more than once: {record.request_id!r}."
            )
        request_ids.add(record.request_id)
        if item.kind != WORLD_SCORED_KIND:
            turns.append({"request_id": record.request_id, "reward_delta": 0.0})
            continue
        responses: list[dict[str, Any]] = []
        for turn_index in item.world_turn_indices:
            if turn_index in scored_indices:
                raise RuntimeError(
                    f"World turn {turn_index} maps to multiple model requests."
                )
            scored_indices.add(turn_index)
            responses.append(
                _load_json(
                    capture.rollout_root.joinpath(
                        "world",
                        "turns",
                        f"{turn_index:06d}",
                        "response.json",
                    ),
                    context=f"world turn {turn_index} response",
                )
            )
        rewards = [response.get("turn_reward") for response in responses]
        if any(
            not isinstance(reward, (int, float))
            or isinstance(reward, bool)
            or not -1.0 <= float(reward) <= 1.0
            for reward in rewards
        ):
            raise RuntimeError("Mapped world action has invalid turn_reward.")
        # One sampled assistant block may issue several parallel tool calls.
        # Average their scores so parallelism does not multiply credit merely by
        # splitting one decision into more calls.
        reward = sum(float(value) for value in rewards) / len(rewards)
        row: dict[str, Any] = {
            "request_id": record.request_id,
            "reward_delta": reward,
        }
        assessments = [
            response.get("reasoning_assessment")
            for response in responses
            if response.get("reasoning_assessment") is not None
        ]
        if assessments:
            if any(not isinstance(value, dict) for value in assessments):
                raise RuntimeError(
                    "Mapped world action reasoning assessment must be an object or null."
                )
            row["reasoning_quality"] = sum(
                float(value["quality"]) for value in assessments
            ) / len(assessments)
            row["reasoning_efficiency"] = sum(
                float(value["efficiency"]) for value in assessments
            ) / len(assessments)
        turns.append(row)
    if not turns:
        raise RuntimeError("Captured rollout has no Tinker model records.")
    return RolloutTurnFeedbackSpec.from_dict(
        {"schema": "shellsmith.rollout_turn_feedback.v0", "turns": turns}
    )


def _sampled_token_count(record: TrainableRolloutRecordSpec) -> int:
    return len(record.sampled_tokens)


def captures_from_vat_log(
    *, records: list[object], rollout_root: Path
) -> tuple[CapturedModelTurn, ...]:
    """Align ordered live Tinker records with durable Pi assistant events.

    Assistant message blocks expose their exact tool-call IDs. World requests
    store those IDs as action IDs, so this mapping is deterministic even when a
    single sampled block issues parallel tools. Multiple tools that land in the
    same GM move collapse to one turn index (the move is the scored unit).
    Compaction calls and terminal assistant text are explicitly neutral. Blank
    assistant message_end shells (empty content, no tools/thinking) are dropped
    — they are not samples. Any unexplained model record fails.
    """
    trajectory = PiCompactionAdapter().adapt(
        rollout_id=rollout_root.name,
        records=records,
        evidence=rollout_root,
    )
    return tuple(
        CapturedModelTurn(
            record=turn.record,
            kind=turn.kind,  # type: ignore[arg-type]
            world_turn_indices=turn.world_turn_indices,
        )
        for turn in trajectory.turns
    )


def build_tinker_group(
    *,
    deps: Any,
    group_id: str,
    step: int,
    rollouts: tuple[CapturedRollout, ...],
    turn_config: TurnAdvantageConfig = TurnAdvantageConfig(),
    advantage_std_floor: float = 0.25,
    advantage_cap: float = 3.0,
    include_action_mask: bool = True,
) -> TinkerBridgeResult:
    """Validate one exact group and build optimizer-ready Tinker datums.

    Main callers:
        The trainer after a GROUP_READY world is banked.
    """
    if len(rollouts) < 2:
        # Relative (group-normalized) advantages need at least two rollouts to
        # carry any signal. The trainer's minimum_trainable policy decides
        # how many beyond that a production run demands; a hardcoded four here
        # contradicted --minimum-trainable 2 and crashed disposable runs.
        raise RuntimeError(
            "A shellsmith Tinker group requires at least two rollouts; "
            f"got {len(rollouts)}."
        )
    identity_keys = {rollout.group_identity_key for rollout in rollouts}
    if len(identity_keys) != 1:
        raise RuntimeError("Tinker bridge cannot mix trainer rollout-group keys.")
    if not next(iter(identity_keys)):
        raise RuntimeError("Tinker bridge group identity key must be non-empty.")
    mlx_records = [capture.record for rollout in rollouts for capture in rollout.captures
                   if getattr(capture.record, "backend", "tinker") in {"mlx","studio"}]
    if mlx_records:
        all_records = [capture.record for rollout in rollouts for capture in rollout.captures]
        revisions = {(record.policy_ref, json.dumps(record.policy_identity, sort_keys=True)) for record in mlx_records}
        if len(mlx_records) != len(all_records) or len(revisions) != 1 or not mlx_records[0].policy_ref:
            raise RuntimeError("MLX group cannot mix policy identities or revisions.")

    task_ids: list[str] = []
    rewards: list[float] = []
    feedback: list[RolloutTurnFeedbackSpec] = []
    request_ids: list[list[str]] = []
    request_kinds: list[list[str]] = []
    token_counts: list[list[int]] = []
    trajectories: list[CompactionAwareTrajectory] = []
    for rollout in rollouts:
        task_id, reward = accepted_rollout_metadata(rollout.rollout_root)
        task_ids.append(task_id)
        rewards.append(reward)
        rollout_feedback = feedback_for_capture(rollout)
        feedback.append(rollout_feedback)
        records = [item.validate() for item in rollout.captures]
        request_ids.append([record.request_id for record in records])
        request_kinds.append([item.kind for item in rollout.captures])
        token_counts.append([_sampled_token_count(record) for record in records])
        if rollout.rollout_root.joinpath("logs", "vat.jsonl").is_file():
            # Rebuild seams from durable evidence, also on capture resume. A
            # split compaction has two generators but advances only once.
            trajectory = PiCompactionAdapter().adapt(
                rollout_id=rollout.rollout_root.name,
                records=[capture.record for capture in rollout.captures],
                evidence=rollout.rollout_root,
            )
            if any(
                (turn.kind, turn.world_turn_indices) != (capture.kind, capture.world_turn_indices)
                for turn, capture in zip(trajectory.turns, rollout.captures, strict=True)
            ):
                raise RuntimeError("Captured turn attribution differs from the durable Pi journal.")
            trajectories.append(trajectory)
        elif any(
            capture.kind == "compaction_generation"
            and (getattr(capture.record, "protocol_row", None) or getattr(capture.record, "payload", None))
            for capture in rollout.captures
        ):
            raise RuntimeError("Pi compaction captures require the durable vat journal to reconstruct boundaries.")
        else:
            # Legacy hand-assembled captures have no journal and one generator
            # per seam. Never use this fallback when journal evidence exists.
            sequence_index = 0
            trajectory_turns: list[CompactionTurn] = []
            for capture in rollout.captures:
                trajectory_turns.append(
                    CompactionTurn(
                        record=capture.record,
                        kind=capture.kind,
                        world_turn_indices=capture.world_turn_indices,
                        sequence_index=sequence_index,
                    )
                )
                if capture.kind == "compaction_generation":
                    sequence_index += 1
            trajectories.append(
                CompactionAwareTrajectory(
                    rollout_id=rollout.rollout_root.name,
                    harness_id="pi",
                    turns=tuple(trajectory_turns),
                ).validate()
            )
    if len(set(task_ids)) != 1:
        raise RuntimeError("Tinker bridge cannot mix task environments.")

    outcomes = tuple(
        RolloutOutcomeSpec(
            rollout_index=index,
            run_id=rollout.rollout_root.name,
            rollout_path=str(rollout.rollout_root),
            reward=rewards[index],
            trainability="trainable",
            trainable_for_update=True,
            accepted_record_indices=tuple(range(len(rollout.captures))),
        )
        for index, rollout in enumerate(rollouts)
    )
    rollout_group = RolloutGroupSpec(
        schema="shellsmith.rollout_group.v0",
        group_id=group_id,
        step=step,
        world_spec_id=task_ids[0],
        advantage_baseline="group_mean",
        advantage_normalization="capped_standardization",
        advantage_std_floor=advantage_std_floor,
        advantage_cap=advantage_cap,
        outcomes=outcomes,
    )
    rollout_advantages = rollout_group.advantages()
    turn_advantages = compute_group_turn_advantages(
        rollout_advantages=list(rollout_advantages),
        feedback_by_rollout=list(feedback),
        request_ids_by_rollout=request_ids,
        sampled_token_counts_by_rollout=token_counts,
        request_kinds_by_rollout=request_kinds,
        config=turn_config,
    )

    datums: list[BridgedDatum] = []
    skipped: list[str] = []
    masked_replays: list[str] = []
    segments, history_rewrites = [], []
    datum_index = 0
    rollout_token_counts = tuple(
        trajectory.trainable_sampled_token_count for trajectory in trajectories
    )
    for rollout_index, (trajectory, rollout_turn_advantages) in enumerate(
        zip(trajectories, turn_advantages, strict=True)
    ):
        advantage_by_request = {
            item.request_id: item.final_advantage for item in rollout_turn_advantages
        }
        normalized = build_normalized_tinker_trajectory(
            deps=deps,
            trajectory=trajectory,
            advantages_by_request_id=advantage_by_request,
            group_id=group_id,
            step=step,
            rollout_index=rollout_index,
            reward=rewards[rollout_index],
            datum_index_start=datum_index,
            include_action_mask=include_action_mask,
        )
        segments.extend(normalized.segments)
        history_rewrites.extend({**row, "rollout_index": rollout_index} for row in normalized.history_rewrites)
        skipped.extend(normalized.skipped_zero_advantage_requests)
        masked_replays.extend(normalized.masked_compaction_replay_requests)
        turn_advantage_by_request = {
            item.request_id: item for item in rollout_turn_advantages
        }
        for item in normalized.datums:
            turn_advantage = turn_advantage_by_request[item.request_id]
            datums.append(
                BridgedDatum(
                    rollout_index=rollout_index,
                    record_index=item.record_index,
                    request_id=item.request_id,
                    reward=rewards[rollout_index],
                    rollout_advantage=rollout_advantages[rollout_index],
                    turn_advantage=turn_advantage,
                    normalized_advantage=item.normalized_advantage,
                    rollout_trainable_token_count=item.rollout_trainable_token_count,
                    rollout_normalization_scale=item.rollout_normalization_scale,
                    datum=item.datum,
                )
            )
        datum_index += len(normalized.datums)
    return TinkerBridgeResult(
        group_id=group_id,
        world_id=task_ids[0],
        group_identity_key=next(iter(identity_keys)),
        rollout_rewards=tuple(rewards),
        rollout_advantages=tuple(rollout_advantages),
        datums=tuple(datums),
        segments=tuple(segments), history_rewrites=tuple(history_rewrites),
        skipped_zero_advantage_requests=tuple(skipped),
        masked_compaction_replay_requests=tuple(masked_replays),
        rollout_trainable_token_counts=rollout_token_counts,
        rollout_sequence_counts=tuple(item.sequence_count for item in trajectories),
        rollout_compaction_counts=tuple(
            len({
                ("boundary", turn.boundary_id) if turn.boundary_id is not None else ("request", turn.request_id)
                for turn in item.turns if turn.kind == "compaction_generation"
            })
            for item in trajectories
        ),
    )


def build_token_group(**kwargs: Any) -> TinkerBridgeResult:
    """Reuse audited credit and compaction semantics, returning portable token sequences."""
    return build_tinker_group(deps=None, **kwargs)
