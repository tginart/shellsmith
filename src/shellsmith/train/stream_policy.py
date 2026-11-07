"""Adaptive stream policy for one campaign: per-world decisions, never mix models.

Architecture map
----------------
    world outcomes (after a collect wave)
            |
            v
    evaluate_world_outcomes(..., provider_down=tracker.is_down(runtime))
            |
            +-- HALT_STREAM          (provider DOWN + halt mode)
            +-- DISCARD_AND_REDO     (provider DOWN + fallback: fresh backup group)
            +-- TOP_UP               (additional_attempts > 0)
            +-- GROUP_READY          (enough accepted for GRPO / scoreboard)
            +-- SKIP_INSUFFICIENT    (max attempts, still short)

Provider DOWN comes from campaign ``ProviderHealthTracker`` (consecutive hard
fails → cooldown → second trip), not from error-message markers.

Invariant (non-negotiable)
--------------------------
Never mix GM models inside one GRPO group. On provider fallback, discard
the in-progress group entirely and redo the same world file as a **new** group
on the backup GM (new identity, start again at initial_attempts).

Main callers:
    ``stream_engine.collect_adaptive_world``; unit tests for the decision table.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from shellsmith.campaign.provider_outage import (
    PROVIDER_OUTAGE_MODES,
    resolve_backup_world,
)

# Fresh campaigns default to maximum_attempts=8. ``--retry`` may raise that by
# ``--retry-extra-attempts`` (default +4) up to this absolute ceiling.
ABSOLUTE_MAXIMUM_ATTEMPTS = 12
MLX_EVALUATION_MAXIMUM_ATTEMPTS = 24
DEFAULT_RETRY_EXTRA_ATTEMPTS = 4


class WorldAction(str, Enum):
    """What the stream engine should do next for this world file group."""

    TOP_UP = "top_up"
    GROUP_READY = "group_ready"
    SKIP_INSUFFICIENT = "skip_insufficient"
    DISCARD_AND_REDO_BACKUP = "discard_and_redo_backup"
    HALT_STREAM = "halt_stream"


@dataclass(frozen=True)
class AdaptivePolicy:
    """Small rollout top-up policy for one exact simulator/task group."""

    initial_attempts: int = 4
    maximum_attempts: int = 8
    minimum_trainable: int = 4
    minimum_reward_range: float = 0.15
    low_spread_top_up: int = 4
    attempt_ceiling: int = ABSOLUTE_MAXIMUM_ATTEMPTS

    def validate(self) -> None:
        if not 1 <= self.initial_attempts <= self.maximum_attempts:
            raise ValueError("initial_attempts must be within maximum_attempts")
        if self.attempt_ceiling not in (ABSOLUTE_MAXIMUM_ATTEMPTS, MLX_EVALUATION_MAXIMUM_ATTEMPTS):
            raise ValueError("Unsupported adaptive attempt ceiling")
        if self.maximum_attempts > self.attempt_ceiling:
            raise ValueError(
                "maximum_attempts cannot exceed "
                f"{self.attempt_ceiling}"
            )
        if not 1 <= self.minimum_trainable <= self.maximum_attempts:
            raise ValueError("minimum_trainable must be within maximum_attempts")
        if not 1 <= self.low_spread_top_up <= 4:
            raise ValueError("low_spread_top_up must be between one and four")
        if self.minimum_reward_range < 0:
            raise ValueError("minimum_reward_range must be non-negative")


def additional_attempts(task: Mapping[str, Any], policy: AdaptivePolicy) -> int:
    """Return a bounded 1-4 top-up based on trainability and reward variation.

    Preserve exactly:
        remaining = maximum_attempts - attempt_count
        need_trainable = max(0, minimum_trainable - accepted_count)
        spread_top_up when accepted >= min and reward_range < minimum_reward_range
        return min(4, remaining, max(need_trainable, spread_top_up))
    """
    policy.validate()
    attempts = int(task.get("attempt_count") or 0)
    accepted = int(task.get("accepted_count") or 0)
    remaining = max(0, policy.maximum_attempts - attempts)
    if remaining == 0:
        return 0
    needed_for_trainable = max(0, policy.minimum_trainable - accepted)
    reward = task.get("reward") if isinstance(task.get("reward"), Mapping) else {}
    values = reward.get("values") if isinstance(reward.get("values"), list) else []
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    reward_range = max(numeric) - min(numeric) if len(numeric) >= 2 else 0.0
    spread_top_up = (
        policy.low_spread_top_up
        if accepted >= policy.minimum_trainable
        and reward_range < policy.minimum_reward_range
        else 0
    )
    requested = max(needed_for_trainable, spread_top_up)
    return min(4, remaining, requested)


@dataclass(frozen=True)
class WorldDecision:
    """Result of evaluating one world's outcomes after a wave."""

    action: WorldAction
    top_up_attempts: int = 0
    reason: str = ""
    backup_runtime: str | None = None
    backup_model: str | None = None
    backup_provider: str | None = None


@dataclass(frozen=True)
class StreamPolicyConfig:
    """Top-level trainer knobs for adaptive stream + outage policy.

    Built from CLI flags:

        --initial-attempts / --maximum-attempts / --minimum-trainable
        --minimum-reward-range / --low-spread-top-up
        --provider-outage {halt,fallback}
        --parallelism / --parallelism-per-provider
        groups_per_update (hero slate or default 2)
    """

    initial_attempts: int = 4
    maximum_attempts: int = 8
    minimum_trainable: int = 4
    minimum_reward_range: float = 0.15
    low_spread_top_up: int = 4
    provider_outage: str = "halt"
    parallelism: int = 8
    parallelism_per_provider: int = 4
    groups_per_update: int = 2
    campaign_mode: str = "inference"  # inference | tinker_train
    attempt_ceiling: int = ABSOLUTE_MAXIMUM_ATTEMPTS

    def adaptive_policy(self) -> AdaptivePolicy:
        return AdaptivePolicy(
            initial_attempts=self.initial_attempts,
            maximum_attempts=self.maximum_attempts,
            minimum_trainable=self.minimum_trainable,
            minimum_reward_range=self.minimum_reward_range,
            low_spread_top_up=self.low_spread_top_up,
            attempt_ceiling=self.attempt_ceiling,
        )

    def validate(self) -> None:
        if self.attempt_ceiling != ABSOLUTE_MAXIMUM_ATTEMPTS and self.campaign_mode != "inference":
            raise ValueError("Only inference may use the larger evaluation attempt ceiling")
        self.adaptive_policy().validate()
        if self.provider_outage not in PROVIDER_OUTAGE_MODES:
            raise RuntimeError(
                f"Unsupported provider_outage {self.provider_outage!r}."
            )
        if self.campaign_mode not in ("inference", "tinker_train", "mlx_train", "studio_train"):
            raise RuntimeError(
                f"Unsupported campaign_mode {self.campaign_mode!r}."
            )
        if self.groups_per_update < 1:
            raise RuntimeError("groups_per_update must be at least one.")
        if self.parallelism < 1:
            raise RuntimeError("parallelism must be at least one.")
        if not 1 <= self.parallelism_per_provider <= self.parallelism:
            raise RuntimeError(
                "parallelism_per_provider must be between 1 and parallelism."
            )


def stream_config_from_namespace(args: Any) -> StreamPolicyConfig:
    """Map argparse trainer args into the typed stream config."""
    parallelism = int(getattr(args, "parallelism", 8))
    config = StreamPolicyConfig(
        initial_attempts=int(getattr(args, "initial_attempts", 4)),
        maximum_attempts=int(getattr(args, "maximum_attempts", 8)),
        minimum_trainable=int(getattr(args, "minimum_trainable", 4)),
        minimum_reward_range=float(getattr(args, "minimum_reward_range", 0.15)),
        low_spread_top_up=int(getattr(args, "low_spread_top_up", 4)),
        provider_outage=str(getattr(args, "provider_outage", "halt")),
        parallelism=parallelism,
        parallelism_per_provider=int(
            getattr(args, "parallelism_per_provider", min(4, parallelism))
        ),
        groups_per_update=int(getattr(args, "groups_per_update", 2)),
        campaign_mode=(
            "inference"
            if getattr(args, "simulation_only", False)
            else "tinker_train"
        ),
    )
    config.validate()
    return config


def task_stats_from_outcomes(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Count bankable accepts from ``accepted_count`` only.

    Why this exists:
        Adaptive GROUP_READY / top-up decisions must match CaptureStore reality.
        Audit ``accepted_for_training`` alone is not enough: Tinker capture build
        can fail after a green audit (record/block mismatch). Counting audit-only
        accepts previously marked worlds GROUP_READY and then crashed when the
        store had no capture.
    """
    accepted = [
        outcome for outcome in outcomes if outcome.get("accepted_count") == 1
    ]
    rewards: list[float] = []
    for outcome in accepted:
        audit = outcome.get("audit")
        if not isinstance(audit, dict):
            continue
        final_task = audit.get("final_task")
        if isinstance(final_task, dict) and isinstance(
            final_task.get("reward"), (int, float)
        ):
            rewards.append(float(final_task["reward"]))
    return {
        "attempt_count": len(outcomes),
        "accepted_count": len(accepted),
        "reward": {"values": rewards},
    }


def evaluate_world_outcomes(
    outcomes: list[dict[str, Any]],
    *,
    config: StreamPolicyConfig,
    gm_runtime: str,
    gm_model: str,
    gm_provider: str,
    gm_fallback_applied: bool,
    provider_down: bool = False,
) -> WorldDecision:
    """Apply the adaptive / outage decision table to one world's outcomes.

    Fallback rule:
        On provider DOWN + fallback mode, return DISCARD_AND_REDO_BACKUP. Caller
        must drop all captures/outcomes for this identity and start a fresh
        group — never append backup rollouts onto a mixed group.

    ``provider_down`` comes from the campaign ``ProviderHealthTracker`` (second
    consecutive-fail trip after cooldown), not from error-string markers.
    """
    config.validate()
    if provider_down:
        if config.provider_outage == "halt" or gm_fallback_applied:
            reason = (
                "provider marked down after backup GM already applied"
                if gm_fallback_applied
                else (
                    f"provider {gm_runtime!r} marked down after repeated "
                    "failure streaks under halt mode"
                )
            )
            return WorldDecision(action=WorldAction.HALT_STREAM, reason=reason)
        backup = resolve_backup_world(
            runtime=gm_runtime,
            model=gm_model,
            provider=gm_provider,
        )
        return WorldDecision(
            action=WorldAction.DISCARD_AND_REDO_BACKUP,
            reason=(
                f"provider {gm_runtime!r} marked down; discard group and redo "
                f"on backup {gm_model!r} -> {backup.model!r}"
            ),
            backup_runtime=backup.runtime,
            backup_model=backup.model,
            backup_provider=backup.provider,
        )

    stats = task_stats_from_outcomes(outcomes)
    policy = config.adaptive_policy()
    top_up = additional_attempts(stats, policy)
    if top_up > 0:
        return WorldDecision(
            action=WorldAction.TOP_UP,
            top_up_attempts=top_up,
            reason=f"adaptive top-up requests {top_up} more attempt(s)",
        )
    if int(stats["accepted_count"]) >= config.minimum_trainable:
        return WorldDecision(
            action=WorldAction.GROUP_READY,
            reason="accepted rollouts meet minimum_trainable",
        )
    return WorldDecision(
        action=WorldAction.SKIP_INSUFFICIENT,
        reason=(
            "no further top-ups available and accepted "
            f"{stats['accepted_count']} < minimum_trainable "
            f"{config.minimum_trainable}"
        ),
    )


@dataclass(frozen=True)
class InsufficientGroupsRetryPlan:
    """How ``--retry`` rewinds one failed ``insufficient_groups`` bank.

    Why this exists:
        Plain resume advances ``next_world_offset`` past the failed update and
        orphans GROUP_READY stragglers that were only waiting for a pair.
        Retry keeps those ready worlds, re-opens SKIP worlds under an elevated
        attempt cap, and pins Player policy_ref so top-ups do not mix models.
    """

    failed_step_index: int
    start_world_offset: int
    end_world_offset: int
    ready_task_indexes: tuple[int, ...]
    retry_task_indexes: tuple[int, ...]
    elevated_maximum_attempts: int


def elevate_maximum_attempts_for_retry(
    *,
    base_maximum_attempts: int,
    extra_attempts: int = DEFAULT_RETRY_EXTRA_ATTEMPTS,
) -> int:
    """Raise the adaptive attempt cap for ``--retry`` (default 8+4 → 12)."""
    if extra_attempts < 1:
        raise ValueError("retry extra_attempts must be at least one")
    if base_maximum_attempts < 1:
        raise ValueError("base_maximum_attempts must be at least one")
    elevated = base_maximum_attempts + extra_attempts
    if elevated > ABSOLUTE_MAXIMUM_ATTEMPTS:
        return ABSOLUTE_MAXIMUM_ATTEMPTS
    return elevated


def pinned_policy_from_outcomes(
    outcomes: list[Mapping[str, Any]],
) -> tuple[str, str | None, int]:
    """Return the single (policy_kind, policy_ref, rft) shared by a group.

    Raises if the group mixed Player policies — retry must not continue a mix.
    """
    if not outcomes:
        raise RuntimeError("Cannot pin Player policy from an empty outcome list.")
    kinds: set[str] = set()
    refs: set[str | None] = set()
    rfts: set[int] = set()
    for outcome in outcomes:
        kind = outcome.get("policy_kind")
        if kind is None:
            kind = "base"
        if not isinstance(kind, str) or not kind:
            raise RuntimeError(
                "Outcome is missing a usable policy_kind for retry pinning."
            )
        ref = outcome.get("policy_ref")
        if ref is not None and not isinstance(ref, str):
            raise RuntimeError(
                "Outcome policy_ref must be a string or null for retry pinning."
            )
        rft_raw = outcome.get("policy_rft_updates_completed")
        if rft_raw is None:
            rft = 0
        elif isinstance(rft_raw, int) and not isinstance(rft_raw, bool):
            rft = rft_raw
        else:
            raise RuntimeError(
                "Outcome policy_rft_updates_completed must be an int for retry."
            )
        kinds.add(kind)
        refs.add(ref)
        rfts.add(rft)
    if len(kinds) != 1 or len(refs) != 1 or len(rfts) != 1:
        raise RuntimeError(
            "Cannot retry a group with mixed Player policies "
            f"(kinds={sorted(kinds)}, refs={sorted(str(r) for r in refs)}, "
            f"rfts={sorted(rfts)})."
        )
    kind = next(iter(kinds))
    ref = next(iter(refs))
    if kind == "sampler_checkpoint" and not ref:
        raise RuntimeError(
            "sampler_checkpoint outcomes require a non-empty policy_ref to retry."
        )
    if kind == "base" and ref is not None:
        raise RuntimeError("base-policy outcomes must not carry a policy_ref.")
    return kind, ref, next(iter(rfts))


@dataclass(frozen=True)
class RetryWorldIdentity:
    """The exact GM identity one ``--retry`` world must continue on.

    Why this exists:
        A world that already fell back to the backup GM stores its
        rollouts under ``{task}__backup_{model}/rollout_NN`` and was sampled by
        the backup runtime/model/provider. ``--retry`` seeds those outcomes and
        schedules more attempts. Without this record the retry would re-derive
        the world identity from the CLI namespace (always the *primary*
        provider) and from ``world_file.stem`` (always the *primary* folder), so
        the top-ups would run on the wrong provider and land in the wrong
        directory — a GRPO group with two different GM models in it.

    Main callers:
        ``trainer._run_tinker_steps`` while assembling the retry maps.
    """

    rollout_world_id: str
    gm_runtime: str
    gm_model: str
    gm_provider: str
    gm_fallback_applied: bool
    original_gm_model: str | None


def retry_world_identity_from_task_row(
    task_row: Mapping[str, Any],
) -> RetryWorldIdentity:
    """Read back the settled GM identity of one recorded step task row.

    Steps, in order:
        1. read ``rollout_world_id`` (the folder that owns ``rollout_NN``);
        2. read the runtime / model / provider actually used for those rollouts;
        3. if the row says a fallback was applied, demand the primary model it
           was remapped away from, so the retry keeps ``gm_fallback_applied``
           truthful and cannot fall back a second time.

    Every missing field raises. A trainer_state row written before this field
    existed cannot be retried safely, and guessing the identity is exactly the
    silent mismatch this function exists to prevent.

    Main callers:
        ``trainer._run_tinker_steps`` (``--retry``).
    """
    rollout_world_id = task_row.get("rollout_world_id")
    if not isinstance(rollout_world_id, str) or not rollout_world_id:
        raise RuntimeError(
            "--retry task row is missing rollout_world_id; it was recorded "
            "before the fallback-aware retry contract and cannot be resumed "
            "without risking a mixed-provider group."
        )
    fields: dict[str, str] = {}
    for name in ("gm_runtime", "gm_model", "gm_provider"):
        value = task_row.get(name)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"--retry task row is missing {name}.")
        fields[name] = value
    fallback_applied = task_row.get("gm_fallback_applied")
    if not isinstance(fallback_applied, bool):
        raise RuntimeError(
            "--retry task row is missing a boolean gm_fallback_applied."
        )
    original_gm_model = task_row.get("original_gm_model")
    if original_gm_model is not None and not isinstance(original_gm_model, str):
        raise RuntimeError(
            "--retry task row original_gm_model must be a string or null."
        )
    if fallback_applied and not original_gm_model:
        raise RuntimeError(
            "--retry task row claims gm_fallback_applied but has no "
            "original_gm_model; the pre-fallback GM is unknown."
        )
    if not fallback_applied and original_gm_model:
        raise RuntimeError(
            "--retry task row carries original_gm_model without "
            "gm_fallback_applied; the fallback state is inconsistent."
        )
    return RetryWorldIdentity(
        rollout_world_id=rollout_world_id,
        gm_runtime=fields["gm_runtime"],
        gm_model=fields["gm_model"],
        gm_provider=fields["gm_provider"],
        gm_fallback_applied=fallback_applied,
        original_gm_model=original_gm_model,
    )


def build_insufficient_groups_retry_plan(
    state: Mapping[str, Any],
    *,
    base_maximum_attempts: int,
    extra_attempts: int = DEFAULT_RETRY_EXTRA_ATTEMPTS,
) -> InsufficientGroupsRetryPlan:
    """Inspect trainer_state and describe how to retry the failed bank.

    Main callers:
        ``run_tinker`` when ``--retry`` is set.
    """
    if state.get("status") != "insufficient_groups":
        raise RuntimeError(
            "--retry requires trainer_state status insufficient_groups; "
            f"got {state.get('status')!r}."
        )
    steps = state.get("steps")
    if not isinstance(steps, list) or not steps:
        raise RuntimeError(
            "--retry found insufficient_groups status but no steps to rewind."
        )
    failed_step_index = len(steps) - 1
    failed = steps[failed_step_index]
    if not isinstance(failed, dict):
        raise RuntimeError("--retry failed step must be an object.")
    if failed.get("skip_reason") != "insufficient_remaining_complete_groups":
        raise RuntimeError(
            "--retry last step skip_reason must be "
            "insufficient_remaining_complete_groups; "
            f"got {failed.get('skip_reason')!r}."
        )
    tasks = failed.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError("--retry failed step has no tasks to rewind.")

    end_offset_raw = failed.get("next_world_offset")
    if end_offset_raw is None:
        end_offset_raw = state.get("next_world_offset")
    if not isinstance(end_offset_raw, int) or isinstance(end_offset_raw, bool):
        raise RuntimeError(
            "--retry could not resolve next_world_offset for the failed bank."
        )
    end_offset = int(end_offset_raw)

    start_raw = failed.get("start_world_offset")
    if start_raw is None:
        start_offset = end_offset - len(tasks)
    elif isinstance(start_raw, int) and not isinstance(start_raw, bool):
        start_offset = int(start_raw)
    else:
        raise RuntimeError("--retry start_world_offset must be an int when present.")
    if start_offset < 0:
        raise RuntimeError("--retry reconstructed a negative start_world_offset.")
    if end_offset - start_offset != len(tasks):
        raise RuntimeError(
            "--retry bank width does not match failed step task count: "
            f"start={start_offset}, end={end_offset}, tasks={len(tasks)}."
        )

    ready: list[int] = []
    retry: list[int] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise RuntimeError(f"--retry task row {index} must be an object.")
        decision = task.get("decision")
        if decision == WorldAction.GROUP_READY.value:
            ready.append(index)
        elif decision == WorldAction.SKIP_INSUFFICIENT.value:
            retry.append(index)
            # Fail fast if this skip already mixed policies.
            pinned_policy_from_outcomes(list(task.get("attempts") or []))
            # Fail fast if we cannot tell which GM owns these rollouts.
            retry_world_identity_from_task_row(task)
        else:
            raise RuntimeError(
                "--retry only supports replaying group_ready + skip_insufficient "
                f"rows; task {index} has decision {decision!r}."
            )
    if not retry:
        raise RuntimeError(
            "--retry found no skip_insufficient worlds in the failed bank."
        )

    return InsufficientGroupsRetryPlan(
        failed_step_index=failed_step_index,
        start_world_offset=start_offset,
        end_world_offset=end_offset,
        ready_task_indexes=tuple(ready),
        retry_task_indexes=tuple(retry),
        elevated_maximum_attempts=elevate_maximum_attempts_for_retry(
            base_maximum_attempts=base_maximum_attempts,
            extra_attempts=extra_attempts,
        ),
    )
