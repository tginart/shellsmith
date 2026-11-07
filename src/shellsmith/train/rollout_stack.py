"""Shared, preflighted rollout machinery for training and inference campaigns.

Selection, merged policies, canonical world/GM identities, filesystem names,
Pi/Codex bindings and the Player token budget are checked before campaign I/O.
Callers preparing a sampler before authentication pass it through unchanged.
The endpoint credential is handed privately to the runner, never to os.environ.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from shellsmith.campaign.draft_catalog import classify
from shellsmith.episode.gm_process.python_runtime import current_python_runtime
from shellsmith.campaign.provider_outage import ProviderHealthTracker
from shellsmith.campaign.rollout_campaign import RolloutRunner, outcome_is_settled
from shellsmith.campaign.stream_campaign import (
    CampaignHandle,
    OpenCampaignSpec,
    finalize_campaign,
    open_campaign,
)
from shellsmith.policy.sampler import (
    PlayerSamplingConfig,
    TinkerPlayerSampler,
)
from shellsmith.policy.server import PlayerEndpoint
from shellsmith.train.capture_store import CaptureStore
from shellsmith.train.config import (
    TrainConfig,
    merged_stream_policy,
    validate_train_config,
    validate_training_gm,
)
from shellsmith.train.durable_capture import resolve_policy_stamp
from shellsmith.train.harness_identity import (
    codex_identity as resolve_codex_identity,
    pi_sdk_identity,
)
from shellsmith.train.name_budget import (
    validate_component_name,
    validate_training_names,
)
from shellsmith.train.runner import make_tinker_rollout_runner
from shellsmith.train.stream_engine import WorldCollectResult, collect_adaptive_world
from shellsmith.train.stream_policy import StreamPolicyConfig
from shellsmith.train.world_selection import (
    FrozenWorld,
    SelectedWorld,
    WorldStream,
    freeze_worlds,
    select_world_stream,
)
from shellsmith.train.backend import PolicySampler, player_profile, profile_hash


def episode_args(
    config: TrainConfig, world: SelectedWorld | None = None
) -> argparse.Namespace:
    """Episode settings, preserving the world's canonical identity and GM."""
    adaptive = config.adaptive
    args = argparse.Namespace(
        world_id=world.world_id if world else None,
        gm_runtime=world.gm_runtime if world else config.gm.runtime,
        gm_model=world.gm_model if world else config.gm.model,
        gm_provider=world.gm_provider if world else config.gm.provider,
        gm_reasoning_effort=world.gm_reasoning_effort
        if world
        else config.gm.reasoning_effort,
        gm_api_key_env=config.gm.api_key_env,
        max_world_steps=config.gm.max_world_steps,
        timeout_seconds=config.gm.timeout_seconds,
        gm_api_idle_timeout_seconds=config.gm.api_idle_timeout_seconds,
        player_prompt_timeout_seconds=config.player.prompt_timeout_seconds,
        protocol_retry=asdict(config.retry),
        player_thinking="high",
        player_max_tokens=config.player.max_tokens,
        parallelism=adaptive.parallelism,
        initial_attempts=adaptive.initial_attempts,
        maximum_attempts=adaptive.maximum_attempts,
        minimum_trainable=adaptive.minimum_trainable,
        minimum_reward_range=adaptive.minimum_reward_range,
        low_spread_top_up=adaptive.low_spread_top_up,
    )
    if config.compute.backend in {"mlx", "studio"}:
        profile = player_profile(config)
        args.player_thinking = profile["reasoning_effort"]
        args.player_reasoning_tokens = profile.get("resolved_reasoning_tokens", profile["thinking_tokens"])
        args.player_response_tokens = profile["response_tokens"]
        args.player_max_tokens = profile["max_tokens"]
        args.player_context_window = profile["context_window"]
        args.player_compaction_reserve_tokens = profile["compaction_reserve_tokens"]
        args.player_compaction_keep_recent_tokens = profile[
            "compaction_keep_recent_tokens"
        ]
    if world is not None:
        vars(args).update(config.world_overrides.get(world.world_id, {}))
    return args


def world_task_runs(
    config: TrainConfig,
    worlds: Iterable[SelectedWorld | Path],
    *,
    campaign_mode: str = "tinker_train",
) -> list[tuple[Path, argparse.Namespace]]:
    """Validate and pair selected worlds with merged, per-GM episode settings.

    Overrides are keyed by parsed contract ID, not filename stem. Plain paths
    inherit the global GM; SelectedWorld objects retain hero GM assignments.
    All policies are validated locally before a caller authenticates or writes.
    The stack/trainer budget composed names using the restored campaign ID.
    """
    validate_train_config(config, campaign_mode=campaign_mode)
    gm = config.gm
    selected = tuple(
        world
        if isinstance(world, SelectedWorld)
        else SelectedWorld(
            world_file=Path(world),
            gm_runtime=gm.runtime,
            gm_model=gm.model,
            gm_provider=gm.provider,
            gm_reasoning_effort=gm.reasoning_effort,
        )
        for world in worlds
    )
    WorldStream(
        worlds=selected, provenance={}
    )  # IDs, collisions and reserved campaign paths.
    unknown = sorted(
        set(config.world_overrides) - {world.world_id for world in selected}
    )
    if unknown:
        raise ValueError(
            f"[world_overrides] names worlds not in this run: {', '.join(unknown)}"
        )
    for world in selected:
        validate_component_name(world.world_id, label="Selected world ID")
        if world.gm_runtime != "codex":
            raise RuntimeError(
                "Tinker Player credentials require the confined Codex GM for every selected world; "
                f"{world.world_file} selects {world.gm_runtime!r}."
            )
        validate_training_gm(
            runtime=world.gm_runtime,
            model=world.gm_model,
            provider=world.gm_provider,
            reasoning_effort=world.gm_reasoning_effort,
            label=f"Selected world {world.world_file.name} GM",
        )
        merged_stream_policy(
            config,
            config.world_overrides.get(world.world_id),
            campaign_mode=campaign_mode,
            label=f"[world_overrides.{world.world_id}]",
        )
    return [(world.world_file, episode_args(config, world)) for world in selected]


def stream_config(
    config: TrainConfig, *, campaign_mode: str, groups_per_update: int
) -> StreamPolicyConfig:
    """Only explicit stream fields; outage circuit thresholds live on the handle."""
    return merged_stream_policy(
        config, campaign_mode=campaign_mode, groups_per_update=groups_per_update
    )


def summarize_world(result: WorldCollectResult) -> dict[str, Any]:
    """
    Purpose: Count completed episodes for trainer metrics and campaign reports.
    Input: The collected world result, which can contain slots that never started.
    Output: One summary row that excludes those slots from episode counts and rewards.
    """
    outcomes = [row for row in result.outcomes if outcome_is_settled(row)]
    discarded = [row for row in result.discarded_outcomes if outcome_is_settled(row)]
    return {
        "world_id": result.world_id,
        "decision": result.decision.action.value,
        "reason": result.decision.reason,
        "gm_model": result.gm_model,
        "gm_fallback_applied": result.gm_fallback_applied,
        "original_gm_model": result.original_gm_model,
        "attempts": len(outcomes),
        "attempted_total": len(outcomes) + len(discarded),
        "accepted": sum(
            1 for outcome in outcomes if outcome.get("accepted_count") == 1
        ),
        "discarded": len(discarded),
        "failed": sum(bool(outcome.get("error")) for outcome in outcomes),
        "rejected": sum(not outcome.get("accepted_count") and not outcome.get("error")
                        for outcome in outcomes),
        "rewards": [
            ((outcome.get("audit") or {}).get("final_task") or {}).get("reward")
            for outcome in outcomes
        ],
    }


@dataclass
class RolloutStack:
    """One open campaign plus the Player endpoint and runner feeding it."""

    handle: CampaignHandle
    capture_store: CaptureStore
    endpoint: PlayerEndpoint
    args: argparse.Namespace
    stream: StreamPolicyConfig
    group_identities: dict[str, Any]
    runner: RolloutRunner
    world_inputs: dict[Path, FrozenWorld]
    staged_inputs: dict[Path, Path]

    def classify_world(self, world_file: Path) -> str:
        frozen = self.world_inputs[world_file]
        return classify(frozen.spec, path=frozen.source_path)

    async def collect_one(
        self,
        world_file: Path,
        task_args: argparse.Namespace,
        world_config: StreamPolicyConfig,
        flavor: str,
    ) -> WorldCollectResult:
        if world_file not in self.staged_inputs:
            # Stage only admitted worlds: sparse-resume siblings and worlds past
            # the target must remain unstarted, with no speculative output paths.
            self.staged_inputs[world_file] = self.world_inputs[world_file].stage(
                self.handle.root
            )
        return await collect_adaptive_world(
            handle=self.handle,
            args=task_args,
            config=world_config,
            world_file=self.staged_inputs[world_file],
            world_id=task_args.world_id,
            flavor=flavor,
            gm_runtime=task_args.gm_runtime,
            gm_model=task_args.gm_model,
            gm_provider=task_args.gm_provider,
            gm_reasoning_effort=task_args.gm_reasoning_effort,
            rollout_runner=self.runner,
            capture_store=self.capture_store,
        )

    def close(self, *, status: str, halt: dict[str, Any] | None) -> None:
        try:
            finalize_campaign(self.handle, status=status, provider_outage=halt)
        finally:
            self.endpoint.stop()


def open_rollout_stack(
    config: TrainConfig,
    *,
    deps: Any,
    sampling_client: Any,
    root: Path,
    campaign_id: str,
    campaign_mode: str,
    groups_per_update: int,
    state: dict[str, Any],
    pinned_sampler_path: str | None = None,
    prepared_sampler: PolicySampler | None = None,
    pi_sdk_module_path: Path | None = None,
    codex_identity: dict[str, Any] | None = None,
    task_runs: list[tuple[Path, argparse.Namespace]] | None = None,
    frozen_worlds: tuple[FrozenWorld, ...] | None = None,
) -> RolloutStack:
    """Preflight before output, then open a campaign and privately bound endpoint.

    Trainer/inference callers pass the sampler they validated before SDK auth,
    and the exact Pi/Codex bindings whose provenance they already captured.
    Standalone callers get the same local preflight before any campaign output.
    """
    if frozen_worlds is None:
        selected = (
            select_world_stream(config, campaign_mode=campaign_mode).worlds
            if task_runs is None
            else tuple(
                SelectedWorld(
                    world_file=path,
                    gm_runtime=args.gm_runtime,
                    gm_model=args.gm_model,
                    gm_provider=args.gm_provider,
                    gm_reasoning_effort=args.gm_reasoning_effort,
                )
                for path, args in task_runs
            )
        )
        frozen_worlds = freeze_worlds(selected)
    selected = tuple(frozen.selected for frozen in frozen_worlds)
    if task_runs is not None and [path for path, _ in task_runs] != [
        world.world_file for world in selected
    ]:
        raise ValueError(
            "Frozen world inputs must match the selected task queue in order."
        )
    world_task_runs(config, selected, campaign_mode=campaign_mode)
    stream = stream_config(
        config, campaign_mode=campaign_mode, groups_per_update=groups_per_update
    )
    validate_training_names(
        config=config, worlds=selected, campaign_id=campaign_id, segment_root=root
    )
    for frozen in frozen_worlds:
        frozen.validate_names(root)
    if pi_sdk_module_path is None:
        sdk = pi_sdk_identity(config.player.pi_sdk_module_path)
        if sdk["module_sha256"] is None:
            raise RuntimeError(
                "The pinned Pi SDK is not installed; configure player.pi_sdk_module_path."
            )
        pi_sdk_module_path = Path(sdk["module_path"])
    if codex_identity is None:
        codex_identity = resolve_codex_identity()
    if codex_identity["executable_sha256"] is None:
        raise RuntimeError(
            "Codex executable not found on PATH; install Codex before starting a campaign."
        )
    gm_python_runtime = (
        current_python_runtime() if config.compute.backend in {"mlx", "studio"} else None
    )
    player = config.player
    profile = player_profile(config)
    sampler = prepared_sampler
    if sampler is None:
        if config.compute.backend in {"mlx", "studio"}:
            raise RuntimeError("This rollout stack requires a prepared policy sampler.")
        sampler = TinkerPlayerSampler(
            config=PlayerSamplingConfig(
                base_model=player.base_model,
                thinking_renderer=player.thinking_renderer,
                plain_renderer=player.plain_renderer,
                temperature=player.temperature,
                top_p=player.top_p,
                max_tokens=player.max_tokens,
                thinking_anomaly_attempts=config.retry.thinking_anomaly_attempts,
            ),
            deps=deps,
            sampling_client=None,
        )
        sampler.validate_thinking_budget()
    if config.compute.backend == "tinker":
        sampler.set_sampling_client(sampling_client)
    endpoint = PlayerEndpoint(sampler=sampler, records_dir=root / "player_records")
    handle = open_campaign(
        root,
        OpenCampaignSpec(
            campaign_id=campaign_id,
            mode=campaign_mode,
            parallelism=stream.parallelism,
            parallelism_per_provider=stream.parallelism_per_provider,
            provider_outage=stream.provider_outage,
            timeout_seconds=config.gm.timeout_seconds,
        ),
    )
    handle.health = ProviderHealthTracker(
        consecutive_failure_threshold=config.adaptive.outage_failure_threshold,
        cooldown_seconds=config.adaptive.outage_cooldown_seconds,
    )
    capture_store = CaptureStore(_run_dir=root)
    state.setdefault("group_identities", {})
    # Exact bytes remain in memory until admission. collect_one stages them
    # only after startup/segment publication, preserving snapshot-only retries.
    world_inputs = {frozen.selected.world_file: frozen for frozen in frozen_worlds}
    by_staged_path = {
        frozen.launch_path(handle.root): frozen for frozen in frozen_worlds
    }

    def identity_for(job: Any) -> Any:
        frozen = by_staged_path[job.world_file]
        if job.world_file.read_bytes() != frozen.contents:
            raise RuntimeError(f"Frozen world input was modified: {job.world_file}.")
        identity = frozen.group_identity(
            gm_runtime=job.gm_runtime,
            gm_model=job.gm_model,
            gm_provider=job.gm_provider,
            gm_reasoning_effort=job.gm_reasoning_effort,
            max_world_steps=config.gm.max_world_steps,
            player_model=profile["base_model"],
            renderer_name=profile["thinking_renderer"],
            max_tokens=profile["max_tokens"],
            temperature=player.temperature,
            top_p=player.top_p,
            **(
                {
                    "backend": config.compute.backend,
                    "profile_sha256": profile_hash(profile),
                }
                if config.compute.backend in {"mlx", "studio"}
                else {}
            ),
        )
        state["group_identities"][identity.key] = identity.to_jsonable()
        return identity

    def policy_stamp(episode_id: str, group_key: str) -> Any:
        if config.compute.backend == "tinker" and hasattr(sampler, "pin"):
            sampler.pin(episode_id)
        pinned = (
            {
                "pinned_policy_kind": "sampler_checkpoint",
                "pinned_policy_ref": pinned_sampler_path,
            }
            if pinned_sampler_path
            else {}
        )
        provenance = resolve_policy_stamp(
            state=state,
            model_name=profile["base_model"],
            lora_rank=profile["lora_rank"],
            episode_id=episode_id,
            group_identity_key=group_key,
            **pinned,
        )
        if config.compute.backend in {"mlx", "studio"}:
            policy_ref = (
                sampler.pin(episode_id)
                if config.compute.backend == "studio"
                else state.get("active_policy_ref")
            )
            provenance = replace(
                provenance,
                backend=config.compute.backend,
                policy_ref=policy_ref,
                policy_identity=state.get("policy_identity", {}),
            )
        return provenance

    runner = make_tinker_rollout_runner(
        endpoint=endpoint,
        capture_store=capture_store,
        run_dir=root,
        identity_for=identity_for,
        policy_stamp=policy_stamp,
        player_renderer=profile["thinking_renderer"],
        player_context_window=profile["context_window"],
        pi_sdk_module_path=pi_sdk_module_path,
        codex_executable=Path(codex_identity["executable_path"]),
        gm_python_runtime=gm_python_runtime,
        prompt_inputs_for=lambda job: by_staged_path[job.world_file].prompt_inputs,
    )
    stack = RolloutStack(
        handle=handle,
        capture_store=capture_store,
        endpoint=endpoint,
        args=episode_args(config),
        stream=stream,
        group_identities=state["group_identities"],
        runner=runner,
        world_inputs=world_inputs,
        staged_inputs={},
    )
    try:
        endpoint.start()
    except BaseException:
        stack.close(status="failed", halt=None)
        raise
    return stack
