"""Run one world file through the adaptive lifecycle inside an open campaign.

Architecture map
----------------
    collect_adaptive_world(...)
        schedule initial_attempts jobs (wave)
        evaluate_world_outcomes
            TOP_UP -> more waves (same identity)
            GROUP_READY -> return captures/outcomes
            SKIP_INSUFFICIENT -> return skip
            DISCARD_AND_REDO_BACKUP -> drop outcomes/captures; fresh group on backup
            HALT_STREAM -> propagate halt

    collect_update_worlds(...)
        within one sequential update boundary:
            admit up to groups_per_update in-flight worlds from the stream
            each world runs collect_adaptive_world (waves share DualCapPool)
            bank GROUP_READY; skip/fallback continue; HALT cancels peers

Never mixes GM models in one GRPO group: fallback discards the group.

Main callers:
    Trainer ``run_tinker`` collection loop; inference stream helpers; tests.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shellsmith.campaign.rollout_campaign import (
    RolloutRunner,
    run_live_job,
)
from shellsmith.campaign.stream_campaign import (
    CampaignHandle,
    build_task_attempt_jobs,
    discard_campaign_outcomes,
    run_campaign_wave,
)
from shellsmith.train.stream_policy import (
    StreamPolicyConfig,
    WorldAction,
    WorldDecision,
    evaluate_world_outcomes,
)
from shellsmith.train.capture_store import (
    CaptureStore,
)


@dataclass
class WorldCollectResult:
    """Settled result for one world file group inside a campaign stream.

    ``world_id`` vs ``rollout_world_id`` (the stage boundary that matters):
        ``world_id`` is the *stream* identity — the slate task this world
        belongs to. It stays stable across a provider fallback so the campaign
        scoreboard and the stream cursor keep talking about the same slate row.
        ``rollout_world_id`` is the *storage* identity — the id that actually
        owns ``{campaign_root}/{id}/rollout_NN`` directories and episode ids.
        After ``DISCARD_AND_REDO_BACKUP`` those two differ: the redo writes under
        ``{task}__backup_{model}``. Anything that resumes or tops up this group
        later (``--retry``) must use ``rollout_world_id``, never
        ``world_id``, or it will append backup-provider outcomes to
        primary-provider directories and mix GM models in one GRPO group.
    """

    world_id: str
    decision: WorldDecision
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    gm_runtime: str = ""
    gm_model: str = ""
    gm_provider: str = ""
    gm_fallback_applied: bool = False
    discarded_outcomes: list[dict[str, Any]] = field(default_factory=list)
    rollout_world_id: str = ""
    original_gm_model: str | None = None


@dataclass(frozen=True)
class UpdateWorldResult:
    """One settled world inside a multi-world update, pinned to stream order."""

    stream_offset: int
    world_file: Path
    task_args: argparse.Namespace
    result: WorldCollectResult


@dataclass
class UpdateCollectResult:
    """World results and the cursor to commit with their optimizer update.

    ``next_world_offset`` is the first unfinished world, not the admission
    high-water mark. ``settled_world_offsets`` preserves completed worlds beyond
    that gap so resuming does not train them twice. Persist both fields with the
    matching optimizer checkpoint; HALT and cancelled worlds remain unfinished.
    """

    worlds: list[UpdateWorldResult]
    next_world_offset: int
    halted: bool = False
    settled_world_offsets: list[int] = field(default_factory=list)


WorldCollector = Callable[
    [Path, argparse.Namespace, StreamPolicyConfig, str],
    Awaitable[WorldCollectResult],
]


async def collect_adaptive_world(
    *,
    handle: CampaignHandle,
    args: argparse.Namespace,
    config: StreamPolicyConfig,
    world_file: Path,
    world_id: str,
    flavor: str,
    gm_runtime: str,
    gm_model: str,
    gm_provider: str,
    gm_reasoning_effort: str,
    gm_fallback_applied: bool = False,
    original_gm_model: str | None = None,
    rollout_runner: RolloutRunner = run_live_job,
    capture_store: CaptureStore | None = None,
    seed_outcomes: list[dict[str, Any]] | None = None,
) -> WorldCollectResult:
    """Collect initial + top-up waves for one world; maybe discard-and-redo once.

    On DISCARD_AND_REDO_BACKUP this function recursively collects a fresh group
    on the backup GM and returns that result (discarded outcomes attached).
    Captures for discarded episodes are removed from ``capture_store`` when set.

    ``seed_outcomes``:
        When set (``--retry`` path), skip the initial wave and continue top-ups
        from attempt ``len(seed_outcomes) + 1`` under the elevated max. Durable
        replay still applies to any newly scheduled attempt dirs.
    """
    config.validate()
    runtime = gm_runtime
    model = gm_model
    provider = gm_provider
    fallback_applied = gm_fallback_applied
    original_model = original_gm_model

    if seed_outcomes is None:
        attempt_start = 1
        world_outcomes: list[dict[str, Any]] = []
        initial_jobs = build_task_attempt_jobs(
            campaign_id=handle.campaign_id,
            campaign_root=handle.root,
            world_file=world_file,
            world_id=world_id,
            flavor=flavor,
            attempt_start=attempt_start,
            attempt_count=config.initial_attempts,
            gm_runtime=runtime,
            gm_model=model,
            gm_provider=provider,
            gm_reasoning_effort=gm_reasoning_effort,
            gm_fallback_applied=fallback_applied,
            original_gm_model=original_model,
        )
        initial_outcomes = await run_campaign_wave(
            handle, initial_jobs, args=args, rollout_runner=rollout_runner
        )
        world_outcomes.extend(initial_outcomes)
        attempt_start += config.initial_attempts
    else:
        world_outcomes = list(seed_outcomes)
        attempt_start = len(world_outcomes) + 1
        if attempt_start < 1:
            raise RuntimeError("seed_outcomes produced an invalid attempt_start.")

    while True:
        decision = evaluate_world_outcomes(
            world_outcomes,
            config=config,
            gm_runtime=runtime,
            gm_model=model,
            gm_provider=provider,
            gm_fallback_applied=fallback_applied,
            provider_down=handle.health.is_down(runtime),
        )
        if decision.action == WorldAction.TOP_UP:
            top_up_jobs = build_task_attempt_jobs(
                campaign_id=handle.campaign_id,
                campaign_root=handle.root,
                world_file=world_file,
                world_id=world_id,
                flavor=flavor,
                attempt_start=attempt_start,
                attempt_count=decision.top_up_attempts,
                gm_runtime=runtime,
                gm_model=model,
                gm_provider=provider,
                gm_reasoning_effort=gm_reasoning_effort,
                gm_fallback_applied=fallback_applied,
                original_gm_model=original_model,
            )
            top_up_outcomes = await run_campaign_wave(
                handle, top_up_jobs, args=args, rollout_runner=rollout_runner
            )
            world_outcomes.extend(top_up_outcomes)
            attempt_start += decision.top_up_attempts
            continue

        if decision.action == WorldAction.DISCARD_AND_REDO_BACKUP:
            discarded = list(world_outcomes)
            if capture_store is not None:
                capture_store.discard_episodes(
                    [str(item["episode_id"]) for item in discarded]
                )
            # Dropping the captures is not enough: the durable outcomes and the
            # campaign scoreboard still claim these episodes are trainable. A
            # resume would replay that claim, bank a group whose captures no
            # longer exist, and wedge on "Accepted outcome ... has no capture".
            discard_campaign_outcomes(
                handle,
                discarded,
                reason=decision.reason or "discarded for GM fallback",
            )
            backup_task_id = f"{world_id}__backup_{decision.backup_model}"
            redo = await collect_adaptive_world(
                handle=handle,
                args=args,
                config=config,
                world_file=world_file,
                world_id=backup_task_id,
                flavor=flavor,
                gm_runtime=str(decision.backup_runtime),
                gm_model=str(decision.backup_model),
                gm_provider=str(decision.backup_provider),
                gm_reasoning_effort=gm_reasoning_effort,
                gm_fallback_applied=True,
                original_gm_model=model,
                rollout_runner=rollout_runner,
                capture_store=capture_store,
            )
            redo.discarded_outcomes = discarded + list(redo.discarded_outcomes)
            # Restore the *stream* identity only. ``rollout_world_id`` keeps
            # pointing at ``backup_task_id`` so a later --retry tops up inside
            # the backup folder on the backup provider.
            redo.world_id = world_id
            return redo

        return WorldCollectResult(
            world_id=world_id,
            decision=decision,
            outcomes=world_outcomes,
            gm_runtime=runtime,
            gm_model=model,
            gm_provider=provider,
            gm_fallback_applied=fallback_applied,
            rollout_world_id=world_id,
            original_gm_model=original_model,
        )


async def collect_update_worlds(
    *,
    task_runs: list[tuple[Path, argparse.Namespace]],
    start_offset: int,
    groups_per_update: int,
    base_config: StreamPolicyConfig,
    classify_flavor: Callable[[Path], str],
    collect_one: WorldCollector,
    preloaded_worlds: list[UpdateWorldResult] | None = None,
    priority_offsets: set[int] | None = None,
    settled_world_offsets: set[int] | None = None,
    on_world_completed: Callable[[UpdateWorldResult], None] | None = None,
) -> UpdateCollectResult:
    """Greedily collect worlds for one update under shared DualCapPool waves.

    Main callers:
        ``run_tinker`` inside each sequential RFT update boundary.

    Why this exists:
        Updates stay sequential (bank N groups then optim), but within an update
        several worlds may run at once. Episode concurrency is enforced by the
        campaign DualCapPool inside each world's waves.

    ``preloaded_worlds`` / ``priority_offsets`` (``--retry``):
        Keep already GROUP_READY stragglers banked toward ``groups_per_update``,
        re-collect incomplete (skip) offsets first, then admit new stream worlds
        only if the bank is still short.

    ``settled_world_offsets`` (durable resume):
        Worlds beyond ``start_offset`` already consumed by a committed update.
        Unlike preloads, these are neither returned nor counted toward the new
        update's ready groups. Pass the sparse offsets from the same checkpoint
        as ``start_offset``; persist the returned pair only with its update.

    ``on_world_completed``:
        Synchronously observe each newly banked, actual result (not preloads).
        Inference retains these even if a sibling or the enclosing window is
        interrupted before this function returns. All pending tasks are drained
        before propagating an interruption; completed siblings are still banked.
        This notification alone does not commit a training cursor/checkpoint.
    """
    if groups_per_update < 1:
        raise RuntimeError("groups_per_update must be at least one.")
    if start_offset < 0:
        raise RuntimeError("start_offset must be non-negative.")

    pending: dict[
        asyncio.Task[WorldCollectResult], tuple[int, Path, argparse.Namespace]
    ] = {}
    completed: list[UpdateWorldResult] = []
    settled_offsets = set(settled_world_offsets or ())
    for stream_offset in settled_offsets:
        if type(stream_offset) is not int or not 0 <= stream_offset < len(task_runs):
            raise RuntimeError("settled_world_offsets must contain valid stream offsets.")
    if settled_offsets.intersection(priority_offsets or ()):
        raise RuntimeError("priority_offsets cannot retry already settled worlds.")
    occupied_offsets = set(settled_offsets)
    for item in preloaded_worlds or []:
        if item.stream_offset in settled_offsets:
            raise RuntimeError("preloaded_worlds cannot include already settled worlds.")
        if item.stream_offset in occupied_offsets:
            raise RuntimeError(
                "preloaded_worlds contains a duplicate stream_offset "
                f"{item.stream_offset}."
            )
        completed.append(item)
        occupied_offsets.add(item.stream_offset)

    offset = start_offset
    ready_count = sum(
        1
        for item in completed
        if item.result.decision.action == WorldAction.GROUP_READY
    )
    halted = False
    priority_queue = sorted(
        offset_value
        for offset_value in (priority_offsets or set())
        if offset_value not in occupied_offsets
    )

    def _slots_open() -> bool:
        return (
            not halted
            and ready_count < groups_per_update
            and (ready_count + len(pending)) < groups_per_update
            and (
                bool(priority_queue)
                or offset < len(task_runs)
            )
        )

    def _admit_one(stream_offset: int) -> None:
        world_file, task_args = task_runs[stream_offset]
        flavor = classify_flavor(world_file)
        world_config = StreamPolicyConfig(
            initial_attempts=int(task_args.initial_attempts),
            maximum_attempts=int(task_args.maximum_attempts),
            minimum_trainable=int(task_args.minimum_trainable),
            minimum_reward_range=float(task_args.minimum_reward_range),
            low_spread_top_up=int(task_args.low_spread_top_up),
            provider_outage=base_config.provider_outage,
            parallelism=base_config.parallelism,
            parallelism_per_provider=base_config.parallelism_per_provider,
            groups_per_update=base_config.groups_per_update,
            campaign_mode=base_config.campaign_mode,
            attempt_ceiling=base_config.attempt_ceiling,
        )
        task = asyncio.create_task(
            collect_one(world_file, task_args, world_config, flavor)
        )
        pending[task] = (stream_offset, world_file, task_args)
        occupied_offsets.add(stream_offset)

    def _admit() -> None:
        nonlocal offset
        while _slots_open():
            if priority_queue:
                stream_offset = priority_queue.pop(0)
                if stream_offset in {meta[0] for meta in pending.values()}:
                    continue
                if stream_offset in {
                    item.stream_offset for item in completed
                }:
                    continue
                _admit_one(stream_offset)
                continue
            while offset < len(task_runs) and offset in occupied_offsets:
                offset += 1
            if offset >= len(task_runs):
                break
            stream_offset = offset
            offset += 1
            _admit_one(stream_offset)

    def _bank(
        meta: tuple[int, Path, argparse.Namespace],
        result: WorldCollectResult,
    ) -> None:
        stream_offset, world_file, task_args = meta
        item = UpdateWorldResult(
            stream_offset=stream_offset,
            world_file=world_file,
            task_args=task_args,
            result=result,
        )
        completed.append(item)
        if on_world_completed is not None:
            on_world_completed(item)

    async def _settle_pending_after_stop(*, propagate_errors: bool = True) -> None:
        """Drain peers on HALT or interruption, preserving their real results.

        A stopped window does not invalidate siblings that already completed,
        including tasks in the same wait batch as a collector failure. Bank all
        successes before raising any error. During exception cleanup preserve
        the original failure rather than replacing it with a sibling's error.
        """
        tasks = list(pending)
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors: list[BaseException] = []
        for task, result in zip(tasks, results, strict=True):
            meta = pending.pop(task)
            if isinstance(result, asyncio.CancelledError):
                continue
            if isinstance(result, BaseException):
                errors.append(result)
                continue
            try:
                _bank(meta, result)
            except BaseException as error:
                errors.append(error)
        if errors and propagate_errors:
            raise errors[0]

    try:
        while True:
            _admit()
            if not pending:
                break
            done, _ = await asyncio.wait(
                set(pending.keys()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Drain the whole batch before reacting to a halt: every task in
            # `done` has already produced a real result.
            for task in done:
                meta = pending.pop(task)
                try:
                    result = await task
                except asyncio.CancelledError:
                    continue
                _bank(meta, result)
                if result.decision.action == WorldAction.HALT_STREAM:
                    halted = True
                elif result.decision.action == WorldAction.GROUP_READY:
                    ready_count += 1
            if halted:
                await _settle_pending_after_stop()
                break
            if ready_count >= groups_per_update:
                # Prefer finish-in-flight; do not admit more worlds.
                if pending:
                    remaining_done, _ = await asyncio.wait(set(pending.keys()))
                    for task in remaining_done:
                        meta = pending.pop(task)
                        try:
                            result = await task
                        except asyncio.CancelledError:
                            continue
                        _bank(meta, result)
                        if result.decision.action == WorldAction.HALT_STREAM:
                            halted = True
                        elif result.decision.action == WorldAction.GROUP_READY:
                            ready_count += 1
                break
    except BaseException:
        await _settle_pending_after_stop(propagate_errors=False)
        raise

    completed.sort(key=lambda item: item.stream_offset)
    settled_offsets.update(
        item.stream_offset
        for item in completed
        if item.result.decision.action in {
            WorldAction.GROUP_READY, WorldAction.SKIP_INSUFFICIENT,
        }
    )
    # Admission says nothing about completion: both cancelled peers and the
    # halting world need another collection. Keep any completed siblings beyond
    # the first hole as sparse progress, rather than skipping holes or training
    # those siblings again after a checkpoint resume.
    retry_offsets = priority_offsets or set()
    next_offset = min({start_offset} | (retry_offsets - settled_offsets))
    # If an explicit retry reopens an older offset, preserve the other worlds
    # already consumed below the old cursor in the new sparse checkpoint too.
    settled_offsets.update(
        value
        for value in range(next_offset + 1, min(start_offset, len(task_runs)))
        if value not in retry_offsets
    )
    while next_offset < len(task_runs) and next_offset in settled_offsets:
        next_offset += 1
    return UpdateCollectResult(
        worlds=completed,
        next_world_offset=next_offset,
        halted=halted,
        settled_world_offsets=sorted(
            value for value in settled_offsets if value > next_offset
        ),
    )
