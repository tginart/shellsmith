"""Durable cursor regressions through real adaptive collectors and campaign waves.

Only the rollout runner is a fixture. Events control actual in-flight episode
and world tasks; no collector, policy decision, or update result is mocked.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from shellsmith.campaign.provider_outage import ProviderHealthTracker
from shellsmith.campaign.rollout_campaign import RolloutJob
from shellsmith.campaign.stream_campaign import OpenCampaignSpec, open_campaign
from shellsmith.train.stream_engine import (
    UpdateCollectResult,
    UpdateWorldResult,
    WorldCollectResult,
    collect_adaptive_world,
    collect_update_worlds,
)
from shellsmith.train.stream_policy import StreamPolicyConfig, WorldAction


class ControlledCampaign:
    """Gate rollout I/O, letting the real policy decide READY, SKIP, or HALT."""

    def __init__(
        self,
        root: Path,
        names: tuple[str, ...],
        *,
        failing: tuple[str, ...] = (),
        rejected: tuple[str, ...] = (),
        return_on_cancel: tuple[str, ...] = (),
    ) -> None:
        self.handle = open_campaign(
            root,
            OpenCampaignSpec(
                campaign_id=root.name,
                mode="tinker_train",
                parallelism=8,
                parallelism_per_provider=8,
            ),
        )
        self.handle.health = ProviderHealthTracker(
            consecutive_failure_threshold=1, cooldown_seconds=0,
        )
        self.config = StreamPolicyConfig(
            initial_attempts=2,
            maximum_attempts=2,
            minimum_trainable=2,
            parallelism=8,
            parallelism_per_provider=8,
            campaign_mode="tinker_train",
        )
        self.task_runs = [
            (
                root / f"{name}.md",
                argparse.Namespace(
                    initial_attempts=2,
                    maximum_attempts=2,
                    minimum_trainable=2,
                    minimum_reward_range=0.15,
                    low_spread_top_up=2,
                    gm_runtime="codex" if name in failing else "claude",
                ),
            )
            for name in names
        ]
        self.failing = set(failing)
        self.rejected = set(rejected)
        self.return_on_cancel = set(return_on_cancel)
        self.release = {name: asyncio.Event() for name in names}
        self.running = {name: asyncio.Event() for name in names}
        self.finished = {name: asyncio.Event() for name in names}
        self.jobs: list[tuple[str, int]] = []
        self.calls: list[str] = []
        self.cancelled_jobs: set[tuple[str, int]] = set()
        self.cancelled_worlds: set[str] = set()
        self.cancelled_after_result: set[str] = set()
        self.tasks: dict[str, asyncio.Task[Any]] = {}
        self.seeds: dict[str, list[dict[str, Any]]] = {}

    async def runner(self, job: RolloutJob, args: argparse.Namespace) -> dict[str, Any]:
        key = (job.world_id, job.attempt)
        self.jobs.append(key)
        if job.attempt == 2:
            self.running[job.world_id].set()
        try:
            await self.release[job.world_id].wait()
        except asyncio.CancelledError:
            self.cancelled_jobs.add(key)
            raise
        failed = job.world_id in self.failing
        return {
            "episode_id": job.episode_id,
            "world_id": job.world_id,
            "flavor": job.flavor,
            "attempt": job.attempt,
            "rollout_dir": str(job.output_dir),
            "error": "provider unavailable" if failed else None,
            "provider_attributable": failed,
            "audit": None if failed else {
                "accepted_for_training": job.world_id not in self.rejected,
                "final_task": {"reward": 0.2 * job.attempt},
            },
        }

    async def collect_one(
        self,
        world_file: Path,
        task_args: argparse.Namespace,
        world_config: StreamPolicyConfig,
        flavor: str,
    ) -> WorldCollectResult:
        name = world_file.stem
        self.calls.append(name)
        task = asyncio.current_task()
        assert task is not None
        self.tasks[name] = task
        try:
            result = await collect_adaptive_world(
                handle=self.handle,
                args=task_args,
                config=world_config,
                world_file=world_file,
                world_id=name,
                flavor=flavor,
                gm_runtime=task_args.gm_runtime,
                gm_model="test-model",
                gm_provider="test-provider",
                gm_reasoning_effort="low",
                rollout_runner=self.runner,
                seed_outcomes=self.seeds.get(name),
            )
            self.finished[name].set()
            if name in self.return_on_cancel:
                # A real result exists, but its task has not returned it yet.
                # Exercise the halt gather path rather than the initial wait batch.
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled_after_result.add(name)
            return result
        except asyncio.CancelledError:
            self.cancelled_worlds.add(name)
            raise

    async def collect(
        self,
        *,
        start_offset: int = 0,
        groups: int = 2,
        **kwargs: Any,
    ) -> UpdateCollectResult:
        return await collect_update_worlds(
            task_runs=self.task_runs,
            start_offset=start_offset,
            groups_per_update=groups,
            base_config=self.config,
            classify_flavor=lambda _path: "agentic",
            collect_one=self.collect_one,
            **kwargs,
        )

    def release_all(self) -> None:
        for event in self.release.values():
            event.set()

    def assert_drained(self) -> None:
        assert all(task.done() for task in self.tasks.values())
        assert self.handle.pool._in_flight_total == 0
        assert not self.handle.pool._in_flight_provider


def _checkpoint_cursor(update: UpdateCollectResult) -> dict[str, Any]:
    # Exercise the JSON-shaped contract needed alongside optimizer checkpoints.
    return json.loads(json.dumps({
        "next_world_offset": update.next_world_offset,
        "settled_world_offsets": update.settled_world_offsets,
    }))


@pytest.mark.parametrize("halting", ["left", "right"])
def test_halt_retries_both_cancelled_sibling_and_halted_world(
    tmp_path: Path, halting: str,
) -> None:
    async def run() -> None:
        names = ("past", "left", "right", "tail")
        campaign = ControlledCampaign(tmp_path / "first", names, failing=(halting,))
        collecting = asyncio.create_task(campaign.collect(start_offset=1))
        await asyncio.gather(campaign.running["left"].wait(), campaign.running["right"].wait())
        campaign.release[halting].set()
        update = await collecting
        sibling = "right" if halting == "left" else "left"
        assert update.halted
        assert [item.result.world_id for item in update.worlds] == [halting]
        assert update.worlds[0].result.decision.action == WorldAction.HALT_STREAM
        assert campaign.cancelled_worlds == {sibling}
        assert campaign.cancelled_jobs == {(sibling, 1), (sibling, 2)}
        assert campaign.tasks[sibling].cancelled()
        campaign.assert_drained()
        assert update.next_world_offset == 1  # Neither HALT nor cancellation is consumed.
        assert update.settled_world_offsets == []

        saved = _checkpoint_cursor(update)
        resumed = ControlledCampaign(tmp_path / "resumed", names)
        resumed.release_all()
        retried = await resumed.collect(
            start_offset=saved["next_world_offset"],
            settled_world_offsets=set(saved["settled_world_offsets"]),
        )
        assert resumed.calls == ["left", "right"]
        assert retried.next_world_offset == 3
        assert retried.settled_world_offsets == []
        assert not retried.halted
        resumed.assert_drained()

    asyncio.run(asyncio.wait_for(run(), timeout=5))


@pytest.mark.parametrize("return_on_cancel", [False, True])
def test_halt_keeps_later_completed_world_without_retraining_on_resume(
    tmp_path: Path, return_on_cancel: bool,
) -> None:
    async def run() -> None:
        names = ("skip", "gap", "halt", "ready", "tail")
        campaign = ControlledCampaign(
            tmp_path / "first", names,
            failing=("halt",), rejected=("skip",),
            return_on_cancel=("ready",) if return_on_cancel else (),
        )
        collecting = asyncio.create_task(campaign.collect(groups=3))
        await asyncio.gather(*(campaign.running[name].wait() for name in names[:3]))
        campaign.release["skip"].set()
        await campaign.running["ready"].wait()  # The skipped world admitted a replacement.
        campaign.release["ready"].set()
        await campaign.finished["ready"].wait()  # Offset 3 finishes before 1 and 2.
        campaign.release["halt"].set()
        update = await collecting
        assert campaign.cancelled_worlds == {"gap"}
        assert campaign.tasks["gap"].cancelled()
        assert campaign.cancelled_after_result == ({"ready"} if return_on_cancel else set())
        campaign.assert_drained()
        assert [(item.stream_offset, item.result.decision.action) for item in update.worlds] == [
            (0, WorldAction.SKIP_INSUFFICIENT),
            (2, WorldAction.HALT_STREAM),
            (3, WorldAction.GROUP_READY),
        ]
        assert update.next_world_offset == 1
        assert update.settled_world_offsets == [3]

        saved = _checkpoint_cursor(update)
        resumed = ControlledCampaign(tmp_path / "resumed", names)
        resumed.release_all()
        retried = await resumed.collect(
            start_offset=saved["next_world_offset"], groups=3,
            settled_world_offsets=set(saved["settled_world_offsets"]),
        )
        # Previously updated sparse offsets neither run nor count toward this update.
        assert resumed.calls == ["gap", "halt", "tail"]
        assert [item.stream_offset for item in retried.worlds] == [1, 2, 4]
        assert retried.next_world_offset == len(names)
        assert retried.settled_world_offsets == []
        updated_offsets = [
            item.stream_offset
            for batch in (update, retried)
            for item in batch.worlds
            if item.result.decision.action == WorldAction.GROUP_READY
        ]
        assert sorted(updated_offsets) == [1, 2, 3, 4]
        resumed.assert_drained()

    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_same_batch_halt_and_ready_leave_halted_world_open(tmp_path: Path) -> None:
    async def run() -> None:
        campaign = ControlledCampaign(tmp_path / "same-batch", ("halt", "ready"), failing=("halt",))
        collecting = asyncio.create_task(campaign.collect())
        await asyncio.gather(*(event.wait() for event in campaign.running.values()))
        campaign.release_all()
        update = await collecting
        assert update.halted
        assert [item.result.decision.action for item in update.worlds] == [
            WorldAction.HALT_STREAM, WorldAction.GROUP_READY,
        ]
        assert not campaign.cancelled_worlds
        assert update.next_world_offset == 0
        assert update.settled_world_offsets == [1]
        campaign.assert_drained()

    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_uninterrupted_out_of_order_results_advance_contiguous_cursor(tmp_path: Path) -> None:
    async def run() -> None:
        campaign = ControlledCampaign(
            tmp_path / "normal", ("past", "slow", "skip", "ready", "tail"),
            rejected=("skip",),
        )
        collecting = asyncio.create_task(campaign.collect(start_offset=1))
        await asyncio.gather(campaign.running["slow"].wait(), campaign.running["skip"].wait())
        campaign.release["skip"].set()
        await campaign.running["ready"].wait()
        campaign.release["ready"].set()
        await campaign.finished["ready"].wait()
        assert not collecting.done()
        campaign.release["slow"].set()
        update = await collecting
        assert not update.halted
        assert [item.stream_offset for item in update.worlds] == [1, 2, 3]
        assert update.next_world_offset == 4
        assert update.settled_world_offsets == []
        campaign.assert_drained()

    asyncio.run(asyncio.wait_for(run(), timeout=5))


@pytest.mark.parametrize("start_offset", [0, 2])
def test_retry_preloads_unupdated_ready_and_tops_up_priority_skip(
    tmp_path: Path, start_offset: int,
) -> None:
    async def run() -> None:
        campaign = ControlledCampaign(tmp_path / "retry", ("ready", "skip"), rejected=("skip",))
        campaign.release_all()
        initial = await campaign.collect()
        assert initial.next_world_offset == 2
        ready, skipped = initial.worlds
        assert ready.result.decision.action == WorldAction.GROUP_READY
        assert skipped.result.decision.action == WorldAction.SKIP_INSUFFICIENT
        # Insufficient-group retry reuses the real ready result, not a committed
        # sparse offset, and seeds only the skipped world's actual prior attempts.
        campaign.task_runs.append((tmp_path / "extra.md", argparse.Namespace()))
        campaign.rejected.clear()
        campaign.seeds["skip"] = skipped.result.outcomes
        campaign.task_runs[1][1].maximum_attempts = 4
        retried = await campaign.collect(
            start_offset=start_offset, preloaded_worlds=[ready], priority_offsets={1},
        )
        assert retried.worlds[0] is ready
        assert [item.result.decision.action for item in retried.worlds] == [
            WorldAction.GROUP_READY, WorldAction.GROUP_READY,
        ]
        assert campaign.calls == ["ready", "skip", "skip"]
        assert [attempt for name, attempt in campaign.jobs if name == "skip"] == [1, 2, 3, 4]
        assert retried.next_world_offset == 2
        assert retried.settled_world_offsets == []
        campaign.assert_drained()

    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_preloaded_ready_worlds_do_not_consume_an_unattempted_gap(tmp_path: Path) -> None:
    async def run() -> None:
        campaign = ControlledCampaign(tmp_path / "preloaded", ("ready0", "gap", "ready2"))
        campaign.release_all()
        offsets = (0, 2)
        results = await asyncio.gather(*(
            campaign.collect_one(*campaign.task_runs[offset], campaign.config, "agentic")
            for offset in offsets
        ))
        preloaded = [
            UpdateWorldResult(offset, *campaign.task_runs[offset], result)
            for offset, result in zip(offsets, results, strict=True)
        ]
        update = await campaign.collect(preloaded_worlds=preloaded)
        assert campaign.calls == ["ready0", "ready2"]
        assert update.worlds == preloaded
        assert update.next_world_offset == 1
        assert update.settled_world_offsets == [2]
        campaign.assert_drained()

        resumed = ControlledCampaign(tmp_path / "resumed", ("ready0", "gap", "ready2"))
        resumed.release_all()
        retried = await resumed.collect(
            start_offset=update.next_world_offset, groups=1,
            settled_world_offsets=set(update.settled_world_offsets),
        )
        assert resumed.calls == ["gap"]
        assert retried.next_world_offset == 3
        assert retried.settled_world_offsets == []
        resumed.assert_drained()

    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_halted_priority_retry_preserves_previously_consumed_later_offsets(tmp_path: Path) -> None:
    async def run() -> None:
        names = ("past", "retry", "already", "tail")
        original = ControlledCampaign(tmp_path / "original", names, rejected=("retry",))
        original.release_all()
        initial = await original.collect()
        assert initial.next_world_offset == 3
        original.assert_drained()

        retry = ControlledCampaign(tmp_path / "retry", names, failing=("retry",))
        retry.release_all()
        halted = await retry.collect(
            start_offset=initial.next_world_offset, groups=1, priority_offsets={1},
        )
        assert halted.halted
        assert retry.calls == ["retry"]
        assert halted.next_world_offset == 1
        assert halted.settled_world_offsets == [2]
        retry.assert_drained()

        saved = _checkpoint_cursor(halted)
        resumed = ControlledCampaign(tmp_path / "resumed", names)
        resumed.release_all()
        update = await resumed.collect(
            start_offset=saved["next_world_offset"],
            settled_world_offsets=set(saved["settled_world_offsets"]),
        )
        assert resumed.calls == ["retry", "tail"]
        assert update.next_world_offset == 4
        assert update.settled_world_offsets == []
        resumed.assert_drained()

    asyncio.run(asyncio.wait_for(run(), timeout=5))
