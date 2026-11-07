"""Admission windows through the shipped campaign, collectors, HTTP and Node.

Only the external SDK/Codex/Pi interfaces are fixtures. Codex events and gates
observe actual admission and top-ups; no collector or outcome is substituted.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import os
import textwrap

import pytest

from shellsmith.campaign import inference
from shellsmith.policy.deps import load_tinker_deps
from shellsmith.train.rollout_stack import world_task_runs
from test_campaign_inference_integration import (
    _assert_captures, _assert_no_training, _codex, _edit_codex, _read, _worlds,
)
from test_player_credentials import _observe_processes
from test_trainer_resume import _runtime, external_tinker_sdk, training_sdk


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _staged(config):
    return {path.parent.name for path in (config.resolved_log_dir() / "campaign").glob("*/input")}


def _instrument_codex(tmp_path, config, *, on_start="", before_response=""):
    """Instrument only the external executable, leaving its real artifacts intact."""
    audit = tmp_path / "gm-window-events.jsonl"
    root = config.resolved_log_dir() / "campaign"
    source = _codex(tmp_path).read_text()
    needle = "attempt = int(episode['episode_id'].rsplit('-', 1)[1])"
    assert source.count(needle) == 1
    hook = textwrap.dedent(f"""
        import time
        campaign_root = pathlib.Path({str(root)!r})
        phase = 'final' if 'resume' in args else 'bootstrap'
        def record(kind):
            with pathlib.Path({str(audit)!r}).open('a') as log:
                log.write(json.dumps({{'kind': kind, 'world': episode['world_id'],
                                      'attempt': attempt, 'phase': phase,
                                      'staged': sorted(p.parent.name for p in campaign_root.glob('*/input'))}}) + '\\n')
        def wait_for(predicate):
            deadline = time.monotonic() + 45
            while not predicate():
                if time.monotonic() >= deadline:
                    raise RuntimeError('external Codex fixture gate timed out')
                time.sleep(0.01)
        record('start')
    """)
    _codex(tmp_path).write_text(source.replace(needle, needle + "\n" + hook + on_start))
    _edit_codex(tmp_path, before_response + "\nrecord('finish')")
    return audit


def _campaign(config):
    return inference.run_inference_campaign(config, deps=load_tinker_deps(), sampler_path=None)


async def _wait(task, predicate):
    async with asyncio.timeout(40):
        while not predicate():
            if task.done():
                await task
                pytest.fail("Campaign ended before the external fixture reached its gate")
            await asyncio.sleep(0.01)


async def _drain(task):
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def _started(events, world, attempt):
    return any(row["kind"] == "start" and row["phase"] == "bootstrap"
               and row["world"] == world and row["attempt"] == attempt for row in events)


def _world_tasks(world_id):
    # Observe/cancel actual asyncio collectors, never replace their implementation
    # or synthesize a sparse cursor. Used for the real child-cancellation case.
    return [task for task in asyncio.all_tasks()
            if task.get_coro().__qualname__.endswith(".collect_one")
            and (frame := task.get_coro().cr_frame) is not None
            and getattr(frame.f_locals.get("task_args"), "world_id", None) == world_id]


def _finished(config, world_id, count):
    report = config.resolved_log_dir() / "campaign/campaign_report.json"
    if not report.exists():
        return False
    rows = [row for row in _read(report)["rollouts"] if row["world_id"] == world_id]
    return len(rows) == count and not _world_tasks(world_id)


def _assert_window_order(events, windows):
    allowed = set()
    for index, window in enumerate(windows):
        allowed.update(window)
        current = [row for row in events if row["world"] in window]
        assert current and all(set(row["staged"]) <= allowed for row in current)
        if index:
            earlier = {world for group in windows[:index] for world in group}
            last_final = max(i for i, row in enumerate(events) if row["world"] in earlier
                             and row["kind"] == "finish" and row["phase"] == "final")
            first_start = min(i for i, row in enumerate(events) if row["world"] in window
                              and row["kind"] == "start")
            assert last_final < first_start


@pytest.mark.parametrize("global_initial,pool,initial,window_indexes", [
    (4, 8, [8, 8], [[0], [1]]),
    (8, 8, [2, 3, 3, 8, 1], [[0, 1, 2], [3], [4]]),
    (4, 4, [6, 1], [[0], [1]]),
    (4, 8, [3, 6, 1], [[0], [1, 2]]),
], ids=["override-above-global", "small-overrides-fill-pool", "first-batch-exceeds-pool", "queue-order-no-bin-packing"])
def test_merged_initial_batches_bound_each_window_and_preserve_concurrency(
    tmp_path, monkeypatch, training_sdk, global_initial, pool, initial, window_indexes,
):
    config, pi_audit = _runtime(tmp_path, monkeypatch, count=len(initial))
    ids = [world.world_id for world in _worlds(config)]
    windows = [[ids[i] for i in indexes] for indexes in window_indexes]
    config = replace(config,
                     adaptive=replace(config.adaptive, initial_attempts=global_initial, maximum_attempts=12,
                                      minimum_trainable=1, low_spread_top_up=1,
                                      parallelism=pool, parallelism_per_provider=pool),
                     world_overrides={world: {"initial_attempts": count, "maximum_attempts": count + 1}
                                      for world, count in zip(ids, initial, strict=True)})
    release = tmp_path / "release-initial-window"
    audit = _instrument_codex(
        tmp_path, config,
        on_start=f"if phase == 'bootstrap' and episode['world_id'] in {windows[0]!r}:\n"
                 f"    wait_for(lambda: pathlib.Path({str(release)!r}).exists())\n",
        before_response="if phase == 'final':\n    payload['final_reward'] = 0.5\n    payload['completion_percentage'] = 50",
    )
    first_jobs = [(ids[i], attempt) for i in window_indexes[0] for attempt in range(1, initial[i] + 1)][:pool]

    async def run():
        task = asyncio.create_task(_campaign(config))
        try:
            await _wait(task, lambda: all(_started(_events(audit), world, attempt) for world, attempt in first_jobs))
            # All fitting initial jobs are live together, even with small overrides.
            # Later worlds must not even be staged while those jobs hold the pool.
            assert _staged(config) == set(windows[0])
            assert not any(row["kind"] == "finish" for row in _events(audit))
            release.touch()
            async with asyncio.timeout(90):
                return await task
        finally:
            await _drain(task)

    summary = asyncio.run(run())
    assert summary["status"] == "complete"
    assert summary["worlds_started"] == summary["worlds_completed"] == len(ids)
    assert summary["worlds_incomplete"] == summary["worlds_not_started"] == 0
    assert summary["world_window"] == max(map(len, windows))  # Peak actual window, not a global estimate.
    assert [row["world_id"] for row in summary["worlds"]] == ids
    assert [row["initial_attempts"] for row in summary["worlds"]] == initial
    assert [row["attempts"] for row in summary["worlds"]] == [count + 1 for count in initial]
    assert all(row["decision"] == "group_ready" and row["rewards"] == [0.5] * row["attempts"]
               for row in summary["worlds"])
    assert summary["totals"]["worlds_topped_up"] == len(ids)
    assert summary["totals"]["accepted_episodes"] == sum(initial) + len(ids)
    assert summary == _read(config.resolved_log_dir() / inference.SUMMARY_FILE)
    _assert_window_order(_events(audit), windows)
    assert len(_events(audit)) == 4 * summary["totals"]["episodes"]
    assert len(_events(pi_audit)) == 2 * summary["totals"]["episodes"]
    _assert_captures(config, summary)
    _assert_no_training(training_sdk)
    # Existing one-argument callers still get the uniform-default capacity;
    # runtime callers explicitly supply the already-merged queue instead.
    assert inference.world_window(config) == max(1, pool // global_initial)
    tasks = world_task_runs(config, _worlds(config), campaign_mode="inference")
    assert inference.world_window(config, tasks) == len(windows[0])


def test_skip_cannot_refill_past_window_while_sibling_needs_top_up(tmp_path, monkeypatch, training_sdk):
    config, _ = _runtime(tmp_path, monkeypatch, count=3)
    skip, slow, later = [world.world_id for world in _worlds(config)]
    config = replace(config,
                     adaptive=replace(config.adaptive, minimum_trainable=1, maximum_attempts=5,
                                      low_spread_top_up=1, parallelism=4, parallelism_per_provider=4),
                     world_overrides={skip: {"initial_attempts": 1, "maximum_attempts": 1},
                                      slow: {"initial_attempts": 3, "maximum_attempts": 4},
                                      later: {"initial_attempts": 4}})
    release = tmp_path / "release-top-up"
    report = config.resolved_log_dir() / "campaign/campaign_report.json"
    audit = _instrument_codex(tmp_path, config, on_start=(
        f"if episode['world_id'] == {slow!r} and phase == 'final' and attempt <= 3:\n"
        f"    wait_for(lambda: any(row['world_id'] == {skip!r} for row in json.loads(pathlib.Path({str(report)!r}).read_text())['rollouts']))\n"
        f"if episode['world_id'] == {slow!r} and phase == 'bootstrap' and attempt == 4:\n"
        f"    wait_for(lambda: pathlib.Path({str(release)!r}).exists())\n"
    ), before_response=(
        "if phase == 'final':\n    payload['final_reward'] = 0.5\n    payload['completion_percentage'] = 50\n"
        f"    if episode['world_id'] == {skip!r}:\n        payload['trainable'] = False"
    ))

    async def run():
        task = asyncio.create_task(_campaign(config))
        try:
            await _wait(task, lambda: _started(_events(audit), slow, 4))
            assert _staged(config) == {skip, slow}
            assert not any(row["world"] == later for row in _events(audit))
            release.touch()
            async with asyncio.timeout(60):
                return await task
        finally:
            await _drain(task)

    summary = asyncio.run(run())
    assert summary["status"] == "complete" and summary["world_window"] == 2
    assert [row["decision"] for row in summary["worlds"]] == ["skip_insufficient", "group_ready", "group_ready"]
    assert [row["attempts"] for row in summary["worlds"]] == [1, 4, 5]
    assert [row["accepted"] for row in summary["worlds"]] == [0, 4, 5]
    assert summary["totals"]["worlds_topped_up"] == 2 and summary["totals"]["episodes"] == 10
    _assert_window_order(_events(audit), [[skip, slow], [later]])
    _assert_captures(config, summary)
    _assert_no_training(training_sdk)


def test_override_halt_never_stages_the_next_window(tmp_path, monkeypatch, training_sdk):
    config, _ = _runtime(tmp_path, monkeypatch, count=2)
    first, later = [world.world_id for world in _worlds(config)]
    config = replace(config, gm=replace(config.gm, timeout_seconds=2),
                     adaptive=replace(config.adaptive, initial_attempts=4, maximum_attempts=8,
                                      parallelism=8, parallelism_per_provider=8,
                                      outage_failure_threshold=1, outage_cooldown_seconds=0),
                     world_overrides={world: {"initial_attempts": 8} for world in (first, later)})
    audit = _instrument_codex(tmp_path, config, on_start="time.sleep(60)\n")
    calls = _observe_processes(monkeypatch, dict(os.environ))
    summary = asyncio.run(_campaign(config))
    assert summary["status"] == "halted" and summary["halt"]["world_id"] == first
    assert summary["worlds_started"] == summary["worlds_completed"] == 1
    assert summary["worlds_not_started"] == 1 and summary["worlds_incomplete"] == 0
    assert summary["worlds"][0]["decision"] == "halt_stream"
    assert summary["worlds"][0]["initial_attempts"] == summary["worlds"][0]["attempts"] == 8
    assert _staged(config) == {first} and {row["world"] for row in _events(audit)} == {first}
    report = _read(config.resolved_log_dir() / "campaign/campaign_report.json")
    assert report["status"] == "halted" and any(row["provider_attributable"] for row in report["rollouts"])
    assert calls and all(call.process.returncode is not None for call in calls)
    _assert_no_training(training_sdk)


@pytest.mark.parametrize("completed_index", [1, 2], ids=["completed-before-gap", "completed-after-gap"])
def test_cancellation_keeps_absolute_summary_offsets_without_later_admission(
    tmp_path, monkeypatch, training_sdk, completed_index,
):
    config, _ = _runtime(tmp_path, monkeypatch, count=4)
    ids = [world.world_id for world in _worlds(config)]
    initial = [4, 3, 1, 4]
    completed, blocked = ids[completed_index], ids[3 - completed_index]
    config = replace(config,
                     adaptive=replace(config.adaptive, initial_attempts=4, maximum_attempts=4,
                                      minimum_trainable=1, minimum_reward_range=0,
                                      parallelism=4, parallelism_per_provider=4),
                     world_overrides={world: {"initial_attempts": count} for world, count in zip(ids, initial, strict=True)})
    audit = _instrument_codex(tmp_path, config, on_start=f"if episode['world_id'] == {blocked!r}:\n    time.sleep(60)\n")
    calls = _observe_processes(monkeypatch, dict(os.environ))

    async def interrupt():
        task = asyncio.create_task(_campaign(config))
        try:
            await _wait(task, lambda: _started(_events(audit), blocked, 1)
                        and _finished(config, completed, initial[completed_index]))
            assert _staged(config) == set(ids[:3])
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not [pending for pending in asyncio.all_tasks() if pending is not asyncio.current_task()]
            assert calls and all(call.process.returncode is not None for call in calls)
        finally:
            await _drain(task)

    asyncio.run(interrupt())
    summary = _read(config.resolved_log_dir() / inference.SUMMARY_FILE)
    assert summary["status"] == "interrupted" and summary["world_window"] == 2
    assert summary["worlds_started"] == 3 and summary["worlds_completed"] == 2
    assert summary["worlds_incomplete"] == summary["worlds_not_started"] == 1
    assert summary["incomplete_world_ids"] == [blocked]
    assert [row["world_id"] for row in summary["worlds"]] == [ids[0], completed]
    assert [row["attempts"] for row in summary["worlds"]] == [4, initial[completed_index]]
    assert not any(row["world"] == ids[3] for row in _events(audit))
    _assert_captures(config, summary)
    _assert_no_training(training_sdk)


def test_sparse_child_cancellation_finishes_original_window_before_readmission(tmp_path, monkeypatch, training_sdk):
    config, _ = _runtime(tmp_path, monkeypatch, count=5)
    ids = [world.world_id for world in _worlds(config)]
    initial = [4, 1, 2, 1, 1]
    gap, later = ids[2], ids[4]
    config = replace(config,
                     adaptive=replace(config.adaptive, initial_attempts=4, maximum_attempts=4,
                                      minimum_trainable=1, minimum_reward_range=0,
                                      parallelism=4, parallelism_per_provider=4),
                     world_overrides={world: {"initial_attempts": count} for world, count in zip(ids, initial, strict=True)})
    release = tmp_path / "release-retried-gap"
    audit = _instrument_codex(tmp_path, config, on_start=(
        f"if episode['world_id'] == {gap!r}:\n"
        f"    wait_for(lambda: pathlib.Path({str(release)!r}).exists())\n"
    ))

    async def run():
        task = asyncio.create_task(_campaign(config))
        try:
            await _wait(task, lambda: all(_started(_events(audit), gap, attempt) for attempt in (1, 2))
                        and _finished(config, ids[1], 1) and _finished(config, ids[3], 1))
            collectors = _world_tasks(gap)
            assert len(collectors) == 1
            collectors[0].cancel()
            # The real collector returns a gap at absolute offset 2, with offset
            # 3 already settled. Its retry encounters genuine incomplete output
            # dirs, then tops up a fresh attempt rather than inventing a result.
            await _wait(task, lambda: _started(_events(audit), gap, 3))
            partial = _read(config.resolved_log_dir() / inference.SUMMARY_FILE)
            assert [row["world_id"] for row in partial["worlds"]] == [ids[0], ids[1], ids[3]]
            assert partial["incomplete_world_ids"] == [gap]
            assert _staged(config) == set(ids[:4])
            assert not any(row["world"] == later for row in _events(audit))
            release.touch()
            async with asyncio.timeout(60):
                return await task
        finally:
            await _drain(task)

    summary = asyncio.run(run())
    assert summary["status"] == "complete" and summary["world_window"] == 3
    assert [row["world_id"] for row in summary["worlds"]] == ids
    assert [row["attempts"] for row in summary["worlds"]] == [4, 1, 3, 1, 1]
    assert summary["worlds_incomplete"] == summary["worlds_not_started"] == 0
    assert summary["totals"]["worlds_topped_up"] == 1
    assert summary["totals"]["accepted_episodes"] == 8
    _assert_window_order(_events(audit), [[ids[0]], ids[1:4], [later]])
    # Already-settled siblings must not run again when the sparse cursor retries.
    for world in (ids[1], ids[3]):
        assert sum(row["kind"] == "start" and row["phase"] == "bootstrap" and row["world"] == world
                   for row in _events(audit)) == 1
    gap_outcomes = [_read(config.resolved_log_dir() / "campaign" / gap / f"rollout_{attempt:02d}/logs/durable_outcome.json")["outcome"]
                    for attempt in range(1, 4)]
    assert [row["accepted_count"] for row in gap_outcomes] == [0, 0, 1]
    assert all(row["error_type"] == "CancelledError" for row in gap_outcomes[:2])
    capture_index = _read(config.resolved_log_dir() / "campaign/capture_index.json")["episodes"]
    assert sum(row["accepted"] for row in capture_index.values()) == 8
    _assert_no_training(training_sdk)
