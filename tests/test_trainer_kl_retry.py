"""Pre-submission KL retry through the shipped trainer/HTTP/Node/credit path.

Only external Tinker/Cookbook/torch and Codex/Pi interfaces are fixtures, shared
with test_trainer_resume. Reference scoring really suspends before failure or
task cancellation; collection, submission, metrics and resume are not replaced.
The sparse base case is explicitly seeded durable input, not prior live work.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import replace
import json
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from shellsmith.policy.deps import load_tinker_deps
from shellsmith.process_io import write_json
from shellsmith.train import trainer
from test_player_credentials import _assert_environment, _observe_processes
from test_trainer_resume import (
    _assert_private_runtime,
    _no_sdk,
    _runtime,
    external_tinker_sdk,
    training_sdk,
)


@pytest.fixture
def reference_sdk(monkeypatch, training_sdk):
    """Explicit external KL/scoring, Adam-construction and ambiguous-RPC faults."""
    deps = load_tinker_deps()
    control = SimpleNamespace(
        deps=deps, state_path=None, failure=None, fail_after_updates=0, ambiguous=None,
        score_states=[], adam_states=[], submission_states=[], kl_batches=[],
    )

    def durable_state():
        return json.loads(control.state_path.read_text())

    base = deps.tinker.ServiceClient

    class ReferenceService(base):
        def create_sampling_client(self, **kwargs):
            client = super().create_sampling_client(**kwargs)

            async def compute_logprobs_async(prompt):
                control.score_states.append(durable_state())
                await asyncio.sleep(0)
                if training_sdk.updates == control.fail_after_updates:
                    failure = control.failure
                    if failure in {"reference-error", "reference-cancel"}:
                        control.failure = None
                        if failure == "reference-error":
                            raise RuntimeError("external reference scoring failure")
                        asyncio.current_task().cancel()
                        await asyncio.sleep(0)
                        pytest.fail("Reference scoring should have been cancelled")
                return [-0.5] * len(prompt.to_ints())

            client.compute_logprobs_async = compute_logprobs_async
            return client

        def _observe_submission(self, client):
            forward, optim = client.forward_backward, client.optim_step

            def forward_backward(*args, **kwargs):
                control.submission_states.append(durable_state())
                result = forward(*args, **kwargs)
                if control.ambiguous == "forward-submit":
                    raise RuntimeError("external forward submission result lost")
                return result

            def optim_step(adam):
                control.submission_states.append(durable_state())
                result = optim(adam)
                if control.ambiguous == "optim-submit":
                    raise RuntimeError("external optimizer submission result lost")
                if control.ambiguous == "optim-result":
                    def lost_result():
                        raise RuntimeError("external optimizer result lost after commit")
                    return SimpleNamespace(result=lost_result)
                return result

            client.forward_backward, client.optim_step = forward_backward, optim_step
            return client

        def create_lora_training_client(self, **kwargs):
            return self._observe_submission(super().create_lora_training_client(**kwargs))

        def create_training_client_from_state_with_optimizer(self, path):
            return self._observe_submission(super().create_training_client_from_state_with_optimizer(path))

    adam_params = deps.types.AdamParams

    def adam(**kwargs):
        control.adam_states.append(durable_state())
        if control.failure == "adam":
            control.failure = None
            raise RuntimeError("external local Adam setup failure")
        return adam_params(**kwargs)

    async def incorporate_kl_penalty(datums, reference_client, coef, discount):
        assert reference_client is not None and coef == 0.1 and discount == 0.8
        assert datums and all("mask" in datum.loss_fn_inputs for datum in datums)
        control.kl_batches.append(len(datums))
        for datum in datums:
            assert await reference_client.compute_logprobs_async(datum.model_input)

    rl, metrics = ModuleType("tinker_cookbook.rl"), ModuleType("tinker_cookbook.rl.metrics")
    metrics.incorporate_kl_penalty = incorporate_kl_penalty
    rl.metrics = metrics
    monkeypatch.setitem(sys.modules, rl.__name__, rl)
    monkeypatch.setitem(sys.modules, metrics.__name__, metrics)
    monkeypatch.setattr(sys.modules["tinker_cookbook"], "rl", rl, raising=False)
    monkeypatch.setattr(deps.tinker, "ServiceClient", ReferenceService)
    monkeypatch.setattr(deps.types, "AdamParams", adam)
    return control


def _position(state):
    return state["optimizer_update_in_flight"], state["next_world_offset"], state["settled_world_offsets"]


def _rows(root):
    path = root / trainer.METRICS_FILE
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _collected(root, segment):
    outcomes = [json.loads(path.read_text())["outcome"] for path in
                (root / f"segments/{segment:03d}").rglob("durable_outcome.json")]
    assert outcomes and all(outcome["accepted_count"] == 1 for outcome in outcomes)
    return Counter(outcome["world_id"] for outcome in outcomes)


def _sparse_base_input(config, worlds):
    """Offsets 0/3 previously skipped; offset 1 and the later hole remain due."""
    root = config.resolved_log_dir()
    write_json(root / trainer.SNAPSHOT_FILE, trainer.snapshot(config, trainer.select_world_stream(config)))
    write_json(root / trainer.STATE_FILE, {
        "run_id": config.run.run_id, "status": "halted", "step": 0, "rft_updates_completed": 0,
        "optimizer_update_in_flight": False, "next_world_offset": 1, "settled_world_offsets": [3],
        "checkpoints": [], "segments": [], "group_identities": {},
    })
    row = {"step": 0, "datums": 0, "next_world_offset": 1, "settled_world_offsets": [3],
           "worlds": [{"world_id": worlds[i], "decision": "skip_insufficient"} for i in (0, 3)],
           "skipped": "prior no-datum sparse progress (durable-input fixture)"}
    prefix = (json.dumps(row, sort_keys=True) + "\n").encode()
    (root / trainer.METRICS_FILE).write_bytes(prefix)
    return prefix


@pytest.mark.parametrize("scenario,failure", [
    ("fresh", "reference-error"),
    ("fresh", "reference-cancel"),
    ("fresh", "adam"),
    ("sparse-base", "reference-error"),
    ("sparse-base", "reference-cancel"),
    ("checkpoint", "reference-error"),
    ("checkpoint", "reference-cancel"),
])
def test_pre_submission_failure_retries_original_worlds_and_cursor(
    tmp_path, monkeypatch, training_sdk, reference_sdk, scenario, failure,
):
    config, pi_audit = _runtime(tmp_path, monkeypatch, count=4 if scenario == "sparse-base" else 2,
                                checkpoint_every=1)
    # Filling both sparse holes would advance to EOF and clear [3]. A failed
    # KL pass must preserve that sparse set as well as the original offset.
    config = replace(config, train=replace(config.train, kl_penalty_coef=0.1, kl_discount_factor=0.8,
                                          groups_per_update=1 if scenario == "checkpoint" else 2))
    root = config.resolved_log_dir()
    worlds = [world.world_id for world in trainer.select_world_stream(config).worlds]
    prior_prefix = _sparse_base_input(config, worlds) if scenario == "sparse-base" else b""
    prior_updates = int(scenario == "checkpoint")
    offset, sparse = (0, []) if scenario == "fresh" else (1, [3] if scenario == "sparse-base" else [])
    reference_sdk.state_path = root / trainer.STATE_FILE
    reference_sdk.failure = failure
    reference_sdk.fail_after_updates = prior_updates
    environment = dict(os.environ)
    training_sdk.observe = lambda: _assert_environment(environment)
    processes = _observe_processes(monkeypatch, environment)

    error = asyncio.CancelledError if failure == "reference-cancel" else RuntimeError
    with pytest.raises(error, match=None if failure == "reference-cancel" else "external"):
        asyncio.run(trainer.run_train(config, deps=reference_sdk.deps))
    failed = json.loads(reference_sdk.state_path.read_text())
    assert failed["status"] == "failed" and failed["run_id"] == config.run.run_id
    assert failed["step"] == failed["rft_updates_completed"] == training_sdk.updates == prior_updates
    for kind in ("forward_backward", "optim_step"):
        assert sum(call[0] == kind for call in training_sdk.calls) == prior_updates
    rows_before = _rows(root)
    assert sum(row["datums"] > 0 for row in rows_before) == prior_updates
    metrics_path = root / trainer.METRICS_FILE
    metrics_before = metrics_path.read_bytes() if metrics_path.exists() else b""
    assert metrics_before.startswith(prior_prefix)
    if scenario == "sparse-base":
        assert metrics_before == prior_prefix
    collected_before = _collected(root, 0)
    first_worlds = worlds[1:3] if scenario == "sparse-base" else worlds
    assert collected_before == Counter({world: 2 for world in first_worlds})
    _assert_private_runtime(tmp_path, environment, processes)

    # Both parts are transactional: moving only the marker would skip this batch
    # on base retry, and resetting the cursor to zero would replay settled work.
    assert _position(failed) == (False, offset, sparse)
    assert _position(reference_sdk.score_states[-1]) == (False, offset, sparse)
    assert all(not state["optimizer_update_in_flight"] for state in reference_sdk.adam_states)
    checkpoint = trainer.latest_full_checkpoint(failed)
    assert (checkpoint is not None) == bool(prior_updates)
    snapshot = (root / trainer.SNAPSHOT_FILE).read_bytes()
    later = replace(config, run=replace(config.run, run_id="run-later-kl-invocation"))
    before = len(training_sdk.calls)
    resumed = asyncio.run(trainer.run_train(later, deps=reference_sdk.deps))
    assert resumed["run_id"] == config.run.run_id and resumed["status"] == "complete"
    assert resumed["step"] == resumed["rft_updates_completed"] == (2 if scenario == "checkpoint" else 1)
    assert resumed["resumes"] == (2 if scenario == "sparse-base" else 1)
    assert _position(resumed) == (False, len(worlds), [])
    assert (root / trainer.SNAPSHOT_FILE).read_bytes() == snapshot
    assert metrics_path.read_bytes().startswith(metrics_before)
    rows = _rows(root)
    trained_worlds = [world["world_id"] for row in rows if row["datums"] for world in row["worlds"]]
    assert trained_worlds == (worlds[1:3] if scenario == "sparse-base" else worlds)
    assert [row["datums"] for row in rows if row["datums"]] == ([2, 2] if scenario == "checkpoint" else [4])
    assert all(row["worlds"][0]["rewards"] == [0.2, 0.8] for row in rows if row["datums"])
    retry_worlds = worlds if scenario == "fresh" else worlds[1:3] if scenario == "sparse-base" else worlds[1:]
    collected_retry = _collected(root, 1)
    assert collected_retry == Counter({world: 2 for world in retry_worlds})
    assert len(pi_audit.read_text().splitlines()) == 2 * (sum(collected_before.values()) + sum(collected_retry.values()))
    if checkpoint is not None:
        assert ("restore", checkpoint["path"], 1) in training_sdk.calls[before:]
    else:
        assert not any(call[0] == "restore" for call in training_sdk.calls)
        assert sum(call[0] == "lora" for call in training_sdk.calls) == 2
    full = trainer.latest_full_checkpoint(resumed)
    assert full["step"] == training_sdk.checkpoints[full["path"]] == resumed["step"]
    assert full["next_world_offset"] == len(worlds) and full["settled_world_offsets"] == []
    assert training_sdk.sampler_steps[full["sampler_path"]] == resumed["step"]
    assert all(row["name"].startswith(config.run.run_id + "-") for row in resumed["checkpoints"])
    campaign = json.loads((root / "segments/001/campaign_manifest.json").read_text())
    assert campaign["campaign_id"] == config.run.run_id + "-s001"
    assert all(state["optimizer_update_in_flight"] for state in reference_sdk.submission_states)
    assert all(not state["optimizer_update_in_flight"] for state in reference_sdk.score_states)
    assert all(not state["optimizer_update_in_flight"] for state in reference_sdk.adam_states)
    _assert_private_runtime(tmp_path, environment, processes)


@pytest.mark.parametrize("failure", ["forward-submit", "optim-submit", "optim-result"])
def test_kl_success_then_ambiguous_optimizer_failure_still_fails_closed(
    tmp_path, monkeypatch, training_sdk, reference_sdk, failure,
):
    config, _ = _runtime(tmp_path, monkeypatch)
    config = replace(config, train=replace(config.train, kl_penalty_coef=0.1, kl_discount_factor=0.8))
    root = config.resolved_log_dir()
    reference_sdk.state_path = root / trainer.STATE_FILE
    reference_sdk.ambiguous = failure
    environment = dict(os.environ)
    processes = _observe_processes(monkeypatch, environment)
    with pytest.raises(RuntimeError, match="external .*lost"):
        asyncio.run(trainer.run_train(config, deps=reference_sdk.deps))
    failed = json.loads(reference_sdk.state_path.read_text())
    assert failed["status"] == "failed" and failed["step"] == failed["rft_updates_completed"] == 0
    assert _position(failed) == (True, 1, []) and failed["checkpoints"] == []
    assert training_sdk.updates == int(failure != "forward-submit")
    assert sum(call[0] == "forward_backward" for call in training_sdk.calls) == 1
    assert not _rows(root)
    assert reference_sdk.kl_batches == [2] and len(reference_sdk.score_states) == 2
    assert reference_sdk.submission_states and all(
        _position(state) == (True, 1, []) for state in reference_sdk.submission_states)
    original_state = reference_sdk.state_path.read_bytes()
    calls_before = list(training_sdk.calls)
    later = replace(config, run=replace(config.run, run_id="run-later-kl-invocation"))
    with pytest.raises(RuntimeError, match="no full checkpoint"):
        asyncio.run(trainer.run_train(later, deps=_no_sdk()))
    assert reference_sdk.state_path.read_bytes() == original_state
    assert training_sdk.calls == calls_before and not _rows(root)
    _assert_private_runtime(tmp_path, environment, processes)
    assert all(_position(state) == (False, 0, []) for state in reference_sdk.score_states)
    assert len(reference_sdk.adam_states) == 1
    assert _position(reference_sdk.adam_states[0]) == (False, 0, [])
