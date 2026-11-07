"""Retry startup and provably untouched step-zero runs through the real trainer.

Tinker/Cookbook/torch and the external Codex/Pi interfaces are fixtures. The
collector, acceptance audit, credit bridge, checkpoint logic, lock and trainer
loop are never replaced. The outage case starts from explicit durable input
rather than waiting for the provider circuit's production cooldown.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import os
from types import SimpleNamespace

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


def _later_invocation(config, *, explicit=False):
    return replace(config, run=replace(config.run, run_id="run-later-invocation"),
                   run_id_generated=not explicit)


def _snapshot_only(config):
    snapshot = trainer.snapshot(config, trainer.select_world_stream(config))
    write_json(config.resolved_log_dir() / trainer.SNAPSHOT_FILE, snapshot)
    return snapshot


def _halted_zero_state(config):
    root = config.resolved_log_dir()
    return {
        "run_id": config.run.run_id, "status": "halted", "step": 0,
        "rft_updates_completed": 0, "optimizer_update_in_flight": False,
        "next_world_offset": 0, "settled_world_offsets": [], "checkpoints": [],
        "segments": [{"index": 0, "root": str(root / "segments/000"), "start_step": 0}],
        "group_identities": {}, "halt": {"world_id": "fixture-outage", "reason": "provider down"},
    }


@pytest.mark.parametrize("failure", ["lora", "base-sampler"])
def test_real_pipeline_retries_transient_sdk_startup_with_original_id(
    tmp_path, monkeypatch, training_sdk, failure,
):
    config, pi_audit = _runtime(tmp_path, monkeypatch)
    root = config.resolved_log_dir()
    environment = dict(os.environ)
    training_sdk.observe = lambda: _assert_environment(environment)
    calls = _observe_processes(monkeypatch, environment)
    sdk = load_tinker_deps()
    base = sdk.tinker.ServiceClient
    failed = []

    class FlakyStartup(base):
        def create_lora_training_client(self, **kwargs):
            if failure == "lora" and not failed:
                failed.append(failure)
                raise RuntimeError("transient external LoRA creation failure")
            return super().create_lora_training_client(**kwargs)

        def create_sampling_client(self, **kwargs):
            if failure == "base-sampler" and "base_model" in kwargs and not failed:
                failed.append(failure)
                raise RuntimeError("transient external base sampling failure")
            return super().create_sampling_client(**kwargs)

    monkeypatch.setattr(sdk.tinker, "ServiceClient", FlakyStartup)
    with pytest.raises(RuntimeError, match="transient external"):
        asyncio.run(trainer.run_train(config, deps=sdk))
    assert failed == [failure] and not calls and not pi_audit.exists()
    assert training_sdk.updates == 0
    assert {path.name for path in root.iterdir()} == {
        trainer.SNAPSHOT_FILE, "backprop", "backprop.json", "observability",
        "telemetry-health.jsonl",
    }
    observations = [json.loads(line) for line in
                    (root / "observability/events.jsonl").read_text().splitlines()]
    assert all(row["train/optimizer_updates_committed"] == 0 for row in observations)
    assert all(row["kind"] in {"phase", "health"} for row in observations)
    original_snapshot = (root / trainer.SNAPSHOT_FILE).read_bytes()
    _assert_environment(environment)

    state = asyncio.run(trainer.run_train(_later_invocation(config), deps=sdk))
    assert state["run_id"] == config.run.run_id and state["status"] == "complete"
    assert state["step"] == state["rft_updates_completed"] == training_sdk.updates == 1
    assert state["next_world_offset"] == 1 and not state["optimizer_update_in_flight"]
    assert (root / trainer.SNAPSHOT_FILE).read_bytes() == original_snapshot
    assert all(row["name"].startswith(config.run.run_id + "-") for row in state["checkpoints"])
    metrics = [json.loads(line) for line in (root / trainer.METRICS_FILE).read_text().splitlines()]
    assert metrics[0]["datums"] == 2 and metrics[0]["worlds"][0]["rewards"] == [0.2, 0.8]
    _assert_private_runtime(tmp_path, environment, calls)


def test_snapshot_state_publication_crash_window_retries_real_pipeline(tmp_path, monkeypatch, training_sdk):
    config, _ = _runtime(tmp_path, monkeypatch)
    snapshot = _snapshot_only(config)  # process died after the first durable file
    root = config.resolved_log_dir()
    state = asyncio.run(trainer.run_train(_later_invocation(config), deps=load_tinker_deps()))
    assert state["run_id"] == config.run.run_id and state["step"] == 1
    assert state["status"] == "complete" and state["next_world_offset"] == 1
    assert json.loads((root / trainer.SNAPSHOT_FILE).read_text()) == snapshot
    assert len(state["segments"]) == 1 and state["segments"][0]["index"] == 0


@pytest.mark.parametrize("problem", ["extra-file", "segments", "explicit-id", "provenance"])
def test_snapshot_only_recovery_rejects_unrelated_or_changed_input_before_auth(
    tmp_path, monkeypatch, training_sdk, problem,
):
    config, _ = _runtime(tmp_path, monkeypatch)
    saved = _snapshot_only(config)
    root = config.resolved_log_dir()
    candidate = _later_invocation(config, explicit=problem == "explicit-id")
    if problem == "extra-file":
        (root / "user-data.txt").write_text("must not delete this file\n")
    elif problem == "segments":
        (root / "segments").mkdir()
        (root / "segments/previous-work.json").write_text('{"step":1}\n')
    elif problem == "provenance":
        saved["source_sha256"] = "different-source"
        write_json(root / trainer.SNAPSHOT_FILE, saved)
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(RuntimeError, match="not empty|does not match saved run_id|metadata changed"):
        asyncio.run(trainer.run_train(candidate, deps=_no_sdk()))
    assert before == {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert not training_sdk.calls


def test_halted_step_zero_resumes_real_collection_with_original_id_and_sparse_cursor(
    tmp_path, monkeypatch, training_sdk,
):
    config, _ = _runtime(tmp_path, monkeypatch, count=3)
    root = config.resolved_log_dir()
    _snapshot_only(config)
    state = _halted_zero_state(config)
    # Offsets 0 and 1 were halted/cancelled; sibling offset 2 settled with no
    # optimizer datums. A pre-update halt must neither skip holes nor replay it.
    state["settled_world_offsets"] = [2]
    write_json(root / trainer.STATE_FILE, state)
    evidence = root / "segments/000/halt-evidence.json"
    write_json(evidence, {"provider_outage": "halt", "optimizer_updates": 0})
    before = evidence.read_bytes()
    environment = dict(os.environ)
    calls = _observe_processes(monkeypatch, environment)
    actual = asyncio.run(trainer.run_train(_later_invocation(config), deps=load_tinker_deps()))
    assert actual["run_id"] == config.run.run_id and actual["status"] == "complete"
    assert actual["step"] == actual["rft_updates_completed"] == training_sdk.updates == 2
    assert actual["next_world_offset"] == 3 and actual["settled_world_offsets"] == []
    assert actual["resumes"] == 1 and actual["halt"] is None
    assert actual["segments"][1]["start_step"] == 0
    worlds = trainer.select_world_stream(config).worlds
    metrics = [json.loads(line) for line in (root / trainer.METRICS_FILE).read_text().splitlines()]
    assert [row["worlds"][0]["world_id"] for row in metrics] == [world.world_file.stem for world in worlds[:2]]
    assert not (root / "segments/001" / worlds[2].world_file.stem).exists()
    assert evidence.read_bytes() == before and not any(call[0] == "restore" for call in training_sdk.calls)
    assert sum(call[0] == "lora" for call in training_sdk.calls) == 1
    _assert_private_runtime(tmp_path, environment, calls)


@pytest.mark.parametrize("problem", ["step", "completed", "sampler", "in-flight", "missing-marker"])
def test_checkpoint_free_resume_never_reinitializes_optimizer_work(
    tmp_path, monkeypatch, training_sdk, problem,
):
    config, _ = _runtime(tmp_path, monkeypatch)
    _snapshot_only(config)
    state = _halted_zero_state(config)
    if problem == "step":
        state["step"] = 1
    elif problem == "completed":
        state["rft_updates_completed"] = 1
    elif problem == "sampler":
        state["checkpoints"] = [{"kind": "sampler", "path": "tinker://updated-weights", "rft_update": 1}]
    elif problem == "in-flight":
        state["optimizer_update_in_flight"] = True
    else:
        del state["optimizer_update_in_flight"]
    root = config.resolved_log_dir()
    write_json(root / trainer.STATE_FILE, state)
    before = (root / trainer.STATE_FILE).read_bytes()
    with pytest.raises(RuntimeError, match="no full checkpoint"):
        asyncio.run(trainer.run_train(config, deps=_no_sdk()))
    assert not training_sdk.calls and (root / trainer.STATE_FILE).read_bytes() == before


def test_actual_ambiguous_first_optimizer_result_is_durable_and_cannot_restart_from_base(
    tmp_path, monkeypatch, training_sdk,
):
    config, _ = _runtime(tmp_path, monkeypatch)
    root = config.resolved_log_dir()
    sdk = load_tinker_deps()
    base = sdk.tinker.ServiceClient

    class AmbiguousOptimizer(base):
        def create_lora_training_client(self, **kwargs):
            client = super().create_lora_training_client(**kwargs)
            original = client.optim_step

            def optim_step(adam):
                before = json.loads((root / trainer.STATE_FILE).read_text())
                assert before["optimizer_update_in_flight"] is True
                assert before["step"] == before["rft_updates_completed"] == 0
                original(adam)  # external optimizer committed but its result is lost

                def result():
                    raise RuntimeError("external optimizer result lost after commit")

                return SimpleNamespace(result=result)

            client.optim_step = optim_step
            return client

    monkeypatch.setattr(sdk.tinker, "ServiceClient", AmbiguousOptimizer)
    with pytest.raises(RuntimeError, match="external optimizer result lost after commit"):
        asyncio.run(trainer.run_train(config, deps=sdk))
    state = json.loads((root / trainer.STATE_FILE).read_text())
    assert state["status"] == "failed" and state["step"] == 0
    assert training_sdk.updates == 1 and state["checkpoints"] == []
    assert state["optimizer_update_in_flight"] is True
    before = (root / trainer.STATE_FILE).read_bytes()
    with pytest.raises(RuntimeError, match="no full checkpoint"):
        asyncio.run(trainer.run_train(_later_invocation(config), deps=_no_sdk()))
    assert (root / trainer.STATE_FILE).read_bytes() == before
    assert sum(call[0] == "lora" for call in training_sdk.calls) == 1
