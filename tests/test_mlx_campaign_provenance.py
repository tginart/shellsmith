"""Campaign provenance with the real shared stack and adaptive collectors.

External model/runtime readiness, the local HTTP server and episode execution
are fixtures; selection, policy stamps and durable campaign artifacts are real.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from shellsmith.campaign import inference
from shellsmith.episode.gm_process.python_runtime import PythonRuntime
import shellsmith.mlx
import shellsmith.mlx.checkpoint
from shellsmith.train import rollout_stack, trainer
from shellsmith.train.config import load_train_config
from shellsmith.train.durable_capture import write_policy_provenance
from shellsmith.train.world_selection import select_world_stream


def _read(path: Path):
    return json.loads(path.read_text())


@pytest.fixture
def gm_python_runtime(tmp_path):
    prefix = (tmp_path / "approved-python").resolve()
    prefix.mkdir(mode=0o700)
    (prefix / "bin").mkdir(mode=0o700)
    stdlib = prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    stdlib.mkdir(parents=True, mode=0o700)
    (stdlib / "os.py").write_text("# standard library marker\n")
    executable = prefix / "bin" / "python3"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    runtime = PythonRuntime(executable, prefix)
    runtime.validate()
    return runtime


@pytest.mark.parametrize("adapter_selected", [False, True], ids=["base", "adapter-checkpoint"])
def test_mlx_campaign_persists_base_or_selected_adapter_provenance(tmp_path, monkeypatch, adapter_selected,
                                                                 gm_python_runtime):
    config = load_train_config(None, log_dir=tmp_path / "campaign-run", campaign_mode="inference")
    world = select_world_stream(config, campaign_mode="inference").worlds[0]
    config = replace(config, compute=replace(config.compute, backend="mlx", gpu_lease_root=str(tmp_path / "lease")),
                     worlds=replace(config.worlds, select="ids", ids=(world.world_id,), count=1),
                     adaptive=replace(config.adaptive, initial_attempts=2, maximum_attempts=2,
                                      minimum_trainable=2, parallelism=2, parallelism_per_provider=2))
    identity = {"identity_sha256": "frozen-native-model"}
    base_ref = "mlx-base:" + identity["identity_sha256"]
    adapter_path = tmp_path / "selected-adapter" if adapter_selected else None
    verified, loaded, lifecycle = [], [], []
    session = SimpleNamespace(policy_identity=identity, policy_ref=base_ref, sampler=object())

    def load_adapter(path):
        loaded.append(path)
        session.policy_ref = str(path.resolve())

    session.load_adapter = load_adapter
    session.close = lambda: lifecycle.append("session-closed")

    async def create_session(config, *, checkpoint_root):
        return session

    def verify_checkpoint(path, *, kind, expected_identity):
        verified.append((path, kind, expected_identity))

    monkeypatch.setattr(shellsmith.mlx, "create_session", create_session)
    monkeypatch.setattr(shellsmith.mlx.checkpoint, "verify_checkpoint", verify_checkpoint)
    monkeypatch.setattr(rollout_stack, "current_python_runtime", lambda: gm_python_runtime)
    monkeypatch.setattr(trainer, "snapshot", lambda *args, **kwargs: {
        "policy_identity": identity,
        "pi_sdk": {"module_path": str(tmp_path / "external-pi.js"), "module_sha256": "pinned-pi"},
        "codex": {"executable_path": str(tmp_path / "external-codex"), "executable_sha256": "pinned-codex"},
    })

    class Endpoint:
        def __init__(self, *, sampler, records_dir):
            assert sampler is session.sampler

        def start(self):
            lifecycle.append("endpoint-started")

        def stop(self):
            lifecycle.append("endpoint-stopped")

    monkeypatch.setattr(rollout_stack, "PlayerEndpoint", Endpoint)

    def episode_runner(*, identity_for, policy_stamp, **kwargs):
        assert kwargs["gm_python_runtime"] is gm_python_runtime
        async def run(job, args):
            group = identity_for(job)
            write_policy_provenance(job.output_dir, policy_stamp(job.episode_id, group.key))
            return {
                "episode_id": job.episode_id, "world_id": job.world_id, "attempt": job.attempt,
                "flavor": job.flavor, "rollout_dir": str(job.output_dir), "group_identity_key": group.key,
                "accepted_count": 1,
                "audit": {"accepted_for_training": True, "final_task": {"reward": 0.2 * job.attempt}},
            }
        return run

    monkeypatch.setattr(rollout_stack, "make_tinker_rollout_runner", episode_runner)
    summary = asyncio.run(inference.run_inference_campaign(config, adapter_path=adapter_path))
    root = config.resolved_log_dir(kind="campaigns")
    expected_ref = str(adapter_path.resolve()) if adapter_path else base_ref
    assert summary["status"] == "complete" and summary["totals"]["accepted_episodes"] == 2
    assert summary["sampler_path"] == (expected_ref if adapter_selected else None)
    assert summary["policy_identity"] == identity
    assert summary == _read(root / inference.SUMMARY_FILE)
    snapshot = _read(root / inference.SNAPSHOT_FILE)
    assert snapshot["policy_ref"] == expected_ref and snapshot["policy_identity"] == identity
    assert snapshot["adapter_path"] == (str(adapter_path.resolve()) if adapter_path else None)
    for attempt in (1, 2):
        rollout = root / "campaign" / world.world_id / f"rollout_{attempt:02d}"
        provenance = _read(rollout / "policy_provenance.json")
        outcome = _read(rollout / "logs/durable_outcome.json")["outcome"]
        assert provenance["policy_kind"] == ("sampler_checkpoint" if adapter_selected else "base")
        assert provenance["policy_ref"] == expected_ref
        assert provenance["backend"] == "mlx" and provenance["policy_identity"] == identity
        assert provenance["rft_updates_completed"] == 0
        assert provenance["episode_id"] == outcome["episode_id"]
        assert provenance["group_identity_key"] == outcome["group_identity_key"]
        assert provenance["group_identity_key"] in summary["group_identities"]
    assert loaded == ([adapter_path] if adapter_selected else [])
    assert verified == ([(adapter_path, "sampler", identity)] if adapter_selected else [])
    assert lifecycle == ["endpoint-started", "endpoint-stopped", "session-closed"]
