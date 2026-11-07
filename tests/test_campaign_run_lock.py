"""Campaign ownership through real entry, process contention and publication.

Only external SDK/Codex/Pi interfaces are fixtures in runtime tests. The campaign
entry, snapshot, collectors, actual episodes, and lock implementation are real.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from shellsmith.campaign import inference
from shellsmith.policy.deps import load_tinker_deps
from shellsmith.train.config import load_train_config
from shellsmith.train.run_lock import TrainRunLockedError, lock_train_run
from test_campaign_inference_integration import _assert_no_training, _run
from test_trainer_resume import _runtime, external_tinker_sdk, training_sdk


_CONTENDER = """
import asyncio, json, pathlib, sys
from shellsmith.campaign.inference import run_inference_campaign
from shellsmith.train.config import load_train_config
from shellsmith.train.run_lock import TrainRunLockedError
class NoSDK:
    def __getattr__(self, name):
        raise AssertionError('Contender reached SDK: ' + name)
config = load_train_config(None, log_dir=pathlib.Path(sys.argv[1]), campaign_mode='inference')
try:
    asyncio.run(run_inference_campaign(config, deps=NoSDK(), sampler_path=None))
except TrainRunLockedError:
    print(json.dumps({'status': 'locked'}))
else:
    raise AssertionError('Contender entered an owned campaign')
"""


@pytest.mark.parametrize("contents", ["missing", "empty", "snapshot"])
def test_other_process_cannot_inspect_or_publish_in_owned_campaign(tmp_path, contents):
    root = tmp_path / "campaign-run"
    if contents != "missing":
        root.mkdir()
    sentinel = root / inference.SNAPSHOT_FILE
    if contents == "snapshot":
        sentinel.write_bytes(b"owner snapshot; deliberately not parseable")
    with lock_train_run(root):
        process = subprocess.run([sys.executable, "-c", _CONTENDER, str(root)],
                                 text=True, capture_output=True, timeout=10)
        assert process.returncode == 0, process.stdout + process.stderr
        assert json.loads(process.stdout.splitlines()[-1]) == {"status": "locked"}
        if contents == "missing":
            assert not root.exists()
        elif contents == "empty":
            assert list(root.iterdir()) == []
        else:
            assert list(root.iterdir()) == [sentinel]
            assert sentinel.read_bytes() == b"owner snapshot; deliberately not parseable"


def test_concurrent_real_campaign_entries_have_one_owner_and_snapshot(tmp_path, monkeypatch, training_sdk):
    config, audit = _runtime(tmp_path, monkeypatch)
    contender = replace(config, run=replace(config.run, run_id=config.run.run_id + "-contender"))

    async def both():
        return await asyncio.gather(
            inference.run_inference_campaign(config, deps=load_tinker_deps(), sampler_path=None),
            inference.run_inference_campaign(contender, deps=load_tinker_deps(), sampler_path=None),
            return_exceptions=True,
        )

    winner, rejected = asyncio.run(both())
    assert isinstance(rejected, TrainRunLockedError), repr(rejected)
    assert isinstance(winner, dict) and winner["status"] == "complete", repr(winner)
    assert winner["run_id"] == config.run.run_id
    root = config.resolved_log_dir(kind="campaigns")
    saved = json.loads((root / inference.SNAPSHOT_FILE).read_text())
    assert saved["config"]["run"]["run_id"] == config.run.run_id
    assert sum(call[0] == "service" for call in training_sdk.calls) == 1
    assert winner["totals"]["accepted_episodes"] == 2 and audit.is_file()
    assert len(list(root.rglob("durable_outcome.json"))) == 2
    _assert_no_training(training_sdk)


def test_real_campaign_holds_ownership_through_snapshot_sdk_and_finalization(tmp_path, monkeypatch, training_sdk):
    config, _ = _runtime(tmp_path, monkeypatch)
    root = config.resolved_log_dir(kind="campaigns")
    observed = []

    def require_owner(phase):
        with pytest.raises(TrainRunLockedError):
            with lock_train_run(root):
                pytest.fail(f"Campaign released ownership during {phase}")
        observed.append(phase)

    training_sdk.observe = lambda: require_owner("external-sdk")
    real_replace = Path.replace

    def check_publication(path, target):
        target = Path(target)
        if target in {root / inference.SNAPSHOT_FILE, root / inference.SUMMARY_FILE,
                      root / "campaign/campaign_report.json"}:
            require_owner(target.name)
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", check_publication)
    result = _run(config)
    assert result["status"] == "complete"
    assert {"external-sdk", inference.SNAPSHOT_FILE, inference.SUMMARY_FILE,
            "campaign_report.json"} <= set(observed)
    with lock_train_run(root):
        pass  # Released only after the final summary/report publication.
    _assert_no_training(training_sdk)


@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
def test_startup_failure_releases_campaign_ownership(tmp_path, monkeypatch, training_sdk, error_type):
    config, _ = _runtime(tmp_path, monkeypatch)
    root = config.resolved_log_dir(kind="campaigns")
    failure = error_type("external service creation failed")
    deps = load_tinker_deps()
    original = deps.tinker.ServiceClient

    class FailedService(original):
        def __init__(self):
            assert (root / inference.SNAPSHOT_FILE).is_file()
            with pytest.raises(TrainRunLockedError):
                with lock_train_run(root):
                    pytest.fail("Campaign was unlocked at authentication")
            raise failure

    monkeypatch.setattr(deps.tinker, "ServiceClient", FailedService)
    with pytest.raises(error_type) as caught:
        _run(config)
    assert caught.value is failure
    with lock_train_run(root):
        assert {path.name for path in root.iterdir()} == {inference.SNAPSHOT_FILE}
