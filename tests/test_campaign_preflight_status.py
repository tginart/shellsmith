"""Campaign engine ceilings and terminal states through shipped entry logic.

The external SDK/Codex/Pi fixtures are shared with the full trainer pipeline.
Publication faults are injected only at the filesystem replace boundary.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path

import pytest

from shellsmith.campaign import inference
from shellsmith.campaign.rollout_campaign import MAX_PARALLELISM
from shellsmith.campaign.stream_campaign import OpenCampaignSpec
from shellsmith.policy.deps import load_tinker_deps
from shellsmith.train import trainer
from shellsmith.train.config import load_train_config, validate_train_config
from shellsmith.train.run_lock import lock_train_run
from test_campaign_inference_integration import _assert_no_training, _run
from test_player_credentials import _assert_endpoint_closed
from test_trainer_resume import _runtime, external_tinker_sdk, training_sdk


class _NoSDK:
    def __getattr__(self, name):
        pytest.fail(f"Parallelism preflight reached external SDK: {name}")


@pytest.mark.parametrize("mode", ["inference", "tinker_train"])
@pytest.mark.parametrize("parallelism", [MAX_PARALLELISM + 1, MAX_PARALLELISM * 2])
def test_engine_parallelism_ceiling_precedes_snapshot_output_and_sdk(
    tmp_path, monkeypatch, training_sdk, mode, parallelism,
):
    config, _ = _runtime(tmp_path, monkeypatch)
    config = replace(config, adaptive=replace(config.adaptive, parallelism=parallelism,
                                              parallelism_per_provider=parallelism))
    with pytest.raises(ValueError, match="parallelism must be between"):
        validate_train_config(config, campaign_mode=mode)
    operation = (inference.run_inference_campaign(config, deps=_NoSDK(), sampler_path=None)
                 if mode == "inference" else trainer.run_train(config, deps=_NoSDK()))
    with pytest.raises(ValueError, match="parallelism must be between"):
        asyncio.run(operation)
    assert not config.resolved_log_dir().exists()
    assert not training_sdk.calls


@pytest.mark.parametrize("mode", ["inference", "tinker_train"])
def test_toml_parallelism_ceiling_matches_actual_campaign_spec(tmp_path, mode):
    path = tmp_path / "config.toml"
    path.write_text(f"[adaptive]\nparallelism = {MAX_PARALLELISM}\nparallelism_per_provider = {MAX_PARALLELISM}\n")
    config = load_train_config(path, campaign_mode=mode)
    spec = OpenCampaignSpec(campaign_id=config.run.run_id, mode=mode,
                            parallelism=config.adaptive.parallelism,
                            parallelism_per_provider=config.adaptive.parallelism_per_provider)
    assert spec.parallelism == MAX_PARALLELISM
    path.write_text(f"[adaptive]\nparallelism = {MAX_PARALLELISM + 1}\nparallelism_per_provider = 1\n")
    with pytest.raises(ValueError, match="parallelism must be between"):
        load_train_config(path, campaign_mode=mode)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("error_type,expected_status", [
    (OSError, "failed"), (RuntimeError, "failed"), (SystemExit, "failed"),
    (KeyboardInterrupt, "interrupted"), (asyncio.CancelledError, "interrupted"),
])
@pytest.mark.parametrize("publication", ["progress", "final"])
def test_campaign_summary_and_report_agree_on_failure_or_interruption(
    tmp_path, monkeypatch, training_sdk, error_type, expected_status, publication,
):
    config, pi_audit = _runtime(tmp_path, monkeypatch)
    root = config.resolved_log_dir()
    real_replace = Path.replace
    faults = []
    failure = error_type("injected campaign summary publication boundary")

    def fail_first_summary_replace(path, target):
        if Path(target) == root / inference.SUMMARY_FILE and not faults:
            # Real episodes have finished; this is publication I/O, not a fake
            # collector or fabricated campaign result.
            candidate = json.loads(path.read_text())
            if publication == "final" and candidate["status"] != "complete":
                return real_replace(path, target)
            assert candidate["worlds_completed"] == 1
            assert candidate["totals"]["accepted_episodes"] == 2
            faults.append(type(failure).__name__)
            raise failure
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_first_summary_replace)
    with pytest.raises(error_type) as caught:
        _run(config)
    assert caught.value is failure and faults == [error_type.__name__]
    summary = json.loads((root / inference.SUMMARY_FILE).read_text())
    report = json.loads((root / "campaign/campaign_report.json").read_text())
    assert summary["status"] == expected_status
    assert report["status"] == expected_status
    assert summary["worlds_completed"] == 1 and summary["worlds_incomplete"] == 0
    assert summary["totals"]["accepted_episodes"] == 2 and pi_audit.is_file()
    assert summary["worlds"][0]["rewards"] == [0.2, 0.8]
    with lock_train_run(root):
        pass  # Final publication failures must also release directory ownership.
    for log in root.rglob("vat.jsonl"):
        for line in log.read_text().splitlines():
            message = json.loads(line).get("message", {})
            if message.get("type") == "initialize":
                _assert_endpoint_closed(message["model"]["base_url"])
    _assert_no_training(training_sdk)


def test_persistent_final_publication_failure_never_reports_success(tmp_path, monkeypatch, training_sdk):
    config, _ = _runtime(tmp_path, monkeypatch)
    root = config.resolved_log_dir()
    real_replace = Path.replace
    failure = OSError("persistent final-summary publication failure")
    failed_statuses = []

    def fail_terminal_summary(path, target):
        if Path(target) == root / inference.SUMMARY_FILE:
            candidate = json.loads(path.read_text())
            if candidate["status"] != "running":
                failed_statuses.append(candidate["status"])
                raise failure
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_terminal_summary)
    with pytest.raises(OSError) as caught:
        _run(config)
    assert caught.value is failure and failed_statuses == ["complete", "failed"]
    # The disk boundary refuses every terminal summary write: preserve the last
    # published progress record rather than claiming it was durably repaired.
    assert json.loads((root / inference.SUMMARY_FILE).read_text())["status"] == "running"
    report = json.loads((root / "campaign/campaign_report.json").read_text())
    assert report["status"] == "failed"
    with lock_train_run(root):
        pass
    for log in root.rglob("vat.jsonl"):
        for line in log.read_text().splitlines():
            message = json.loads(line).get("message", {})
            if message.get("type") == "initialize":
                _assert_endpoint_closed(message["model"]["base_url"])
    _assert_no_training(training_sdk)
