"""Offline tests for the read-only shellsmith run dashboard."""

from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import pytest

from shellsmith.viewer.app import create_app
from shellsmith.viewer.loader import load_catalog, load_run


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(root: Path) -> Path:
    run = root / "batch_a" / "rollout_01"
    _write(
        run / "episode.json",
        {"episode_id": "ep-1", "world_id": "checkout_idempotency_001", "status": "complete", "next_action_index": 1},
    )
    _write(run / "run_config.json", {"player_model": "qwen-3.6", "gm_model": "gpt-6.1-sol"})
    _write(run.parent / "batch_report.json", {"rollout_count": 1})
    _write(
        run / "world/final/response.json",
        {"final_reward": 0.75, "completion_percentage": 90, "trainable": True, "integrity_issues": []},
    )
    _write(run / "acceptance_audit.json", {"accepted_for_training": True, "simulator_integrity": {"passed": True, "issues": []}})
    _write(
        run / "world/turns/000000/request.json",
        {
            "step_index": 0,
            "action": {"id": "c1", "kind": "tool_call", "name": "bash", "arguments": {"command": "pwd"}},
            "trajectory_delta": [{"payload": {"message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "Inspect first."}]}}}],
        },
    )
    _write(
        run / "world/turns/000000/response.json",
        {
            "tool_response": {"content": "/workspace", "is_error": False},
            "turn_reward": 0.4,
            "completion_percentage": 10,
            "reasoning_assessment": {"quality": 0.9, "efficiency": 0.8},
            "trainable": True,
            "integrity_issues": [],
        },
    )
    return run


def test_catalog_and_detail_normalize_durable_artifacts(tmp_path: Path) -> None:
    _fixture(tmp_path)
    catalog = load_catalog(tmp_path)
    assert catalog["error"] is None
    assert catalog["runs"] == [
        {
            "id": "batch_a/rollout_01",
            "episode_id": "ep-1",
            "task": "checkout_idempotency_001",
            "player_model": "qwen-3.6",
            "gm_model": "gpt-6.1-sol",
            "run_dir": "batch_a",
            "campaign": None,
            "batch": "batch_a",
            "status": "complete",
            "action_count": 1,
            "reward": 0.75,
            "mean_quality": None,
            "mean_efficiency": None,
            "completion_percentage": 90,
            "trainable": True,
            "integrity_passed": True,
            "integrity_issue_count": 0,
            "accepted_for_training": True,
            "end_kind": None,
            "end_label": None,
            "has_dangling_agent_final": False,
            "agent_final_preview": None,
            "policy_kind": None,
            "policy_ref": None,
            "policy_rft_updates_completed": None,
            "gm_fallback_applied": None,
            "original_gm_model": None,
            "episode_not_started": False,
            "discarded_by_provider_fallback": False,
            "discard_reason": None,
            "warnings": [],
            "errors": [],
        }
    ]
    detail, error = load_run(tmp_path, "batch_a/rollout_01")
    assert error is None
    assert detail is not None
    assert detail["summary"]["id"] == "batch_a/rollout_01"
    assert detail["summary"]["episode_id"] == "ep-1"
    assert detail["summary"]["player_model"] == "qwen-3.6"
    assert detail["summary"]["gm_model"] == "gpt-6.1-sol"
    assert detail["summary"]["run_dir"] == "batch_a"
    assert detail["summary"]["reward"] == 0.75
    assert detail["turns"][0]["action"]["name"] == "bash"
    assert detail["turns"][0]["reasoning"] == [{"type": "thinking", "text": "Inspect first."}]
    assert detail["turns"][0]["reasoning_assessment"]["quality"] == 0.9


def test_default_viewer_root_contains_all_command_output_kinds(monkeypatch) -> None:
    from shellsmith.campaign.rollout_campaign import DEFAULT_OUTPUT_PARENT
    from shellsmith.paths import DEFAULT_LOG_ROOT
    from shellsmith.viewer.app import default_runs_root

    monkeypatch.delenv("SHELLSMITH_RUNS_ROOT", raising=False)
    assert default_runs_root() == DEFAULT_LOG_ROOT
    assert DEFAULT_OUTPUT_PARENT.parent == default_runs_root()
    from shellsmith.episode.smoke import DEFAULT_OUTPUT_PARENT as EPISODES_ROOT
    from shellsmith.train.config import load_train_config

    config = load_train_config(None)
    assert EPISODES_ROOT.parent == default_runs_root()
    assert config.resolved_log_dir().parent.parent == default_runs_root()
    assert config.resolved_log_dir(kind="campaigns").parent.parent == default_runs_root()


@pytest.mark.parametrize("entry", [("shellsmith.viewer.app",), ("shellsmith.cli", "view")], ids=["module", "cli"])
@pytest.mark.parametrize("launch", range(2))
def test_viewer_module_serves_real_artifacts(tmp_path: Path, launch: int, entry: tuple[str, ...]) -> None:
    """Exercise both shipped entry points twice through real HTTP, not just Flask's client."""
    _fixture(tmp_path)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, "-m", *entry, "--runs-root", str(tmp_path), "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 15
        while True:
            try:
                with urlopen(base + "/api/runs", timeout=1) as response:
                    catalog = json.load(response)
                break
            except OSError:
                if proc.poll() is not None or time.monotonic() >= deadline:
                    pytest.fail("viewer did not start its HTTP server")
                time.sleep(0.05)
        assert catalog["runs"][0]["id"] == "batch_a/rollout_01"
        assert catalog["runs"][0]["reward"] == 0.75
        with urlopen(base + "/api/runs/batch_a/rollout_01", timeout=2) as response:
            detail = json.load(response)
        assert detail["turns"][0]["tool_response"]["content"] == "/workspace"
        with urlopen(base, timeout=2) as response:
            assert "Trajectory Observatory" in response.read().decode()
        print(f"launch={launch}: HTTP catalog reward={catalog['runs'][0]['reward']}, tool_response=/workspace")
    finally:
        proc.terminate()
        output, _ = proc.communicate(timeout=10)
        print(output)


def test_flask_api_and_dashboard(tmp_path: Path) -> None:
    _fixture(tmp_path)
    app = create_app(runs_root=tmp_path)
    app.testing = True
    client = app.test_client()
    index = client.get("/")
    assert index.status_code == 200
    assert "Trajectory Observatory" in index.get_data(as_text=True)
    assert client.get("/api/runs").get_json()["runs"][0]["reward"] == 0.75
    detail = client.get("/api/runs/batch_a/rollout_01")
    assert detail.status_code == 200
    assert detail.get_json()["summary"]["reward"] == 0.75
    assert detail.get_json()["turns"][0]["tool_response"]["content"] == "/workspace"


def test_catalog_identifies_nearest_trainer_run_directory(tmp_path: Path) -> None:
    run = tmp_path / "hero" / "baby_v3"
    _write(run / "trainer_state.json", {"status": "running"})
    rollout = run / "step_0000" / "task_a" / "rollout_01"
    _write(
        rollout / "episode.json",
        {"episode_id": "ep-v3", "world_id": "task_a", "status": "complete"},
    )

    summary = load_catalog(tmp_path)["runs"][0]

    assert summary["run_dir"] == "hero/baby_v3"


def test_detail_supports_relative_viewer_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "runs"
    run = _fixture(root)
    monkeypatch.chdir(tmp_path)

    detail, error = load_run(Path("runs"), "batch_a/rollout_01")

    assert error is None
    assert detail is not None
    assert detail["summary"]["run_dir"] == "batch_a"
    assert detail["summary"]["id"] == run.relative_to(root).as_posix()


def test_traversal_and_malformed_artifacts_are_loud(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    (run / "run_config.json").write_text("{broken", encoding="utf-8")
    catalog = load_catalog(tmp_path)
    assert "could not read run_config.json" in catalog["runs"][0]["errors"][0]
    app = create_app(runs_root=tmp_path)
    app.testing = True
    assert app.test_client().get("/api/runs/../../etc").status_code == 404


def test_viewer_rejects_nested_symlinked_turn_directory(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    outside_turn = tmp_path / "outside-private-turn"
    _write(outside_turn / "request.json", {"action": {"name": "read"}})
    _write(outside_turn / "response.json", {"tool_response": {"content": "secret"}})
    (run / "world/turns/000001").symlink_to(outside_turn, target_is_directory=True)

    detail, error = load_run(tmp_path, "batch_a/rollout_01")

    assert error is None
    assert detail is not None
    assert len(detail["turns"]) == 1
    assert "refusing symlinked turn directory" in detail["artifact_errors"][-1]


def test_viewer_treats_terminal_untrainable_as_integrity_failure_without_audit(
    tmp_path: Path,
) -> None:
    run = _fixture(tmp_path)
    (run / "acceptance_audit.json").unlink()
    _write(
        run / "world/final/response.json",
        {
            "final_reward": 0.4,
            "completion_percentage": 60,
            "trainable": False,
            "integrity_issues": [],
        },
    )
    _write(
        run / "episode.json",
        {
            "episode_id": "ep-1",
            "world_id": "checkout_idempotency_001",
            "status": "untrainable",
            "next_action_index": 0,
        },
    )

    summary = load_catalog(tmp_path)["runs"][0]

    assert summary["integrity_passed"] is False
    assert summary["action_count"] == 1


# ---------------------------------------------------------------------------
# Artifacts this branch added must actually reach the dashboard.
#
# policy_provenance.json says which checkpoint sampled Player;
# logs/durable_outcome.json is the only record of a provider fallback, of an
# attempt slot that never started, and of a group discarded and redone on the
# backup GM. Without them, all three read as ordinary failing rollouts.
# ---------------------------------------------------------------------------


def _write_durable_outcome(run: Path, outcome: dict) -> None:
    _write(
        run / "logs" / "durable_outcome.json",
        {
            "schema": "shellsmith.durable_outcome.v1",
            "outcome": outcome,
        },
    )


def test_viewer_surfaces_policy_provenance(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    _write(
        run / "policy_provenance.json",
        {
            "schema": "shellsmith.policy_provenance.v1",
            "model_name": "Qwen/Qwen3.6-35B-A3B",
            "lora_rank": 64,
            "policy_kind": "sampler_checkpoint",
            "policy_ref": "tinker://run/sampler_weights/step-3",
            "rft_updates_completed": 3,
            "group_identity_key": "k",
            "episode_id": "ep-1",
        },
    )

    summary = load_catalog(tmp_path)["runs"][0]
    assert summary["policy_kind"] == "sampler_checkpoint"
    assert summary["policy_ref"] == "tinker://run/sampler_weights/step-3"
    assert summary["policy_rft_updates_completed"] == 3
    assert summary["warnings"] == []

    detail, error = load_run(tmp_path, "batch_a/rollout_01")
    assert error is None
    assert detail["policy_provenance"]["policy_ref"] == (
        "tinker://run/sampler_weights/step-3"
    )


def test_viewer_flags_a_never_started_attempt_slot(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    _write_durable_outcome(
        run,
        {
            "episode_id": "ep-1",
            "accepted_count": 0,
            "episode_not_started": True,
            "error": "RuntimeError: GM provider 'codex' is marked down.",
        },
    )

    summary = load_catalog(tmp_path)["runs"][0]
    assert summary["episode_not_started"] is True
    assert any("never started" in text for text in summary["warnings"])


def test_viewer_flags_a_discarded_fallback_group(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    _write_durable_outcome(
        run,
        {
            "episode_id": "ep-1",
            "accepted_count": 0,
            "discarded_by_provider_fallback": True,
            "discard_reason": "provider 'codex' marked down",
        },
    )

    summary = load_catalog(tmp_path)["runs"][0]
    assert summary["discarded_by_provider_fallback"] is True
    assert summary["discard_reason"] == "provider 'codex' marked down"
    assert any("discarded" in text for text in summary["warnings"])


def test_viewer_names_the_backup_gm_after_a_fallback(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    _write(run / "run_config.json", {"player_model": "qwen-3.6", "gm_model": "opus"})
    _write_durable_outcome(
        run,
        {
            "episode_id": "ep-1",
            "accepted_count": 1,
            "gm_fallback_applied": True,
            "original_gm_model": "gpt-6.1-sol",
        },
    )

    summary = load_catalog(tmp_path)["runs"][0]
    assert summary["gm_fallback_applied"] is True
    assert summary["original_gm_model"] == "gpt-6.1-sol"
    warning = "".join(summary["warnings"])
    assert "opus" in warning and "gpt-6.1-sol" in warning


def test_viewer_flags_base_policy_sampling_after_rft_updates(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    _write(
        run / "policy_provenance.json",
        {
            "schema": "shellsmith.policy_provenance.v1",
            "model_name": "Qwen/Qwen3.6-35B-A3B",
            "lora_rank": 64,
            "policy_kind": "base",
            "policy_ref": None,
            "rft_updates_completed": 4,
            "group_identity_key": "k",
            "episode_id": "ep-1",
        },
    )

    summary = load_catalog(tmp_path)["runs"][0]
    assert any("base policy" in text for text in summary["warnings"])
