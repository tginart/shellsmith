"""Public backend dispatch and durable remote destination contracts."""
from dataclasses import asdict
import json

import pytest

from shellsmith import cli
from shellsmith.jobs import Endpoint


def test_mlx_dry_run_does_not_load_a_player_or_tinker(tmp_path, monkeypatch, capsys):
    config = tmp_path / "run.toml"
    config.write_text('[compute]\nbackend="mlx"\nhost="studio"\n')
    import shellsmith.jobs
    import shellsmith.policy.deps

    monkeypatch.setattr(shellsmith.policy.deps, "load_tinker_deps", lambda: pytest.fail("Tinker import"))
    class Client:
        def __init__(self, endpoint):
            assert endpoint.host == "studio"
        def check(self, config):
            assert config["mlx"]["context_window"] == 32768
            return {"ready": False, "blockers": ["Studio login required"]}
    monkeypatch.setattr(shellsmith.jobs, "JobClient", Client)
    assert cli.main(["train", "--config", str(config), "--dry-run"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["backend"] == "mlx"
    assert report["profile"]["compaction_reserve_tokens"] == 12288


def test_remote_training_requires_studio_job_submission(tmp_path, monkeypatch):
    config = tmp_path / "run.toml"
    config.write_text('[compute]\nbackend="mlx"\nhost="studio"\n')
    import shellsmith.train.trainer
    monkeypatch.setattr(shellsmith.train.trainer, "run_train", lambda *a, **k: pytest.fail("local work"))
    with pytest.raises(SystemExit, match="job submit"):
        cli.main(["train", "--config", str(config)])


def test_studio_login_failure_is_reported_before_model_work(tmp_path, monkeypatch):
    config = tmp_path / "run.toml"
    config.write_text('[compute]\nbackend="mlx"\n')
    import shellsmith.train.preflight
    import shellsmith.train.trainer
    monkeypatch.setattr(shellsmith.train.preflight, "probe_document", lambda document: {
        "ready": False, "blockers": ["Authorize Codex on the Studio"]})
    monkeypatch.setattr(shellsmith.train.trainer, "run_train", lambda *a, **k: pytest.fail("model work"))
    with pytest.raises(SystemExit, match="Authorize Codex"):
        cli.main(["train", "--config", str(config)])


@pytest.mark.parametrize("flag,backend,reason", [
    ("--adapter-path", "tinker", "requires"),
    ("--sampler-path", "mlx", "Tinker reference"),
])
def test_campaign_rejects_cross_backend_artifacts_before_dependencies(tmp_path, monkeypatch, flag, backend, reason):
    config = tmp_path / "run.toml"
    config.write_text(f'[compute]\nbackend="{backend}"\n')
    import shellsmith.policy.deps
    monkeypatch.setattr(shellsmith.policy.deps, "load_tinker_deps", lambda: pytest.fail("dependency work"))
    with pytest.raises(SystemExit, match=reason):
        cli.main(["campaign", "--config", str(config), flag, "artifact"])


def test_mlx_episode_rejects_docker_before_session_creation(tmp_path, monkeypatch):
    import shellsmith.mlx
    monkeypatch.setattr(shellsmith.mlx, "create_session", lambda *a, **k: pytest.fail("model work"))
    with pytest.raises(SystemExit, match="protected host execution"):
        cli.main(["episode", "--player", "mlx", "--docker", "--log-dir", str(tmp_path / "run")])


def test_job_status_recovers_destination_after_lost_submission_response(tmp_path, monkeypatch, capsys):
    import shellsmith.jobs
    endpoint = Endpoint("original-studio", "/private/job-home", "/private/env/python")
    (tmp_path / "job-one.json").write_text(json.dumps({"endpoint": asdict(endpoint)}))
    class Client:
        def __init__(self, selected, cache_dir):
            assert selected == endpoint
        def status(self, job_id):
            assert job_id == "job-one"
            return {"status": "running"}
    monkeypatch.setattr(shellsmith.jobs, "JobClient", Client)
    assert cli.main(["job", "status", "job-one", "--cache-dir", str(tmp_path), "--home", endpoint.home]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "running"
    with pytest.raises(SystemExit, match="destination is pinned"):
        cli.main(["job", "status", "job-one", "--cache-dir", str(tmp_path), "--host", "another-studio"])


@pytest.mark.parametrize("command", ["train", "campaign"])
def test_job_submission_and_check_validate_the_selected_execution_mode(tmp_path, monkeypatch, command):
    import shellsmith.jobs
    config = tmp_path / "run.toml"
    config.write_text('[compute]\nbackend="mlx"\n' + ('[adaptive]\ninitial_attempts=16\nmaximum_attempts=24\nminimum_trainable=16\n' if command == "campaign" else ''))
    class Client:
        def __init__(self, selected, cache_dir):
            pass
        def submit(self, **kwargs):
            assert kwargs.get("command", "train") == command
            return {"job_id": "job-one", "status": "starting"}
        def check(self, config):
            assert config["compute"]["backend"] == "mlx"
            return {"ready": True}
    monkeypatch.setattr(shellsmith.jobs, "JobClient", Client)
    monkeypatch.setenv("SHELLSMITH_STUDIO_HOST", "user@studio-host")
    common = ["--config", str(config), "--cache-dir", str(tmp_path), "--command", command]
    assert cli.main(["job", "submit", *common, "--request-id", command]) == 0
    assert cli.main(["job", "check", *common]) == 0
