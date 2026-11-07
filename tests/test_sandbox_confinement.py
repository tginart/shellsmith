"""Docker must retain the Codex tool boundary and fail closed without it.

These tests verify the real configuration/command/preflight paths. They do not
claim to prove Linux namespace or filesystem enforcement without a Docker host.
"""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from shellsmith.episode.gm_process.codex import CodexWorldConfig, CodexWorldProcess, SANDBOX_PERMISSION_PROFILE
from shellsmith.paths import DEFAULT_WORLD, PROJECT_ROOT
from shellsmith.sandbox import sandbox_rollout
from shellsmith.sandbox.docker_launcher import DockerEpisodeConfig, container_environment


def _environment(monkeypatch, tmp_path) -> None:
    config = DockerEpisodeConfig(
        image="fixture", gm_runtime="codex", player_base_url="http://127.0.0.1:1/v1",
        player_model="fixture", player_provider="openai", player_api_key_env="TEST_PLAYER_KEY",
        smoke_env={"SHELLSMITH_WORLD_FILE": str(DEFAULT_WORLD),
                   "SHELLSMITH_OUTPUT_DIR": str(tmp_path / "run"),
                   "SHELLSMITH_CODEX_SANDBOX": "danger-full-access"},
    )
    for key, value in container_environment(config).items():
        if key.startswith("SHELLSMITH_"):
            monkeypatch.setenv(key, value)
    monkeypatch.setenv("TEST_PLAYER_KEY", "synthetic-secret")


def test_container_environment_cannot_disable_initial_or_resumed_confinement(tmp_path, monkeypatch) -> None:
    _environment(monkeypatch, tmp_path)
    args = sandbox_rollout.args_from_env()
    assert args.codex_sandbox == SANDBOX_PERMISSION_PROFILE
    process = CodexWorldProcess(CodexWorldConfig(
        executable=Path("codex"), model=args.gm_model, workspace_dir=tmp_path / "world",
        log_dir=tmp_path / "logs", bootstrap_response_schema_path=tmp_path / "schema.json",
        sandbox_mode=args.codex_sandbox,
    ))
    for resume in (False, True):
        command = process._base_command(resume=resume)
        assert 'default_permissions="game_master"' in command
        assert '\":minimal\"=\"read\"' in " ".join(command)
        assert 'shell_environment_policy.inherit="core"' in command
        assert "--ignore-user-config" in command
        assert "danger-full-access" not in " ".join(command)


@pytest.mark.parametrize("failure", ["denied", "missing", "timeout"])
def test_namespace_failure_stops_real_entry_before_any_model_call(tmp_path, monkeypatch, failure) -> None:
    _environment(monkeypatch, tmp_path)
    calls = []

    def unshare(command, **kwargs):
        calls.append(command)
        assert command == ["/usr/bin/unshare", "--user", "--map-root-user", "--mount", "--", "/bin/true"]
        assert kwargs["timeout"] == 10
        if failure == "missing":
            raise FileNotFoundError("unshare missing")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 10)
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="Operation not permitted")

    async def forbidden_model_call(args):
        pytest.fail("No model may run after failed sandbox preflight")

    monkeypatch.setattr(sandbox_rollout.subprocess, "run", unshare)
    monkeypatch.setattr(sandbox_rollout, "run_live_smoke", forbidden_model_call)
    summary = tmp_path / "summary.json"
    monkeypatch.setattr(sandbox_rollout, "SUMMARY_ARTIFACT", summary)
    with pytest.raises(RuntimeError, match="nested user namespaces"):
        sandbox_rollout.main()
    assert len(calls) == 1
    assert not summary.exists()


@pytest.mark.skipif(sys.platform != "darwin" or shutil.which("codex") is None, reason="requires the installed macOS Codex sandbox; Linux Docker is a separate boundary")
def test_real_codex_profile_denies_outside_credentials_and_symlink_escape(tmp_path) -> None:
    world = tmp_path / "world"
    world.mkdir()
    private = tmp_path / "synthetic-auth.json"
    private.write_text("synthetic-account-secret")
    assert subprocess.check_output(["/bin/cat", str(private)], text=True) == "synthetic-account-secret"
    link = world / "outside-link"
    link.symlink_to(private)
    process = CodexWorldProcess(CodexWorldConfig(
        executable=Path(shutil.which("codex")), model="unused", workspace_dir=world,
        log_dir=tmp_path / "logs", bootstrap_response_schema_path=world / "schema.json",
    ))
    base = [str(process.config.executable), *process._sandbox_args(), "sandbox", "-P", "game_master", "-C", str(world), "--"]
    allowed = subprocess.run(base + ["/bin/sh", "-c", "printf allowed > permitted.txt; cat permitted.txt"], capture_output=True, text=True, timeout=30)
    assert allowed.returncode == 0 and allowed.stdout == "allowed", allowed.stderr
    assert (world / "permitted.txt").read_text() == "allowed"
    for target in (private, link):
        denied = subprocess.run(base + ["/bin/cat", str(target)], capture_output=True, text=True, timeout=30)
        assert denied.returncode != 0
        assert "synthetic-account-secret" not in denied.stdout


def test_successful_namespace_screen_does_not_change_sandbox_policy(monkeypatch) -> None:
    calls = []
    def unshare(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    monkeypatch.setattr(sandbox_rollout.subprocess, "run", unshare)
    sandbox_rollout._require_codex_namespaces()
    assert len(calls) == 1
    assert "util-linux" in (PROJECT_ROOT / "docker/Dockerfile").read_text()


@pytest.mark.skipif(sys.platform != "darwin" or shutil.which("codex") is None, reason="requires installed Codex; this does not prove Linux Docker enforcement")
def test_real_no_model_filesystem_compatibility_probe() -> None:
    sandbox_rollout._require_codex_filesystem(shutil.which("codex"))


@pytest.mark.parametrize("failure", ["denied", "missing", "timeout"])
def test_filesystem_failure_stops_before_model_call(tmp_path, monkeypatch, failure) -> None:
    _environment(monkeypatch, tmp_path)
    calls = []

    def external_process(command, **kwargs):
        calls.append(command)
        if command[0] == "/usr/bin/unshare":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        assert "sandbox" in command and "game_master" in command
        if failure == "missing":
            raise FileNotFoundError("codex missing")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 30)
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="mount denied")

    async def forbidden_model_call(args):
        pytest.fail("No model may run after failed filesystem preflight")

    monkeypatch.setattr(sandbox_rollout.subprocess, "run", external_process)
    monkeypatch.setattr(sandbox_rollout, "run_live_smoke", forbidden_model_call)
    with pytest.raises(RuntimeError, match="filesystem preflight"):
        sandbox_rollout.main()
    assert len(calls) == 2


@pytest.mark.skipif(sys.platform != "darwin" or shutil.which("codex") is None, reason="requires installed macOS Codex sandbox; Linux Docker is a separate boundary")
def test_real_codex_profile_denies_sibling_player_credential_inspection(tmp_path) -> None:
    """Private child env is not protection from unconfined sibling inspection."""
    world = tmp_path / "world"
    world.mkdir()
    sentinel = "synthetic-private-player-credential"
    # A ready, nonsystem executable gives a valid positive control on macOS:
    # protected system executables can hide their environment even outside the
    # sandbox. No real provider credentials are given to this synthetic child.
    sibling = subprocess.Popen(
        [sys.executable, "-u", "-c", 'import sys; print("ready", flush=True); sys.stdin.read()'],
        env={"PATH": "/usr/bin:/bin", "SHELLSMITH_PLAYER_ENDPOINT_KEY": sentinel},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert sibling.stdout.readline() == "ready\n"
        probe = ["/bin/ps", "eww", "-p", str(sibling.pid)]
        positive = subprocess.run(probe, capture_output=True, text=True, check=True, timeout=10)
        assert sentinel in positive.stdout
        process = CodexWorldProcess(CodexWorldConfig(
            executable=Path(shutil.which("codex")), model="unused", workspace_dir=world,
            log_dir=tmp_path / "logs", bootstrap_response_schema_path=world / "schema.json",
        ))
        command = [str(process.config.executable), *process._sandbox_args(), "sandbox", "-P", "game_master", "-C", str(world), "--", *probe]
        denied = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert denied.returncode != 0
        assert sentinel not in denied.stdout + denied.stderr
    finally:
        sibling.communicate(timeout=5)
