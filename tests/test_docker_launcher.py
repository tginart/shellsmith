"""Docker launcher: auth sanitizing, mounts, secret handling, stale-image refusal."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from shellsmith.cli import build_parser
from shellsmith.sandbox import docker_launcher as launcher
from shellsmith.sandbox.docker_launcher import DockerEpisodeConfig


@pytest.fixture(autouse=True)
def local_docker_endpoint(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "unix:///synthetic-docker.sock")
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)


def _unpinned_arguments(command):
    if command[0] == "docker":
        assert command[1:3] == ["--host", "unix:///synthetic-docker.sock"]
        return ["docker", *command[3:]]
    return command


def _auth(**extra: object) -> bytes:
    return json.dumps({
        "auth_mode": "chatgpt",
        "OPENAI_API_KEY": "sk-should-be-dropped",
        "tokens": {"access_token": "a", "refresh_token": "r"},
        "last_refresh": "2026-10-03",
        **extra,
    }).encode()


def _config(**overrides: object) -> DockerEpisodeConfig:
    values = dict(
        image="shellsmith:abc",
        gm_runtime="codex",
        player_base_url="https://openrouter.ai/api/v1",
        player_model="qwen/qwen3.8-27b",
        player_provider="openrouter",
        player_api_key_env="TEST_PLAYER_KEY",
        smoke_env={"SHELLSMITH_GM_MODEL": "gpt-6.1-sol"},
    )
    values.update(overrides)
    return DockerEpisodeConfig(**values)  # type: ignore[arg-type]


def test_sanitize_keeps_only_chatgpt_oauth() -> None:
    cleaned = json.loads(launcher.sanitize_codex_auth_bytes(_auth()))
    assert cleaned == {
        "auth_mode": "chatgpt",
        "tokens": {"access_token": "a", "refresh_token": "r"},
        "last_refresh": "2026-10-03",
    }


@pytest.mark.parametrize(
    "raw",
    [b"not json", json.dumps({"auth_mode": "apikey"}).encode(),
     json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "a"}}).encode()],
)
def test_sanitize_rejects_non_oauth_or_incomplete(raw: bytes) -> None:
    with pytest.raises(ValueError):
        launcher.sanitize_codex_auth_bytes(raw)


def test_stage_episode_copies_world_and_sanitized_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = tmp_path / "w.md"
    world.write_text("# world\n")
    auth = tmp_path / "auth.json"
    auth.write_bytes(_auth())
    auth.chmod(0o600)
    monkeypatch.setenv(launcher.CODEX_AUTH_PATH_ENV, str(auth))
    monkeypatch.setattr(launcher.tempfile, "tempdir", str(tmp_path))
    with launcher.stage_episode(tmp_path / "run", world, gm_runtime="codex") as staged:
        assert staged["world_file"].read_text() == "# world\n"
        assert staged["artifacts"].is_dir()
        assert "sk-should-be-dropped" not in (staged["codex_home"] / "auth.json").read_text()
        assert not staged["codex_home"].is_relative_to(tmp_path / "run")
        assert (staged["codex_home"] / "auth.json").stat().st_mode & 0o777 == 0o600
        temporary_home = staged["codex_home"]
    assert not temporary_home.exists()
    assert not (tmp_path / "run/codex_home").exists()
    with pytest.raises(RuntimeError, match="not empty"):
        with launcher.stage_episode(tmp_path / "run", world, gm_runtime="codex"):
            pytest.fail("nonempty output must not be accepted")


def test_docker_command_forwards_secrets_by_name_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_PLAYER_KEY", "super-secret-value")
    staged = {"world_file": tmp_path / "w.md", "artifacts": tmp_path / "a", "codex_home": tmp_path / "c"}
    staged["seccomp_profile"] = tmp_path / "seccomp.json"
    command = launcher.docker_command(_config(), staged, name="shellsmith-x")
    joined = " ".join(command)
    assert "super-secret-value" not in joined
    assert command[command.index("TEST_PLAYER_KEY") - 1] == "-e"
    assert "--mount" not in command and "--volume" not in command
    assert "SHELLSMITH_GM_MODEL=gpt-6.1-sol" in command
    assert f"CODEX_HOME={launcher.CONTAINER_CODEX_HOME}" in command
    assert command[-3:] == ["python", "-m", "shellsmith.sandbox.sandbox_rollout"]


def test_missing_secret_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_PLAYER_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TEST_PLAYER_KEY"):
        launcher.secret_names(_config())


def test_stale_image_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        command = _unpinned_arguments(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout="oldtree\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="newtree\n", stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="Rebuild with docker/build.sh"):
        launcher.require_fresh_image("shellsmith:old")


def test_episode_cli_defaults_to_protected_host_with_docker_opt_in() -> None:
    args = build_parser().parse_args(["episode"])
    assert args.host is True
    assert build_parser().parse_args(["episode", "--docker"]).host is False
    assert args.gm_model == "gpt-6.1-sol"
    assert args.player_model == "qwen/qwen3.8-27b"
    assert args.world_file.name == "checkout_idempotency_001.md"


def _capture_engine(tmp_path, monkeypatch, *, outcome="success", label=None, observe=None):
    """Replace only external git/Docker processes, not launcher functions."""
    calls, homes = [], []
    tree = "a" * 40
    stopped = False

    def run(command, **kwargs):
        nonlocal stopped
        command = _unpinned_arguments(command)
        if observe is not None:
            observe(command, kwargs)
        calls.append(command)
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, stdout=tree if "rev-parse" in command else "", stderr="")
        if command[:3] == ["docker", "image", "inspect"]:
            identity = launcher.current_image_identity()
            return subprocess.CompletedProcess(command, 0, stdout=identity if label is None else label.replace("{current}", identity), stderr="")
        if command[:3] == ["docker", "rm", "--force"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:2] == ["docker", "create"]:
            assert "--mount" not in command and "--volume" not in command
            assert command[command.index("--user") + 1] == "0:0"
            return subprocess.CompletedProcess(command, 7 if outcome == "create-failure" else 0)
        if command[:2] == ["docker", "stop"]:
            stopped = True
            return subprocess.CompletedProcess(command, 0)
        if command[:2] == ["docker", "cp"]:
            assert "-a" not in command and "--archive" not in command
            if command[2].endswith(launcher.CONTAINER_ARTIFACTS + "/."):
                if outcome in {"timeout", "timeout-rotated", "interrupt", "failure"}:
                    assert stopped, "must stop execution before exporting failed output"
                if outcome in {"success", "rotated"}:
                    (Path(command[3]) / "smoke_summary.json").write_text(json.dumps({"final_reward": 0.75, "trainable": True}))
            elif command[2].endswith(":" + launcher.CONTAINER_CODEX_HOME + "/auth.json"):
                returned = json.loads((homes[-1] / "auth.json").read_text())
                if outcome in {"rotated", "timeout-rotated"}:
                    returned["tokens"].update(access_token="renewed-access", refresh_token="renewed-refresh")
                    returned["last_refresh"] = "2026-10-06"
                Path(command[3]).write_text(json.dumps(returned))
                Path(command[3]).chmod(0o600)
            elif command[2].endswith(":/logs/in-progress/."):
                assert stopped
                partial = Path(command[3]) / "stopped-episode"
                partial.mkdir()
                (partial / "episode.json").write_text('{"status":"running"}')
                (partial / "outside-link").symlink_to(tmp_path / "input-auth.json")
                scratch = partial / "world/runtime_scratch"
                scratch.mkdir(parents=True)
                (scratch / "excluded.txt").write_text("not archived")
            elif command[3].endswith(":" + launcher.CONTAINER_WORLD_FILE):
                assert Path(command[2]).is_file()
                assert Path(command[2]).stat().st_mode & 0o777 == 0o444
            else:
                home = Path(command[2]); homes.append(home)
                assert (home / "auth.json").is_file()
                assert command[3].endswith(":" + launcher.CONTAINER_CODEX_HOME)
            return subprocess.CompletedProcess(command, 0)
        assert command[:3] == ["docker", "start", "--attach"]
        if outcome in {"timeout", "timeout-rotated"}:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if outcome == "interrupt":
            raise KeyboardInterrupt()
        return subprocess.CompletedProcess(command, 7 if outcome == "failure" else 0)

    auth = tmp_path / "input-auth.json"
    auth.write_bytes(_auth())
    auth.chmod(0o600)
    monkeypatch.setenv(launcher.CODEX_AUTH_PATH_ENV, str(auth))
    monkeypatch.setenv("TEST_PLAYER_KEY", "test-player-secret")
    monkeypatch.setattr(launcher.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(launcher.subprocess, "run", run)
    return calls, homes


@pytest.mark.parametrize("outcome", ["success", "rotated", "timeout", "timeout-rotated", "interrupt", "failure", "missing-summary"])
def test_real_launcher_cleans_auth_and_stops_failed_containers(tmp_path, monkeypatch, outcome) -> None:
    calls, homes = _capture_engine(tmp_path, monkeypatch, outcome=outcome)
    world = tmp_path / "world.md"
    world.write_text("representative world")
    run = tmp_path / "nested" / "run with spaces, commas"
    if outcome in {"success", "rotated"}:
        summary = launcher.run_episode_in_docker(world, run, _config())
        assert summary == {"final_reward": 0.75, "trainable": True}
    else:
        error = subprocess.TimeoutExpired if outcome in {"timeout", "timeout-rotated"} else KeyboardInterrupt if outcome == "interrupt" else RuntimeError
        with pytest.raises(error):
            launcher.run_episode_in_docker(world, run, _config())
    assert homes and all(not home.exists() for home in homes)
    assert all(not home.is_relative_to(run) for home in homes)
    assert not list(run.rglob("auth.json"))
    launch = next(command for command in calls if command[:2] == ["docker", "create"])
    name = launch[launch.index("--name") + 1]
    import re
    assert re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", name)
    removed = ["docker", "rm", "--force", name]
    assert removed in calls
    updated_auth = json.loads((tmp_path / "input-auth.json").read_text())
    if outcome in {"rotated", "timeout-rotated"}:
        assert updated_auth["tokens"]["refresh_token"] == "renewed-refresh"
        assert updated_auth["OPENAI_API_KEY"] == "sk-should-be-dropped"
        returned_copy = next(command for command in calls if command[:2] == ["docker", "cp"] and command[2].endswith(launcher.CONTAINER_CODEX_HOME + "/auth.json"))
        assert calls.index(returned_copy) < calls.index(removed)
    else:
        assert updated_auth == json.loads(_auth())
    if outcome not in {"success", "rotated"}:
        stopped = ["docker", "stop", "--time", "30", name]
        recovery = next(command for command in calls if command[:2] == ["docker", "cp"] and command[2].endswith(":/logs/in-progress/."))
        assert calls.index(stopped) < calls.index(recovery) < calls.index(removed)
        partial = run / "artifacts/partial/stopped-episode"
        assert json.loads((partial / "episode.json").read_text()) == {"status": "running"}
        assert not (partial / "outside-link").exists()
        assert not (partial / "world/runtime_scratch").exists()
        assert json.loads((partial / "archive_sanitization.json").read_text())["skipped_entries"]


@pytest.mark.parametrize("previous_key", [None, "older-ambient-player-key"], ids=["unset", "prior-key"])
@pytest.mark.parametrize("outcome", ["success", "create-failure", "failure", "timeout", "interrupt"])
def test_in_memory_player_key_reaches_only_create_and_survives_cleanup(
    tmp_path, monkeypatch, capsys, previous_key, outcome,
) -> None:
    from shellsmith.config import PLAYER_ENDPOINT_KEY_ENV

    private_key = "in-memory-create-only-player-secret"
    observations = []
    parent_environment = {}

    def observe(command, kwargs):
        changed = {name for name in set(parent_environment) | set(os.environ)
                   if parent_environment.get(name) != os.environ.get(name)}
        assert not changed, f"Docker launcher changed parent environment: {sorted(changed)}"
        assert private_key not in os.environ.values()
        assert private_key not in " ".join(command)
        actual = kwargs.get("env", os.environ)
        expected = dict(parent_environment)
        if command[0] == "docker":
            expected.pop("DOCKER_HOST", None)
            expected.pop("DOCKER_CONTEXT", None)
        create = command[:2] == ["docker", "create"]
        if create:
            expected[PLAYER_ENDPOINT_KEY_ENV] = private_key
        differences = {name for name in set(expected) | set(actual)
                       if expected.get(name) != actual.get(name)}
        assert not differences, f"Unexpected child environment: {sorted(differences)}"
        observations.append((command[:2], actual.get(PLAYER_ENDPOINT_KEY_ENV)))

    calls, homes = _capture_engine(tmp_path, monkeypatch, outcome=outcome, observe=observe)
    if previous_key is None:
        monkeypatch.delenv(PLAYER_ENDPOINT_KEY_ENV, raising=False)
    else:
        monkeypatch.setenv(PLAYER_ENDPOINT_KEY_ENV, previous_key)
    monkeypatch.setenv("CREDENTIAL_TEST_OTHER_SETTING", "keep-this-setting")
    parent_environment.update(os.environ)
    config = _config(player_api_key_env=PLAYER_ENDPOINT_KEY_ENV, player_api_key=private_key)
    assert config.player_api_key == private_key
    assert private_key not in repr(config)
    assert launcher.secret_names(config) == [PLAYER_ENDPOINT_KEY_ENV]
    world = tmp_path / "world.md"
    world.write_text("credential fixture world")
    output = tmp_path / "run"
    if outcome == "success":
        assert launcher.run_episode_in_docker(world, output, config)["trainable"] is True
    else:
        error = (subprocess.TimeoutExpired if outcome == "timeout" else
                 KeyboardInterrupt if outcome == "interrupt" else RuntimeError)
        with pytest.raises(error):
            launcher.run_episode_in_docker(world, output, config)
    assert os.environ.get(PLAYER_ENDPOINT_KEY_ENV) == previous_key
    assert len([row for row in observations if row[0] == ["docker", "create"]]) == 1
    for command, key in observations:
        assert key == (private_key if command == ["docker", "create"] else previous_key)
    create = next(command for command in calls if command[:2] == ["docker", "create"])
    assert create[create.index(PLAYER_ENDPOINT_KEY_ENV) - 1] == "-e"
    name = create[create.index("--name") + 1]
    assert calls[-1] == ["docker", "rm", "--force", name]
    assert all(not home.exists() for home in homes)
    assert not list(tmp_path.glob("shellsmith-codex-*"))
    assert not list(tmp_path.glob("shellsmith-seccomp-*"))
    assert not list(output.rglob("auth.json"))
    for path in output.rglob("*"):
        if path.is_file():
            leaked = private_key.encode() in path.read_bytes()
            assert not leaked, f"Player credential persisted in {path}"
    captured = capsys.readouterr()
    assert private_key not in captured.out + captured.err


@pytest.mark.parametrize("key", ["", " \t\n"])
def test_empty_in_memory_key_does_not_fall_back_to_ambient(monkeypatch, key) -> None:
    monkeypatch.setenv("TEST_PLAYER_KEY", "valid-ambient-key")
    with pytest.raises(RuntimeError, match="Required credentials"):
        launcher.secret_names(_config(player_api_key=key))
    assert os.environ["TEST_PLAYER_KEY"] == "valid-ambient-key"


def test_auth_is_removed_when_command_validation_fails(tmp_path, monkeypatch) -> None:
    _capture_engine(tmp_path, monkeypatch)
    monkeypatch.delenv("TEST_PLAYER_KEY")
    world = tmp_path / "world.md"
    world.write_text("world")
    with pytest.raises(RuntimeError, match="Required credentials"):
        launcher.run_episode_in_docker(world, tmp_path / "run", _config())
    assert not list(tmp_path.glob("shellsmith-codex-*"))
    assert not list((tmp_path / "run").rglob("auth.json"))


@pytest.mark.parametrize("allow,label,accepted", [
    (False, "{current}", True),
    (False, "{current}+pi=1.2.3+codex=4.5.6+platform=linux/amd64", False),
    (True, "{current}+pi=1.2.3+codex=4.5.6+platform=linux/amd64", True),
    (True, "b" * 40 + "+pi=1.2.3+codex=4.5.6+platform=linux/amd64", False),
    (True, "{current}+unrecognized=1", False),
])
def test_explicit_overrides_still_require_current_source(tmp_path, monkeypatch, allow, label, accepted) -> None:
    _capture_engine(tmp_path, monkeypatch, label=label)
    if accepted:
        launcher.require_fresh_image("custom:image", allow_overrides=allow)
    else:
        with pytest.raises(RuntimeError, match="Rebuild"):
            launcher.require_fresh_image("custom:image", allow_overrides=allow)


def test_image_identity_covers_dockerfile_and_refuses_dirty_recipe(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "docker").mkdir()
    (repo / "src/app.py").write_text("version = 1\n")
    recipe = repo / "docker/Dockerfile"
    recipe.write_text("FROM scratch\n")
    real_run = subprocess.run
    real_run(["git", "init", "-q", str(repo)], check=True)

    def commit():
        real_run(["git", "-C", str(repo), "add", "src", "docker"], check=True)
        real_run(["git", "-C", str(repo), "-c", "user.name=Tests", "-c", "user.email=tests@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"], check=True)

    commit()
    monkeypatch.setattr(launcher, "PROJECT_ROOT", repo)
    first = launcher.current_image_identity()

    def inspect(command, **kwargs):
        command = _unpinned_arguments(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=first, stderr="")
        return real_run(command, **kwargs)

    monkeypatch.setattr(launcher.subprocess, "run", inspect)
    launcher.require_fresh_image("fixture")
    recipe.write_text("FROM scratch\nENV CHANGED=1\n")
    with pytest.raises(RuntimeError, match="uncommitted"):
        launcher.require_fresh_image("fixture")
    commit()
    second = launcher.current_image_identity()
    assert second != first
    with pytest.raises(RuntimeError, match="Rebuild"):
        launcher.require_fresh_image("fixture")
    (repo / "src/app.py").write_text("version = 2\n")
    commit()
    assert launcher.current_image_identity() != second


def test_container_names_are_unique_for_identical_basenames(tmp_path) -> None:
    first = launcher._container_name(tmp_path / "a" / "run with spaces")
    second = launcher._container_name(tmp_path / "b" / "run with spaces")
    assert first != second
    assert " " not in first + second


def test_docker_preserves_daemon_uid_namespaces_and_uses_copied_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TEST_PLAYER_KEY", "secret")
    staged = {"world_file": tmp_path / "w", "artifacts": tmp_path / "a", "codex_home": tmp_path / "c"}
    staged["seccomp_profile"] = tmp_path / "seccomp.json"
    command = launcher.docker_command(_config(), staged, name="test")
    assert command[command.index("--user") + 1] == "0:0"
    joined = " ".join(command)
    assert f"src={staged['artifacts']}" not in joined
    assert f"src={staged['codex_home']}" not in joined
    assert "HOME=/tmp/shellsmith-home" in command
    for unsafe in ("--userns", "--privileged", "--cap-add", "seccomp=unconfined", "apparmor=unconfined"):
        assert unsafe not in joined


def test_pi_docker_rejected_before_credentials_or_external_execution(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "gm-secret")
    monkeypatch.setenv("TEST_PLAYER_KEY", "player-secret")
    with pytest.raises(ValueError, match="Pi GM execution is disabled"):
        launcher.secret_names(_config(gm_runtime="pi"))
    with pytest.raises(ValueError, match="Pi GM execution is disabled"):
        launcher.run_episode_in_docker(tmp_path / "nonexistent.md", tmp_path / "run", _config(gm_runtime="pi"))
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("backend", ["--host", "--docker"])
def test_claude_is_rejected_before_episode_launch(backend, capsys) -> None:
    from shellsmith.cli import main
    with pytest.raises(SystemExit) as error:
        main(["episode", backend, "--gm-runtime", "claude", "--gm-model", "sonnet"])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "only the Codex GM is supported" in captured.err
    assert not captured.out


def test_host_entry_forwards_chat_only_preflight_option(tmp_path, monkeypatch) -> None:
    from shellsmith import cli
    from shellsmith.episode import smoke
    observed = []

    async def external_episode(args, *, player_api_key=None):
        assert player_api_key is None
        observed.append(args)
        return tmp_path, {"final_reward": 0.4}

    monkeypatch.setattr(smoke, "run_live_smoke", external_episode)
    args = build_parser().parse_args(["episode", "--host", "--skip-player-models-preflight"])
    assert cli._run_host(args, tmp_path) == {"final_reward": 0.4}
    assert observed[0].skip_player_models_preflight is True


def test_default_log_directories_do_not_collide_within_one_second() -> None:
    from shellsmith.cli import _resolve_log_dir
    args = build_parser().parse_args(["episode"])
    assert _resolve_log_dir(args) != _resolve_log_dir(args)


def test_missing_sdk_recovery_names_the_real_npm_package(tmp_path, monkeypatch) -> None:
    from shellsmith.episode import smoke
    monkeypatch.setattr(smoke, "PINNED_PI_SDK", tmp_path / "missing.js")
    with pytest.raises(RuntimeError) as error:
        smoke._resolve_pi_sdk(smoke._parser().parse_args([]))
    assert str(smoke.VAT_RUNNER) in str(error.value)
    assert (smoke.VAT_RUNNER / "package.json").is_file()
    assert "node_modules" not in str(error.value)


@pytest.mark.parametrize("options", [["--image", "custom:tag"], ["--docker-timeout-seconds", "80000"]])
def test_docker_only_options_cannot_silently_run_on_host(options) -> None:
    from shellsmith.cli import main
    with pytest.raises(SystemExit, match="require --docker"):
        main(["episode", *options])


def test_outer_deadline_covers_player_and_six_gm_turns() -> None:
    from shellsmith.episode.smoke import _parser
    defaults = _parser().parse_args([])
    expected = defaults.player_prompt_timeout_seconds + 6 * defaults.timeout_seconds + 300
    assert launcher.episode_timeout_seconds(_config()) == expected > 12 * 3600
    custom = _config(smoke_env={"SHELLSMITH_TIMEOUT_SECONDS": "2000", "SHELLSMITH_PLAYER_PROMPT_TIMEOUT_SECONDS": "80000"})
    assert launcher.episode_timeout_seconds(custom) == 92300
    assert launcher.episode_timeout_seconds(_config(timeout_seconds=100000)) == 100000


@pytest.mark.parametrize("timeout", [14400, -1, float("nan"), float("inf")])
def test_outer_deadline_cannot_silently_shorten_supported_trajectory(timeout) -> None:
    with pytest.raises(ValueError, match="Docker timeout"):
        launcher.episode_timeout_seconds(_config(timeout_seconds=timeout))


def test_durable_container_staging_survives_until_archival(tmp_path) -> None:
    from shellsmith.episode.smoke import _create_staging_root
    parent = tmp_path / "container/logs/in-progress"
    staging = _create_staging_root("example", parent=parent)
    assert staging.parent == parent
    assert staging.is_dir()
    assert staging.stat().st_mode & 0o777 == 0o700


def test_sigterm_handler_is_restored_after_unwinding() -> None:
    import signal
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(ValueError, match="unwind"):
        with launcher._sigterm_cleanup():
            assert signal.getsignal(signal.SIGTERM) != previous
            raise ValueError("unwind")
    assert signal.getsignal(signal.SIGTERM) == previous


def test_docker_worker_thread_is_rejected_before_credentials(tmp_path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(launcher.run_episode_in_docker, tmp_path / "world", tmp_path / "run", _config())
        with pytest.raises(RuntimeError, match="main-thread worker process"):
            future.result(timeout=5)
    assert not (tmp_path / "run").exists()


def test_seccomp_policy_is_pinned_and_frozen_for_the_docker_client(tmp_path) -> None:
    import hashlib
    policy = launcher._stage_seccomp_policy(tmp_path)
    assert policy.is_file() and not policy.is_symlink()
    assert policy.stat().st_mode & 0o777 == 0o444
    assert hashlib.sha256(policy.read_bytes()).hexdigest() == launcher.SECCOMP_POLICY_SHA256
    assert json.loads(policy.read_text())["defaultAction"] == "SCMP_ACT_ERRNO"


@pytest.mark.parametrize("changed", ["symlink", "bytes"])
def test_unreviewed_or_linked_seccomp_policy_cannot_launch(tmp_path, monkeypatch, changed) -> None:
    repo = tmp_path / "repo"
    (repo / "docker").mkdir(parents=True)
    target = repo / "docker/seccomp-codex.json"
    if changed == "symlink":
        target.symlink_to(launcher.PROJECT_ROOT / "docker/seccomp-codex.json")
    else:
        target.write_text('{"defaultAction":"SCMP_ACT_ALLOW"}')
    monkeypatch.setattr(launcher, "PROJECT_ROOT", repo)
    with pytest.raises((RuntimeError, OSError)):
        launcher._stage_seccomp_policy(tmp_path)
    assert not (tmp_path / "seccomp.json").exists()
