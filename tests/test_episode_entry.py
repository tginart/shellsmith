"""Repeat the real episode CLI against a captured Docker transport.

The Docker daemon is external; all CLI, staging, credential, mount, freshness,
artifact-reading, and in-container argument code is the shipped implementation.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from shellsmith.paths import DEFAULT_WORLD, PROJECT_ROOT


DOCKER_TRANSPORT = r'''
import csv, hashlib, json, os, pathlib, re, signal, sys, time
args = sys.argv[1:]
if args[:1] == ['--host']:
    assert args[1] == 'unix:///synthetic-docker.sock'
    assert 'DOCKER_HOST' not in os.environ and 'DOCKER_CONTEXT' not in os.environ
    args = args[2:]
else:
    assert args[0] == 'build', 'daemon operations must use the pinned local host'
if os.environ.get('TEST_ACTIONS'):
    with pathlib.Path(os.environ['TEST_ACTIONS']).open('a') as output:
        output.write(json.dumps(args) + '\n')
if args[0] == 'build':
    key, value = args[args.index('--label')+1].split('=',1)
    assert key == 'shellsmith.image_identity'
    pathlib.Path(os.environ['TEST_IMAGE_LABEL']).write_text(value)
    raise SystemExit(0)
if args[:2] == ['image', 'inspect']:
    if os.environ.get('TEST_HANG_IMAGE_INSPECT'):
        signal.pause()
    print(pathlib.Path(os.environ['TEST_IMAGE_LABEL']).read_text())
    raise SystemExit(0)
state_path = pathlib.Path(os.environ['TEST_STATE'])
state = json.loads(state_path.read_text()) if state_path.exists() else {}
if args[0] == 'create':
    name = args[args.index('--name') + 1]
    assert name not in state
    assert re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]+', name)
    mounts, settings, secrets = {}, {}, []
    for i, value in enumerate(args):
        if value == '--mount':
            fields = next(csv.reader([args[i+1]]))
            assert 'readonly' in fields
            row = dict(field.split('=',1) for field in fields if '=' in field)
            mounts[row['dst']] = pathlib.Path(row['src'])
        if value == '-e':
            field = args[i+1]
            if '=' in field:
                key, value = field.split('=',1); settings[key] = value
            else:
                secrets.append(field)
                assert os.environ.get(field)
                assert os.environ[field] not in args
    assert not mounts and '--volume' not in args
    security = [args[i+1] for i, value in enumerate(args) if value == '--security-opt']
    assert 'no-new-privileges=true' in security
    policy = pathlib.Path(next(value.split('=',1)[1] for value in security if value.startswith('seccomp=')))
    assert policy.is_absolute() and policy.is_file() and not policy.is_symlink()
    assert json.loads(policy.read_text())['defaultAction'] == 'SCMP_ACT_ERRNO'
    assert policy.read_bytes() == (pathlib.Path(os.environ['TEST_REPO'])/'docker/seccomp-codex.json').read_bytes()
    assert args[args.index('--user')+1] == '0:0'
    state[name] = {'name':name,'auth_home':None,'settings':settings,'secrets':secrets,'world_sha256':None}
elif args[0] == 'cp':
    assert '-a' not in args and '--archive' not in args
    source, destination = args[1:]
    if source.endswith(':/logs/artifacts/.'):
        name = source.split(':',1)[0]
        record = state[name]
        assert record.pop('started') is True
        if not os.environ.get('TEST_HANG_ON_START'):
            (pathlib.Path(destination)/'smoke_summary.json').write_text(os.environ['TEST_DOCKER_SUMMARY'])
        with pathlib.Path(os.environ['TEST_CAPTURE']).open('a') as output:
            output.write(json.dumps(record)+'\n')
    elif source.endswith(':/tmp/shellsmith-codex/auth.json'):
        name = source.split(':',1)[0]
        auth = json.loads((pathlib.Path(state[name]['auth_home'])/'auth.json').read_text())
        if os.environ.get('TEST_ROTATE_AUTH'):
            auth['tokens'].update(access_token='rotated-access', refresh_token='rotated-refresh')
            auth['last_refresh'] = '2026-10-06'
        if os.environ.get('TEST_SIGNAL_AT_AUTH'):
            pathlib.Path(os.environ['TEST_READY']).write_text('auth-return')
            time.sleep(0.3)
        pathlib.Path(destination).write_text(json.dumps(auth))
        pathlib.Path(destination).chmod(0o600)
    elif source.endswith(':/logs/in-progress/.'):
        partial = pathlib.Path(destination)/'stopped-episode'
        partial.mkdir()
        (partial/'episode.json').write_text('{"status":"running"}')
    else:
        name, target = destination.split(':',1)
        if target == '/opt/shellsmith/world_file.md':
            world = pathlib.Path(source).read_bytes()
            assert world == pathlib.Path(os.environ['TEST_WORLD']).read_bytes()
            assert pathlib.Path(source).stat().st_mode & 0o777 == 0o444
            state[name]['world_sha256'] = hashlib.sha256(world).hexdigest()
        else:
            assert target == '/tmp/shellsmith-codex'
            home = pathlib.Path(source)
            auth = json.loads((home/'auth.json').read_text())
            assert 'OPENAI_API_KEY' not in auth
            assert auth['tokens']['access_token'] in ({'synthetic-access', 'rotated-access'} if os.environ.get('TEST_ROTATE_AUTH') else {'synthetic-access'})
            state[name]['auth_home'] = str(home)
elif args[:2] == ['start','--attach']:
    record = state[args[2]]
    assert record['world_sha256'] is not None
    if record['settings']['SHELLSMITH_GM_RUNTIME'] == 'codex':
        assert record['auth_home'] is not None
    record['started'] = True
    if os.environ.get('TEST_HANG_ON_START'):
        state_path.write_text(json.dumps(state))
        pathlib.Path(os.environ['TEST_READY']).write_text('attached')
        signal.pause()
elif args[0] == 'stop':
    assert state[args[-1]]['started'] is True
    state[args[-1]]['stopped'] = True
elif args[:2] == ['rm','--force']:
    state.pop(args[2], None)
else:
    raise AssertionError(args)
state_path.write_text(json.dumps(state))
'''


def _entry_fixture(tmp_path):
    repo = tmp_path / "checkout"
    # An isolated committed checkout keeps the real freshness guard enabled,
    # even when a developer runs this regression before committing a fix.
    shutil.copytree(PROJECT_ROOT / "src", repo / "src", ignore=shutil.ignore_patterns("node_modules", "__pycache__"))
    shutil.copytree(PROJECT_ROOT / "docker", repo / "docker")
    shutil.copyfile(PROJECT_ROOT / ".gitignore", repo / ".gitignore")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "src", "docker", ".gitignore"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Shellsmith Tests", "-c", "user.email=tests@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "production source fixture"], check=True)
    tools = tmp_path / "bin"
    tools.mkdir()
    docker = tools / "docker"
    docker.write_text(f"#!{sys.executable}\n" + DOCKER_TRANSPORT)
    docker.chmod(0o755)
    auth = tmp_path / "source-auth.json"
    auth.write_text(json.dumps({"auth_mode": "chatgpt", "OPENAI_API_KEY": "discard-me", "tokens": {"access_token": "synthetic-access", "refresh_token": "synthetic-refresh"}}))
    auth.chmod(0o600)
    capture = tmp_path / "transport.jsonl"
    temp = tmp_path / "temp"
    temp.mkdir()
    summary = {"final_reward": 0.75, "trainable": True, "completion_percentage": 90, "tool_turn_count": 2, "integrity_issues": []}
    env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"],
               PYTHONPATH=str(repo / "src"), SHELLSMITH_ROOT=str(repo), TMPDIR=str(temp),
               SHELLSMITH_CODEX_AUTH_PATH=str(auth), TEST_PLAYER_KEY="synthetic-player",
               OPENROUTER_API_KEY="synthetic-gm", TEST_REPO=str(repo), TEST_WORLD=str(DEFAULT_WORLD),
               TEST_CAPTURE=str(capture), TEST_DOCKER_SUMMARY=json.dumps(summary),
               TEST_IMAGE_LABEL=str(tmp_path / "image-label.txt"), TEST_STATE=str(tmp_path / "containers.json"))
    # Exercise the default image identity independently of the developer's
    # shell/App build overrides; override tests opt in after this fixture.
    for key in ("PI_CODING_AGENT_VERSION", "CODEX_VERSION", "DOCKER_PLATFORM", "IMAGE_TAG"):
        env.pop(key, None)
    env["DOCKER_HOST"] = "unix:///synthetic-docker.sock"
    env.pop("DOCKER_CONTEXT", None)
    # Drive the real label producer too; the transport only records its output.
    build = subprocess.run(["bash", str(repo / "docker/build.sh")], env=env, text=True, capture_output=True)
    assert build.returncode == 0, build.stderr
    entry = Path(sys.executable).parent / "shellsmith"
    command = [str(entry)] if entry.is_file() else [sys.executable, "-m", "shellsmith.cli"]
    return command, env, capture, temp, summary


@pytest.mark.parametrize("runtime", ["codex"])
@pytest.mark.parametrize("rotate", [False, True])
def test_episode_cli_repeatedly_consumes_real_staged_output(tmp_path, monkeypatch, runtime, rotate) -> None:
    command, env, capture, temp, summary = _entry_fixture(tmp_path)
    if rotate:
        env["TEST_ROTATE_AUTH"] = "1"
    for attempt in range(2):
        output = tmp_path / str(attempt) / "run with spaces, commas"
        result = subprocess.run(command + ["episode", "--docker", "--world-file", str(DEFAULT_WORLD), "--log-dir", str(output),
            "--gm-runtime", runtime, "--timeout-seconds", "2400", "--player-prompt-timeout-seconds", "60000", "--docker-timeout-seconds", "80000", "--player-api-key-env", "TEST_PLAYER_KEY", "--skip-player-models-preflight"],
            env=env, text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr
        actual = json.loads(result.stdout[result.stdout.index("{"):])
        assert actual == summary
        assert json.loads((output / "artifacts/smoke_summary.json").read_text()) == summary
        assert not list(output.rglob("auth.json"))
        host_auth = json.loads(Path(env["SHELLSMITH_CODEX_AUTH_PATH"]).read_text())
        assert host_auth["OPENAI_API_KEY"] == "discard-me"
        assert host_auth["tokens"]["refresh_token"] == ("rotated-refresh" if rotate else "synthetic-refresh")
        assert not list(temp.glob("shellsmith-codex-*"))
        assert json.loads((tmp_path / "containers.json").read_text()) == {}
        print(f"runtime={runtime} launch={attempt}: {json.dumps(actual, sort_keys=True)}")
    records = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(records) == 2 and records[0]["name"] != records[1]["name"]
    assert records[0]["world_sha256"] == records[1]["world_sha256"]
    for record in records:
        if record["auth_home"]:
            assert not Path(record["auth_home"]).exists()
        assert record["settings"]["SHELLSMITH_SKIP_PLAYER_MODELS_PREFLIGHT"] == "1"
        assert record["secrets"] == ["TEST_PLAYER_KEY"]
        assert record["settings"]["SHELLSMITH_TIMEOUT_SECONDS"] == "2400.0"
        assert record["settings"]["SHELLSMITH_PLAYER_PROMPT_TIMEOUT_SECONDS"] == "60000.0"
    # Consume the actual environment produced by the outer CLI with the real
    # in-container parser, not a duplicate argument mapping in the test.
    from shellsmith.sandbox.sandbox_rollout import args_from_env
    for key, value in records[-1]["settings"].items():
        if key.startswith("SHELLSMITH_"):
            monkeypatch.setenv(key, value)
    monkeypatch.setenv("TEST_PLAYER_KEY", "synthetic-player")
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-gm")
    args = args_from_env()
    assert args.skip_player_models_preflight is True
    assert args.gm_runtime == runtime
    assert args.gm_provider == "codex-cli"
    assert args.timeout_seconds == 2400
    assert args.player_prompt_timeout_seconds == 60000
    assert args.staging_parent == Path("/logs/in-progress")


@pytest.mark.skipif(os.name != "posix", reason="local POSIX Docker transport timeout")
def test_image_inspect_hang_is_bounded_before_credentials_are_staged(tmp_path) -> None:
    _command, env, _capture, temp, _summary = _entry_fixture(tmp_path)
    env["TEST_HANG_IMAGE_INSPECT"] = "1"
    output = tmp_path / "must-not-stage"
    # Shorten only the external I/O deadline, not the launcher or client under
    # test. This drives their real subprocess timeout and actual CLI handler.
    bootstrap = "from shellsmith.sandbox import docker_launcher; docker_launcher.IMAGE_INSPECT_TIMEOUT_SECONDS = 0.1; from shellsmith.cli import main; main()"
    result = subprocess.run([sys.executable, "-c", bootstrap, "episode", "--docker",
        "--world-file", str(DEFAULT_WORLD), "--log-dir", str(output)],
        env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "Docker image inspection timed out after 0.1s" in result.stderr
    assert "Check or restart the selected local Docker daemon" in result.stderr
    assert not output.exists()
    assert not list(temp.glob("shellsmith-codex-*"))


def test_real_docker_entry_rejects_remote_before_staging_or_daemon_access(tmp_path) -> None:
    command, env, _capture, temp, _summary = _entry_fixture(tmp_path)
    actions = tmp_path / "actions.jsonl"
    env.update(DOCKER_HOST="ssh://remote.invalid", TEST_ACTIONS=str(actions))
    output = tmp_path / "must-not-exist"
    result = subprocess.run(command + ["episode", "--docker", "--world-file", str(DEFAULT_WORLD),
        "--log-dir", str(output), "--player-api-key-env", "TEST_PLAYER_KEY"],
        env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "Remote Docker endpoints are unsupported" in result.stderr
    assert not output.exists()
    assert not actions.exists()
    assert not list(temp.rglob("auth.json"))


@pytest.mark.skipif(os.name != "posix", reason="real POSIX SIGTERM lifecycle test")
@pytest.mark.parametrize("stage", ["episode", "auth-return"])
def test_real_sigterm_unwinds_docker_stop_export_remove_and_secret_cleanup(tmp_path, stage) -> None:
    import signal
    import time
    command, env, capture, temp, _summary = _entry_fixture(tmp_path)
    ready, actions = tmp_path / "ready", tmp_path / "actions.jsonl"
    env.update(TEST_READY=str(ready), TEST_ACTIONS=str(actions), TEST_ROTATE_AUTH="1")
    env["TEST_HANG_ON_START" if stage == "episode" else "TEST_SIGNAL_AT_AUTH"] = "1"
    output = tmp_path / "interrupted"
    process = subprocess.Popen(command + ["episode", "--docker", "--world-file", str(DEFAULT_WORLD),
        "--log-dir", str(output), "--player-api-key-env", "TEST_PLAYER_KEY"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), process.communicate(timeout=2)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 128 + signal.SIGTERM, (stdout, stderr)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    assert json.loads((tmp_path / "containers.json").read_text()) == {}
    assert not [path for path in temp.iterdir() if path.name.startswith(("shellsmith-codex-", "shellsmith-seccomp-", "shellsmith-recovery-"))], "temporary credentials and policy must be removed"
    assert not list(temp.rglob("auth.json"))
    assert not list(output.rglob("auth.json"))
    calls = [json.loads(line) for line in actions.read_text().splitlines()]
    removed = next(i for i, args in enumerate(calls) if args[:2] == ["rm", "--force"])
    auth_returned = next(i for i, args in enumerate(calls) if args[0] == "cp" and args[1].endswith(":/tmp/shellsmith-codex/auth.json"))
    assert auth_returned < removed
    if stage == "episode":
        assert json.loads((output / "artifacts/partial/stopped-episode/episode.json").read_text()) == {"status": "running"}
        stopped = next(i for i, args in enumerate(calls) if args[0] == "stop")
        recovered = next(i for i, args in enumerate(calls) if args[0] == "cp" and args[1].endswith(":/logs/in-progress/."))
        assert stopped < recovered < auth_returned
    else:
        assert (output / "artifacts/smoke_summary.json").is_file()
    host_auth = json.loads(Path(env["SHELLSMITH_CODEX_AUTH_PATH"]).read_text())
    assert host_auth["tokens"]["refresh_token"] == "rotated-refresh"
    assert host_auth["OPENAI_API_KEY"] == "discard-me"


def test_entry_fixture_isolates_ambient_build_overrides(tmp_path, monkeypatch):
    keys = ("PI_CODING_AGENT_VERSION", "CODEX_VERSION", "DOCKER_PLATFORM", "IMAGE_TAG")
    for key in keys:
        monkeypatch.setenv(key, "ambient-override")
    _command, env, _capture, _temp, _summary = _entry_fixture(tmp_path)
    assert all(key not in env for key in keys)
    label = Path(env["TEST_IMAGE_LABEL"]).read_text().strip()
    assert "+pi=" not in label and "+codex=" not in label and "+platform=" not in label
