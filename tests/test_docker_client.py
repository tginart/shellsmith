"""Exercise the real resolver/run methods through a Docker executable on PATH."""

from __future__ import annotations

from dataclasses import dataclass, FrozenInstanceError
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import traceback

import pytest

from shellsmith.sandbox.docker_client import DockerClient


LOCAL_HOST = "unix:///local/docker.sock"
API_KEY = "SHELLSMITH_ENDPOINT_TEST_API_KEY"
OAUTH_TOKEN = "SHELLSMITH_ENDPOINT_TEST_OAUTH_TOKEN"
TRACKED_ENV = (
    "DOCKER_CONTEXT", "DOCKER_HOST", "HOME", "DOCKER_CONFIG", "CODEX_HOME",
    "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH",
    "DOCKER_API_VERSION", "TMPDIR", "PATH", API_KEY, OAUTH_TOKEN,
    "SHELLSMITH_ENDPOINT_TEST_EXTRA",
)


@dataclass
class DockerTransport:
    metadata: Path
    log: Path

    def configure(self, **changes: object) -> None:
        state = json.loads(self.metadata.read_text())
        state.update(changes)
        self.metadata.write_text(json.dumps(state))

    @property
    def calls(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]


@pytest.fixture
def docker_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DockerTransport:
    """Replace the external transport only; never mock resolve/run/subprocess.

    The executable reads synthetic endpoint metadata, not Docker configuration.
    No HOME, credential/configuration home, or cache directory is redirected.
    Only explicitly named synthetic secrets are recorded, never the whole env.
    """
    if os.name == "nt":
        pytest.skip("The external executable fixture requires POSIX shebang support.")
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    transport = DockerTransport(tmp_path / "endpoint-metadata.json", tmp_path / "docker-calls.jsonl")
    transport.metadata.write_text(json.dumps({
        "current": "desktop-linux",
        "contexts": {
            "desktop-linux": LOCAL_HOST,
            "default": "unix:///var/run/docker.sock",
            "remote": "ssh://remote.example.invalid",
        },
    }))
    executable = binary_dir / "docker"
    executable.write_text(
        f"#!{sys.executable}\nTRACKED_ENV = {TRACKED_ENV!r}\n"
        + textwrap.dedent(r'''
            import json
            import os
            from pathlib import Path
            import sys

            args = sys.argv[1:]
            metadata_path = Path(os.environ["SHELLSMITH_TEST_DOCKER_METADATA"])
            state = json.loads(metadata_path.read_text())
            metadata_call = args[:1] == ["context"]
            entry = {
                "args": args,
                "kind": "metadata" if metadata_call else "daemon",
                "env": {key: os.environ[key] for key in TRACKED_ENV if key in os.environ},
                "cwd": os.getcwd(),
                "forwarded": {},
            }
            if not metadata_call:
                pinned = args[1] if args[:1] == ["--host"] else None
                daemon_args = args[2:] if pinned else args
                context = os.environ.get("DOCKER_CONTEXT")
                entry["endpoint"] = (
                    state["contexts"].get(context) if context else
                    pinned or os.environ.get("DOCKER_HOST") or state["contexts"][state["current"]]
                )
                if daemon_args[:1] == ["create"]:
                    for index, arg in enumerate(daemon_args[:-1]):
                        if arg == "-e":
                            name = daemon_args[index + 1]
                            if name in TRACKED_ENV and name in os.environ:
                                entry["forwarded"][name] = os.environ[name]
                if state.get("read_stdin"):
                    entry["stdin"] = sys.stdin.read()
            with Path(os.environ["SHELLSMITH_TEST_DOCKER_LOG"]).open("a") as log:
                log.write(json.dumps(entry) + "\n")
            if metadata_call:
                if state.get("metadata_failure") == args[1]:
                    # A trusted CLI may include sensitive config in diagnostics.
                    secret = os.environ.get("SHELLSMITH_ENDPOINT_TEST_API_KEY", "failure")
                    print(secret)
                    print(secret, file=sys.stderr)
                    sys.exit(23)
                if args == ["context", "show"]:
                    sys.stdout.write(state.get("show_output", state["current"] + "\n"))
                    if "after_show" in state:
                        state.update(state.pop("after_show"))
                        metadata_path.write_text(json.dumps(state))
                elif args[:3] == ["context", "inspect", "--"] and len(args) == 4:
                    name = args[3]
                    if "inspect_output" in state:
                        sys.stdout.write(state["inspect_output"])
                    elif name in state["contexts"]:
                        host = state["contexts"][name]
                        if name == "default":
                            host = os.environ.get("DOCKER_HOST") or host
                        print(json.dumps([{"Name": name, "Endpoints": {"docker": {"Host": host}}}]))
                    else:
                        sys.exit(24)
                else:
                    sys.exit(25)
            else:
                sys.stdout.write(state.get("daemon_stdout", "daemon output\n"))
                sys.stderr.write(state.get("daemon_stderr", ""))
                sys.exit(state.get("daemon_returncode", 0))
        ''')
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(binary_dir) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("SHELLSMITH_TEST_DOCKER_METADATA", str(transport.metadata))
    monkeypatch.setenv("SHELLSMITH_TEST_DOCKER_LOG", str(transport.log))
    monkeypatch.setenv(API_KEY, "synthetic-api-secret")
    monkeypatch.setenv(OAUTH_TOKEN, "synthetic-oauth-secret")
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    return transport


def _assert_metadata_only(transport: DockerTransport) -> None:
    assert all(call["kind"] == "metadata" for call in transport.calls)
    for call in transport.calls:
        assert "DOCKER_HOST" not in call["env"]
        assert "DOCKER_CONTEXT" not in call["env"]
        assert call["forwarded"] == {}
        assert "synthetic-api-secret" not in json.dumps(call["args"])
        assert "synthetic-oauth-secret" not in json.dumps(call["args"])


@pytest.mark.parametrize("context", ["desktop-linux", "default"])
def test_context_env_overrides_remote_host_and_current_context(docker_transport, monkeypatch, context) -> None:
    docker_transport.configure(current="remote")
    monkeypatch.setenv("DOCKER_CONTEXT", context)
    monkeypatch.setenv("DOCKER_HOST", "tcp://remote.example.invalid:2376")
    client = DockerClient.resolve()
    assert client.host == (LOCAL_HOST if context == "desktop-linux" else "unix:///var/run/docker.sock")
    assert [call["args"] for call in docker_transport.calls] == [["context", "inspect", "--", context]]
    _assert_metadata_only(docker_transport)


@pytest.mark.parametrize("empty_context", [False, True])
def test_host_env_overrides_current_context_without_metadata_access(docker_transport, monkeypatch, empty_context) -> None:
    docker_transport.configure(current="remote")
    monkeypatch.setenv("DOCKER_HOST", LOCAL_HOST)
    if empty_context:
        monkeypatch.setenv("DOCKER_CONTEXT", "")
    assert DockerClient.resolve().host == LOCAL_HOST
    assert docker_transport.calls == []


def test_empty_selectors_use_current_context_show_then_explicit_inspect(docker_transport, monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_HOST", "")
    monkeypatch.setenv("DOCKER_CONTEXT", "")
    assert DockerClient.resolve().host == LOCAL_HOST
    assert [call["args"] for call in docker_transport.calls] == [
        ["context", "show"], ["context", "inspect", "--", "desktop-linux"],
    ]
    _assert_metadata_only(docker_transport)


def test_current_context_change_between_show_and_inspect_does_not_reselect(docker_transport) -> None:
    docker_transport.configure(after_show={"current": "remote"})
    assert DockerClient.resolve().host == LOCAL_HOST
    assert docker_transport.calls[-1]["args"] == ["context", "inspect", "--", "desktop-linux"]
    _assert_metadata_only(docker_transport)


@pytest.mark.parametrize("selection", ["environment", "current"])
@pytest.mark.parametrize("endpoint", [
    "tcp://remote.example.invalid:2376", "ssh://remote.example.invalid",
    "npipe:////remote-server/pipe/docker_engine", "unix:///tmp/\x00socket",
])
def test_unsupported_context_is_rejected_before_any_daemon_call(docker_transport, monkeypatch, selection, endpoint) -> None:
    docker_transport.configure(current="unsafe", contexts={"unsafe": endpoint})
    if selection == "environment":
        monkeypatch.setenv("DOCKER_CONTEXT", "unsafe")
        monkeypatch.setenv("DOCKER_HOST", LOCAL_HOST)
    with pytest.raises(RuntimeError, match="Remote Docker endpoints are unsupported"):
        DockerClient.resolve()
    assert len(docker_transport.calls) == (1 if selection == "environment" else 2)
    _assert_metadata_only(docker_transport)


@pytest.mark.parametrize("host", [
    "tcp://remote.example.invalid:2376", "tcp://127.0.0.1:2375", "tcp://[::1]:2375",
    "ssh://user:embedded-secret@remote.example.invalid", "http://localhost:2375",
    "https://remote.example.invalid", "remote.example.invalid:2375", "/var/run/docker.sock",
    "unix://relative.sock", "unix://server/path.sock", "unix://", "unix:///",
    "unix:////server/share/docker.sock", "unix://@abstract", "unix:///tmp/socket?query",
    "unix:///tmp/socket#fragment", "unix:///tmp/socket\n",
    "unix:///tmp/\\remote\\socket", " unix:///tmp/docker.sock",
    "npipe:////server/pipe/docker_engine", "npipe:////localhost/pipe/docker_engine",
    r"npipe://\\server\pipe\docker_engine", "npipe:////./pipe/../remote",
    "npipe:////./pipe/docker_engine/remote", "npipe:////./pipe/docker_engine\\remote",
    "npipe:////./pipe/", "npipe:////./pipe/docker_engine.", "npipe:////?/pipe/docker_engine",
])
def test_unsupported_host_is_rejected_without_even_metadata_calls(docker_transport, monkeypatch, host) -> None:
    monkeypatch.setenv("DOCKER_HOST", host)
    with pytest.raises(RuntimeError, match="local Docker endpoint") as error:
        DockerClient.resolve()
    assert docker_transport.calls == []
    assert "embedded-secret" not in str(error.value)


@pytest.mark.parametrize("host,expected", [
    ("unix:///var/run/docker.sock", "unix:///var/run/docker.sock"),
    ("unix:///run/user/1000/docker.sock", "unix:///run/user/1000/docker.sock"),
    ("unix:///Users/local user/.docker/run/docker.sock", "unix:///Users/local user/.docker/run/docker.sock"),
    ("npipe:////./pipe/docker_engine", "npipe:////./pipe/docker_engine"),
    ("npipe:////./pipe/dockerDesktopLinuxEngine", "npipe:////./pipe/dockerDesktopLinuxEngine"),
    (r"npipe://\\.\pipe\docker_engine", "npipe:////./pipe/docker_engine"),
])
def test_safe_endpoints_are_pinned_in_real_subprocess(docker_transport, monkeypatch, host, expected) -> None:
    monkeypatch.setenv("DOCKER_HOST", host)
    client = DockerClient.resolve()
    result = client.run(["docker", "image", "inspect", "local-image"], capture_output=True, text=True, check=True)
    assert result.stdout == "daemon output\n"
    assert client.host == expected
    assert docker_transport.calls[0]["args"] == ["--host", expected, "image", "inspect", "local-image"]
    assert docker_transport.calls[0]["endpoint"] == expected
    assert "DOCKER_HOST" not in docker_transport.calls[0]["env"]


def test_all_lifecycle_commands_stay_pinned_after_context_and_environment_changes(docker_transport, monkeypatch) -> None:
    client = DockerClient.resolve()
    docker_transport.configure(current="remote", contexts={
        "desktop-linux": "tcp://changed.example.invalid:2376",
        "remote": "ssh://remote.example.invalid",
    })
    monkeypatch.setenv("DOCKER_CONTEXT", "remote")
    monkeypatch.setenv("DOCKER_HOST", "tcp://another.example.invalid:2376")
    monkeypatch.setenv(API_KEY, "rotated-api-secret")
    monkeypatch.setenv(OAUTH_TOKEN, "rotated-oauth-secret")
    commands = [
        ["docker", "image", "inspect", "--format", "{{.Id}}", "local-image"],
        ["docker", "create", "--name", "episode", "-e", API_KEY, "local-image", "python", "-m", "episode"],
        ["docker", "cp", "/staged/world.md", "episode:/world.md"],
        ["docker", "cp", "/staged/oauth", "episode:/oauth"],
        ["docker", "start", "--attach", "episode"],
        ["docker", "stop", "--time", "30", "episode"],
        ["docker", "cp", "episode:/logs/artifacts/.", "/output"],
        ["docker", "cp", "episode:/logs/in-progress/.", "/recovery"],
        ["docker", "rm", "--force", "episode"],
    ]
    for command in commands:
        original = list(command)
        client.run(command, capture_output=True, text=True, check=True, timeout=5)
        assert command == original
    calls = docker_transport.calls
    assert len(calls) == 2 + len(commands)
    for call, command in zip(calls[2:], commands, strict=True):
        assert call["kind"] == "daemon"
        assert call["args"] == ["--host", LOCAL_HOST, *command[1:]]
        assert call["endpoint"] == LOCAL_HOST
        assert "DOCKER_CONTEXT" not in call["env"]
        assert "DOCKER_HOST" not in call["env"]
        assert call["env"][API_KEY] == "rotated-api-secret"
        assert call["env"][OAUTH_TOKEN] == "rotated-oauth-secret"
        assert "rotated-api-secret" not in json.dumps(call["args"])
        assert "rotated-oauth-secret" not in json.dumps(call["args"])
        assert call["forwarded"] == ({API_KEY: "rotated-api-secret"} if command[1] == "create" else {})
    assert os.environ["DOCKER_CONTEXT"] == "remote"
    assert os.environ["DOCKER_HOST"] == "tcp://another.example.invalid:2376"
    assert "secret" not in repr(client)


def test_metadata_and_run_preserve_nonselector_env_without_redirecting_homes(docker_transport, monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_API_VERSION", "1.45")
    monkeypatch.setenv("SHELLSMITH_ENDPOINT_TEST_EXTRA", "required-env-value")
    expected = {key: os.environ[key] for key in TRACKED_ENV if key in os.environ}
    client = DockerClient.resolve()
    client.run(["docker", "image", "inspect", "local-image"], capture_output=True, check=True, env=None)
    assert len(docker_transport.calls) == 3
    assert all(call["env"] == expected for call in docker_transport.calls)
    assert all(not call["forwarded"] for call in docker_transport.calls)
    assert dict(os.environ).items() >= expected.items()


@pytest.mark.parametrize("bytes_environment", [False, True])
def test_explicit_child_environment_is_copied_scrubbed_and_not_cached(docker_transport, monkeypatch, bytes_environment) -> None:
    client = DockerClient.resolve()
    environment = dict(os.environ)
    environment.update({
        "DOCKER_CONTEXT": "remote", "DOCKER_HOST": "ssh://remote.example.invalid",
        API_KEY: "explicit-api-secret", "SHELLSMITH_ENDPOINT_TEST_EXTRA": "only-this-child",
    })
    environment.pop(OAUTH_TOKEN)
    if bytes_environment:
        environment = {os.fsencode(key): os.fsencode(value) for key, value in environment.items()}
    original = environment.copy()
    monkeypatch.setenv(OAUTH_TOKEN, "ambient-must-not-be-merged")
    client.run(["docker", "create", "-e", API_KEY, "local-image"], env=environment, capture_output=True, check=True)
    assert environment == original
    call = docker_transport.calls[-1]
    assert call["endpoint"] == LOCAL_HOST
    assert call["forwarded"] == {API_KEY: "explicit-api-secret"}
    assert call["env"]["SHELLSMITH_ENDPOINT_TEST_EXTRA"] == "only-this-child"
    assert OAUTH_TOKEN not in call["env"]
    assert "DOCKER_CONTEXT" not in call["env"]
    assert "DOCKER_HOST" not in call["env"]
    assert "explicit-api-secret" not in json.dumps(call["args"])


@pytest.mark.parametrize("name", ["", "--help", "--host=tcp://remote", "bad/name", "bad\nname", " name"])
def test_invalid_current_context_name_does_not_reach_inspect_or_daemon(docker_transport, name) -> None:
    docker_transport.configure(show_output=name + "\n")
    with pytest.raises(RuntimeError, match="context name"):
        DockerClient.resolve()
    assert [call["args"] for call in docker_transport.calls] == [["context", "show"]]


def test_invalid_explicit_context_name_is_rejected_without_cli_execution(docker_transport, monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_CONTEXT", "--host=tcp://remote.example.invalid")
    monkeypatch.setenv("DOCKER_HOST", LOCAL_HOST)
    with pytest.raises(RuntimeError, match="context name"):
        DockerClient.resolve()
    assert docker_transport.calls == []


@pytest.mark.parametrize("output", [
    "not-json synthetic-api-secret", "null", "{}", "[]", "[{}, {}]", "[{}]",
    '[{"Name":"wrong-context","Endpoints":{"docker":{"Host":"unix:///local/docker.sock"}}}]',
    '[{"Name":"desktop-linux","Endpoints":{}}]',
    '[{"Name":"desktop-linux","Endpoints":{"docker":{}}}]',
    '[{"Name":"desktop-linux","Endpoints":{"docker":{"Host":null}}}]',
    '[{"Name":"desktop-linux","Endpoints":{"docker":{"Host":42}}}]',
    '[{"Name":"desktop-linux","Endpoints":{"docker":{"Host":""}}}]',
])
def test_malformed_context_metadata_fails_closed_without_leaking_output(docker_transport, output, capsys) -> None:
    docker_transport.configure(inspect_output=output)
    with pytest.raises(RuntimeError, match="local Docker") as error:
        DockerClient.resolve()
    _assert_metadata_only(docker_transport)
    assert len(docker_transport.calls) == 2
    assert "synthetic-api-secret" not in "".join(traceback.format_exception(error.value))
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("step", ["show", "inspect"])
def test_metadata_failure_does_not_fallback_or_echo_credentials(docker_transport, step, capsys) -> None:
    docker_transport.configure(metadata_failure=step)
    with pytest.raises(RuntimeError, match="local Docker context metadata") as error:
        DockerClient.resolve()
    assert "synthetic-api-secret" not in "".join(traceback.format_exception(error.value))
    assert capsys.readouterr() == ("", "")
    assert len(docker_transport.calls) == (1 if step == "show" else 2)
    _assert_metadata_only(docker_transport)


def test_missing_context_does_not_fallback_to_host(docker_transport, monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_CONTEXT", "missing-context")
    monkeypatch.setenv("DOCKER_HOST", LOCAL_HOST)
    with pytest.raises(RuntimeError, match="local Docker context metadata"):
        DockerClient.resolve()
    assert [call["args"] for call in docker_transport.calls] == [["context", "inspect", "--", "missing-context"]]


def test_missing_docker_cli_fails_closed(docker_transport, monkeypatch, tmp_path) -> None:
    empty_path = tmp_path / "empty-bin"
    empty_path.mkdir()
    monkeypatch.setenv("PATH", str(empty_path))
    with pytest.raises(RuntimeError, match="local Docker context metadata"):
        DockerClient.resolve()
    assert docker_transport.calls == []


def test_client_is_immutable_and_direct_construction_cannot_allow_remote(docker_transport) -> None:
    client = DockerClient.resolve()
    with pytest.raises(FrozenInstanceError):
        client.host = "ssh://remote.example.invalid"
    with pytest.raises(RuntimeError, match="local Docker endpoint"):
        DockerClient("ssh://remote.example.invalid")
    _assert_metadata_only(docker_transport)


@pytest.mark.parametrize("command", [
    "docker image inspect local-image", [], ["docker"], ["other-cli", "inspect"],
    ["docker", "--host", "ssh://remote", "image", "inspect"],
    ["docker", "image", "inspect", "--host", "ssh://remote"],
    ["docker", "image", "inspect", "--host=ssh://remote"],
    ["docker", "image", "inspect", "--context", "remote"],
    ["docker", "image", "inspect", "--context=remote"],
    ["docker", "image", "inspect", "-Hssh://remote"],
    ["docker", "image", "inspect", "-c", "remote"],
    ["docker", "image", "inspect", "-cremote"],
])
def test_command_cannot_override_the_pinned_transport(docker_transport, command) -> None:
    client = DockerClient.resolve()
    with pytest.raises(ValueError):
        client.run(command)
    _assert_metadata_only(docker_transport)


@pytest.mark.parametrize("kwargs", [{"shell": True}, {"executable": "/bin/sh"}])
def test_subprocess_overrides_cannot_bypass_the_pinned_cli(docker_transport, kwargs) -> None:
    client = DockerClient.resolve()
    with pytest.raises(ValueError):
        client.run(["docker", "image", "inspect", "local-image"], **kwargs)
    _assert_metadata_only(docker_transport)


def test_run_preserves_subprocess_input_cwd_output_and_check_semantics(docker_transport, tmp_path) -> None:
    docker_transport.configure(read_stdin=True, daemon_stdout="stdout-value", daemon_stderr="stderr-value")
    client = DockerClient.resolve()
    result = client.run(
        ["docker", "image", "inspect", "local-image"], input="stdin-value",
        capture_output=True, text=True, check=True, timeout=5, cwd=tmp_path,
    )
    assert result.stdout == "stdout-value"
    assert result.stderr == "stderr-value"
    assert result.args == ["docker", "--host", LOCAL_HOST, "image", "inspect", "local-image"]
    assert docker_transport.calls[-1]["stdin"] == "stdin-value"
    assert Path(docker_transport.calls[-1]["cwd"]).resolve() == tmp_path.resolve()
    docker_transport.configure(read_stdin=False, daemon_returncode=19)
    with pytest.raises(subprocess.CalledProcessError) as error:
        client.run(["docker", "rm", "--force", "episode"], check=True, capture_output=True, text=True)
    assert error.value.returncode == 19
    assert error.value.stderr == "stderr-value"
    assert error.value.cmd[:3] == ["docker", "--host", LOCAL_HOST]
    assert client.run(["docker", "rm", "episode"], capture_output=True).returncode == 19


def test_run_preserves_launcher_log_redirection(docker_transport, tmp_path) -> None:
    client = DockerClient.resolve()
    docker_transport.configure(daemon_stdout="stdout-value\n", daemon_stderr="stderr-value\n")
    log_path = tmp_path / "container.log"
    with log_path.open("wb") as log:
        result = client.run(["docker", "start", "--attach", "episode"], stdout=log, stderr=subprocess.STDOUT, timeout=5)
    assert result.returncode == 0
    assert sorted(log_path.read_text().splitlines()) == ["stderr-value", "stdout-value"]
    assert docker_transport.calls[-1]["endpoint"] == LOCAL_HOST
