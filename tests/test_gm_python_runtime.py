"""Portable proofs for the MLX-only interpreter grant and private GM guidance."""

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from shellsmith.episode.gm_process.codex import CodexWorldConfig, CodexWorldProcess
from shellsmith.episode.gm_process.python_runtime import PythonRuntime, current_python_runtime
from shellsmith.episode.gm_process.shared import WorldHarnessConfig, WorldHarnessLauncher


@pytest.fixture
def runtime(tmp_path):
    prefix = (tmp_path / "approved-python").resolve()
    (prefix / "bin").mkdir(parents=True)
    stdlib = prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    stdlib.mkdir(parents=True)
    (stdlib / "os.py").write_text("# standard library marker\n")
    executable = prefix / "bin" / "python3"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    return PythonRuntime(executable, prefix)


def process(tmp_path, runtime=None):
    return CodexWorldProcess(CodexWorldConfig(
        executable=tmp_path / "codex", model="fixture", workspace_dir=tmp_path / "episode",
        log_dir=tmp_path / "logs", bootstrap_response_schema_path=tmp_path / "schema.json",
        python_runtime=runtime,
    ))


def settings(command):
    document = {}
    for index, argument in enumerate(command[:-1]):
        if argument == "-c":
            override = tomllib.loads(command[index + 1])
            for key, value in override.items():
                document.setdefault(key, {}).update(value) if isinstance(value, dict) else document.update({key: value})
    return document


def test_default_profile_and_environment_are_unchanged(tmp_path):
    command = process(tmp_path)._base_command(resume=False)
    config = settings(command)
    profile = config["permissions"]["game_master"]
    assert profile == {
        "description": "Episode-only Game Master",
        "filesystem": {":minimal": "read", ":workspace_roots": {".": "write"}},
        "network": {"enabled": True},
    }
    assert config["shell_environment_policy"] == {"inherit": "core"}


@pytest.mark.parametrize("resume", [False, True])
def test_runtime_grant_is_only_the_approved_prefix_and_is_read_only(tmp_path, runtime, resume):
    config = settings(process(tmp_path, runtime)._base_command(resume=resume))
    assert config["permissions"]["game_master"]["filesystem"] == {
        ":minimal": "read", ":workspace_roots": {".": "write"}, str(runtime.prefix): "read",
    }
    assert config["shell_environment_policy"] == {
        "inherit": "core", "set": {"PATH": runtime.shell_path,
                                    "TMPDIR": str(tmp_path / "episode" / "world" / "runtime_scratch"),
                                    "PIP_CACHE_DIR": str(tmp_path / "episode" / "world" / "runtime_scratch" / "pip_cache"),
                                    "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
    }
    assert str(Path.home()) not in config["permissions"]["game_master"]["filesystem"]


def test_launcher_preserves_the_explicit_runtime(tmp_path, runtime):
    instance = WorldHarnessLauncher.create(WorldHarnessConfig(
        runtime="codex", executable=tmp_path / "codex", model="fixture",
        workspace_dir=tmp_path / "episode", log_dir=tmp_path / "logs",
        bootstrap_response_schema_path=tmp_path / "schema.json", python_runtime=runtime,
    ))
    assert instance.config.python_runtime == runtime


@pytest.mark.parametrize("stage", ["prefix-symlink", "executable-escape", "stdlib-escape", "not-executable", "other-writable"])
def test_runtime_rejects_unsafe_installation_paths(tmp_path, runtime, stage):
    if stage == "prefix-symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(runtime.prefix, target_is_directory=True)
        runtime = replace(runtime, prefix=alias)
    elif stage == "executable-escape":
        outside = tmp_path / "python3"
        outside.write_text("#!/bin/sh\nexit 0\n")
        outside.chmod(0o755)
        runtime.executable.unlink()
        runtime.executable.symlink_to(outside)
    elif stage == "stdlib-escape":
        stdlib = next((runtime.prefix / "lib").iterdir())
        outside = tmp_path / "other-stdlib"
        stdlib.rename(outside)
        stdlib.symlink_to(outside, target_is_directory=True)
    elif stage == "not-executable":
        runtime.executable.chmod(0o644)
    else:
        runtime.prefix.chmod(0o777)
    with pytest.raises(ValueError):
        runtime.validate()
    with pytest.raises(ValueError):
        process(tmp_path, runtime)


@pytest.mark.parametrize("prefix", [Path.home(), Path.home().parent, Path("/")])
def test_home_and_ancestor_grants_are_rejected(prefix, runtime):
    with pytest.raises(ValueError, match="broad filesystem root"):
        replace(runtime, prefix=prefix.resolve()).validate()


def test_runtime_must_belong_to_current_account(monkeypatch, runtime):
    monkeypatch.setattr(os, "getuid", lambda: runtime.prefix.stat().st_uid + 1)
    with pytest.raises(ValueError, match="owned by this account"):
        runtime.validate()


def test_current_runtime_resolves_venv_alias_without_allowing_venv_siblings(monkeypatch, tmp_path, runtime):
    alias = tmp_path / "trainer-venv" / "bin" / "python"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(runtime.executable)
    monkeypatch.setattr(sys, "executable", str(alias))
    monkeypatch.setattr(sys, "base_prefix", str(runtime.prefix))
    actual = current_python_runtime()
    assert actual == runtime
    assert not alias.is_relative_to(actual.prefix)


@pytest.mark.parametrize("with_runtime", [False, True])
def test_start_and_resume_capture_exact_private_runtime_guidance(monkeypatch, tmp_path, runtime, with_runtime):
    import shellsmith.episode.gm_process.codex as codex
    captured = []
    async def external_cli(command, **kwargs):
        captured.append(kwargs["input_data"].decode())
        payload = b'{"type":"thread.started","thread_id":"fixture-session"}\n'
        kwargs["stdout_path"].write_bytes(payload)
        kwargs["stderr_path"].write_bytes(b"")
        Path(command[command.index("--output-last-message") + 1]).write_text("{}")
        return 0, payload, b""
    monkeypatch.setattr(codex, "run_captured_process", external_cli)
    instance = process(tmp_path, runtime if with_runtime else None)
    async def run():
        first = await instance.start("bootstrap prompt")
        await instance.resume(first.session_id, "turn prompt")
    asyncio.run(run())
    for index, initial in enumerate(("bootstrap prompt", "turn prompt")):
        saved = json.loads(sorted(instance.config.log_dir.glob("*.request.json"))[index].read_text())
        assert captured[index] == saved["prompt"]
        if with_runtime:
            assert str(runtime.executable) in saved["prompt"]
            assert str(instance.config.workspace_dir / "world" / "simulation_root") in saved["prompt"]
            assert "has already created" in saved["prompt"] and "exists even when empty" in saved["prompt"]
            assert "-m venv --copies" in saved["prompt"]
        else:
            assert saved["prompt"] == initial


def test_readiness_rejects_unusable_runtime_before_mlx_probe(monkeypatch):
    import shellsmith.jobs.readiness as readiness
    import shellsmith.episode.gm_process.python_runtime as module
    events = []
    class Unusable:
        def probe(self):
            events.append("python")
            raise ValueError("fixture standard library unavailable")
    monkeypatch.setattr(module, "current_python_runtime", lambda: Unusable())
    monkeypatch.setattr(readiness, "installed_dependencies", lambda: events.append("dependencies") or {})
    monkeypatch.setattr(readiness, "executable_version", lambda _: None)
    monkeypatch.setattr(readiness.platform, "system", lambda: "Linux")
    report = readiness.probe()
    assert events[:2] == ["python", "dependencies"]
    assert not report["ready"] and report["gm_python_runtime"] is None
    assert any("Confined GM Python runtime is unavailable: fixture standard library unavailable" in item
               for item in report["blockers"])


def test_runtime_probe_requires_standard_computation_modules(monkeypatch, runtime):
    calls = []
    def failing(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 1, b"", b"fixture module failure")
    monkeypatch.setattr(subprocess, "run", failing)
    with pytest.raises(ValueError, match="json, sqlite3 and venv"):
        runtime.probe()
    assert calls[0][0] == [str(runtime.executable), "-I", "-B", "-c", "import json, sqlite3, venv"]
    assert calls[0][1]["env"] == {"PATH": runtime.shell_path}


def test_episode_audit_records_only_the_explicit_runtime_identity(runtime):
    import hashlib
    from shellsmith.episode import smoke
    args = smoke._parser().parse_args([])
    assert "gm_python_runtime" not in smoke._run_config(args, episode_id="default")
    args.gm_python_runtime = runtime
    metadata = smoke._run_config(args, episode_id="mlx")
    assert metadata["gm_python_runtime"] == {
        "executable": str(runtime.executable), "prefix": str(runtime.prefix),
        "executable_sha256": hashlib.sha256(runtime.executable.read_bytes()).hexdigest(),
    }
