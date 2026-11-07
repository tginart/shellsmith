"""Codex-only GM execution at real public boundaries, without live model calls.

Negative tests forbid external I/O and credential access, never replace a
Shellsmith target. Positive process tests use only a local external CLI fixture.
"""

from __future__ import annotations

import asyncio
import builtins
from collections.abc import Mapping
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import urllib.request

import pytest

from shellsmith.campaign import checkout_batch, rollout_campaign
from shellsmith.config import DEFAULT_GM_MODEL
from shellsmith.episode import smoke
from shellsmith.episode.gm_process import (
    ClaudeWorldConfig,
    ClaudeWorldProcess,
    CodexWorldProcess,
    WorldHarnessConfig,
    WorldHarnessLauncher,
    WorldTurnContext,
)
from shellsmith.episode.gm_process.codex import SANDBOX_PERMISSION_PROFILE
from shellsmith.episode.gm_process.shared import (
    CLAUDE_GM_DISABLED_MESSAGE,
    PI_GM_DISABLED_MESSAGE,
    validate_gm_runtime,
)
from shellsmith.paths import PROJECT_ROOT
from shellsmith.sandbox import docker_launcher, sandbox_rollout


class _CredentialGuard(Mapping[str, str]):
    """Settings are readable; secrets and copying the parent environment are not."""

    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        if key.endswith(("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")):
            pytest.fail(f"GM runtime validation must precede credential access: {key}")
        return self.values[key]

    def __iter__(self):
        pytest.fail("GM runtime validation must precede copying the environment")

    def __len__(self):
        return len(self.values)


@pytest.fixture
def forbidden_gm_boundaries(monkeypatch):
    """Enter after fixture setup; all patched objects are external boundaries."""
    @contextmanager
    def guard(*, environment=None):
        def forbidden(*args, **kwargs):
            pytest.fail("GM runtime rejection must precede external I/O or launch setup")

        real_import = builtins.__import__
        real_import_module = importlib.import_module

        def reject_sdk(name):
            if name.split(".")[0] in {"tinker", "tinker_cookbook", "torch"}:
                pytest.fail("GM runtime rejection must precede external SDK loading")

        def guarded_import(name, *args, **kwargs):
            reject_sdk(name)
            return real_import(name, *args, **kwargs)

        def guarded_import_module(name, *args, **kwargs):
            reject_sdk(name)
            return real_import_module(name, *args, **kwargs)

        values = dict(os.environ) if environment is None else environment
        with monkeypatch.context() as patch:
            patch.setattr(os, "environ", _CredentialGuard(values))
            patch.setattr(builtins, "__import__", guarded_import)
            patch.setattr(importlib, "import_module", guarded_import_module)
            patch.setattr(asyncio, "create_subprocess_exec", forbidden)
            patch.setattr(asyncio, "create_subprocess_shell", forbidden)
            patch.setattr(subprocess, "Popen", forbidden)
            patch.setattr(urllib.request, "urlopen", forbidden)
            patch.setattr(socket, "create_connection", forbidden)
            patch.setattr(socket.socket, "connect", forbidden)
            patch.setattr(socket.socket, "connect_ex", forbidden)
            patch.setattr(socket.socket, "bind", forbidden)
            patch.setattr(Path, "open", forbidden)
            patch.setattr(Path, "read_text", forbidden)
            patch.setattr(Path, "mkdir", forbidden)
            patch.setattr(tempfile, "mkdtemp", forbidden)
            yield

    return guard


def _episode_parser():
    # Keep independent GM tests importable while the CLI merge is in progress.
    from shellsmith.cli import build_parser
    return build_parser()


_PARSERS = [
    pytest.param(smoke._parser, [], id="smoke"),
    pytest.param(checkout_batch._parser, [], id="checkout-batch"),
    pytest.param(rollout_campaign._parser, [], id="rollout-campaign"),
    pytest.param(_episode_parser, ["episode", "--host"], id="episode-host"),
    pytest.param(_episode_parser, ["episode", "--docker"], id="episode-docker"),
]


@pytest.mark.parametrize("runtime,message", [
    ("claude", CLAUDE_GM_DISABLED_MESSAGE),
    (" Claude ", CLAUDE_GM_DISABLED_MESSAGE),
    ("pi", PI_GM_DISABLED_MESSAGE),
    (" PI ", PI_GM_DISABLED_MESSAGE),
])
def test_shared_runtime_guard_preserves_disabled_messages(runtime, message, forbidden_gm_boundaries):
    with forbidden_gm_boundaries(), pytest.raises(RuntimeError) as error:
        validate_gm_runtime(runtime)
    assert str(error.value) == message


@pytest.mark.parametrize("runtime", ["codex", " Codex "])
def test_shared_runtime_guard_allows_codex(runtime, forbidden_gm_boundaries):
    with forbidden_gm_boundaries():
        assert validate_gm_runtime(runtime) is None


@pytest.mark.parametrize("factory,prefix", _PARSERS)
@pytest.mark.parametrize("flags", [
    ["--gm-runtime", "claude"],
    ["--gm-runtime", "claude", "--gm-model", "explicit-model"],
    ["--gm-model=explicit-model", "--gm-runtime=claude"],
])
@pytest.mark.parametrize("method", ["parse_args", "parse_known_args"])
def test_real_parsers_reject_claude_before_model_defaults(
    factory, prefix, flags, method, capsys, forbidden_gm_boundaries,
):
    parser = factory()
    with forbidden_gm_boundaries(), pytest.raises(SystemExit) as error:
        getattr(parser, method)(prefix + flags)
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert CLAUDE_GM_DISABLED_MESSAGE in stderr
    assert "--gm-model is required" not in stderr


@pytest.mark.parametrize("factory", [smoke._parser, checkout_batch._parser, rollout_campaign._parser])
@pytest.mark.parametrize("flags", [
    ["--gm-provider", "codex-cli", "--gm-runtime", "claude"],
    ["--gm-runtime", "claude", "--gm-provider", "custom"],
])
def test_provider_override_cannot_enable_claude(factory, flags, capsys, forbidden_gm_boundaries):
    parser = factory()
    with forbidden_gm_boundaries(), pytest.raises(SystemExit) as error:
        parser.parse_args(flags + ["--gm-model", "explicit-model"])
    assert error.value.code == 2
    assert CLAUDE_GM_DISABLED_MESSAGE in capsys.readouterr().err


@pytest.mark.parametrize("factory,prefix", _PARSERS)
@pytest.mark.parametrize("flags,model", [([], DEFAULT_GM_MODEL), (["--gm-model", "explicit-model"], "explicit-model")])
def test_real_parsers_keep_codex_defaults_and_explicit_models(factory, prefix, flags, model, forbidden_gm_boundaries):
    parser = factory()
    with forbidden_gm_boundaries():
        args = parser.parse_args(prefix + ["--gm-runtime", "codex", *flags])
    assert args.gm_runtime == "codex"
    assert args.gm_model == model
    assert args.gm_provider == "codex-cli"
    if prefix:
        assert args.host is ("--host" in prefix)


@pytest.mark.parametrize("runtime", ["claude", " CLAUDE "])
def test_shared_launcher_rejects_claude_before_setup(runtime, tmp_path, forbidden_gm_boundaries):
    config = WorldHarnessConfig(
        runtime=runtime, executable=tmp_path / "must-not-execute",
        model="explicit-model", workspace_dir=tmp_path / "world",
        log_dir=tmp_path / "logs", bootstrap_response_schema_path=tmp_path / "schema.json",
    )
    with forbidden_gm_boundaries(), pytest.raises(RuntimeError) as error:
        WorldHarnessLauncher.create(config)
    assert str(error.value) == CLAUDE_GM_DISABLED_MESSAGE
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("runtime", ["claude", " CLAUDE ", "pi", " PI "])
def test_mutated_smoke_namespace_rejects_before_credentials_or_setup(runtime, tmp_path, forbidden_gm_boundaries):
    args = smoke._parser().parse_args(["--output-dir", str(tmp_path / "output")])
    args.gm_runtime = runtime
    args.gm_model = None
    # Skip-models is not a bypass for the runtime or credential policy.
    args.skip_player_models_preflight = True
    with forbidden_gm_boundaries(), pytest.raises(RuntimeError, match="GM execution is disabled"):
        asyncio.run(smoke.run_live_smoke(args))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("runtime", ["claude", " CLAUDE ", "pi", " PI "])
@pytest.mark.parametrize("backend", ["--host", "--docker"])
@pytest.mark.parametrize("player", ["openai", "tinker"])
def test_mutated_episode_namespace_rejects_before_world_model_or_output_setup(
    runtime, backend, player, tmp_path, capsys, forbidden_gm_boundaries,
):
    from shellsmith import cli
    args = cli.build_parser().parse_args([
        "episode", backend, "--player", player,
        "--world-file", str(tmp_path / "absent-world.md"),
        "--log-dir", str(tmp_path / "output"),
    ])
    args.gm_runtime = runtime
    args.gm_model = None
    args.tinker_model = "must-not-resolve-this-model"
    with forbidden_gm_boundaries(), pytest.raises(RuntimeError, match="GM execution is disabled"):
        cli.run_episode_command(args)
    assert not capsys.readouterr().out
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("runtime", ["claude", " CLAUDE ", "pi", " PI "])
@pytest.mark.parametrize("entrypoint", ["secret_names", "run_episode_in_docker"])
def test_direct_docker_boundary_rejects_disabled_gms_before_credentials(
    runtime, entrypoint, tmp_path, forbidden_gm_boundaries,
):
    config = docker_launcher.DockerEpisodeConfig(
        image="must-not-inspect", gm_runtime=runtime,
        player_base_url="https://player.invalid/v1", player_model="fixture-player",
        player_provider="fixture-provider", player_api_key_env="PLAYER_API_KEY",
    )
    with forbidden_gm_boundaries(), pytest.raises(ValueError, match="only the confined Codex GM") as error:
        if entrypoint == "secret_names":
            docker_launcher.secret_names(config)
        else:
            docker_launcher.run_episode_in_docker(tmp_path / "missing-world.md", tmp_path / "output", config)
    assert "use --host for Claude" not in str(error.value)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("runtime", ["claude", " CLAUDE "])
@pytest.mark.parametrize("settings_present", [False, True])
def test_container_env_rejects_claude_before_player_credentials(runtime, settings_present, forbidden_gm_boundaries):
    environment = {"SHELLSMITH_GM_RUNTIME": runtime}
    if settings_present:
        environment.update({
            "SHELLSMITH_GM_MODEL": "explicit-model",
            "SHELLSMITH_PLAYER_API_KEY_ENV": "PLAYER_API_KEY",
            "PLAYER_API_KEY": "synthetic-player-only",
            "SHELLSMITH_PLAYER_BASE_URL": "https://player.invalid/v1",
            "SHELLSMITH_PLAYER_MODEL": "synthetic-player-model",
        })
    with forbidden_gm_boundaries(environment=environment), pytest.raises(RuntimeError) as error:
        sandbox_rollout.args_from_env()
    assert str(error.value) == CLAUDE_GM_DISABLED_MESSAGE


def test_container_env_keeps_codex_and_separate_pi_player(monkeypatch):
    monkeypatch.setattr(os, "environ", {
        "SHELLSMITH_GM_RUNTIME": "codex",
        "SHELLSMITH_PLAYER_API_KEY_ENV": "PLAYER_API_KEY",
        "PLAYER_API_KEY": "synthetic-player-only",
        "SHELLSMITH_PLAYER_BASE_URL": "https://player.invalid/v1",
        "SHELLSMITH_PLAYER_MODEL": "synthetic-player-model",
        "SHELLSMITH_PI_SDK_MODULE_PATH": "/synthetic/pi-sdk/index.js",
    })
    args = sandbox_rollout.args_from_env()
    assert (args.gm_runtime, args.gm_provider, args.gm_model) == ("codex", "codex-cli", DEFAULT_GM_MODEL)
    assert args.codex_sandbox == SANDBOX_PERMISSION_PROFILE
    assert args.pi_sdk_module_path == Path("/synthetic/pi-sdk/index.js")


@pytest.mark.parametrize("entrypoint", ["start", "resume", "_run_turn", "_run_turn_once"])
@pytest.mark.parametrize("artifact_state", ["absent", "valid", "repair"])
@pytest.mark.parametrize("closed", [False, True])
def test_direct_claude_execution_cannot_bypass_runtime_guard(
    entrypoint, artifact_state, closed, tmp_path, forbidden_gm_boundaries,
):
    process = ClaudeWorldProcess(ClaudeWorldConfig(
        executable=tmp_path / "must-not-execute", model="explicit-model",
        workspace_dir=tmp_path / "world", log_dir=tmp_path / "logs",
        bootstrap_response_schema_path=tmp_path / "schema.json",
    ))
    process._events_by_session["historical-session"] = []
    if closed:
        asyncio.run(process.close())
    artifact = tmp_path / "response.json"
    if artifact_state != "absent":
        artifact.write_text('{"status":"ready"}\n')
    context = WorldTurnContext(
        stage="turn", expected_artifact_path=artifact,
        reuse_valid_artifact=artifact_state != "repair",
        output_schema_path=tmp_path / "schema.json",
    )
    if entrypoint == "start":
        operation = process.start("prompt", turn_context=context)
    elif entrypoint == "resume":
        operation = process.resume("historical-session", "prompt", turn_context=context)
    elif entrypoint == "_run_turn":
        operation = process._run_turn(prompt="prompt", expected_session_id="historical-session", turn_context=context)
    else:
        operation = process._run_turn_once(
            prompt="prompt", expected_session_id="historical-session",
            output_schema_path=context.output_schema_path, stage="turn", turn_index=0, attempt_index=0,
        )
    with forbidden_gm_boundaries(), pytest.raises(RuntimeError) as error:
        asyncio.run(operation)
    assert str(error.value) == CLAUDE_GM_DISABLED_MESSAGE
    assert process._next_turn_index == 0
    assert process._events_by_session == {"historical-session": []}
    assert not process.config.workspace_dir.exists()
    assert not process.config.log_dir.exists()
    if artifact_state != "absent":
        assert json.loads(artifact.read_text()) == {"status": "ready"}
    assert asyncio.run(process.transcript("historical-session")) == ()
    asyncio.run(process.close())


@pytest.mark.parametrize("failure", [False, True], ids=["start-and-resume", "stdout-error"])
def test_codex_launcher_and_process_still_execute_external_cli_fixture(tmp_path, failure):
    executable = tmp_path / "external-codex"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        f"failure = {failure!r}\n"
        "assert 'exec' in sys.argv and '--json' in sys.argv\n"
        "assert 'default_permissions=\"game_master\"' in sys.argv\n"
        "if failure:\n"
        "    print(json.dumps({'type': 'error', 'message': 'fixture-codex-refusal'}))\n"
        "    raise SystemExit(1)\n"
        "pathlib.Path(sys.argv[sys.argv.index('--output-last-message') + 1]).write_text('{\"status\":\"ready\"}')\n"
        "print(json.dumps({'type': 'thread.started', 'thread_id': 'fixture-codex-session'}))\n"
    )
    executable.chmod(0o755)
    workspace = tmp_path / "world"
    workspace.mkdir()
    process = WorldHarnessLauncher.create(WorldHarnessConfig(
        runtime=" Codex ", executable=executable, model="fixture-model",
        workspace_dir=workspace, log_dir=tmp_path / "logs",
        bootstrap_response_schema_path=tmp_path / "unused-schema.json",
    ))
    assert isinstance(process, CodexWorldProcess)
    if failure:
        with pytest.raises(RuntimeError, match="fixture-codex-refusal"):
            asyncio.run(process.start("fixture prompt"))
    else:
        first = asyncio.run(process.start("fixture prompt"))
        resumed = asyncio.run(process.resume(first.session_id, "fixture resume"))
        assert first.session_id == resumed.session_id == "fixture-codex-session"
        assert first.final_response == resumed.final_response == {"status": "ready"}
        assert len(asyncio.run(process.transcript(first.session_id))) == 2
    asyncio.run(process.close())


def test_architecture_marks_claude_gm_and_historical_backups_offline_only():
    guide = (PROJECT_ROOT / "docs/ARCHITECTURE.md").read_text()
    assert "Claude GM execution is disabled" in guide
    assert "GM execution is Codex-only" in guide
    assert "Pi Player SDK remains supported" in guide
