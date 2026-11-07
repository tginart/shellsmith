"""Pi GM fails closed; offline Pi helpers and the separate Player remain usable."""

from __future__ import annotations

import asyncio

import pytest

from shellsmith import cli
from shellsmith.campaign import checkout_batch, rollout_campaign
from shellsmith.episode import smoke
from shellsmith.episode.gm_process import (
    CodexWorldProcess,
    WorldHarnessConfig,
    WorldHarnessLauncher,
    WorldTurnContext,
)
from shellsmith.episode.gm_process.pi import PiWorldConfig, PiWorldProcess
from shellsmith.episode.gm_process.shared import PI_GM_DISABLED_MESSAGE
from shellsmith.sandbox import sandbox_rollout
from shellsmith.paths import PROJECT_ROOT
from test_codex_only_gm import forbidden_gm_boundaries


def test_current_architecture_guide_does_not_advertise_disabled_pi_gm() -> None:
    guide = (PROJECT_ROOT / "docs/ARCHITECTURE.md").read_text()
    assert "--gm-runtime pi" not in guide
    assert "Pi GM execution is disabled" in guide
    assert "Pi Player SDK remains supported" in guide


@pytest.fixture
def synthetic_environment(tmp_path):
    home = tmp_path / "home"
    for relative_path in (".codex/auth.json", ".pi/agent/auth.json"):
        auth_path = home / relative_path
        auth_path.parent.mkdir(parents=True)
        auth_path.write_text('{"token":"synthetic-auth-only"}\n', encoding="utf-8")
    return {
        "HOME": str(home),
        "PI_CODING_AGENT_DIR": str(home / ".pi/agent"),
        "CODEX_HOME": str(home / ".codex"),
        "PLAYER_API_KEY": "synthetic-player-only",
        "GM_API_KEY": "synthetic-gm-only",
        "OPENROUTER_API_KEY": "synthetic-openrouter-only",
    }


@pytest.mark.parametrize("entrypoint", ["start", "resume", "_run_turn"])
@pytest.mark.parametrize("provider,base_url", [
    ("openrouter", None),
    ("custom-provider", "https://gm.invalid/v1"),
])
@pytest.mark.parametrize("enabled_tools", [
    ("read", "bash", "edit", "write", "grep", "find", "ls"),
    ("read",),
    (),
])
@pytest.mark.parametrize("credentials_present", [False, True])
def test_direct_pi_execution_rejects_before_credentials_or_side_effects(
    tmp_path, forbidden_gm_boundaries, synthetic_environment,
    entrypoint, provider, base_url, enabled_tools, credentials_present,
) -> None:
    process = PiWorldProcess(PiWorldConfig(
        executable=tmp_path / "must-not-run-pi",
        model="synthetic-model",
        provider=provider,
        api_key_env="GM_API_KEY",
        base_url=base_url,
        enabled_tools=enabled_tools,
        workspace_dir=tmp_path / "world",
        log_dir=tmp_path / "logs",
        bootstrap_response_schema_path=tmp_path / "schema.json",
    ))
    # A known session and valid response cannot make resume/bootstrap acceptable.
    process._events_by_session["existing-session"] = []
    artifact = tmp_path / "response.json"
    artifact.write_text('{"status":"ready"}\n', encoding="utf-8")
    context = WorldTurnContext(stage="turn", expected_artifact_path=artifact)
    if entrypoint == "start":
        operation = process.start("synthetic prompt", turn_context=context)
    elif entrypoint == "resume":
        operation = process.resume(
            "existing-session", "synthetic prompt", turn_context=context,
            output_schema_path=tmp_path / "schema.json",
        )
    else:
        operation = process._run_turn(
            session_id="existing-session", prompt="synthetic prompt", stage="turn",
        )
    environment = synthetic_environment if credentials_present else {}
    with forbidden_gm_boundaries(environment=environment):
        with pytest.raises(RuntimeError, match="Pi GM execution is disabled") as error:
            asyncio.run(operation)
    assert str(error.value) == PI_GM_DISABLED_MESSAGE
    assert "synthetic-" not in str(error.value)
    assert not process.config.log_dir.exists()
    assert not process.config.workspace_dir.exists()
    assert process._events_by_session == {"existing-session": []}
    asyncio.run(process.close())


@pytest.mark.parametrize("factory,prefix", [
    pytest.param(smoke._parser, [], id="smoke"),
    pytest.param(checkout_batch._parser, [], id="checkout-batch"),
    pytest.param(rollout_campaign._parser, [], id="rollout-campaign"),
    pytest.param(cli.build_parser, ["episode", "--docker"], id="episode-docker"),
    pytest.param(cli.build_parser, ["episode", "--host"], id="episode-host"),
])
@pytest.mark.parametrize("flags", [
    ["--gm-runtime", "pi"],
    ["--gm-runtime", "pi", "--gm-model", "explicit-model"],
    ["--gm-model=explicit-model", "--gm-runtime=pi"],
])
@pytest.mark.parametrize("parse_method", ["parse_args", "parse_known_args"])
def test_shared_cli_rejects_pi_before_defaults(
    factory, prefix, flags, parse_method, capsys, forbidden_gm_boundaries,
) -> None:
    parser = factory()
    with forbidden_gm_boundaries(), pytest.raises(SystemExit) as error:
        getattr(parser, parse_method)(prefix + flags)
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert PI_GM_DISABLED_MESSAGE in stderr
    assert "--gm-model is required" not in stderr


@pytest.mark.parametrize("factory", [
    smoke._parser, checkout_batch._parser, rollout_campaign._parser,
])
@pytest.mark.parametrize("flags", [
    ["--gm-provider", "custom", "--gm-runtime", "pi"],
    ["--gm-runtime", "pi", "--gm-provider", "codex-cli"],
])
def test_provider_override_cannot_enable_pi_gm(
    factory, flags, capsys, forbidden_gm_boundaries,
) -> None:
    parser = factory()
    with forbidden_gm_boundaries(), pytest.raises(SystemExit) as error:
        parser.parse_args(flags + ["--gm-model", "explicit-model"])
    assert error.value.code == 2
    assert PI_GM_DISABLED_MESSAGE in capsys.readouterr().err


@pytest.mark.parametrize("host_flags", [[], ["--host"], ["--docker"]])
def test_episode_cli_rejects_before_launch_or_log_setup(
    host_flags, capsys, forbidden_gm_boundaries,
) -> None:
    with forbidden_gm_boundaries(), pytest.raises(SystemExit) as error:
        cli.main(["episode", *host_flags, "--gm-runtime", "pi", "--gm-model", "explicit-model"])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert PI_GM_DISABLED_MESSAGE in captured.err


@pytest.mark.parametrize("runtime", ["pi", " PI "])
def test_shared_launcher_rejects_pi_before_constructing_adapter(
    runtime, tmp_path, forbidden_gm_boundaries,
) -> None:
    config = WorldHarnessConfig(
        runtime=runtime, executable=tmp_path / "must-not-run-pi",
        model="explicit-model", workspace_dir=tmp_path / "world",
        log_dir=tmp_path / "logs",
        bootstrap_response_schema_path=tmp_path / "schema.json",
    )
    with forbidden_gm_boundaries(), pytest.raises(RuntimeError, match="Pi GM execution is disabled"):
        WorldHarnessLauncher.create(config)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("runtime", ["codex", "claude"])
def test_shared_launcher_only_allows_codex(runtime, tmp_path, forbidden_gm_boundaries) -> None:
    config = WorldHarnessConfig(
        runtime=runtime, executable=tmp_path / "unused-runtime",
        model="explicit-model", workspace_dir=tmp_path / "world",
        log_dir=tmp_path / "logs",
        bootstrap_response_schema_path=tmp_path / "schema.json",
    )
    with forbidden_gm_boundaries():
        if runtime == "claude":
            with pytest.raises(RuntimeError, match="Claude GM execution is disabled"):
                WorldHarnessLauncher.create(config)
        else:
            assert isinstance(WorldHarnessLauncher.create(config), CodexWorldProcess)


@pytest.mark.parametrize("runtime", ["pi", " PI "])
def test_mutated_smoke_namespace_rejects_before_player_preflight(
    runtime, tmp_path, forbidden_gm_boundaries, synthetic_environment,
) -> None:
    args = smoke._parser().parse_args(["--output-dir", str(tmp_path / "output")])
    args.gm_runtime = runtime
    with forbidden_gm_boundaries(environment=synthetic_environment):
        with pytest.raises(RuntimeError, match="Pi GM execution is disabled"):
            asyncio.run(smoke.run_live_smoke(args))
    assert not args.output_dir.exists()


@pytest.mark.parametrize("runtime", ["pi", " PI "])
@pytest.mark.parametrize("player_settings_present", [False, True])
def test_sandbox_environment_rejects_pi_before_reading_credentials(
    runtime, player_settings_present, forbidden_gm_boundaries,
    synthetic_environment,
) -> None:
    environment = {**synthetic_environment, "SHELLSMITH_GM_RUNTIME": runtime}
    if player_settings_present:
        environment.update({
            "SHELLSMITH_PLAYER_API_KEY_ENV": "PLAYER_API_KEY",
            "SHELLSMITH_PLAYER_BASE_URL": "https://player.invalid/v1",
            "SHELLSMITH_PLAYER_MODEL": "synthetic-player-model",
            "SHELLSMITH_GM_API_KEY_ENV": "GM_API_KEY",
            "SHELLSMITH_GM_MODEL": "synthetic-gm-model",
        })
    with forbidden_gm_boundaries(environment=environment):
        with pytest.raises(RuntimeError, match="Pi GM execution is disabled") as error:
            sandbox_rollout.args_from_env()
    assert str(error.value) == PI_GM_DISABLED_MESSAGE
    assert "synthetic-" not in str(error.value)
