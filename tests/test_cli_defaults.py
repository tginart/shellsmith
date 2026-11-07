"""Real execution parsers allow Codex; historical GM metadata stays inspectable."""

import json
from pathlib import Path

import pytest

from shellsmith.campaign import checkout_batch, rollout_campaign
from shellsmith.campaign.provider_outage import resolve_backup_world
from shellsmith.episode import smoke
from shellsmith.episode.gm_process.pi import PiWorldConfig, PiWorldProcess
from shellsmith.episode.gm_process.shared import CLAUDE_GM_DISABLED_MESSAGE
from shellsmith.paths import HERO_RUNS_DIR, PROJECT_ROOT


@pytest.mark.parametrize("factory", [smoke._parser, checkout_batch._parser, rollout_campaign._parser])
@pytest.mark.parametrize("runtime,provider", [("codex", "codex-cli"), ("claude", "claude-cli")])
def test_provider_defaults_follow_supported_runtime(factory, runtime, provider, capsys) -> None:
    flags = ["--gm-runtime", runtime, "--gm-model", "explicit-model"]
    if runtime == "claude":
        with pytest.raises(SystemExit) as error:
            factory().parse_args(flags)
        assert error.value.code == 2
        assert CLAUDE_GM_DISABLED_MESSAGE in capsys.readouterr().err
    else:
        assert factory().parse_args(flags).gm_provider == provider


@pytest.mark.parametrize("factory", [smoke._parser, checkout_batch._parser, rollout_campaign._parser])
@pytest.mark.parametrize("runtime", ["codex", "claude"])
@pytest.mark.parametrize("flags", [
    ["--gm-provider", "custom", "--gm-runtime", "{runtime}"],
    ["--gm-runtime", "{runtime}", "--gm-provider", "custom"],
])
def test_explicit_provider_cannot_override_runtime_policy(factory, runtime, flags, capsys) -> None:
    flags = [value.format(runtime=runtime) for value in flags] + ["--gm-model", "explicit-model"]
    if runtime == "claude":
        with pytest.raises(SystemExit) as error:
            factory().parse_args(flags)
        assert error.value.code == 2
        assert CLAUDE_GM_DISABLED_MESSAGE in capsys.readouterr().err
    else:
        assert factory().parse_args(flags).gm_provider == "custom"


@pytest.mark.parametrize("factory", [smoke._parser, checkout_batch._parser, rollout_campaign._parser])
def test_claude_runtime_is_disabled_before_model_defaults(factory, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        factory().parse_args(["--gm-runtime", "claude"])
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert CLAUDE_GM_DISABLED_MESSAGE in stderr
    assert "--gm-model is required" not in stderr


@pytest.mark.parametrize("model_flags", [[], ["--gm-model", "explicit-model"]])
def test_episode_entry_rejects_claude_with_or_without_model(model_flags, capsys) -> None:
    from shellsmith.cli import build_parser
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["episode", "--host", "--gm-runtime", "claude", *model_flags])
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert CLAUDE_GM_DISABLED_MESSAGE in stderr
    assert "--gm-model is required" not in stderr


def test_pi_offline_command_keeps_provider_and_model_metadata(tmp_path: Path) -> None:
    # Historical Pi configuration stays inspectable without enabling execution.
    process = PiWorldProcess(PiWorldConfig(
        executable=Path("pi"), model="vendor/model", provider="openrouter",
        workspace_dir=tmp_path, log_dir=tmp_path / "logs",
        bootstrap_response_schema_path=tmp_path / "schema.json",
    ))
    command = process._command("session")
    assert command[command.index("--provider") + 1] == "openrouter"
    assert command[command.index("--model") + 1] == "vendor/model"


@pytest.mark.parametrize("name", ["first_hero_v1.json", "first_hero_baby_v1.json"])
def test_immutable_hero_slate_sol_models_still_have_backups(name) -> None:
    slate = json.loads((HERO_RUNS_DIR / name).read_text())
    sol = slate["simulators"]["sol"]
    assert sol["model"] == "gpt-5.6-sol"
    target = resolve_backup_world(runtime=sol["runtime"], model=sol["model"], provider=sol["provider"])
    assert (target.runtime, target.model, target.provider) == ("claude", "opus", "claude-cli")


def test_smoke_documentation_matches_parsed_model_and_player_credentials() -> None:
    args = smoke._parser().parse_args([])
    requirements = (PROJECT_ROOT / "docs/ARCHITECTURE.md").read_text().split("## Live smoke episode", 1)[1].split("Run one", 1)[0]
    for expected in (args.gm_model, args.player_base_url, args.player_api_key_env, args.player_model):
        assert f"`{expected}`" in requirements
    assert "PLAYER_LAMBDA_API_KEY" not in requirements
