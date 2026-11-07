"""Focused contract tests for the Pi/OpenRouter world harness adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shellsmith.episode.gm_process.pi import (
    PiWorldConfig,
    PiWorldProcess,
)
from shellsmith.episode.gm_process import (
    WorldHarnessConfig,
    WorldHarnessLauncher,
)


def _process(tmp_path: Path) -> PiWorldProcess:
    return PiWorldProcess(
        PiWorldConfig(
            executable=Path("/opt/homebrew/bin/pi"),
            provider="openrouter",
            model="openrouter/example-model",
            workspace_dir=tmp_path / "world",
            log_dir=tmp_path / "logs",
            bootstrap_response_schema_path=tmp_path / "boot.schema.json",
            api_key_env="TEST_OPENROUTER_KEY",
        )
    )


def test_pi_command_is_resumable_and_machine_config_independent(tmp_path: Path) -> None:
    process = _process(tmp_path)

    command = process._command("fixed-session-id")

    assert command[0] == "/opt/homebrew/bin/pi"
    assert command[command.index("--provider") + 1] == "openrouter"
    assert command[command.index("--model") + 1] == "openrouter/example-model"
    assert command[command.index("--session-id") + 1] == "fixed-session-id"
    assert command[command.index("--session-dir") + 1] == str(tmp_path / "logs/sessions")
    assert "--no-context-files" not in command
    assert "--no-extensions" in command
    assert "--no-skills" in command
    assert "--no-prompt-templates" in command
    assert "--add-dir" not in command
    assert "TEST_OPENROUTER_KEY" not in command
    assert command[command.index("--tools") + 1] == (
        "read,bash,edit,write,grep,find,ls"
    )


def test_world_launcher_rejects_pi_openrouter_adapter(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Pi GM execution is disabled"):
        WorldHarnessLauncher.create(
            WorldHarnessConfig(
                runtime="pi",
                executable=Path("/opt/homebrew/bin/pi"),
                provider="openrouter",
                model="openrouter/example-model",
                workspace_dir=tmp_path / "world",
                log_dir=tmp_path / "logs",
                bootstrap_response_schema_path=tmp_path / "boot.schema.json",
                api_key_env="TEST_OPENROUTER_KEY",
            )
        )

    assert not (tmp_path / "logs").exists()


def test_pi_custom_provider_is_episode_local_and_secret_free(tmp_path: Path) -> None:
    process = PiWorldProcess(
        PiWorldConfig(
            executable=Path("/opt/homebrew/bin/pi"),
            provider="custom-provider",
            model="custom-model",
            workspace_dir=tmp_path / "world",
            log_dir=tmp_path / "logs",
            bootstrap_response_schema_path=tmp_path / "boot.schema.json",
            api_key_env="CUSTOM_PROVIDER_API_KEY",
            base_url="http://lambda.test:30000/v1",
        )
    )

    agent_home = process._prepare_agent_home()
    assert agent_home == tmp_path / "logs/agent_home"
    models = json.loads((agent_home / "models.json").read_text(encoding="utf-8"))
    provider = models["providers"]["custom-provider"]
    assert provider["baseUrl"] == "http://lambda.test:30000/v1"
    assert provider["apiKey"] == "$CUSTOM_PROVIDER_API_KEY"
    assert "secret" not in json.dumps(models).lower()


def test_pi_output_parser_preserves_events_and_extracts_json(tmp_path: Path) -> None:
    output = tmp_path / "pi.jsonl"
    output.write_text(
        "\n".join(
            [
                json.dumps({"type": "session", "id": "session-1"}),
                json.dumps(
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "thinking", "thinking": "private"},
                                {"type": "text", "text": '{"status":"ready"}'},
                            ],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    events, response = PiWorldProcess._parse_output(output)

    assert [event.type for event in events] == ["pi.session", "pi.message_end"]
    assert response == {"status": "ready"}


def test_pi_output_parser_accepts_one_complete_json_fence(tmp_path: Path) -> None:
    output = tmp_path / "pi.jsonl"
    output.write_text(
        "\n".join(
            [
                json.dumps({"type": "session", "id": "session-1"}),
                json.dumps(
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": '```json\n{"status":"ready"}\n```',
                                }
                            ],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _, response = PiWorldProcess._parse_output(output)

    assert response == {"status": "ready"}


def test_pi_output_parser_does_not_extract_json_from_prose(tmp_path: Path) -> None:
    output = tmp_path / "pi.jsonl"
    text = 'Here is the result:\n```json\n{"status":"ready"}\n```'
    output.write_text(
        json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _, response = PiWorldProcess._parse_output(output)

    assert response == {"text": text}


def test_pi_output_parser_rejects_provider_error(tmp_path: Path) -> None:
    output = tmp_path / "pi.jsonl"
    output.write_text(
        json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [],
                    "errorMessage": "401 invalid key",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="401 invalid key"):
        PiWorldProcess._parse_output(output)
