"""Legacy Claude artifact helpers remain offline-only under Codex-only execution."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from shellsmith.episode.gm_process.claude import (
    ClaudeWorldConfig,
    ClaudeWorldProcess,
    _artifact_is_valid,
    _artifact_response,
    _claude_transient_provider_failure,
    _idempotent_recovery_prompt,
)
from shellsmith.episode.gm_process.shared import WorldTurnContext
from test_codex_only_gm import forbidden_gm_boundaries


def _process(tmp_path: Path) -> ClaudeWorldProcess:
    return ClaudeWorldProcess(
        ClaudeWorldConfig(
            executable=Path("/usr/bin/claude"),
            model="claude-test",
            workspace_dir=tmp_path / "world",
            log_dir=tmp_path / "logs",
            bootstrap_response_schema_path=tmp_path / "boot.schema.json",
        )
    )


@pytest.mark.parametrize("reuse_valid_artifact", [False, True])
def test_boot_repair_cannot_enable_claude_with_a_structurally_valid_artifact(
    tmp_path: Path, forbidden_gm_boundaries, reuse_valid_artifact: bool,
) -> None:
    """Neither repair dispatch nor artifact reuse may bypass the execution guard."""
    process = _process(tmp_path)
    process._events_by_session["session-1"] = []
    artifact_path = tmp_path / "response.json"
    artifact_path.write_text('{"wrong_boot_content": true}', encoding="utf-8")

    with forbidden_gm_boundaries(), pytest.raises(RuntimeError, match="Claude GM execution is disabled"):
        asyncio.run(process.resume(
            "session-1", "Please repair the boot response.",
            turn_context=WorldTurnContext(
                stage="bootstrap_repair_01",
                expected_artifact_path=artifact_path,
                reuse_valid_artifact=reuse_valid_artifact,
            ),
        ))

    assert _artifact_response(artifact_path) == {"wrong_boot_content": True}
    assert not process.config.log_dir.exists()
    assert process._events_by_session == {"session-1": []}


@pytest.mark.parametrize("payload,valid", [
    ('{"status":"ready"}', True), ("[]", False), ("not-json", False), (None, False),
])
def test_legacy_artifact_validation_remains_inspectable(tmp_path, payload, valid):
    artifact_path = tmp_path / "response.json"
    if payload is not None:
        artifact_path.write_text(payload)
    assert _artifact_is_valid(artifact_path) is valid
    if valid:
        assert _artifact_response(artifact_path) == {"status": "ready"}
    else:
        with pytest.raises(RuntimeError, match="World artifact|Expected world artifact"):
            _artifact_response(artifact_path)


def test_legacy_recovery_prompt_and_provider_errors_remain_inspectable():
    prompt = _idempotent_recovery_prompt(idempotency_key="boot-1", validation_error="invalid artifact")
    assert "Idempotency key: boot-1" in prompt
    assert "Validation error: invalid artifact" in prompt
    assert "not a new episode" in prompt
    assert _claude_transient_provider_failure({"terminal_reason": "api_error"}, stderr="")
    assert _claude_transient_provider_failure(None, stderr="Connection closed mid-response")
    assert not _claude_transient_provider_failure(None, stderr="invalid artifact")
