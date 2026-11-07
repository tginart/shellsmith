"""Focused tests for the real empty vat and Python↔Node wire boundary.

Architecture map:
    - Pure contract tests reject malformed or stale JSONL messages.
    - Workspace tests prove the actual cwd is AGENTS.md-only.
    - A dependency-free fake Node runner exercises the asynchronous handshake:
      prompt -> tool_request -> tool_response -> prompt_complete.
    - A mocked Pi SDK exercises the real runner's offline initialization path.

The main integration test intentionally does not import Pi or call a model. It proves
the subprocess and suspended-tool control flow before provider configuration is
introduced.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from shellsmith.paths import PACKAGE_ROOT, PROMPTS_DIR
from test_codex_only_gm import forbidden_gm_boundaries
from shellsmith.episode.failure import (
    ExecutionTimeoutError,
)
from shellsmith.episode.vat_process import (
    VAT_SUBPROCESS_STREAM_LIMIT_BYTES,
    PiSdkVatConfig,
    PiSdkVatProcess,
)
from shellsmith.episode.vat_protocol import (
    VAT_PROTOCOL_VERSION,
    VatInitialized,
    VatToolRequest,
    VatToolResponse,
    decode_runner_message,
    initialize_command,
)
from shellsmith.episode.vat_workspace import (
    EphemeralVatWorkspace,
)
from shellsmith.episode.vat_redaction import (
    redact_private_paths,
)
from shellsmith.episode.bridge import (
    VatWorldToolHandler,
    WorldBridge,
    _render_prompt_template,
)
from shellsmith.episode.consistency_ledger import (
    create_ledger,
    load_ledger,
)
from shellsmith.episode.gm_process import (
    ClaudeWorldConfig,
    ClaudeWorldProcess,
    CodexWorldConfig,
    CodexWorldProcess,
    ProcessTurn,
    WorldTurnContext,
)
from shellsmith.episode import (
    orchestrator as orchestrator_module,
)
from shellsmith.episode.orchestrator import (
    EpisodeStatus,
    EpisodeOrchestrator,
    _terminal_trainability,
)
from shellsmith.episode.smoke import (
    _copy_rollout_without_symlinks,
    _parser,
    _run_config,
)
from shellsmith.episode.world_artifacts import (
    FinalResponseArtifact,
    load_final_response,
    load_turn_response,
)


TURN_PROMPT_PATH = PROMPTS_DIR / "WORLD_TURN_PROMPT.md"


def test_post_result_failures_do_not_overwrite_a_settled_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = object.__new__(EpisodeOrchestrator)
    settled_state = SimpleNamespace(status=EpisodeStatus.COMPLETE)
    orchestrator.episode_id = "episode-1"
    orchestrator.spec = SimpleNamespace(id="task-1")
    orchestrator.paths = SimpleNamespace(log_dir=tmp_path)
    orchestrator._latest_state = settled_state
    cleanup_strict_values: list[bool] = []
    emitted: list[dict[str, object]] = []

    def failed_audit() -> None:
        raise OSError("workspace is locked")

    async def failed_cleanup(*, strict: bool) -> None:
        cleanup_strict_values.append(strict)
        raise RuntimeError("process still running")

    def capture_emit(**kwargs: object) -> None:
        emitted.append(kwargs)

    orchestrator._write_vat_workspace_audit = failed_audit
    orchestrator._cleanup = failed_cleanup
    monkeypatch.setattr(orchestrator_module, "emit_simple", capture_emit)

    asyncio.run(orchestrator._finalize_completed_episode())

    assert settled_state.status is EpisodeStatus.COMPLETE
    assert cleanup_strict_values == [True]
    assert (tmp_path / "post_result_errors.log").read_text(encoding="utf-8") == (
        "vat_workspace_audit: OSError: workspace is locked\n"
        "cleanup: RuntimeError: process still running\n"
    )
    assert emitted[-1]["phase"] == "episode_post_result_warning"


def test_prompt_rendering_preserves_template_syntax_in_player_content() -> None:
    rendered = _render_prompt_template(
        template="Turn summary:\n{{PLAYER_TURN_SUMMARY}}\nPath: {{TURN_PATH}}",
        replacements={
            "{{PLAYER_TURN_SUMMARY}}": "write config: {{API_HOST}}",
            "{{TURN_PATH}}": "/tmp/turn",
        },
        prompt_name="World turn prompt",
    )

    assert rendered == "Turn summary:\nwrite config: {{API_HOST}}\nPath: /tmp/turn"


def test_prompt_rendering_rejects_unknown_template_placeholders() -> None:
    with pytest.raises(RuntimeError, match=r"unresolved placeholder: \{\{UNKNOWN\}\}"):
        _render_prompt_template(
            template="Known {{KNOWN}} unknown {{UNKNOWN}}",
            replacements={"{{KNOWN}}": "value"},
            prompt_name="World turn prompt",
        )


def test_vat_subprocess_stream_limit_reads_over_default_64kib_line() -> None:
    """Host↔vat JSONL uses asyncio readline(); default 64 KiB is too small.

    Why this exists:
        Packaging rollout_08 died with
        `ValueError: Separator is found, but chunk is longer than limit` when a
        forced-close thinking dump produced a ~66 KiB JSON line. Prove the vat
        launch limit can read past asyncio's default.
    """

    async def _run() -> None:
        # One JSONL line larger than the asyncio default StreamReader limit.
        fat_line = ("x" * 70_000) + "\n"
        process = await asyncio.create_subprocess_exec(
            "python3",
            "-c",
            "import sys; sys.stdout.write(sys.stdin.read()); sys.stdout.flush()",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=VAT_SUBPROCESS_STREAM_LIMIT_BYTES,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(fat_line.encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()
        line = await process.stdout.readline()
        await process.wait()
        assert len(line) == len(fat_line)
        assert line.decode("utf-8") == fat_line

    assert VAT_SUBPROCESS_STREAM_LIMIT_BYTES > 65_536
    asyncio.run(_run())


def test_live_smoke_defaults_to_codex_sol_gm_and_openrouter_qwen_player() -> None:
    args = _parser().parse_args([])

    assert args.gm_runtime == "codex"
    assert args.gm_model == "gpt-6.1-sol"
    assert args.gm_provider == "codex-cli"
    assert args.gm_reasoning_effort == "medium"
    assert args.player_provider == "openrouter"
    assert args.player_model == "qwen/qwen3.8-27b"
    assert args.player_base_url == "https://openrouter.ai/api/v1"
    assert args.player_context_window == 65_536
    assert args.max_world_steps == 256
    config = _run_config(args, episode_id="episode-sol-light")
    assert config["gm_reasoning_effort"] == "medium"
    assert config["credential_environment_variable"] == "OPENROUTER_API_KEY"
    assert "credential_value" not in config


def test_claude_world_rejects_before_installing_native_agents_import(
    tmp_path: Path, forbidden_gm_boundaries,
) -> None:
    workspace = tmp_path / "world"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("canonical instructions\n")
    schema = tmp_path / "boot.schema.json"
    schema.write_text("{}\n")
    process = ClaudeWorldProcess(
        ClaudeWorldConfig(
            executable=Path("/usr/bin/false"),
            model="sonnet",
            workspace_dir=workspace,
            log_dir=tmp_path / "logs",
            bootstrap_response_schema_path=schema,
        )
    )

    with forbidden_gm_boundaries(), pytest.raises(RuntimeError, match="Claude GM execution is disabled"):
        asyncio.run(process.start("boot"))

    assert not (workspace / "CLAUDE.md").exists()
    assert (workspace / "AGENTS.md").read_text() == "canonical instructions\n"
    assert not process.config.log_dir.exists()


def test_claude_world_rejects_valid_artifact_instead_of_resuming_execution(
    tmp_path: Path, forbidden_gm_boundaries,
) -> None:
    workspace = tmp_path / "world"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("canonical instructions\n")
    schema = tmp_path / "boot.schema.json"
    schema.write_text("{}\n")
    artifact_path = workspace / "world" / "bootstrap" / "response.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text('{"status": "ready"}\n', encoding="utf-8")
    process = ClaudeWorldProcess(
        ClaudeWorldConfig(
            executable=tmp_path / "must-not-execute-claude",
            model="sonnet",
            workspace_dir=workspace,
            log_dir=tmp_path / "logs",
            bootstrap_response_schema_path=schema,
        )
    )

    with forbidden_gm_boundaries(), pytest.raises(RuntimeError, match="Claude GM execution is disabled"):
        asyncio.run(process.start(
            "boot prompt",
            turn_context=WorldTurnContext(
                stage="bootstrap",
                expected_artifact_path=artifact_path,
                idempotency_key="boot-1",
                output_schema_path=schema,
            ),
        ))

    assert json.loads(artifact_path.read_text()) == {"status": "ready"}
    assert not process.config.log_dir.exists()
    assert not (workspace / "CLAUDE.md").exists()


def test_codex_world_uses_episode_only_permission_profile(tmp_path: Path) -> None:
    process = CodexWorldProcess(
        CodexWorldConfig(
            executable=Path("/usr/bin/codex"),
            model="gpt-test",
            workspace_dir=tmp_path,
            log_dir=tmp_path / "logs",
            bootstrap_response_schema_path=tmp_path / "boot.schema.json",
            additional_writable_dirs=(tmp_path / "vat",),
        )
    )

    command = process._base_command(resume=False)
    resume_command = process._base_command(resume=True)
    joined = " ".join(command)

    assert "--sandbox" not in command
    assert "sandbox_mode" not in joined
    assert 'default_permissions="game_master"' in command
    assert '":minimal"="read"' in joined
    assert '":workspace_roots"={ "."="write" }' in joined
    assert "network={ enabled=true }" in joined
    assert 'shell_environment_policy.inherit="core"' in command
    assert command[command.index("--add-dir") + 1] == str(tmp_path / "vat")
    assert resume_command[resume_command.index("--add-dir") + 1] == str(
        tmp_path / "vat"
    )
    assert resume_command.index("--add-dir") < resume_command.index("resume")


def test_codex_world_full_access_mode_drops_codex_sandbox(tmp_path: Path) -> None:
    from shellsmith.episode.gm_process.codex import (
        SANDBOX_FULL_ACCESS,
    )

    process = CodexWorldProcess(
        CodexWorldConfig(
            executable=Path("/usr/bin/codex"),
            model="gpt-test",
            workspace_dir=tmp_path,
            log_dir=tmp_path / "logs",
            bootstrap_response_schema_path=tmp_path / "boot.schema.json",
            additional_writable_dirs=(tmp_path / "vat",),
            sandbox_mode=SANDBOX_FULL_ACCESS,
        )
    )

    command = process._base_command(resume=False)
    resume_command = process._base_command(resume=True)
    joined = " ".join(command)

    # ``codex exec resume`` rejects ``--sandbox``; the config key works for both.
    assert "--sandbox" not in command
    assert "--sandbox" not in resume_command
    assert command[command.index('sandbox_mode="danger-full-access"') - 1] == "-c"
    assert 'sandbox_mode="danger-full-access"' in resume_command
    assert "default_permissions" not in joined
    assert "permissions.game_master" not in joined
    assert 'shell_environment_policy.inherit="core"' in command
    assert command[command.index("--add-dir") + 1] == str(tmp_path / "vat")


def test_codex_world_rejects_unknown_sandbox_mode(tmp_path: Path) -> None:
    process = CodexWorldProcess(
        CodexWorldConfig(
            executable=Path("/usr/bin/codex"),
            model="gpt-test",
            workspace_dir=tmp_path,
            log_dir=tmp_path / "logs",
            bootstrap_response_schema_path=tmp_path / "boot.schema.json",
            sandbox_mode="read-only",
        )
    )

    with pytest.raises(RuntimeError, match="Unknown Codex world sandbox mode"):
        process._base_command(resume=False)


def test_codex_world_detects_bubblewrap_namespace_failure(tmp_path: Path) -> None:
    """Codex exits 0 when bwrap cannot start; the turn must still fail loudly."""
    from shellsmith.episode.gm_process.codex import (
        BWRAP_NAMESPACE_FAILURE,
        _require_working_sandbox,
    )
    from shellsmith.episode.gm_process.shared import (
        normalized_event,
    )

    healthy = (
        normalized_event({"type": "thread.started", "thread_id": "t1"}),
        normalized_event(
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "exit_code": 0,
                    "aggregated_output": "ok\n",
                },
            }
        ),
    )
    _require_working_sandbox(healthy, stage="bootstrap", log_path=tmp_path / "b.jsonl")

    broken = healthy + (
        normalized_event(
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "exit_code": 1,
                    "aggregated_output": f"{BWRAP_NAMESPACE_FAILURE}, likely ...\n",
                },
            }
        ),
    )
    with pytest.raises(RuntimeError, match="cannot create user namespaces") as error:
        _require_working_sandbox(broken, stage="bootstrap", log_path=tmp_path / "b.jsonl")
    assert "do not disable confinement" in str(error.value)
    assert "danger-full-access" not in str(error.value)


def test_rollout_archive_rejects_model_authored_symlink(tmp_path: Path) -> None:
    source = tmp_path / "staging"
    simulation_root = source / "world" / "simulation_root"
    simulation_root.mkdir(parents=True)
    private_file = tmp_path / "private-host-data.txt"
    private_file.write_text("must not be archived", encoding="utf-8")
    (simulation_root / "leak").symlink_to(private_file)
    (simulation_root / "kept.txt").write_text("real artifact", encoding="utf-8")
    destination = tmp_path / "durable"

    _copy_rollout_without_symlinks(source, destination)

    # The symlink is skipped (never followed, target never copied) and the
    # sanitization manifest records it; the rest of the rollout survives.
    assert not (destination / "world" / "simulation_root" / "leak").exists()
    assert (destination / "world" / "simulation_root" / "kept.txt").is_file()
    manifest = json.loads(
        (destination / "archive_sanitization.json").read_text(encoding="utf-8")
    )
    assert manifest["skipped_entries"] == [
        {"path": "world/simulation_root/leak", "reason": "symlink"}
    ]


def test_rollout_archive_omits_ephemeral_runtime_with_internal_symlinks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "staging"
    runtime = source / "world" / "runtime_scratch" / "node" / "bin"
    runtime.mkdir(parents=True)
    executable = runtime / "node-real"
    executable.write_text("runtime", encoding="utf-8")
    (runtime / "node").symlink_to("node-real")
    durable = source / "world" / "simulation_root" / "result.txt"
    durable.parent.mkdir(parents=True)
    durable.write_text("keep", encoding="utf-8")
    destination = tmp_path / "durable"

    _copy_rollout_without_symlinks(source, destination)

    assert not (destination / "world" / "runtime_scratch").exists()
    assert (destination / "world" / "simulation_root" / "result.txt").read_text() == "keep"


def test_rollout_archive_rejects_non_regular_entry(tmp_path: Path) -> None:
    source = tmp_path / "staging"
    simulation_root = source / "world" / "simulation_root"
    simulation_root.mkdir(parents=True)
    os.mkfifo(simulation_root / "model-created-pipe")

    destination = tmp_path / "durable"
    _copy_rollout_without_symlinks(source, destination)

    assert not (destination / "world" / "simulation_root" / "model-created-pipe").exists()
    manifest = json.loads(
        (destination / "archive_sanitization.json").read_text(encoding="utf-8")
    )
    assert manifest["skipped_entries"][0]["reason"] == "non-regular entry"


def test_rollout_archive_rejects_multiply_linked_file(tmp_path: Path) -> None:
    source = tmp_path / "staging"
    simulation_root = source / "world" / "simulation_root"
    simulation_root.mkdir(parents=True)
    first = simulation_root / "first.txt"
    first.write_text("shared inode", encoding="utf-8")
    os.link(first, simulation_root / "second.txt")

    destination = tmp_path / "durable"
    _copy_rollout_without_symlinks(source, destination)

    manifest = json.loads(
        (destination / "archive_sanitization.json").read_text(encoding="utf-8")
    )
    reasons = {entry["reason"] for entry in manifest["skipped_entries"]}
    assert reasons == {"multiply linked file"}
    # Both hard-linked names are skipped; only unshared files survive.
    archived_root = destination / "world" / "simulation_root"
    assert not (archived_root / "first.txt").exists()
    assert not (archived_root / "second.txt").exists()


def test_protocol_initialization_contains_no_secret_value(tmp_path: Path) -> None:
    command = initialize_command(
        request_id="initialize-1",
        episode_id="episode-1",
        workspace_path=str(tmp_path / "workspace"),
        agent_home_path=str(tmp_path / "agent-home"),
        model_provider="tinker",
        model_id="qwen-3.6",
        model_base_url="https://example.invalid/v1",
        model_api="openai-completions",
        model_context_window=204_800,
        model_max_tokens=32_768,
        model_reasoning=True,
        thinking_level="high",
        api_key_env="TINKER_API_KEY",
    )

    assert command["protocol_version"] == VAT_PROTOCOL_VERSION
    assert command["model"] == {
        "provider": "tinker",
        "id": "qwen-3.6",
        "base_url": "https://example.invalid/v1",
        "api": "openai-completions",
        "context_window": 204_800,
        "max_tokens": 32_768,
        "reasoning": True,
        "thinking_level": "high",
        "api_key_env": "TINKER_API_KEY",
    }
    assert "secret" not in json.dumps(command).lower()


def test_world_facing_json_redacts_real_vat_paths_recursively() -> None:
    private_root = "/private/tmp/shellsmith-vat-123"
    private_workspace = private_root + "/workspace"
    value = {
        "command": f"cd {private_workspace} && pytest",
        "notes": [f"cwd={private_workspace}", f"vat={private_root}"],
    }

    redacted = redact_private_paths(
        value,
        replacements=(
            (private_workspace, "/workspace"),
            (private_root, "/vat"),
        ),
    )

    assert redacted == {
        "command": "cd /workspace && pytest",
        "notes": ["cwd=/workspace", "vat=/vat"],
    }
    assert private_root not in json.dumps(redacted)


def test_protocol_decodes_tool_requests_and_rejects_wrong_versions() -> None:
    decoded = decode_runner_message(
        json.dumps(
            {
                "protocol_version": VAT_PROTOCOL_VERSION,
                "type": "tool_request",
                "tool_call_id": "call-1",
                "tool_name": "bash",
                "arguments": {"command": "pwd"},
            }
        )
    )

    assert isinstance(decoded, VatToolRequest)
    assert decoded.arguments == {"command": "pwd"}

    with pytest.raises(RuntimeError, match="version mismatch"):
        decode_runner_message(
            json.dumps(
                {
                    "protocol_version": 999,
                    "type": "initialized",
                    "request_id": "x",
                    "session_id": "x",
                    "enabled_tools": ["bash"],
                }
            )
        )


def test_ephemeral_workspace_is_empty_then_agents_only(tmp_path: Path) -> None:
    vat = EphemeralVatWorkspace.create(temp_parent=tmp_path / "vats")
    try:
        assert vat.root.parent == (tmp_path / "vats").resolve()
        assert vat.workspace_entries() == ()
        vat.agents_path.write_text("# Workspace instructions\n", encoding="utf-8")
        vat.assert_agents_only()

        (vat.workspace_dir / "simulated.py").write_text("pass\n", encoding="utf-8")
        vat.assert_agents_only()
        vat.agents_path.write_text("# Changed instructions\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="changed after Pi initialization"):
            vat.assert_agents_only()
    finally:
        root = vat.root
        vat.close()
        assert not root.exists()


def test_turn_response_loader_preserves_private_scores_and_public_result(
    tmp_path: Path,
) -> None:
    path = tmp_path / "response.json"
    path.write_text(
        json.dumps(
            {
                "action_id": "call-1",
                "tool_response": {
                    "content": "37 rows",
                    "is_error": False,
                    "details": {"format": "text"},
                },
                "turn_reward": 0.4,
                "difficulty_rating": 0.7,
                "completion_percentage": 25,
                "commentary": "The query used the correct grouping key.",
                "reasoning_assessment": {
                    "quality": 0.9,
                    "efficiency": 0.8,
                    "commentary": "The visible reasoning was direct and causal.",
                },
                "consistency_review": {
                    "reviewed_revision": 0,
                    "observation_claims": [],
                    "new_commitments": [],
                    "conflicts": [],
                    "executed_cross_checks": [],
                },
                "trainable": True,
                "integrity_issues": [],
                "terminate_episode": True,
                "termination_reason": "A prohibited production mutation succeeded.",
                "private_state_delta": {"scope_known": True},
            }
        ),
        encoding="utf-8",
    )

    artifact = load_turn_response(path, expected_action_id="call-1")

    assert artifact.tool_response.content == "37 rows"
    assert artifact.turn_reward == 0.4
    assert artifact.reasoning_assessment is not None
    assert artifact.reasoning_assessment.quality == 0.9
    assert artifact.reasoning_assessment.efficiency == 0.8
    assert artifact.terminate_episode is True
    assert artifact.termination_reason == "A prohibited production mutation succeeded."
    assert artifact.metadata == {"private_state_delta": {"scope_known": True}}

    with pytest.raises(RuntimeError, match="action ID mismatch"):
        load_turn_response(path, expected_action_id="stale-call")


def test_sibling_slots_accept_id_as_action_id_alias(tmp_path: Path) -> None:
    """Live worlds mirror the request's ``actions[].id`` key on response slots.

    The geospatial Gate 6 failure wrote ``tool_results[].id`` three times in a
    row; the loader now accepts that alias (the id-set equality check still
    enforces exact correlation) while conflicting keys stay a hard error.
    """
    path = tmp_path / "response.json"
    payload = {
        "action_id": "call-a",
        "tool_results": [
            {
                "id": "call-a",
                "tool_response": {"content": "obs a", "is_error": False},
            },
            {
                "id": "call-b",
                "tool_response": {"content": "obs b", "is_error": False},
            },
        ],
        "turn_reward": 0.1,
        "difficulty_rating": 0.5,
        "completion_percentage": 10,
        "commentary": "Sibling batch with legacy slot keys.",
        "reasoning_assessment": None,
        "consistency_review": {
            "reviewed_revision": 0,
            "observation_claims": [],
            "new_commitments": [],
            "conflicts": [],
            "executed_cross_checks": [],
        },
        "trainable": True,
        "integrity_issues": [],
        "terminate_episode": False,
        "termination_reason": None,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    artifact = load_turn_response(
        path,
        expected_action_id="call-a",
        expected_action_ids=("call-a", "call-b"),
    )

    assert [slot.action_id for slot in artifact.tool_slots] == ["call-a", "call-b"]
    assert artifact.tool_slots[1].tool_response.content == "obs b"

    payload["tool_results"][0]["action_id"] = "call-other"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="conflicting 'id' and 'action_id'"):
        load_turn_response(
            path,
            expected_action_id="call-a",
            expected_action_ids=("call-a", "call-b"),
        )


def test_final_response_loader_keeps_final_reward_independent(tmp_path: Path) -> None:
    path = tmp_path / "final-response.json"
    path.write_text(
        json.dumps(
            {
                "final_reward": 0.72,
                "completion_percentage": 80,
                "trainable": False,
                "criteria": [{"name": "outcome", "score": 0.9}],
                "process_deductions": [
                    {"reason": "lucky destructive shortcut", "amount": 0.18}
                ],
                "commentary": "The outcome was mostly correct, but the route was poor.",
                "integrity_issues": ["The simulator exposed an inconsistent timestamp."],
                "factual_consistency": {
                    "passed": False,
                    "contradictions": ["Two timestamps disagreed."],
                },
                "task_outcomes": {"example": {"fixed": False}},
                "private_terminal_state": {"ticket_closed": True},
            }
        ),
        encoding="utf-8",
    )

    artifact = load_final_response(path)

    assert artifact.final_reward == 0.72
    assert artifact.trainable is False
    assert artifact.factual_consistency_passed is False
    assert artifact.factual_contradictions == ("Two timestamps disagreed.",)
    assert artifact.task_outcomes == {"example": {"fixed": False}}
    assert artifact.process_deductions[0]["amount"] == 0.18
    assert artifact.metadata == {"private_terminal_state": {"ticket_closed": True}}


def test_turn_response_rejects_object_content_before_vat_delivery(tmp_path: Path) -> None:
    path = tmp_path / "turn-response.json"
    path.write_text(
        json.dumps(
            {
                "action_id": "call-api",
                "tool_response": {
                    "content": {"orders": []},
                    "is_error": False,
                },
                "turn_reward": 0.0,
                "difficulty_rating": 0.2,
                "completion_percentage": 5,
                "commentary": "API discovery.",
                "reasoning_assessment": None,
                "trainable": True,
                "integrity_issues": [],
                "terminate_episode": False,
                "termination_reason": None,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="must be a string"):
        load_turn_response(path, expected_action_id="call-api")


def test_terminal_integrity_overrides_inconsistent_trainable_boolean(
    tmp_path: Path,
) -> None:
    artifact = FinalResponseArtifact(
        final_reward=0.8,
        completion_percentage=90,
        trainable=True,
        criteria=(),
        process_deductions=(),
        commentary="Reward remains independently useful.",
        integrity_issues=("Simulator contradicted its ledger.",),
        factual_consistency_passed=True,
        factual_contradictions=(),
        task_outcomes={},
        metadata={},
        path=tmp_path / "final.json",
    )

    trainable, issues = _terminal_trainability(artifact, ())

    assert trainable is False
    assert issues == ("Simulator contradicted its ledger.",)

    factual_failure = FinalResponseArtifact(
        **{
            **artifact.__dict__,
            "integrity_issues": (),
            "factual_consistency_passed": False,
            "factual_contradictions": ("Two file sizes conflict.",),
        }
    )
    trainable, issues = _terminal_trainability(factual_failure, ())

    assert trainable is False
    assert issues == ()


def test_python_host_services_a_suspended_node_tool_request(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the vat JSONL integration test.")

    fake_runner = tmp_path / "fake-vat-runner.mjs"
    fake_runner.write_text(
        textwrap.dedent(
            """
            import readline from "node:readline";

            const protocolVersion = 1;
            const send = (message) => {
              process.stdout.write(JSON.stringify({ protocol_version: protocolVersion, ...message }) + "\\n");
            };
            const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
            let promptRequestId = null;
            for await (const line of input) {
              const message = JSON.parse(line);
              if (message.type === "initialize") {
                send({
                  type: "initialized",
                  request_id: message.request_id,
                  session_id: "fake-pi-session",
                  enabled_tools: message.enabled_tools,
                });
              } else if (message.type === "prompt") {
                promptRequestId = message.request_id;
                send({
                  type: "agent_event",
                  event_type: "message_end",
                  payload: { message: { role: "assistant", content: [
                    { type: "toolCall", id: "call-pwd", name: "bash", arguments: { command: "pwd" } }
                  ] } },
                });
                send({
                  type: "tool_request",
                  tool_call_id: "call-pwd",
                  tool_name: "bash",
                  arguments: { command: "pwd" },
                });
              } else if (message.type === "tool_response") {
                send({
                  type: "agent_event",
                  event_type: "tool_execution_end",
                  payload: { toolCallId: message.tool_call_id },
                });
                send({
                  type: "prompt_complete",
                  request_id: promptRequestId,
                  final_text: "Observed the simulated cwd.",
                  message_count: 3,
                  terminated: false,
                });
              } else if (message.type === "shutdown") {
                send({ type: "shutdown_complete", request_id: message.request_id });
                process.exit(0);
              }
            }
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    async def run() -> None:
        with EphemeralVatWorkspace.create(temp_parent=tmp_path / "vats") as vat:
            vat.agents_path.write_text("# Stay inside the workspace.\n", encoding="utf-8")
            process = PiSdkVatProcess(
                PiSdkVatConfig(
                    node_executable=Path(node),
                    runner_script=fake_runner,
                    episode_id="episode-wire-test",
                    workspace=vat,
                    model_provider="unused",
                    model_id="unused",
                    model_base_url="https://example.invalid/v1",
                    thinking_level="off",
                    raw_log_path=tmp_path / "logs" / "vat.jsonl",
                    enabled_tools=("bash",),
                )
            )
            initialized = await process.launch()
            assert initialized == "fake-pi-session"

            requests: list[VatToolRequest] = []

            async def handle_tool(
                request: VatToolRequest,
                trajectory_delta: tuple[object, ...],
            ) -> VatToolResponse:
                assert len(trajectory_delta) == 1
                requests.append(request)
                return VatToolResponse(
                    tool_call_id=request.tool_call_id,
                    content="/simulated/workspace\n",
                    is_error=False,
                )

            result = await process.prompt("Inspect the cwd.", tool_handler=handle_tool)
            assert result.final_text == "Observed the simulated cwd."
            assert result.message_count == 3
            assert result.terminated is False
            assert requests == [
                VatToolRequest(
                    tool_call_id="call-pwd",
                    tool_name="bash",
                    arguments={"command": "pwd"},
                )
            ]
            assert result.events[-1].event_type == "tool_execution_end"
            vat.assert_agents_only()
            await process.close()

    asyncio.run(run())

    audit_rows = [
        json.loads(line)
        for line in (tmp_path / "logs" / "vat.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    initialized_messages = [
        VatInitialized(
            request_id=row["message"]["request_id"],
            session_id=row["message"]["session_id"],
            enabled_tools=tuple(row["message"]["enabled_tools"]),
        )
        for row in audit_rows
        if row["message"]["type"] == "initialized"
    ]
    assert initialized_messages[0].session_id == "fake-pi-session"
    assert any(row["message"]["type"] == "tool_response" for row in audit_rows)


def test_vat_initialize_timeout_names_the_local_boundary(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the vat initialization timeout test.")

    stalled_runner = tmp_path / "stalled-runner.mjs"
    stalled_runner.write_text(
        "process.stdin.resume(); await new Promise(() => {});\n",
        encoding="utf-8",
    )

    async def run() -> None:
        with EphemeralVatWorkspace.create(temp_parent=tmp_path / "vats") as vat:
            vat.agents_path.write_text("# Remain isolated.\n", encoding="utf-8")
            process = PiSdkVatProcess(
                PiSdkVatConfig(
                    node_executable=Path(node),
                    runner_script=stalled_runner,
                    episode_id="initialize-timeout-test",
                    workspace=vat,
                    model_provider="unused",
                    model_id="unused",
                    model_base_url="http://127.0.0.1:9/v1",
                    thinking_level="off",
                    raw_log_path=tmp_path / "logs" / "vat.jsonl",
                    enabled_tools=("read",),
                    initialize_timeout_seconds=0.05,
                )
            )
            with pytest.raises(ExecutionTimeoutError) as caught:
                await process.launch()
            assert caught.value.component == "player"
            assert caught.value.stage == "player_initialize"
            assert caught.value.timeout_seconds == 0.05
            assert caught.value.provider_attributable is False

    asyncio.run(run())


def test_real_vat_runner_disables_model_catalog_network_during_initialize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setting the local proxy key must not refresh Pi's remote catalogs.

    Pi 0.80.x controls catalog access at ``ModelRuntime.create``; Pi 0.81+
    additionally requires ``allowNetwork: false`` when setting a runtime key.
    Try3 launched eight vats in parallel; every runner then stalled before
    ``initialized`` and hit the host's 30-second initialization deadline.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the real vat-runner integration test.")

    fake_sdk = tmp_path / "fake-pi-sdk"
    fake_dist = fake_sdk / "dist"
    fake_core = fake_dist / "core"
    fake_core.mkdir(parents=True)
    (fake_sdk / "package.json").write_text('{"type":"module"}\n')
    (fake_core / "auth-storage.js").write_text(
        "export class AuthStorage { static inMemory() { return {}; } }\n"
    )
    (fake_core / "http-dispatcher.js").write_text(
        "export function configureHttpDispatcher(timeout) { "
        "globalThis.piHttpTimeout = timeout; }\n"
    )
    (fake_dist / "index.js").write_text(
        textwrap.dedent(
            """
            export const SettingsManager = {
              inMemory: (options) => {
                if (options?.compaction?.reserveTokens !== 65536) {
                  throw new Error("unsafe compaction reserve");
                }
                return {};
              }
            };
            export const SessionManager = { inMemory: () => ({}) };
            export class DefaultResourceLoader {
              constructor() {}
              async reload() {}
            }
            export class ModelRuntime {
              static async create(options) {
                if (globalThis.piHttpTimeout !== 900000) {
                  await new Promise(() => {});
                }
                if (options?.allowModelNetwork !== false) {
                  await new Promise(() => {});
                }
                return {
                  async setRuntimeApiKey(_provider, _key, options) {
                    if (options?.allowNetwork !== false) {
                      await new Promise(() => {});
                    }
                  },
                  registerProvider(_provider, config) { this.model = config.models[0]; },
                  getModel() { return this.model; },
                };
              }
            }
            export function createReadTool() { return { name: "read" }; }
            export async function createAgentSession() {
              return {
                session: {
                  sessionId: "offline-initialized",
                  messages: [],
                  subscribe() { return () => {}; },
                  dispose() {},
                  async abort() {},
                },
                extensionsResult: { errors: [] },
              };
            }
            """
        ).strip()
        + "\n"
    )
    runner = PACKAGE_ROOT / "episode" / "vat_runner" / "runner.js"
    monkeypatch.setenv("VAT_OFFLINE_TEST_KEY", "local-only")

    async def run() -> None:
        with EphemeralVatWorkspace.create(temp_parent=tmp_path / "vats") as vat:
            vat.agents_path.write_text("# Remain isolated.\n", encoding="utf-8")
            process = PiSdkVatProcess(
                PiSdkVatConfig(
                    node_executable=Path(node),
                    runner_script=runner,
                    episode_id="offline-initialize-test",
                    workspace=vat,
                    model_provider="local-test-provider",
                    model_id="local-test-model",
                    model_base_url="http://127.0.0.1:9/v1",
                    thinking_level="high",
                    api_key_env="VAT_OFFLINE_TEST_KEY",
                    pi_sdk_module_path=fake_dist / "index.js",
                    raw_log_path=tmp_path / "logs" / "vat.jsonl",
                    enabled_tools=("read",),
                    initialize_timeout_seconds=2.0,
                )
            )
            assert await process.launch() == "offline-initialized"
            await process.close()

    asyncio.run(run())


def test_sibling_tool_calls_share_their_assistant_message_context(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the vat JSONL integration test.")

    fake_runner = tmp_path / "fake-sibling-runner.mjs"
    fake_runner.write_text(
        textwrap.dedent(
            """
            import readline from "node:readline";
            const send = (message) => process.stdout.write(JSON.stringify({ protocol_version: 1, ...message }) + "\\n");
            const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
            let promptRequestId = null;
            let responseCount = 0;
            const responses = [];
            for await (const line of input) {
              const message = JSON.parse(line);
              if (message.type === "initialize") {
                send({ type: "initialized", request_id: message.request_id, session_id: "siblings", enabled_tools: message.enabled_tools });
              } else if (message.type === "prompt") {
                promptRequestId = message.request_id;
                send({
                  type: "agent_event",
                  event_type: "message_end",
                  payload: { message: { role: "assistant", content: [
                    { type: "thinking", thinking: "Inspect both independent files." },
                    { type: "toolCall", id: "call-a", name: "read", arguments: { path: "a" } },
                    { type: "toolCall", id: "call-b", name: "read", arguments: { path: "b" } }
                  ] } }
                });
                send({ type: "tool_request", tool_call_id: "call-a", tool_name: "read", arguments: { path: "a" } });
                send({ type: "tool_request", tool_call_id: "call-b", tool_name: "read", arguments: { path: "b" } });
              } else if (message.type === "tool_response") {
                responses.push(message);
                responseCount += 1;
                if (responseCount === 2) {
                  if (responses[0].terminate || !responses[1].terminate) {
                    throw new Error("termination was not deferred to the final sibling response");
                  }
                  send({ type: "prompt_complete", request_id: promptRequestId, final_text: "done", message_count: 2, terminated: false });
                }
              } else if (message.type === "shutdown") {
                send({ type: "shutdown_complete", request_id: message.request_id });
                process.exit(0);
              }
            }
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    async def run() -> None:
        with EphemeralVatWorkspace.create(temp_parent=tmp_path / "vats") as vat:
            vat.agents_path.write_text("# Use virtual tools.\n", encoding="utf-8")
            process = PiSdkVatProcess(
                PiSdkVatConfig(
                    node_executable=Path(node),
                    runner_script=fake_runner,
                    episode_id="sibling-context",
                    workspace=vat,
                    model_provider="unused",
                    model_id="unused",
                    model_base_url="https://example.invalid/v1",
                    thinking_level="off",
                    raw_log_path=tmp_path / "logs" / "vat.jsonl",
                    enabled_tools=("read",),
                )
            )
            await process.launch()
            observed: list[tuple[VatToolRequest, tuple[object, ...]]] = []

            async def handle_tool(
                request: VatToolRequest,
                trajectory_delta: tuple[object, ...],
            ) -> VatToolResponse:
                observed.append((request, trajectory_delta))
                return VatToolResponse(
                    tool_call_id=request.tool_call_id,
                    content=f"contents of {request.arguments['path']}",
                    is_error=False,
                    terminate=request.tool_call_id == "call-a",
                )

            await process.prompt("Read both files.", tool_handler=handle_tool)
            await process.close()

            assert [request.tool_call_id for request, _ in observed] == [
                "call-a",
                "call-b",
            ]
            assert observed[0][1] == observed[1][1]
            assert len(observed[0][1]) == 1
            assert observed[0][1][0].event_type == "message_end"

    asyncio.run(run())


def test_vat_world_handler_writes_private_turn_and_returns_only_tool_result(
    tmp_path: Path,
) -> None:
    turns_dir = tmp_path / "world" / "turns"

    class FakeWorldProcess:
        async def start(self, prompt: str) -> ProcessTurn:
            raise AssertionError(prompt)

        async def resume(
            self,
            session_id: str,
            prompt: str,
            *,
            output_schema_path: Path | None = None,
            turn_context=None,
        ) -> ProcessTurn:
            assert session_id == "world-thread-1"
            assert output_schema_path is None
            turn_dir = turns_dir / "000000"
            request = json.loads(
                (turn_dir / "request.json").read_text(encoding="utf-8")
            )
            assert str(turn_dir.resolve()) in prompt
            (turn_dir / "response.json").write_text(
                json.dumps(
                    {
                        "action_id": request["action"]["id"],
                        "tool_response": {
                            "content": "/simulated/workspace\n",
                            "is_error": False,
                            "details": {"exit_code": 0},
                        },
                        "turn_reward": 0.1,
                        "difficulty_rating": 0.05,
                            "completion_percentage": 1,
                            "commentary": "A harmless orientation action.",
                            "consistency_review": {
                                "reviewed_revision": 0,
                                "observation_claims": [],
                                "new_commitments": [],
                                "conflicts": [],
                                "executed_cross_checks": [],
                            },
                            "trainable": True,
                        "integrity_issues": [],
                    }
                ),
                encoding="utf-8",
            )
            return ProcessTurn(
                session_id=session_id,
                events=(),
                final_response={},
                raw_log_path=tmp_path / "world.jsonl",
            )

        async def transcript(self, session_id: str) -> tuple[object, ...]:
            raise AssertionError(session_id)

        async def close(self) -> None:
            return None

    async def run() -> None:
        ledger_path = tmp_path / "world" / "consistency_ledger.json"
        ledger_path.parent.mkdir(parents=True)
        create_ledger(ledger_path, boot_commitments=[])
        bridge = WorldBridge(
            gm_agent_process=FakeWorldProcess(),
            world_session_id="world-thread-1",
            turns_dir=turns_dir,
            simulation_root=tmp_path / "simulation-root",
            consistency_ledger_path=ledger_path,
            private_path_replacements=(),
            turn_prompt_template_path=TURN_PROMPT_PATH,
        )
        handler = VatWorldToolHandler(
            episode_id="episode-1", bridge=bridge, max_world_steps=1
        )

        response = await handler(
            VatToolRequest(
                tool_call_id="call-pwd",
                tool_name="bash",
                arguments={"command": "pwd"},
            ),
            (),
        )

        # max_world_steps=1: the first (and only) world step runs normally,
        # then terminate=True so Player does not get another tool round.
        assert response.content == "/simulated/workspace\n"
        assert response.details == {"exit_code": 0}
        assert response.terminate is True
        assert handler.budget_exhausted is True
        assert handler.next_step_index == 1
        assert handler.termination_reason is not None
        assert "budget" in handler.termination_reason.lower()
        assert handler.observations[0].assessment.turn_reward == 0.1
        assert "turn_reward" not in json.dumps(response.to_wire())
        request = json.loads(
            (turns_dir / "000000" / "request.json").read_text(encoding="utf-8")
        )
        assert request["action"]["name"] == "bash"

        try:
            await handler(
                VatToolRequest(
                    tool_call_id="call-budget-after-cap",
                    tool_name="read",
                    arguments={"path": "README.md"},
                ),
                (),
            )
        except RuntimeError as error:
            assert "already exhausted" in str(error)
        else:
            raise AssertionError(
                "expected RuntimeError when Player tools after the N-step budget"
            )
        assert len(handler.observations) == 1

    asyncio.run(run())


def test_world_bridge_repairs_missing_response_on_same_world_thread(
    tmp_path: Path,
) -> None:
    turns_dir = tmp_path / "world" / "turns"

    class RepairingWorldProcess:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def start(self, prompt: str) -> ProcessTurn:
            raise AssertionError(prompt)

        async def resume(
            self,
            session_id: str,
            prompt: str,
            *,
            output_schema_path: Path | None = None,
            turn_context=None,
        ) -> ProcessTurn:
            assert session_id == "world-thread-repair"
            assert output_schema_path is None
            self.prompts.append(prompt)
            turn_dir = turns_dir / "000000"
            if len(self.prompts) == 2:
                request = json.loads(
                    (turn_dir / "request.json").read_text(encoding="utf-8")
                )
                (turn_dir / "response.json").write_text(
                    json.dumps(
                        {
                            "action_id": request["action"]["id"],
                            "tool_response": {
                                "content": "AGENTS.md\n",
                                "is_error": False,
                            },
                            "turn_reward": 0.0,
                            "difficulty_rating": 0.0,
                                "completion_percentage": 0.0,
                                "commentary": "Orientation only.",
                                "consistency_review": {
                                    "reviewed_revision": 0,
                                    "observation_claims": [],
                                    "new_commitments": [],
                                    "conflicts": [],
                                    "executed_cross_checks": [],
                                },
                                "trainable": True,
                            "integrity_issues": [],
                        }
                    ),
                    encoding="utf-8",
                )
            return ProcessTurn(
                session_id=session_id,
                events=(),
                final_response={"text": "turn complete"},
                raw_log_path=tmp_path / f"world-{len(self.prompts)}.jsonl",
            )

        async def transcript(self, session_id: str) -> tuple[object, ...]:
            raise AssertionError(session_id)

        async def close(self) -> None:
            return None

    async def run() -> None:
        process = RepairingWorldProcess()
        ledger_path = tmp_path / "world" / "consistency_ledger.json"
        ledger_path.parent.mkdir(parents=True)
        create_ledger(ledger_path, boot_commitments=[])
        bridge = WorldBridge(
            gm_agent_process=process,
            world_session_id="world-thread-repair",
            turns_dir=turns_dir,
            simulation_root=tmp_path / "simulation-root",
            consistency_ledger_path=ledger_path,
            private_path_replacements=(),
            turn_prompt_template_path=TURN_PROMPT_PATH,
        )
        handler = VatWorldToolHandler(episode_id="episode-repair", bridge=bridge)

        response = await handler(
            VatToolRequest(
                tool_call_id="call-list",
                tool_name="bash",
                arguments={"command": "find . -maxdepth 1"},
            ),
            (),
        )

        assert response.content == "AGENTS.md\n"
        assert len(process.prompts) == 2
        assert "repair of the same turn" in process.prompts[1]
        assert (turns_dir / "000000" / "repair_01_prompt.md").is_file()

    asyncio.run(run())


def test_world_bridge_quarantines_and_terminates_declared_conflict(
    tmp_path: Path,
) -> None:
    """A simulator conflict must never survive as a trainable continuation."""
    turns_dir = tmp_path / "world" / "turns"
    ledger_path = tmp_path / "world" / "consistency_ledger.json"
    ledger_path.parent.mkdir(parents=True)
    create_ledger(ledger_path, boot_commitments=["The public count is 12."])

    class ConflictedWorldProcess:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def start(self, prompt: str) -> ProcessTurn:
            raise AssertionError(prompt)

        async def resume(
            self,
            session_id: str,
            prompt: str,
            *,
            output_schema_path: Path | None = None,
            turn_context=None,
        ) -> ProcessTurn:
            self.prompts.append(prompt)
            turn_dir = turns_dir / "000000"
            request = json.loads(
                (turn_dir / "request.json").read_text(encoding="utf-8")
            )
            (turn_dir / "response.json").write_text(
                json.dumps(
                    {
                        "action_id": request["action"]["id"],
                        "tool_response": {
                            "content": "count=13\n",
                            "is_error": False,
                        },
                        "turn_reward": 0.0,
                        "difficulty_rating": 0.2,
                        "completion_percentage": 0.0,
                        "commentary": "The generated count conflicts with boot truth.",
                        "reasoning_assessment": None,
                        "consistency_review": {
                            "reviewed_revision": 0,
                            "observation_claims": ["The public count is 13."],
                            "new_commitments": [],
                            "conflicts": ["Boot fixed the public count at 12."],
                            "executed_cross_checks": ["Compared against boot commitment."],
                        },
                        # The first draft is internally contradictory. The
                        # repair preserves the actual conflict while aligning
                        # the integrity and termination fields.
                        "trainable": len(self.prompts) == 1,
                        "integrity_issues": (
                            []
                            if len(self.prompts) == 1
                            else ["The generated count conflicts with boot truth."]
                        ),
                        "terminate_episode": len(self.prompts) > 1,
                        "termination_reason": (
                            None
                            if len(self.prompts) == 1
                            else "Authoritative counts conflict."
                        ),
                    }
                ),
                encoding="utf-8",
            )
            return ProcessTurn(
                session_id=session_id,
                events=(),
                final_response={},
                raw_log_path=tmp_path / "world.jsonl",
            )

        async def transcript(self, session_id: str) -> tuple[object, ...]:
            raise AssertionError(session_id)

        async def close(self) -> None:
            return None

    async def run() -> None:
        process = ConflictedWorldProcess()
        bridge = WorldBridge(
            gm_agent_process=process,
            world_session_id="world-thread-conflict",
            turns_dir=turns_dir,
            simulation_root=tmp_path / "simulation-root",
            consistency_ledger_path=ledger_path,
            private_path_replacements=(),
            turn_prompt_template_path=TURN_PROMPT_PATH,
        )
        handler = VatWorldToolHandler(episode_id="episode-conflict", bridge=bridge)
        response = await handler(
            VatToolRequest(
                tool_call_id="call-count",
                tool_name="bash",
                arguments={"command": "show-count"},
            ),
            (),
        )

        assert response.terminate is True
        assert handler.observations[0].integrity.trainable is False
        assert handler.observations[0].integrity.issues == (
            "The generated count conflicts with boot truth.",
            "Simulator consistency conflict: Boot fixed the public count at 12.",
        )
        assert handler.termination_reason is not None
        assert load_ledger(ledger_path).revision == 1
        assert len(process.prompts) == 2
        rejected = turns_dir / "000000" / "rejected_01_response.json"
        assert rejected.is_file()
        rejected_payload = json.loads(rejected.read_text(encoding="utf-8"))
        assert rejected_payload["trainable"] is True

    asyncio.run(run())


def test_pi_sdk_resolution_never_guesses_from_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from shellsmith.episode import smoke

    monkeypatch.setattr(smoke, "PINNED_PI_SDK", tmp_path / "missing" / "index.js")
    args = smoke._parser().parse_args([])
    with pytest.raises(RuntimeError, match="npm ci"):
        smoke._resolve_pi_sdk(args)
    explicit = tmp_path / "index.js"
    explicit.write_text("")
    args = smoke._parser().parse_args(["--pi-sdk-module-path", str(explicit)])
    assert smoke._resolve_pi_sdk(args) == explicit.resolve()


def test_world_bridge_repairs_duplicate_ledger_claims_on_same_world_thread(
    tmp_path: Path,
) -> None:
    """A duplicate claim gets a same-thread repair prompt instead of failing the episode."""
    turns_dir = tmp_path / "world" / "turns"

    class DuplicateThenFixedWorldProcess:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def start(self, prompt: str) -> ProcessTurn:
            raise AssertionError(prompt)

        async def resume(self, session_id: str, prompt: str, *, output_schema_path: Path | None = None,
                         turn_context=None) -> ProcessTurn:
            self.prompts.append(prompt)
            turn_dir = turns_dir / "000000"
            request = json.loads((turn_dir / "request.json").read_text(encoding="utf-8"))
            claims = ["ls listed AGENTS.md"] * (2 if len(self.prompts) == 1 else 1)
            (turn_dir / "response.json").write_text(json.dumps({
                "action_id": request["action"]["id"],
                "tool_response": {"content": "AGENTS.md\n", "is_error": False},
                "turn_reward": 0.0, "difficulty_rating": 0.0, "completion_percentage": 0.0,
                "commentary": "Orientation only.",
                "consistency_review": {"reviewed_revision": 0, "observation_claims": claims,
                                       "new_commitments": [], "conflicts": [], "executed_cross_checks": []},
                "trainable": True, "integrity_issues": [],
            }), encoding="utf-8")
            return ProcessTurn(session_id=session_id, events=(), final_response={"text": "turn complete"},
                               raw_log_path=tmp_path / f"world-{len(self.prompts)}.jsonl")

        async def transcript(self, session_id: str) -> tuple[object, ...]:
            raise AssertionError(session_id)

        async def close(self) -> None:
            return None

    async def run() -> None:
        process = DuplicateThenFixedWorldProcess()
        ledger_path = tmp_path / "world" / "consistency_ledger.json"
        ledger_path.parent.mkdir(parents=True)
        create_ledger(ledger_path, boot_commitments=[])
        bridge = WorldBridge(
            gm_agent_process=process, world_session_id="world-thread-dup", turns_dir=turns_dir,
            simulation_root=tmp_path / "simulation-root", consistency_ledger_path=ledger_path,
            private_path_replacements=(), turn_prompt_template_path=TURN_PROMPT_PATH,
        )
        handler = VatWorldToolHandler(episode_id="episode-dup", bridge=bridge)
        response = await handler(
            VatToolRequest(tool_call_id="call-list", tool_name="bash", arguments={"command": "ls"}), ())

        assert response.content == "AGENTS.md\n"
        assert len(process.prompts) == 2
        assert "observation_claims must not contain duplicate strings" in process.prompts[1]
        assert (turns_dir / "000000" / "rejected_01_response.json").is_file()
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert ledger["entries"][0]["claims"] == ["ls listed AGENTS.md"]

    asyncio.run(run())
