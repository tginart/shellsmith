"""Tests for the deliberately minimal generative world file boundary.

Architecture map:
    1. Load the first short Markdown world prompt.
    2. Prove only `Player` is projected into the vat.
    3. Reject structural mistakes without interpreting world quality.
    4. Pin stable content identity and the inert episode-state data model.

Why this exists:
    The parser should protect privacy and obvious authoring errors while leaving
    simulation judgment to the persistent Game Master.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shellsmith.world_file import (
    WorldFileSpec,
    load_world_file,
)
from shellsmith.episode.orchestrator import (
    EpisodePaths,
    EpisodeState,
    EpisodeStatus,
)
from shellsmith.paths import DEFAULT_WORLD, DRAFTS_DIR, PROMPTS_DIR
from shellsmith.episode.schemas import (
    BOOT_RESPONSE_SCHEMA,
    FINAL_RESPONSE_SCHEMA,
    TURN_RESPONSE_SCHEMA,
)


FIRST_ENV_PATH = DEFAULT_WORLD
PROMPT_ROOT = PROMPTS_DIR
GAME_MASTER_AGENTS_PATH = PROMPT_ROOT / "game_master" / "AGENTS.md"
WORLD_BOOTSTRAP_PROMPT_PATH = PROMPT_ROOT / "WORLD_BOOTSTRAP_PROMPT.md"
WORLD_TURN_PROMPT_PATH = PROMPT_ROOT / "WORLD_TURN_PROMPT.md"
WORLD_FINAL_PROMPT_PATH = PROMPT_ROOT / "WORLD_FINAL_PROMPT.md"
PUBLIC_REPO_DRAFT_ROOT = DRAFTS_DIR


def test_first_world_file_loads_as_four_markdown_sections() -> None:
    spec = load_world_file(FIRST_ENV_PATH)

    assert spec.id == "checkout_idempotency_001"
    assert spec.title == "Duplicate orders after checkout retries"
    assert "Northstar Commerce" in spec.world
    assert "active checkout incident" in spec.player
    assert "Score the complete trajectory" in spec.reward
    assert "Successful:" in spec.termination_conditions
    assert len(spec.markdown.splitlines()) < 175
    assert len(spec.content_hash()) == 64


def test_public_repo_replay_meta_spec_and_instances_are_complete() -> None:
    meta = (PUBLIC_REPO_DRAFT_ROOT / "PUBLIC_REPO_REPLAY_META_SPEC.md").read_text(
        encoding="utf-8"
    )
    assert "authoring template, not a runnable world file" in meta
    assert "**Executed.**" in meta
    assert "**Upstream replay.**" in meta
    assert "**Faithful synthesis.**" in meta
    assert "upstream_patch_copied" in meta
    assert "~/miniconda3/bin/conda" in meta
    assert "world/simulation_root/.miniconda" in meta

    paths = sorted(PUBLIC_REPO_DRAFT_ROOT.glob("public_repo_*_001.md"))
    assert {
        "public_repo_go_jose_multisig_key_selection_001.md",
        "public_repo_golang_jwt_epoch_expiry_001.md",
        "public_repo_hapi_content_duplicate_parameters_001.md",
        "public_repo_node_tmp_path_escape_001.md",
        "public_repo_packaging_marker_parentheses_001.md",
        "public_repo_zipp_iterdir_exception_001.md",
    }.issubset({path.name for path in paths})
    for path in paths:
        spec = WorldFileSpec.from_markdown(
            task_id=path.stem,
            markdown=path.read_text(encoding="utf-8"),
        )
        normalized_world = " ".join(spec.world.split())
        assert "vulnerable commit" in normalized_world
        assert "oracle" in spec.world.lower()
        assert "task_outcomes.public_repo_repair" in spec.reward
        assert "Successful:" in spec.termination_conditions


def test_player_projection_contains_only_its_section() -> None:
    spec = load_world_file(FIRST_ENV_PATH)
    public = spec.player_projection()

    assert public == spec.player
    assert "ord_dup_000_a" in public
    assert "30-second checkout-lock lease" not in public
    assert "Score the complete trajectory" not in public
    assert "Successful:" not in public


def test_parser_requires_exact_ordered_sections() -> None:
    malformed = """# Example

## Player
Do something.

## World
Something is true.

## Reward
Reward success.

## Termination conditions
Stop when the task reaches a terminal outcome.
"""

    with pytest.raises(RuntimeError, match="ordered H2 sections"):
        WorldFileSpec.from_markdown(task_id="example", markdown=malformed)


def test_parser_rejects_missing_empty_or_extra_sections() -> None:
    missing = (
        "# Example\n\n## World\nWorld.\n\n## Player\nAct.\n\n"
        "## Reward\nScore.\n"
    )
    empty = (
        "# Example\n\n## World\n\n## Player\nAct.\n\n"
        "## Reward\nScore.\n\n## Termination conditions\nStop.\n"
    )
    extra = (
        "# Example\n\n## World\nWorld.\n\n## Player\nAct.\n\n"
        "## Reward\nScore.\n\n## Termination conditions\nStop.\n\n"
        "## Fixtures\nNo.\n"
    )

    for markdown in (missing, empty, extra):
        with pytest.raises(RuntimeError):
            WorldFileSpec.from_markdown(task_id="example", markdown=markdown)


def test_parser_does_not_interpret_world_completeness() -> None:
    deliberately_sparse = """# Tiny world

## World
A room with a locked box.

## Player
Open the box.

## Reward
Full reward if the box is opened without breaking it.

## Termination conditions
Stop when the box is open or cannot be opened safely.
"""

    spec = WorldFileSpec.from_markdown(
        task_id="tiny", markdown=deliberately_sparse
    )

    assert spec.world == "A room with a locked box."


def test_content_hash_changes_with_private_world_text() -> None:
    original = load_world_file(FIRST_ENV_PATH)
    changed = WorldFileSpec.from_markdown(
        task_id=original.id,
        markdown=original.markdown.replace("37 affected", "38 affected", 1),
    )

    assert original.id == changed.id
    assert original.content_hash() != changed.content_hash()


def test_loader_rejects_non_markdown_world_file(tmp_path: Path) -> None:
    path = tmp_path / "task.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match=".md extension"):
        load_world_file(path)


def test_episode_blueprint_derives_paths_and_serializes_recovery_cursor(
    tmp_path: Path,
) -> None:
    spec = load_world_file(FIRST_ENV_PATH)
    paths = EpisodePaths.from_root(tmp_path / "episode-1")
    state = EpisodeState(
        episode_id="episode-1",
        world_id=spec.id,
        world_hash=spec.content_hash(),
        status=EpisodeStatus.CREATED,
    )

    assert paths.world_dir == tmp_path / "episode-1" / "world"
    assert paths.simulation_root == (
        tmp_path / "episode-1" / "world" / "simulation_root"
    )
    assert paths.runtime_scratch == (
        tmp_path / "episode-1" / "world" / "runtime_scratch"
    )
    assert paths.player_agent_prompt_path == (
        tmp_path / "episode-1" / "scenario" / "player_agent_prompt.md"
    )
    assert paths.world_boot_result_path == (
        tmp_path / "episode-1" / "scenario" / "world_boot_result.json"
    )
    assert paths.bootstrap_dir == tmp_path / "episode-1" / "world" / "bootstrap"
    assert paths.bootstrap_response_path == (
        tmp_path / "episode-1" / "world" / "bootstrap" / "response.json"
    )
    assert paths.consistency_ledger_path == (
        tmp_path / "episode-1" / "world" / "consistency_ledger.json"
    )
    assert paths.turns_dir == tmp_path / "episode-1" / "world" / "turns"
    assert paths.final_response_path == (
        tmp_path / "episode-1" / "world" / "final" / "response.json"
    )
    assert state.to_jsonable()["status"] == "created"
    assert state.to_jsonable()["world_hash"] == spec.content_hash()


def test_game_master_agents_prompt_defines_the_universal_boundary() -> None:
    prompt = GAME_MASTER_AGENTS_PATH.read_text(encoding="utf-8")
    normalized_prompt = " ".join(prompt.split())

    assert "Game Master" in prompt
    assert "Player" in prompt
    assert "Write exactly one public file into the vat: `AGENTS.md`" in prompt
    assert "Produce the observation that the attempted action would naturally return" in prompt
    assert "A shell command returns shell output" in normalized_prompt
    assert "Never mention the Game Master, Player" in normalized_prompt
    assert "Do not announce that classification inside the world" in normalized_prompt
    assert "mark the episode untrainable" in normalized_prompt
    assert "a local import exposed in one observation must remain available" in normalized_prompt
    assert "Assume the Player is a highly capable frontier AI agent" in normalized_prompt
    assert "Establish these challenges during boot" in normalized_prompt
    assert "Grade to the frontier-level standard" in normalized_prompt
    assert "Use its Termination conditions" in normalized_prompt


def test_bootstrap_prompt_injects_world_and_creates_only_vat_agents() -> None:
    prompt = WORLD_BOOTSTRAP_PROMPT_PATH.read_text(encoding="utf-8")

    assert prompt.count("{{EPISODE_ID}}") == 1
    assert prompt.count("{{WORLD_MARKDOWN}}") == 1
    assert prompt.count("{{VAT_WORKSPACE_PATH}}") >= 3
    assert prompt.count("{{SIMULATION_ROOT_PATH}}") >= 2
    assert prompt.count("{{BOOT_RESPONSE_PATH}}") >= 2
    assert "{{BOOT_RESPONSE_SCHEMA_PATH}}" in prompt
    assert "Write exactly one file: `{{VAT_WORKSPACE_PATH}}/AGENTS.md`" in prompt
    assert '"private_world_summary"' in prompt
    assert '"private_initial_commitments"' in prompt
    assert '"initial_prompt"' not in prompt
    assert '"vat_files"' not in prompt
    assert '"status"' not in prompt
    assert "Do not report `status`, `ready`, or `failed`" in prompt
    assert "Do not begin the interactive episode" in prompt


def test_turn_prompt_makes_response_file_the_private_transaction_boundary() -> None:
    prompt = WORLD_TURN_PROMPT_PATH.read_text(encoding="utf-8")
    agents_prompt = GAME_MASTER_AGENTS_PATH.read_text(encoding="utf-8")

    for placeholder in (
        "{{EPISODE_ID}}",
        "{{TURN_INDEX}}",
        "{{TURN_DIRECTORY}}",
        "{{TURN_REQUEST_PATH}}",
        "{{PLAYER_TURN_SUMMARY}}",
        "{{TURN_RESPONSE_PATH}}",
        "{{TURN_RESPONSE_SCHEMA_PATH}}",
        "{{SIMULATION_ROOT_PATH}}",
        "{{CONSISTENCY_LEDGER_PATH}}",
        "{{CONSISTENCY_LEDGER_REVISION}}",
        "{{CONSISTENCY_LEDGER_SHA256}}",
    ):
        assert placeholder in prompt
    assert "still read\n`request.json`" in prompt
    assert "Convenience preview of Player's pre-action thinking" in prompt
    assert "only authoritative result of this turn" in prompt
    assert '"tool_response"' in agents_prompt
    assert '"turn_reward"' in agents_prompt
    assert '"difficulty_rating"' in agents_prompt
    assert '"completion_percentage"' in agents_prompt
    assert '"reasoning_assessment"' in agents_prompt
    assert "proportionate to `difficulty_rating`" in agents_prompt
    assert "agent-authored data" in agents_prompt
    assert "Copy exact numbers, versions" in agents_prompt
    assert "Use `conflicts` only" in prompt
    assert "at most 48,000 characters" in prompt
    assert "Never write replacement reasoning" in agents_prompt
    assert "Only its\n   `tool_response` value is shown inside the world" in agents_prompt


def test_format_player_turn_summary_includes_thinking_and_actions() -> None:
    from shellsmith.episode.bridge import (
        format_player_turn_summary,
    )

    summary = format_player_turn_summary(
        {
            "actions": [
                {
                    "id": "call_1",
                    "name": "bash",
                    "arguments": {"command": "ls /workspace"},
                    "harness_rejected": False,
                    "harness_reject_reason": None,
                }
            ],
            "trajectory_delta": [
                {
                    "event_type": "message_end",
                    "payload": {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "thinking",
                                    "thinking": "Survey the workspace first.",
                                },
                                {"type": "text", "text": "Listing files."},
                                {
                                    "type": "toolCall",
                                    "id": "call_1",
                                    "name": "bash",
                                    "arguments": {"command": "ls /workspace"},
                                },
                            ],
                        }
                    },
                }
            ],
        }
    )
    assert "## Pre-action thinking" in summary
    assert "Survey the workspace first." in summary
    assert "## Actions" in summary
    assert "bash id=call_1" in summary
    assert '"command": "ls /workspace"' in summary


def test_format_player_turn_summary_truncates_long_thinking() -> None:
    from shellsmith.episode.bridge import (
        format_player_turn_summary,
    )

    long_thinking = "x" * 50
    summary = format_player_turn_summary(
        {
            "action": {"id": "call_1", "name": "ls", "arguments": {}},
            "trajectory_delta": [
                {
                    "event_type": "message_end",
                    "payload": {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "thinking", "thinking": long_thinking},
                            ],
                        }
                    },
                }
            ],
        },
        thinking_char_cap=20,
    )
    assert "x" * 20 in summary
    assert "truncated 30 chars" in summary
    assert "full thinking in request.json" in summary


def test_final_prompt_keeps_terminal_reward_separate_from_turn_rewards() -> None:
    prompt = WORLD_FINAL_PROMPT_PATH.read_text(encoding="utf-8")
    agents_prompt = GAME_MASTER_AGENTS_PATH.read_text(encoding="utf-8")

    assert "{{FINAL_REQUEST_PATH}}" in prompt
    assert "{{FINAL_RESPONSE_PATH}}" in prompt
    assert "{{FINAL_RESPONSE_SCHEMA_PATH}}" in prompt
    assert "Do not derive it by summing `turn_reward` values" in prompt
    assert '"final_reward"' in agents_prompt
    assert '"factual_consistency"' in agents_prompt
    assert '"task_outcomes"' in agents_prompt
    assert "final-state judgment is independently\n  authoritative" in agents_prompt


def test_world_response_schemas_cover_all_three_artifact_boundaries() -> None:
    assert BOOT_RESPONSE_SCHEMA["additionalProperties"] is False
    assert set(BOOT_RESPONSE_SCHEMA["required"]) == {
        "private_world_summary",
        "private_initial_commitments",
    }
    assert set(BOOT_RESPONSE_SCHEMA["properties"]) == {
        "private_world_summary",
        "private_initial_commitments",
    }

    turn_properties = TURN_RESPONSE_SCHEMA["properties"]
    assert "tool_response" in turn_properties
    content_schema = turn_properties["tool_response"]["properties"]["content"]
    assert content_schema == {"type": "string", "maxLength": 48_000}
    assert "reasoning_assessment" in turn_properties
    reasoning_variants = turn_properties["reasoning_assessment"]["anyOf"]
    reasoning_schema = reasoning_variants[0]
    assert "better_reasoning" not in reasoning_schema["properties"]
    assert TURN_RESPONSE_SCHEMA["additionalProperties"] is True

    assert "final_reward" in FINAL_RESPONSE_SCHEMA["required"]
    assert "factual_consistency" in FINAL_RESPONSE_SCHEMA["required"]
    assert "task_outcomes" in FINAL_RESPONSE_SCHEMA["required"]
    criterion_schema = FINAL_RESPONSE_SCHEMA["properties"]["criteria"]["items"]
    assert criterion_schema["required"] == [
        "criterion",
        "weight",
        "score",
        "evidence",
    ]
    assert FINAL_RESPONSE_SCHEMA["additionalProperties"] is True


def test_load_boot_response_accepts_slim_two_field_artifact(tmp_path: Path) -> None:
    from shellsmith.episode.world_artifacts import (
        load_boot_response,
        parse_boot_response,
    )

    path = tmp_path / "world" / "bootstrap" / "response.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "private_world_summary": "Bank settlement world initialized.",
                "private_initial_commitments": [
                    "Settlement window closes at 17:00 UTC."
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    artifact = load_boot_response(path)
    assert artifact.private_world_summary.startswith("Bank settlement")
    assert artifact.private_initial_commitments == (
        "Settlement window closes at 17:00 UTC.",
    )
    assert artifact.path == path

    try:
        parse_boot_response(
            {
                "private_world_summary": "ok",
                "private_initial_commitments": [],
                "status": "ready",
            }
        )
    except RuntimeError as error:
        assert "unexpected fields" in str(error)
    else:
        raise AssertionError("expected unexpected-fields rejection")

    try:
        parse_boot_response({"private_initial_commitments": []})
    except RuntimeError as error:
        assert "private_world_summary" in str(error)
    else:
        raise AssertionError("expected missing-summary rejection")


def test_reward_from_smoke_summary_rejects_untrainable_episodes() -> None:
    from shellsmith.episode.smoke import (
        reward_from_smoke_summary,
    )

    assert reward_from_smoke_summary({"trainable": True, "final_reward": 0.4}) == 0.4
    with pytest.raises(RuntimeError, match="missing trainable"):
        reward_from_smoke_summary({"final_reward": 0.9})
    assert reward_from_smoke_summary({"trainable": False, "final_reward": 0.9}) == 0.0
    # final_reward must be validated even for untrainable episodes: a truncated
    # artifact must surface the broken contract, not silently score reward 0.
    with pytest.raises(RuntimeError, match="missing final_reward"):
        reward_from_smoke_summary({"trainable": False})
    with pytest.raises(RuntimeError, match="final_reward is not numeric"):
        reward_from_smoke_summary({"trainable": False, "final_reward": "oops"})
    with pytest.raises(RuntimeError, match="final_reward is not numeric"):
        reward_from_smoke_summary({"trainable": True, "final_reward": None})
    # A non-boolean trainable is a broken contract, not an untrainable episode.
    with pytest.raises(RuntimeError, match="trainable must be a boolean"):
        reward_from_smoke_summary({"trainable": 0, "final_reward": 0.7})
    with pytest.raises(RuntimeError, match="trainable must be a boolean"):
        reward_from_smoke_summary({"trainable": "false", "final_reward": 0.7})
    from shellsmith.episode.smoke import (
        smoke_summary_is_trainable,
    )
    with pytest.raises(RuntimeError, match="trainable must be a boolean"):
        smoke_summary_is_trainable({"trainable": 1})
    assert smoke_summary_is_trainable({"trainable": True}) is True
    assert smoke_summary_is_trainable({"trainable": False}) is False
