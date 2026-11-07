"""Focused tests for shellsmith auditing and batch aggregation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from shellsmith.episode.acceptance_audit import (
    audit_rollout,
)
from shellsmith.campaign.checkout_batch import (
    _aggregate,
    audit_existing_batch,
)
from shellsmith.episode.consistency_ledger import (
    append_entry,
    create_ledger,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _rollout(tmp_path: Path) -> Path:
    root = tmp_path / "rollout"
    _write(
        root / "episode.json",
        {
            "episode_id": "checkout-01",
            "world_id": "checkout_idempotency_001",
            "status": "complete",
        },
    )
    _write(
        root / "run_config.json",
        {
            "gm_runtime": "codex",
            "world_permission_profile": "game_master",
            "world_network_access": True,
            "player_renderer": "qwen3_5",
            "player_thinking": "thinking",
        },
    )
    _write(
        root / "scenario" / "vat_workspace_audit.json",
        {
            "passed": True,
            "entries": ["AGENTS.md"],
            "extra_entries": [],
            "agents_sha256": "a" * 64,
            "agents_unchanged": True,
        },
    )
    _write(
        root / "scenario" / "world_boot_result.json",
        {
            "private_world_summary": "Fixture world is consistent.",
            "private_initial_commitments": ["The fixture starts consistent."],
        },
    )
    _write(
        root / "world" / "turns" / "000000" / "request.json",
        {
            "step_index": 0,
            "consistency_ledger": {"expected_revision": 0},
            "action": {
                "id": "call-1",
                "name": "bash",
                "arguments": {
                    "command": "cd /virtual/repo && python -m pytest -q"
                },
            }
        },
    )
    _write(
        root / "world" / "turns" / "000000" / "response.json",
        {
            "action_id": "call-1",
            "tool_response": {"content": "5 passed in 1.0s", "is_error": False},
            "turn_reward": 0.4,
            "difficulty_rating": 0.7,
            "completion_percentage": 90,
            "commentary": "Good test.",
            "reasoning_assessment": {
                "quality": 0.9,
                "efficiency": 0.8,
                "commentary": "Targeted.",
            },
            "consistency_review": {
                "reviewed_revision": 0,
                "observation_claims": ["The regression suite has five passing tests."],
                "new_commitments": [],
                "conflicts": [],
                "executed_cross_checks": ["Ran the requested pytest command."],
            },
            "trainable": True,
            "integrity_issues": [],
        },
    )
    ledger_path = root / "world" / "consistency_ledger.json"
    create_ledger(
        ledger_path, boot_commitments=["The fixture starts consistent."]
    )
    append_entry(
        ledger_path,
        expected_revision=0,
        turn_index=0,
        action_id="call-1",
        observation="5 passed in 1.0s",
        claims=["The regression suite has five passing tests."],
        new_commitments=[],
        cross_checks=["Ran the requested pytest command."],
    )
    _write(
        root / "world" / "final" / "response.json",
        {
            "final_reward": 0.9,
            "completion_percentage": 100,
            "trainable": True,
            "criteria": [
                {"criterion": "Safe fix", "weight": 0.3, "score": 0.9},
                {"criterion": "Verification", "weight": 0.2, "score": 0.9},
            ],
            "process_deductions": [],
            "commentary": "Atomic fix verified.",
            "integrity_issues": [],
            "factual_consistency": {"passed": True, "contradictions": []},
            "task_outcomes": {
                "checkout_idempotency": {
                    "atomic_fix": True,
                    "atomic_fix_evidence": "Overlapping regression passed.",
                }
            },
        },
    )
    vat_path = "/private/var/folders/test/shellsmith-vat-123/workspace"
    (root / "logs").mkdir(exist_ok=True)
    vat_lines = [
        {
            "direction": "host_to_runner",
            "message": {"type": "initialize", "workspace_path": vat_path},
        },
        {
            "direction": "runner_to_host",
            "message": {
                "type": "agent_event",
                "event_type": "message_end",
                "payload": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "Run the regression suite.",
                            },
                            {
                                "type": "toolCall",
                                "id": "call-1",
                                "name": "bash",
                                "arguments": {
                                    "command": "cd /virtual/repo && python -m pytest -q"
                                },
                            },
                        ],
                    }
                },
            },
        },
        {
            "direction": "runner_to_host",
            "message": {
                "type": "tool_request",
                "tool_call_id": "call-1",
                "tool_name": "bash",
                "arguments": {
                    "command": "cd /virtual/repo && python -m pytest -q"
                },
                "harness_rejected": False,
            },
        },
        {
            "direction": "host_to_runner",
            "message": {
                "type": "tool_response",
                "tool_call_id": "call-1",
                "content": "5 passed in 1.0s",
                "is_error": False,
            },
        },
    ]
    (root / "logs" / "vat.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in vat_lines),
        encoding="utf-8",
    )
    _write(
        root / "scenario" / "model_protocol.json",
        {
            "schema": "shellsmith.model_protocol.v1",
            "expected_renderer": "qwen3_5",
            "expected_thinking_mode": "thinking",
            "requests": [
                {
                    "request_id": "req-1",
                    "renderer_name": "qwen3_5",
                    "thinking_mode": "thinking",
                    "parse_termination": "clean",
                    "unparsed_tool_call_count": 0,
                    "assistant_text": "",
                    "assistant_content_types": ["thinking", "toolCall"],
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "name": "bash",
                            "arguments": {
                                "command": "cd /virtual/repo && python -m pytest -q"
                            },
                        }
                    ],
                }
            ],
        },
    )
    codex = root / "logs" / "codex" / "000001.jsonl"
    codex.parent.mkdir(parents=True)
    codex.write_text(
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "/bin/zsh -lc 'python3 -m json.tool world/turns/000000/response.json'",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_acceptance_audit_separates_all_requested_signals(tmp_path: Path) -> None:
    report = audit_rollout(_rollout(tmp_path))

    assert report["accepted_for_training"] is True
    assert report["simulator_integrity"]["passed"] is True
    assert report["final_task"]["reward"] == 0.9
    assert report["final_task"]["atomic_fix"] is True
    assert report["final_task"]["atomic_fix_declared"] is True
    assert report["reasoning"]["mean_quality"] == 0.9
    assert report["reasoning"]["mean_efficiency"] == 0.8
    assert report["executed_verification"]["passed_count"] == 1
    assert report["factual_consistency"]["passed"] is True
    assert report["model_protocol_integrity"]["passed"] is True


def test_acceptance_audit_treats_blank_gm_slots_as_absent(
    tmp_path: Path,
) -> None:
    root = _rollout(tmp_path)
    boot_path = root / "scenario" / "world_boot_result.json"
    boot = json.loads(boot_path.read_text(encoding="utf-8"))
    boot["private_initial_commitments"] = ["", "The fixture starts consistent.", "  "]
    boot_path.write_text(json.dumps(boot) + "\n", encoding="utf-8")
    response_path = root / "world" / "turns" / "000000" / "response.json"
    response = json.loads(response_path.read_text(encoding="utf-8"))
    review = response["consistency_review"]
    review["observation_claims"] = ["", "The regression suite has five passing tests."]
    review["new_commitments"] = [""]
    review["executed_cross_checks"] = ["Ran the requested pytest command.", ""]
    response_path.write_text(json.dumps(response) + "\n", encoding="utf-8")
    ledger_path = root / "world" / "consistency_ledger.json"
    ledger_path.unlink()
    create_ledger(
        ledger_path,
        boot_commitments=boot["private_initial_commitments"],
    )
    append_entry(
        ledger_path,
        expected_revision=0,
        turn_index=0,
        action_id="call-1",
        observation="5 passed in 1.0s",
        claims=review["observation_claims"],
        new_commitments=review["new_commitments"],
        cross_checks=review["executed_cross_checks"],
    )

    report = audit_rollout(root)

    assert report["accepted_for_training"] is True
    assert report["simulator_integrity"]["passed"] is True


def test_acceptance_audit_treats_criterion_score_as_normalized(tmp_path: Path) -> None:
    root = _rollout(tmp_path)
    final_path = root / "world" / "final" / "response.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["criteria"] = [
        {"criterion": "Safe fix", "weight": 0.25, "score": 0.3},
        {"criterion": "Verification", "weight": 0.1, "score": 0.9},
    ]
    final_path.write_text(json.dumps(final) + "\n", encoding="utf-8")

    report = audit_rollout(root)

    assert report["final_task"]["atomic_fix"] is False


@pytest.mark.parametrize("runtime", ["claude", "pi"])
def test_acceptance_audit_ignores_host_path_commands_in_gm_runtime(
    tmp_path: Path, runtime: str
) -> None:
    """Host-path shell commands no longer reject an otherwise coherent rollout."""

    root = _rollout(tmp_path)
    config_path = root / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["gm_runtime"] = runtime
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
    shutil.rmtree(root / "logs" / "codex")
    log_path = root / "logs" / runtime / "000000_bootstrap.jsonl"
    if runtime == "claude":
        _write(
            log_path,
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {
                                "command": (
                                    "ls /Users/example/miniconda3/bin/conda; "
                                    "find / -maxdepth 1"
                                )
                            },
                        }
                    ]
                },
            },
        )
    else:
        _write(
            log_path,
            {
                "type": "tool_execution_start",
                "toolName": "bash",
                "args": {
                    "command": (
                        "ls /Users/example/miniconda3/bin/conda; find / -maxdepth 1"
                    )
                },
            },
        )

    report = audit_rollout(root)

    assert report["accepted_for_training"] is True
    assert "host_boundary_safety" not in report


def test_acceptance_audit_does_not_require_runtime_logs_for_acceptance(
    tmp_path: Path,
) -> None:
    root = _rollout(tmp_path)
    shutil.rmtree(root / "logs" / "codex")

    report = audit_rollout(root)

    assert report["accepted_for_training"] is True
    assert "host_boundary_safety" not in report


def test_acceptance_audit_does_not_reject_malformed_runtime_log(
    tmp_path: Path,
) -> None:
    root = _rollout(tmp_path)
    log_path = root / "logs" / "codex" / "000001.jsonl"
    log_path.write_text("not-json\n", encoding="utf-8")

    report = audit_rollout(root)

    assert report["accepted_for_training"] is True
    assert "host_boundary_safety" not in report


def test_acceptance_audit_recognizes_hapi_lab_success_output(tmp_path: Path) -> None:
    root = _rollout(tmp_path)
    response_path = root / "world" / "turns" / "000000" / "response.json"
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response["tool_response"]["content"] = (
        "57 tests complete\nTest coverage: 100.00%\nLinting: No errors\n"
    )
    response_path.write_text(json.dumps(response) + "\n", encoding="utf-8")

    report = audit_rollout(root)

    assert report["executed_verification"]["passed_count"] == 1


def test_acceptance_audit_allows_inert_world_materialization_in_vat(
    tmp_path: Path,
) -> None:
    root = _rollout(tmp_path)
    codex = root / "logs" / "codex" / "000001.jsonl"
    codex.write_text(
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": (
                        "/bin/zsh -lc 'cat /private/var/folders/test/"
                        "shellsmith-vat-123/workspace/secret.py'"
                    ),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit_rollout(root)

    assert report["accepted_for_training"] is True


def test_acceptance_audit_rejects_stale_action_correlation(tmp_path: Path) -> None:
    root = _rollout(tmp_path)
    response_path = root / "world" / "turns" / "000000" / "response.json"
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response["action_id"] = "stale-call"
    response_path.write_text(json.dumps(response) + "\n", encoding="utf-8")

    report = audit_rollout(root)

    assert report["accepted_for_training"] is False
    assert any(
        "action ID mismatch" in issue
        for issue in report["simulator_integrity"]["issues"]
    )


def test_acceptance_audit_quarantines_malformed_integrity_field(
    tmp_path: Path,
) -> None:
    root = _rollout(tmp_path)
    final_path = root / "world" / "final" / "response.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["integrity_issues"] = 7
    final_path.write_text(json.dumps(final) + "\n", encoding="utf-8")

    report = audit_rollout(root)

    assert report["accepted_for_training"] is False
    assert any(
        "final contract validation failed" in issue
        for issue in report["simulator_integrity"]["issues"]
    )


def test_acceptance_audit_records_public_network_without_rejecting_it(
    tmp_path: Path,
) -> None:
    root = _rollout(tmp_path)
    codex = root / "logs" / "codex" / "000001.jsonl"
    codex.write_text(
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "git clone https://github.com/torvalds/linux world/simulation_root/linux",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit_rollout(root)

    assert report["accepted_for_training"] is True


def test_batch_aggregate_measures_variance_and_atomic_success() -> None:
    def audit(reward: float, actions: int, atomic: bool) -> dict:
        return {
            "accepted_for_training": True,
            "simulator_integrity": {"passed": True},
            "factual_consistency": {"passed": True, "contradiction_count": 0},
            "final_task": {"reward": reward, "atomic_fix": atomic},
            "reasoning": {"mean_quality": 0.8, "mean_efficiency": 0.7},
            "executed_verification": {"executed": True},
            "action_count": actions,
        }

    report = _aggregate(
        "batch-1",
        [
            {"episode_id": "a", "error": None, "audit": audit(0.4, 20, False)},
            {"episode_id": "b", "error": None, "audit": audit(0.8, 30, True)},
        ],
    )

    assert report["reward"]["mean"] == 0.6
    assert report["reward"]["population_stddev"] == 0.2
    assert report["action_count"]["values"] == [20, 30]
    assert report["atomic_fix"] == {
        "evaluated_count": 2,
        "success_count": 1,
        "success_rate": 0.5,
    }


def test_existing_batch_audit_preserves_missing_manifest_rollouts(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "batch_manifest.json",
        {
            "schema": "shellsmith.checkout_batch_manifest.v1",
            "batch_id": "batch-interrupted",
            "requested_rollout_count": 2,
            "parallelism": 2,
            "rollouts": [
                {"episode_id": "ep-1", "directory": "rollout_01"},
                {"episode_id": "ep-2", "directory": "rollout_02"},
            ],
        },
    )

    report = audit_existing_batch(tmp_path, batch_id="batch-interrupted")

    assert report["requested_rollouts"] == 2
    assert report["completed_audits"] == 0
    assert report["infrastructure_failure_count"] == 2
    assert all(item["error"] for item in report["rollouts"])


def test_sibling_turn_acceptance_uses_full_action_set(tmp_path: Path) -> None:
    from shellsmith.episode.protocol_integrity import (
        audit_protocol_integrity,
        populate_model_protocol_requests,
    )

    root = _rollout(tmp_path)
    _write(
        root / "world" / "turns" / "000000" / "request.json",
        {
            "step_index": 0,
            "consistency_ledger": {"expected_revision": 0},
            "action": {
                "id": "call-A",
                "name": "bash",
                "arguments": {"command": "ls"},
            },
            "actions": [
                {
                    "id": "call-A",
                    "name": "bash",
                    "arguments": {"command": "ls"},
                },
                {
                    "id": "call-B",
                    "name": "read_file",
                    "arguments": {"path": "README.md"},
                },
            ],
        },
    )
    _write(
        root / "world" / "turns" / "000000" / "response.json",
        {
            "action_id": "call-A",
            "tool_results": [
                {
                    "action_id": "call-A",
                    "tool_response": {"content": "README.md\n", "is_error": False},
                },
                {
                    "action_id": "call-B",
                    "tool_response": {
                        "content": "# hello\n",
                        "is_error": False,
                    },
                },
            ],
            "turn_reward": 0.4,
            "difficulty_rating": 0.5,
            "completion_percentage": 40,
            "commentary": "Sibling batch ok.",
            "consistency_review": {
                "reviewed_revision": 0,
                "observation_claims": ["Listed and read README."],
                "new_commitments": [],
                "conflicts": [],
                "executed_cross_checks": ["Read the requested file."],
            },
            "trainable": True,
            "integrity_issues": [],
        },
    )
    ledger_path = root / "world" / "consistency_ledger.json"
    ledger_path.unlink(missing_ok=True)
    create_ledger(ledger_path, boot_commitments=["The fixture starts consistent."])
    append_entry(
        ledger_path,
        expected_revision=0,
        turn_index=0,
        action_id="call-A",
        observation="README.md\n",
        claims=["Listed and read README."],
        new_commitments=[],
        cross_checks=["Read the requested file."],
    )
    populate_model_protocol_requests(
        root,
        [
            {
                "request_id": "req-1",
                "renderer_name": "qwen3_5",
                "thinking_mode": "thinking",
                "parse_termination": "clean",
                "unparsed_tool_call_count": 0,
                "tool_calls": [
                    {
                        "id": "call-A",
                        "name": "bash",
                        "arguments": {"command": "ls"},
                    },
                    {
                        "id": "call-B",
                        "name": "read_file",
                        "arguments": {"path": "README.md"},
                    },
                ],
                "assistant_text": "",
                "assistant_content_types": ["tool_call"],
                "sampled_token_count": 12,
            }
        ],
    )
    vat_path = root / "logs" / "vat.jsonl"
    vat_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "direction": "host_to_runner",
                        "message": {
                            "type": "initialize",
                            "workspace_path": "/tmp/vat/workspace",
                        },
                    }
                ),
                json.dumps(
                    {
                        "direction": "runner_to_host",
                        "message": {
                            "type": "agent_event",
                            "event_type": "message_end",
                            "payload": {
                                "message": {
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": "toolCall",
                                            "id": "call-A",
                                            "name": "bash",
                                            "arguments": {"command": "ls"},
                                        },
                                        {
                                            "type": "toolCall",
                                            "id": "call-B",
                                            "name": "read_file",
                                            "arguments": {"path": "README.md"},
                                        },
                                    ],
                                }
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "direction": "runner_to_host",
                        "message": {
                            "type": "tool_response",
                            "tool_call_id": "call-A",
                            "content": "README.md\n",
                        },
                    }
                ),
                json.dumps(
                    {
                        "direction": "runner_to_host",
                        "message": {
                            "type": "tool_response",
                            "tool_call_id": "call-B",
                            "content": "# hello\n",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    protocol = audit_protocol_integrity(root)
    assert protocol["passed"] is True, protocol["problems"]
    assert protocol["world_turn_count"] == 2

    report = audit_rollout(root)
    assert report["model_protocol_integrity"]["passed"] is True
    assert not any(
        "contract validation failed" in item
        for item in report.get("problems", [])
        if isinstance(item, str)
    )
    assert report["consistency_ledger"]["passed"] is True


def test_empty_model_protocol_requests_emits_loud_problem(tmp_path: Path) -> None:
    from shellsmith.episode.protocol_integrity import (
        audit_protocol_integrity,
    )

    root = _rollout(tmp_path)
    _write(
        root / "scenario" / "model_protocol.json",
        {
            "schema": "shellsmith.model_protocol.v1",
            "expected_renderer": "qwen3_5",
            "expected_thinking_mode": "thinking",
            "requests": [],
        },
    )
    report = audit_protocol_integrity(root)
    assert any(
        "model_protocol.requests empty while vat has" in item
        for item in report["problems"]
    )
    # Vat fallback still correlates the single tool call.
    assert report["request_count"] >= 1
    assert "world request 'call-1' has no matching assistant tool call" not in report[
        "problems"
    ]


def _set_parallel_verification(
    root: Path,
    *,
    actions: list[dict],
    results: list[dict],
) -> None:
    request_path = root / "world" / "turns" / "000000" / "request.json"
    response_path = root / "world" / "turns" / "000000" / "response.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    response = json.loads(response_path.read_text(encoding="utf-8"))
    request["actions"] = actions
    request["action"] = actions[0]
    response["action_id"] = actions[0]["id"]
    response["tool_results"] = results
    response["tool_response"] = results[0]["tool_response"]
    request_path.write_text(json.dumps(request) + "\n", encoding="utf-8")
    response_path.write_text(json.dumps(response) + "\n", encoding="utf-8")


def test_acceptance_audit_counts_non_primary_parallel_test(tmp_path: Path) -> None:
    root = _rollout(tmp_path)
    actions = [
        {"id": "call-1", "name": "bash", "arguments": {"command": "echo ready"}},
        {"id": "call-2", "name": "bash", "arguments": {"command": "pytest -q"}},
    ]
    results = [
        {
            "action_id": "call-1",
            "tool_response": {"content": "5 passed in 1.0s", "is_error": False},
        },
        {
            "action_id": "call-2",
            "tool_response": {"content": "7 passed in 2.0s", "is_error": False},
        },
    ]
    _set_parallel_verification(root, actions=actions, results=results)

    report = audit_rollout(root)

    assert report["executed_verification"]["passed_count"] == 1
    assert report["executed_verification"]["executions"][0]["action_id"] == "call-2"


def test_acceptance_audit_correlates_out_of_order_parallel_results(
    tmp_path: Path,
) -> None:
    root = _rollout(tmp_path)
    actions = [
        {"id": "call-1", "name": "bash", "arguments": {"command": "echo ready"}},
        {
            "id": "call-pass",
            "name": "bash",
            "arguments": {"command": "pytest -q tests/pass"},
        },
        {
            "id": "call-fail",
            "name": "bash",
            "arguments": {"command": "pytest -q tests/fail"},
        },
    ]
    results = [
        {
            "action_id": "call-1",
            "tool_response": {"content": "5 passed in 1.0s", "is_error": False},
        },
        {
            "action_id": "call-fail",
            "tool_response": {"content": "1 failed", "is_error": True},
        },
        {
            "action_id": "call-pass",
            "tool_response": {"content": "3 passed", "is_error": False},
        },
    ]
    _set_parallel_verification(root, actions=actions, results=results)

    report = audit_rollout(root)
    verification = report["executed_verification"]

    assert verification["execution_count"] == 2
    assert verification["passed_count"] == 1
    assert verification["failed_count"] == 1
    assert [row["action_id"] for row in verification["executions"]] == [
        "call-pass",
        "call-fail",
    ]
