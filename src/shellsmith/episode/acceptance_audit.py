"""Deterministic acceptance audit for one shellsmith rollout.

Architecture map:
    episode artifacts
      -> simulator integrity and terminal reward
      -> private reasoning-score summary
      -> executed verification evidence
      -> Pi/Tinker/world protocol integrity
      -> task-specific machine-readable outcomes
      -> `acceptance_audit.json`

The audit deliberately keeps trainability separate from task success. A coherent
rollout with a poor fix remains useful RL data; a high-reward rollout with a
simulator contradiction does not.

Host-path heuristics used to gate acceptance (`host_boundary_safety`). That
rejected intact public-repo episodes where GM legitimately touched
allowlisted host conda / temp paths without cracking the simulation. Acceptance
now trusts simulator integrity, factual consistency, and protocol integrity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import statistics
from pathlib import Path
from typing import Any

from shellsmith.episode.consistency_ledger import (
    ConsistencyLedgerError,
    drop_blank_gm_slots,
    load_ledger,
    observation_sha256,
)
from shellsmith.episode.protocol_integrity import (
    audit_protocol_integrity,
)
from shellsmith.episode.world_artifacts import (
    load_final_response,
    load_turn_response,
)


SCHEMA = "shellsmith.acceptance_audit.v1"
TEST_COMMAND = re.compile(
    r"(?:^|\s)(?:python\s+-m\s+pytest|pytest|unittest|go\s+test|cargo\s+test|"
    r"npm\s+(?:run\s+)?test|pnpm\s+(?:run\s+)?test|yarn\s+test|bun\s+test|"
    r"mvn\s+test|gradle\s+test)(?:\s|$)",
    re.IGNORECASE,
)
TEST_FAILURE = re.compile(r"(?:\b[1-9]\d*\s+failed\b|\bFAILED\b|\bFAILURES?\b)")
TEST_SUCCESS = re.compile(
    r"(?:\b\d+\s+passed\b|\btests? passed\b|\b\d+\s+tests? complete\b|\bOK\b)",
    re.I,
)
def _object(path: Path) -> dict[str, Any]:
    try:
        if path.is_symlink():
            raise RuntimeError(f"Refusing symlinked rollout artifact: {path}.")
        if path.exists() and not stat.S_ISREG(path.lstat().st_mode):
            raise RuntimeError(f"Refusing non-regular rollout artifact: {path}.")
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Required rollout artifact is missing: {path}.") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Rollout artifact is malformed JSON: {path}.") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Rollout artifact must contain a JSON object: {path}.")
    return value


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


def _strings(value: Any) -> list[str]:
    """Return only a valid string array; contract errors are reported elsewhere."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _criterion_ratio(criteria: Any, name_fragment: str) -> float | None:
    """Return the normalized score for the matching terminal criterion."""
    if not isinstance(criteria, list):
        return None
    for criterion in criteria:
        if not isinstance(criterion, dict):
            continue
        name = criterion.get("criterion") or criterion.get("name")
        score = criterion.get("score")
        if (
            isinstance(name, str)
            and name_fragment.casefold() in name.casefold()
            and isinstance(score, (int, float))
        ):
            return float(score)
    return None


def _turn_artifacts(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    turns: list[dict[str, Any]] = []
    problems: list[str] = []
    turns_root: Path = root.joinpath("world", "turns")
    if not turns_root.is_dir():
        return turns, ["world/turns directory is missing"]
    for directory in sorted(turns_root.iterdir()):
        if directory.is_symlink():
            problems.append(f"{directory.name} is a forbidden symlinked turn directory")
            continue
        if not directory.is_dir():
            problems.append(f"{directory.name} is not a turn directory")
            continue
        request_path: Path = directory.joinpath("request.json")
        response_path: Path = directory.joinpath("response.json")
        if not request_path.is_file() or not response_path.is_file():
            problems.append(f"{directory.name} lacks a complete request/response pair")
            continue
        request = _object(request_path)
        response = _object(response_path)
        action_ids = _expected_action_ids(request)
        primary_id = action_ids[0] if action_ids else None
        if primary_id is None:
            problems.append(f"{directory.name} request has no valid action ID")
        else:
            try:
                load_turn_response(
                    response_path,
                    expected_action_id=primary_id,
                    expected_action_ids=tuple(action_ids),
                )
            except RuntimeError as error:
                problems.append(f"{directory.name} contract validation failed: {error}")
        if directory.name.isdigit() and request.get("step_index") != int(directory.name):
            problems.append(
                f"{directory.name} request step_index does not match its directory"
            )
        turns.append(
            {
                "index": directory.name,
                "request": request,
                "response": response,
            }
        )
    return turns, problems


def _expected_action_ids(request: dict[str, Any]) -> list[str]:
    """Sibling action ids from request.actions, else primary action.id."""
    ids: list[str] = []
    actions = request.get("actions")
    if isinstance(actions, list):
        for item in actions:
            if not isinstance(item, dict):
                continue
            action_id = item.get("id")
            if isinstance(action_id, str) and action_id and action_id not in ids:
                ids.append(action_id)
    if ids:
        return ids
    action = request.get("action")
    if isinstance(action, dict):
        action_id = action.get("id")
        if isinstance(action_id, str) and action_id:
            return [action_id]
    return []


def _observation_content_for_ledger(
    request: dict[str, Any],
    response: dict[str, Any],
) -> str | None:
    """Observation text that should match the consistency-ledger hash.

    This must reproduce the bridge's ``append_entry(observation=...)`` input
    EXACTLY: the primary (first-sibling) slot's Player-facing content, or the
    synthesized empty string when the primary slot was harness-rejected/omitted
    (``load_turn_response`` substitutes ``content=""`` there). Joining sibling
    contents or falling back to another slot would produce a false hash
    mismatch. ``id`` is accepted as a slot-key alias exactly like the loader.
    """
    action_ids = _expected_action_ids(request)
    primary_id = action_ids[0] if action_ids else None
    results = response.get("tool_results")
    if isinstance(results, list) and results and primary_id is not None:
        for item in results:
            if not isinstance(item, dict):
                continue
            slot_id = item.get("action_id", item.get("id"))
            if slot_id != primary_id:
                continue
            rejected = bool(
                item.get("harness_rejected") or item.get("omit_player_facing")
            )
            tool_response = item.get("tool_response")
            if not rejected and isinstance(tool_response, dict):
                content = tool_response.get("content")
                if isinstance(content, str):
                    return content
            # Rejected/omitted primary slot: bridge hashed the synthesized "".
            return ""
        return None
    tool_response = response.get("tool_response")
    if isinstance(tool_response, dict):
        content = tool_response.get("content")
        if isinstance(content, str):
            return content
    return None


def _correlated_action_results(
    request: dict[str, Any], response: dict[str, Any]
) -> list[tuple[str | None, dict[str, Any], dict[str, Any]]]:
    """Return action/result pairs without ever correlating siblings by position."""
    actions = request.get("actions")
    results = response.get("tool_results")
    if isinstance(actions, list) or isinstance(results, list):
        if not isinstance(actions, list) or not isinstance(results, list):
            return []
        result_by_id: dict[str, dict[str, Any]] = {}
        for item in results:
            if not isinstance(item, dict):
                continue
            action_id = item.get("action_id", item.get("id"))
            if isinstance(action_id, str) and action_id:
                result_by_id[action_id] = item
        pairs: list[tuple[str | None, dict[str, Any], dict[str, Any]]] = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_id = action.get("id")
            if not isinstance(action_id, str) or not action_id:
                continue
            result = result_by_id.get(action_id)
            if result is None:
                continue
            if result.get("harness_rejected") or result.get("omit_player_facing"):
                continue
            tool_response = result.get("tool_response")
            if isinstance(tool_response, dict):
                pairs.append((action_id, action, tool_response))
        return pairs

    action = request.get("action")
    tool_response = response.get("tool_response")
    if not isinstance(action, dict) or not isinstance(tool_response, dict):
        return []
    action_id = action.get("id")
    return [
        (
            action_id if isinstance(action_id, str) and action_id else None,
            action,
            tool_response,
        )
    ]


def _verification_report(turns: list[dict[str, Any]]) -> dict[str, Any]:
    executions: list[dict[str, Any]] = []
    for turn in turns:
        for action_id, action, tool_response in _correlated_action_results(
            turn["request"], turn["response"]
        ):
            if action.get("name") != "bash":
                continue
            arguments = action.get("arguments")
            command = (
                arguments.get("command") if isinstance(arguments, dict) else None
            )
            if not isinstance(command, str) or TEST_COMMAND.search(command) is None:
                continue
            content = str(tool_response.get("content", ""))
            is_error = tool_response.get("is_error") is True
            explicit_failure = TEST_FAILURE.search(content) is not None
            passed = (
                not is_error
                and not explicit_failure
                and TEST_SUCCESS.search(content) is not None
            )
            executions.append(
                {
                    "turn": turn["index"],
                    "action_id": action_id,
                    "command": command,
                    "passed": passed,
                    "is_error": is_error,
                    "output_sha256": hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest(),
                    "output_tail": content[-600:],
                }
            )
    return {
        "executed": bool(executions),
        "execution_count": len(executions),
        "passed_count": sum(item["passed"] is True for item in executions),
        "failed_count": sum(item["passed"] is False for item in executions),
        "executions": executions,
    }


def _consistency_ledger_report(
    root: Path, *, turns: list[dict[str, Any]]
) -> dict[str, Any]:
    """Cross-check the append-only ledger against boot and every public turn.

    Main callers:
        ``audit_rollout`` after strict turn-contract loading.

    Why this exists:
        A syntactically valid ledger is insufficient. Acceptance requires proof
        that it was initialized from this episode and advanced exactly once for
        each observation that entered the vat.
    """
    problems: list[str] = []
    ledger_path: Path = root.joinpath("world", "consistency_ledger.json")
    try:
        ledger = load_ledger(ledger_path)
    except (FileNotFoundError, ConsistencyLedgerError) as error:
        return {
            "passed": False,
            "path": str(ledger_path),
            "revision": None,
            "entry_count": 0,
            "problems": [str(error)],
        }

    boot_path: Path = root.joinpath("scenario", "world_boot_result.json")
    try:
        boot = _object(boot_path)
    except RuntimeError as error:
        problems.append(str(error))
        boot = {}
    boot_commitments = boot.get("private_initial_commitments")
    if not isinstance(boot_commitments, list) or not all(
        isinstance(item, str) for item in boot_commitments
    ):
        problems.append("world boot result has invalid private_initial_commitments")
    elif tuple(drop_blank_gm_slots(boot_commitments)) != ledger.boot_commitments:
        problems.append("ledger boot commitments differ from world boot result")

    if len(ledger.entries) != len(turns):
        problems.append(
            "ledger entry count does not match complete turn count: "
            f"{len(ledger.entries)} != {len(turns)}"
        )

    for position, turn in enumerate(turns):
        if position >= len(ledger.entries):
            break
        entry = ledger.entries[position]
        request = turn["request"]
        response = turn["response"]
        # Derive the id the SAME way the rest of the module does (via
        # _expected_action_ids), which accepts both a singular ``action`` and a
        # plural ``actions`` list. Reading only ``request["action"]`` here made a
        # parallel-action turn compare the ledger's primary action id against
        # None, flagging a spurious "action ID mismatch" and failing acceptance.
        expected_action_ids = _expected_action_ids(request)
        primary_action_id = expected_action_ids[0] if expected_action_ids else None
        receipt = response.get("consistency_review")
        receipt = receipt if isinstance(receipt, dict) else {}
        content = _observation_content_for_ledger(request, response)
        expected_revision = request.get("consistency_ledger")
        expected_revision = (
            expected_revision.get("expected_revision")
            if isinstance(expected_revision, dict)
            else None
        )
        checks = (
            (entry.turn_index, request.get("step_index"), "turn index"),
            (entry.action_id, primary_action_id, "action ID"),
            (receipt.get("reviewed_revision"), position, "reviewed revision"),
            (expected_revision, position, "requested ledger revision"),
            (
                entry.claims,
                tuple(drop_blank_gm_slots(receipt.get("observation_claims", []))),
                "claims",
            ),
            (
                entry.new_commitments,
                tuple(drop_blank_gm_slots(receipt.get("new_commitments", []))),
                "new commitments",
            ),
            (
                entry.cross_checks,
                tuple(drop_blank_gm_slots(receipt.get("executed_cross_checks", []))),
                "cross-checks",
            ),
        )
        for actual, expected, label in checks:
            if actual != expected:
                problems.append(f"turn {position} ledger {label} mismatch")
        if isinstance(content, str):
            if entry.observation_hash != observation_sha256(content):
                problems.append(f"turn {position} observation hash mismatch")
        else:
            problems.append(f"turn {position} has no hashable tool response")
        conflicts = receipt.get("conflicts")
        if isinstance(conflicts, list) and conflicts:
            problems.append(f"turn {position} declared consistency conflicts")

    return {
        "passed": not problems,
        "path": str(ledger_path),
        "revision": ledger.revision,
        "entry_count": len(ledger.entries),
        "problems": problems,
    }


def audit_rollout(root: Path) -> dict[str, Any]:
    """Recompute one audit entirely from durable episode artifacts.

    Main callers:
        The batch launcher and the standalone CLI.
    """
    root = root.expanduser().resolve()
    state = _object(root.joinpath("episode.json"))
    turns, turn_problems = _turn_artifacts(root)
    final_path: Path = root.joinpath("world", "final", "response.json")
    final = _object(final_path) if final_path.is_file() else {}
    final_problems: list[str] = []
    if not final:
        final_problems.append("world/final/response.json is missing")
    else:
        try:
            load_final_response(final_path)
        except RuntimeError as error:
            final_problems.append(f"final contract validation failed: {error}")

    turn_integrity_issues = [
        issue
        for turn in turns
        for issue in _strings(turn["response"].get("integrity_issues"))
    ]
    untrainable_turns = [
        turn["index"]
        for turn in turns
        if turn["response"].get("trainable") is not True
    ]
    final_issues = _strings(final.get("integrity_issues"))
    consistency_ledger = _consistency_ledger_report(root, turns=turns)
    simulator_passed = (
        state.get("status") == "complete"
        and final.get("trainable") is True
        and not turn_problems
        and not final_problems
        and not untrainable_turns
        and not turn_integrity_issues
        and not final_issues
        and consistency_ledger["passed"]
    )

    reasoning = [
        turn["response"].get("reasoning_assessment")
        for turn in turns
        if isinstance(turn["response"].get("reasoning_assessment"), dict)
    ]
    qualities = [float(item["quality"]) for item in reasoning if isinstance(item.get("quality"), (int, float))]
    efficiencies = [float(item["efficiency"]) for item in reasoning if isinstance(item.get("efficiency"), (int, float))]
    rewards = [
        float(turn["response"]["turn_reward"])
        for turn in turns
        if isinstance(turn["response"].get("turn_reward"), (int, float))
    ]

    factual = final.get("factual_consistency")
    factual = factual if isinstance(factual, dict) else {}
    contradictions = _strings(factual.get("contradictions"))
    factual_passed = factual.get("passed") is True and not contradictions
    protocol_integrity = audit_protocol_integrity(root)
    verification = _verification_report(turns)
    task_outcomes = final.get("task_outcomes")
    task_outcomes = task_outcomes if isinstance(task_outcomes, dict) else {}
    checkout_outcome = task_outcomes.get("checkout_idempotency")
    checkout_outcome = checkout_outcome if isinstance(checkout_outcome, dict) else {}
    atomic_declared = checkout_outcome.get("atomic_fix")
    safe_fix_ratio = _criterion_ratio(final.get("criteria"), "safe fix")
    verification_ratio = _criterion_ratio(final.get("criteria"), "verification")
    atomic_validated = (
        atomic_declared is True
        and safe_fix_ratio is not None
        and safe_fix_ratio >= 0.8
        and verification_ratio is not None
        and verification_ratio >= 0.8
        and verification["passed_count"] > 0
    )

    # Trainability gates on world coherence only. Host-path heuristics were
    # removed: they false-rejected intact public-repo sims that used allowlisted
    # host conda / temp paths without breaking the fiction.
    accepted = (
        simulator_passed
        and factual_passed
        and protocol_integrity["passed"]
    )
    return {
        "schema": SCHEMA,
        "rollout_root": str(root),
        "episode_id": state.get("episode_id"),
        "world_id": state.get("world_id"),
        "accepted_for_training": accepted,
        "simulator_integrity": {
            "passed": simulator_passed,
            "episode_status": state.get("status"),
            "final_trainable": final.get("trainable") is True,
            "untrainable_turns": untrainable_turns,
            "issues": (
                turn_problems
                + final_problems
                + turn_integrity_issues
                + final_issues
                + consistency_ledger["problems"]
            ),
        },
        "final_task": {
            "reward": final.get("final_reward"),
            "completion_percentage": final.get("completion_percentage"),
            "criteria": final.get("criteria", []),
            "process_deductions": final.get("process_deductions", []),
            "commentary": final.get("commentary"),
            "task_outcomes": task_outcomes,
            "atomic_fix_declared": atomic_declared,
            "atomic_fix": atomic_validated,
            "atomic_fix_evidence": checkout_outcome.get("atomic_fix_evidence"),
            "atomic_fix_validation": {
                "safe_fix_score_ratio": safe_fix_ratio,
                "verification_score_ratio": verification_ratio,
                "passing_test_execution_count": verification["passed_count"],
            },
        },
        "reasoning": {
            "assessed_turn_count": len(reasoning),
            "total_turn_count": len(turns),
            "assessment_coverage": round(len(reasoning) / len(turns), 6) if turns else 0.0,
            "mean_quality": _mean(qualities),
            "mean_efficiency": _mean(efficiencies),
            "mean_turn_reward": _mean(rewards),
        },
        "executed_verification": verification,
        "factual_consistency": {
            "passed": factual_passed,
            "contradiction_count": len(contradictions),
            "contradictions": contradictions,
        },
        "consistency_ledger": consistency_ledger,
        "model_protocol_integrity": protocol_integrity,
        "action_count": len(turns),
    }


def write_audit(root: Path) -> dict[str, Any]:
    report = audit_rollout(root)
    path: Path = root.expanduser().resolve().joinpath("acceptance_audit.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit one shellsmith rollout.")
    parser.add_argument("rollout_root", type=Path)
    args = parser.parse_args(argv)
    report = write_audit(args.rollout_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["accepted_for_training"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
