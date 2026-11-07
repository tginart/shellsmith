"""Parse and validate GM's on-disk JSON answers (boot / turn / final).

What this is (plain English)
----------------------------
GM's authoritative outputs are **files**, not chat prose. This module is
the gate: read JSON from disk → check required fields / types / ranges → return
a frozen dataclass, or **raise**. It never invents a tool observation or reward.

    GM writes a file
            |
            v
    world_artifacts (this file)     # "is this a valid contract?"
            |
            +---> bridge / orchestrator use the typed object
            |
            +---> Player only ever sees tool_response (bridge strips the rest)

Three artifact kinds
--------------------
::

    Stage     On disk                              Loader                 Typed result
    -----     ------                               ------                 ------------
    Boot      world/bootstrap/response.json        load_boot_response     BootResponseArtifact
    Turn      world/turns/<NNNNNN>/response.json   load_turn_response     TurnResponseArtifact
    Final     world/final/response.json            load_final_response    FinalResponseArtifact

Turn shape (the hot path — every Player tool call)
--------------------------------------------------
``load_turn_response`` requires (among other fields)::

    {
      "action_id": "...",
      "tool_response": { "content": "...", "is_error": false, ... },  # Player-facing
      "turn_reward": 0.1,           # private
      "difficulty_rating": 0.2,     # private
      "completion_percentage": 5.0, # private
      "commentary": "...",          # private
      "consistency_review": {       # private; bridge feeds consistency_ledger.append_entry
        "reviewed_revision": 2,
        "observation_claims": [...],
        "new_commitments": [...],
        "conflicts": [],
        "executed_cross_checks": [...]
      },
      "trainable": true,
      "integrity_issues": [],
      ...
    }

Final shape (episode judgment)
------------------------------
``load_final_response`` requires (among other fields)::

    {
      "final_reward": 0.4,
      "completion_percentage": 55.0,
      "trainable": false,
      "criteria": [...],
      "process_deductions": [...],
      "commentary": "...",
      "integrity_issues": [...],
      "factual_consistency": {      # episode-level; NOT the same as consistency_review
        "passed": true,
        "contradictions": []
      },
      "task_outcomes": {...}
    }

How this relates to consistency_ledger.py
-----------------------------------------
    this module     = validate the GM JSON form
    consistency_ledger = append durable receipts AFTER a valid turn artifact

    consistency_review     → per-turn field here → bridge calls append_entry
    factual_consistency    → final-only field here → judge / trainability
    consistency_ledger.json → separate Python-owned file (GM must not edit it)

Who calls what
--------------
    orchestrator.bootstrap_world  → load_boot_response
    WorldBridge.step              → load_turn_response
    orchestrator.judge_episode    → load_final_response

Semi-structured policy
----------------------
Required fields are strict. Extra unknown keys are kept as ``metadata`` so
GM can add private notes without a world file schema change. Missing or
wrong-typed required fields fail loudly — no silent defaults.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shellsmith.types import JsonValue
from shellsmith.episode.schemas import (
    MAX_TOOL_RESPONSE_CHARS,
)


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Game Master response artifact is missing: {path}.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Game Master wrote malformed JSON at {path}.") from error
    if not isinstance(raw, dict):
        raise RuntimeError(f"Game Master response must be a JSON object: {path}.")
    return raw


def _string(raw: dict[str, Any], key: str, *, context: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context}.{key} must be a non-empty string.")
    return value


def _bool(raw: dict[str, Any], key: str, *, context: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise RuntimeError(f"{context}.{key} must be a boolean.")
    return value


def _number(
    raw: dict[str, Any],
    key: str,
    *,
    minimum: float,
    maximum: float,
    context: str,
) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError(f"{context}.{key} must be numeric.")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise RuntimeError(
            f"{context}.{key} must be in [{minimum}, {maximum}], got {value!r}."
        )
    return number


def _nonnegative_int(raw: dict[str, Any], key: str, *, context: str) -> int:
    """Read an exact non-negative integer without accepting booleans or floats."""
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{context}.{key} must be a non-negative integer.")
    return value


def _string_tuple(raw: dict[str, Any], key: str, *, context: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeError(f"{context}.{key} must be a string array.")
    return tuple(value)


def _object_tuple(
    raw: dict[str, Any], key: str, *, context: str
) -> tuple[dict[str, JsonValue], ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise RuntimeError(f"{context}.{key} must be an array of JSON objects.")
    return tuple(
        {str(child_key): child_value for child_key, child_value in item.items()}
        for item in value
    )


@dataclass(frozen=True)
class ParsedToolResponse:
    """The only turn-result projection authorized to enter the Node vat."""

    content: JsonValue
    is_error: bool
    details: dict[str, JsonValue] | None


@dataclass(frozen=True)
class ParsedReasoningAssessment:
    """Optional private grading of reasoning visible before the action."""

    quality: float
    efficiency: float
    commentary: str


@dataclass(frozen=True)
class ParsedConsistencyReview:
    """Private receipt proving the observation was checked against committed truth."""

    reviewed_revision: int
    observation_claims: tuple[str, ...]
    new_commitments: tuple[str, ...]
    conflicts: tuple[str, ...]
    executed_cross_checks: tuple[str, ...]


@dataclass(frozen=True)
class BootResponseArtifact:
    """Private world initialization written to ``world/bootstrap/response.json``.

    Boot success/failure is not a model field. The orchestrator computes pass when
    this artifact is at the expected path with correct structure *and* the vat
    contains only ``AGENTS.md``. Player's first prompt comes from the world file,
    not from this artifact.
    """

    private_world_summary: str
    private_initial_commitments: tuple[str, ...]
    path: Path


@dataclass(frozen=True)
class ParsedToolSlot:
    """One per-tool Player-facing slot inside a sibling world turn."""

    action_id: str
    tool_response: ParsedToolResponse | None
    omit_player_facing: bool
    harness_rejected: bool


@dataclass(frozen=True)
class TurnResponseArtifact:
    """One validated public observation plus private dense supervision."""

    action_id: str
    tool_response: ParsedToolResponse
    turn_reward: float
    difficulty_rating: float
    completion_percentage: float
    commentary: str
    reasoning_assessment: ParsedReasoningAssessment | None
    consistency_review: ParsedConsistencyReview
    trainable: bool
    integrity_issues: tuple[str, ...]
    terminate_episode: bool
    termination_reason: str | None
    metadata: dict[str, JsonValue]
    path: Path
    tool_slots: tuple[ParsedToolSlot, ...] = ()


@dataclass(frozen=True)
class FinalResponseArtifact:
    """Authoritative final-state reward and simulator-integrity decision."""

    final_reward: float
    completion_percentage: float
    trainable: bool
    criteria: tuple[dict[str, JsonValue], ...]
    process_deductions: tuple[dict[str, JsonValue], ...]
    commentary: str
    integrity_issues: tuple[str, ...]
    factual_consistency_passed: bool
    factual_contradictions: tuple[str, ...]
    task_outcomes: dict[str, JsonValue]
    metadata: dict[str, JsonValue]
    path: Path


def parse_boot_response(
    raw: dict[str, JsonValue], *, path: Path | None = None
) -> BootResponseArtifact:
    """Validate the slim two-field boot JSON object.

    Main callers:
        ``load_boot_response`` after reading ``world/bootstrap/response.json``.

    Why this exists:
        Boot is artifact-first. The model must not report status/ready/failed;
        the orchestrator imputes success from this parse plus the vat AGENTS check.
    """
    context = "boot response"
    allowed_keys = {"private_world_summary", "private_initial_commitments"}
    unexpected = sorted(set(raw) - allowed_keys)
    if unexpected:
        raise RuntimeError(
            f"{context} has unexpected fields: {unexpected}."
        )
    return BootResponseArtifact(
        private_world_summary=_string(raw, "private_world_summary", context=context),
        private_initial_commitments=_string_tuple(
            raw, "private_initial_commitments", context=context
        ),
        path=path if path is not None else Path("<in-memory>"),
    )


def load_boot_response(path: Path) -> BootResponseArtifact:
    """Load and validate ``world/bootstrap/response.json`` from disk.

    Main callers:
        ``EpisodeOrchestrator._validate_boot_turn``.
    """
    return parse_boot_response(_load_object(path), path=path)


def load_turn_response(
    path: Path,
    *,
    expected_action_id: str,
    expected_action_ids: tuple[str, ...] | None = None,
) -> TurnResponseArtifact:
    """Validate one per-action artifact and reject stale correlation IDs.

    When ``expected_action_ids`` is provided (sibling batch), ``tool_results``
    must cover every id. Harness-rejected slots may omit Player-facing
    ``tool_response`` content.
    """
    raw = _load_object(path)
    context = "turn response"
    action_id = _string(raw, "action_id", context=context)
    if action_id != expected_action_id:
        raise RuntimeError(
            "World turn action ID mismatch: "
            f"expected {expected_action_id!r}, got {action_id!r}."
        )
    expected_ids = expected_action_ids or (expected_action_id,)
    tool_slots = _parse_tool_slots(raw, expected_ids=expected_ids, context=context)
    primary_slot = next(
        (slot for slot in tool_slots if slot.action_id == expected_action_id),
        tool_slots[0],
    )
    if primary_slot.tool_response is None:
        tool_response = ParsedToolResponse(
            content="",
            is_error=True,
            details={"harness_rejected": True},
        )
    else:
        tool_response = primary_slot.tool_response
    content_raw = str(tool_response.content)
    if len(content_raw) > MAX_TOOL_RESPONSE_CHARS:
        raise RuntimeError(
            "turn response.tool_response.content exceeds the transport ceiling: "
            f"{len(content_raw)} > {MAX_TOOL_RESPONSE_CHARS} characters. Truncate "
            "the in-world output and expose continuation offsets or a narrower "
            "command instead."
        )
    terminate_episode = raw.get("terminate_episode", False)
    if not isinstance(terminate_episode, bool):
        raise RuntimeError("turn response.terminate_episode must be a boolean.")
    termination_reason = raw.get("termination_reason")
    if termination_reason is not None and (
        not isinstance(termination_reason, str) or not termination_reason.strip()
    ):
        raise RuntimeError(
            "turn response.termination_reason must be a non-empty string or null."
        )
    if terminate_episode and termination_reason is None:
        raise RuntimeError(
            "turn response.termination_reason is required when terminate_episode is true."
        )
    reasoning_raw = raw.get("reasoning_assessment")
    reasoning_assessment: ParsedReasoningAssessment | None = None
    if reasoning_raw is not None:
        if not isinstance(reasoning_raw, dict):
            raise RuntimeError(
                "turn response.reasoning_assessment must be an object or null."
            )
        reasoning_assessment = ParsedReasoningAssessment(
            quality=_number(
                reasoning_raw,
                "quality",
                minimum=0.0,
                maximum=1.0,
                context="reasoning_assessment",
            ),
            efficiency=_number(
                reasoning_raw,
                "efficiency",
                minimum=0.0,
                maximum=1.0,
                context="reasoning_assessment",
            ),
            commentary=_string(
                reasoning_raw, "commentary", context="reasoning_assessment"
            ),
        )

    consistency_raw = raw.get("consistency_review")
    if not isinstance(consistency_raw, dict):
        raise RuntimeError("turn response.consistency_review must be a JSON object.")
    consistency_review = ParsedConsistencyReview(
        reviewed_revision=_nonnegative_int(
            consistency_raw,
            "reviewed_revision",
            context="turn response.consistency_review",
        ),
        observation_claims=_string_tuple(
            consistency_raw,
            "observation_claims",
            context="turn response.consistency_review",
        ),
        new_commitments=_string_tuple(
            consistency_raw,
            "new_commitments",
            context="turn response.consistency_review",
        ),
        conflicts=_string_tuple(
            consistency_raw,
            "conflicts",
            context="turn response.consistency_review",
        ),
        executed_cross_checks=_string_tuple(
            consistency_raw,
            "executed_cross_checks",
            context="turn response.consistency_review",
        ),
    )

    trainable = _bool(raw, "trainable", context=context)
    integrity_issues = _string_tuple(raw, "integrity_issues", context=context)
    if (
        tool_response.is_error
        and not content_raw.strip()
        and not primary_slot.omit_player_facing
        and not primary_slot.harness_rejected
    ):
        raise RuntimeError(
            "turn response.tool_response.content must describe the in-world "
            "failure when tool_response.is_error is true."
        )
    if consistency_review.conflicts:
        if trainable:
            raise RuntimeError(
                "turn response.trainable must be false when "
                "consistency_review.conflicts is non-empty."
            )
        if not terminate_episode:
            raise RuntimeError(
                "turn response.terminate_episode must be true when "
                "consistency_review.conflicts is non-empty."
            )
    if not trainable and not terminate_episode:
        raise RuntimeError(
            "turn response.terminate_episode must be true when trainable is false."
        )
    if not trainable and not integrity_issues:
        raise RuntimeError(
            "turn response.integrity_issues must explain why trainable is false."
        )

    required = {
        "action_id",
        "turn_reward",
        "difficulty_rating",
        "completion_percentage",
        "commentary",
        "reasoning_assessment",
        "consistency_review",
        "trainable",
        "integrity_issues",
        "terminate_episode",
        "termination_reason",
    }
    metadata = {
        str(key): value
        for key, value in raw.items()
        if key not in required and key not in {"tool_response", "tool_results"}
    }
    return TurnResponseArtifact(
        action_id=action_id,
        tool_response=tool_response,
        turn_reward=_number(
            raw, "turn_reward", minimum=-1.0, maximum=1.0, context=context
        ),
        difficulty_rating=_number(
            raw, "difficulty_rating", minimum=0.0, maximum=1.0, context=context
        ),
        completion_percentage=_number(
            raw,
            "completion_percentage",
            minimum=0.0,
            maximum=100.0,
            context=context,
        ),
        commentary=_string(raw, "commentary", context=context),
        reasoning_assessment=reasoning_assessment,
        consistency_review=consistency_review,
        trainable=trainable,
        integrity_issues=integrity_issues,
        terminate_episode=terminate_episode,
        termination_reason=termination_reason,
        metadata=metadata,
        path=path,
        tool_slots=tool_slots,
    )


def _parse_tool_slots(
    raw: dict[str, Any],
    *,
    expected_ids: tuple[str, ...],
    context: str,
) -> tuple[ParsedToolSlot, ...]:
    """Parse per-tool Player-facing slots, or synthesize from legacy tool_response."""

    results_raw = raw.get("tool_results")
    if results_raw is None:
        tool_raw = raw.get("tool_response")
        if not isinstance(tool_raw, dict):
            raise RuntimeError(f"{context}.tool_response must be a JSON object.")
        if "content" not in tool_raw:
            raise RuntimeError(f"{context}.tool_response.content is required.")
        content_raw = tool_raw["content"]
        if not isinstance(content_raw, str):
            raise RuntimeError(f"{context}.tool_response.content must be a string.")
        details_raw = tool_raw.get("details")
        if details_raw is not None and not isinstance(details_raw, dict):
            raise RuntimeError(f"{context}.tool_response.details must be an object.")
        if len(expected_ids) != 1:
            raise RuntimeError(
                f"{context}.tool_results is required when the sibling batch has "
                f"{len(expected_ids)} tools."
            )
        return (
            ParsedToolSlot(
                action_id=expected_ids[0],
                tool_response=ParsedToolResponse(
                    content=content_raw,
                    is_error=_bool(tool_raw, "is_error", context="tool_response"),
                    details=(
                        {str(key): value for key, value in details_raw.items()}
                        if details_raw is not None
                        else None
                    ),
                ),
                omit_player_facing=False,
                harness_rejected=False,
            ),
        )

    if not isinstance(results_raw, list) or not results_raw:
        raise RuntimeError(f"{context}.tool_results must be a non-empty array.")
    slots: list[ParsedToolSlot] = []
    seen: set[str] = set()
    for index, item in enumerate(results_raw):
        if not isinstance(item, dict):
            raise RuntimeError(f"{context}.tool_results[{index}] must be an object.")
        # World models frequently mirror the request's ``actions[].id`` key when
        # naming a slot. Accept ``id`` as an alias for ``action_id`` when the
        # canonical key is absent: the id-set equality check below still forces
        # every slot to correlate exactly with the request, so nothing is
        # guessed. Conflicting keys stay a hard error.
        if "action_id" not in item and "id" in item:
            item = {**item, "action_id": item["id"]}
        elif "id" in item and item.get("id") != item.get("action_id"):
            raise RuntimeError(
                f"{context}.tool_results[{index}] has conflicting 'id' and "
                f"'action_id' values: {item.get('id')!r} != {item.get('action_id')!r}."
            )
        slot_id = _string(item, "action_id", context=f"{context}.tool_results[{index}]")
        if slot_id in seen:
            raise RuntimeError(f"{context}.tool_results has duplicate action_id {slot_id!r}.")
        seen.add(slot_id)
        harness_rejected = bool(item.get("harness_rejected", False))
        omit = bool(item.get("omit_player_facing", harness_rejected))
        tool_raw = item.get("tool_response")
        if omit or harness_rejected:
            slots.append(
                ParsedToolSlot(
                    action_id=slot_id,
                    tool_response=None,
                    omit_player_facing=True,
                    harness_rejected=True,
                )
            )
            continue
        if not isinstance(tool_raw, dict):
            raise RuntimeError(
                f"{context}.tool_results[{index}].tool_response is required for "
                "Player-facing slots."
            )
        content_raw = tool_raw.get("content")
        if not isinstance(content_raw, str):
            raise RuntimeError(
                f"{context}.tool_results[{index}].tool_response.content must be a string."
            )
        details_raw = tool_raw.get("details")
        if details_raw is not None and not isinstance(details_raw, dict):
            raise RuntimeError(
                f"{context}.tool_results[{index}].tool_response.details must be an object."
            )
        slots.append(
            ParsedToolSlot(
                action_id=slot_id,
                tool_response=ParsedToolResponse(
                    content=content_raw,
                    is_error=_bool(
                        tool_raw,
                        "is_error",
                        context=f"tool_results[{index}].tool_response",
                    ),
                    details=(
                        {str(key): value for key, value in details_raw.items()}
                        if details_raw is not None
                        else None
                    ),
                ),
                omit_player_facing=False,
                harness_rejected=False,
            )
        )
    if set(expected_ids) != seen:
        raise RuntimeError(
            f"{context}.tool_results action_id set mismatch: "
            f"expected {list(expected_ids)!r}, got {sorted(seen)!r}."
        )
    order = {action_id: index for index, action_id in enumerate(expected_ids)}
    slots.sort(key=lambda slot: order[slot.action_id])
    return tuple(slots)



def load_final_response(path: Path) -> FinalResponseArtifact:
    """Validate final reward without deriving it from per-turn rewards."""
    raw = _load_object(path)
    context = "final response"
    required = {
        "final_reward",
        "completion_percentage",
        "trainable",
        "criteria",
        "process_deductions",
        "commentary",
        "integrity_issues",
        "factual_consistency",
        "task_outcomes",
    }
    factual_raw = raw.get("factual_consistency")
    if not isinstance(factual_raw, dict):
        raise RuntimeError("final response.factual_consistency must be an object.")
    task_outcomes_raw = raw.get("task_outcomes")
    if not isinstance(task_outcomes_raw, dict):
        raise RuntimeError("final response.task_outcomes must be an object.")
    return FinalResponseArtifact(
        final_reward=_number(
            raw, "final_reward", minimum=0.0, maximum=1.0, context=context
        ),
        completion_percentage=_number(
            raw,
            "completion_percentage",
            minimum=0.0,
            maximum=100.0,
            context=context,
        ),
        trainable=_bool(raw, "trainable", context=context),
        criteria=_object_tuple(raw, "criteria", context=context),
        process_deductions=_object_tuple(
            raw, "process_deductions", context=context
        ),
        commentary=_string(raw, "commentary", context=context),
        integrity_issues=_string_tuple(raw, "integrity_issues", context=context),
        factual_consistency_passed=_bool(
            factual_raw, "passed", context="final response.factual_consistency"
        ),
        factual_contradictions=_string_tuple(
            factual_raw,
            "contradictions",
            context="final response.factual_consistency",
        ),
        task_outcomes={str(key): value for key, value in task_outcomes_raw.items()},
        metadata={str(key): value for key, value in raw.items() if key not in required},
        path=path,
    )
