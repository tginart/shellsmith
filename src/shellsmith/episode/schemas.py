"""Canonical JSON schemas for every authoritative world response.

Architecture map:
    1. These repository constants are the single schema source.
    2. The orchestrator copies all three schemas into each episode audit.
    3. Boot/turn/final prompts point the world agent at on-disk response paths;
       adapters do not force CLI ``--json-schema`` / ``--output-schema`` for boot.
    4. `world_artifacts.py` performs the corresponding semantic validation.

Boot is a closed two-field form (`additionalProperties: false`).
Turn and final artifacts remain extensible through `additionalProperties: true`.
"""

from __future__ import annotations

from shellsmith.types import JsonValue


MAX_TOOL_RESPONSE_CHARS = 48_000

BOOT_RESPONSE_SCHEMA: dict[str, JsonValue] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "World boot response",
    "type": "object",
    "properties": {
        "private_world_summary": {"type": "string", "minLength": 1},
        "private_initial_commitments": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "private_world_summary",
        "private_initial_commitments",
    ],
    "additionalProperties": False,
}


TOOL_RESPONSE_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {
        "content": {"type": "string", "maxLength": MAX_TOOL_RESPONSE_CHARS},
        "is_error": {"type": "boolean"},
        "details": {"type": ["object", "null"]},
    },
    "required": ["content", "is_error"],
    "additionalProperties": True,
}


REASONING_ASSESSMENT_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {
        "quality": {"type": "number", "minimum": 0, "maximum": 1},
        "efficiency": {"type": "number", "minimum": 0, "maximum": 1},
        "commentary": {"type": "string", "minLength": 1},
    },
    "required": ["quality", "efficiency", "commentary"],
    "additionalProperties": False,
}


CONSISTENCY_REVIEW_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {
        "reviewed_revision": {"type": "integer", "minimum": 0},
        "observation_claims": {
            "type": "array",
            "items": {"type": "string"},
        },
        "new_commitments": {
            "type": "array",
            "items": {"type": "string"},
        },
        "conflicts": {
            "type": "array",
            "items": {"type": "string"},
        },
        "executed_cross_checks": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "reviewed_revision",
        "observation_claims",
        "new_commitments",
        "conflicts",
        "executed_cross_checks",
    ],
    "additionalProperties": False,
}


TURN_RESPONSE_SCHEMA: dict[str, JsonValue] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Per-action world response",
    "type": "object",
    "properties": {
        "action_id": {"type": "string", "minLength": 1},
        "tool_response": TOOL_RESPONSE_SCHEMA,
        "tool_results": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "action_id": {"type": "string", "minLength": 1},
                    "tool_response": TOOL_RESPONSE_SCHEMA,
                    "omit_player_facing": {"type": "boolean"},
                    "harness_rejected": {"type": "boolean"},
                },
                "required": ["action_id"],
                "additionalProperties": True,
            },
        },
        "turn_reward": {"type": "number", "minimum": -1, "maximum": 1},
        "difficulty_rating": {"type": "number", "minimum": 0, "maximum": 1},
        "completion_percentage": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
        },
        "commentary": {"type": "string", "minLength": 1},
        "reasoning_assessment": {
            "anyOf": [REASONING_ASSESSMENT_SCHEMA, {"type": "null"}]
        },
        "consistency_review": CONSISTENCY_REVIEW_SCHEMA,
        "trainable": {"type": "boolean"},
        "integrity_issues": {"type": "array", "items": {"type": "string"}},
        "terminate_episode": {"type": "boolean"},
        "termination_reason": {"type": ["string", "null"], "minLength": 1},
    },
    "required": [
        "action_id",
        "turn_reward",
        "difficulty_rating",
        "completion_percentage",
        "commentary",
        "consistency_review",
        "trainable",
        "integrity_issues",
    ],
    "additionalProperties": True,
}


FINAL_RESPONSE_SCHEMA: dict[str, JsonValue] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Terminal world judgment",
    "type": "object",
    "properties": {
        "final_reward": {"type": "number", "minimum": 0, "maximum": 1},
        "completion_percentage": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
        },
        "trainable": {"type": "boolean"},
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion": {"type": "string", "minLength": 1},
                    "weight": {"type": "number", "minimum": 0, "maximum": 1},
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "string", "minLength": 1},
                },
                "required": ["criterion", "weight", "score", "evidence"],
                "additionalProperties": False,
            },
        },
        "process_deductions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "deduction": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string", "minLength": 1},
                },
                "required": ["deduction", "reason"],
                "additionalProperties": False,
            },
        },
        "commentary": {"type": "string", "minLength": 1},
        "integrity_issues": {"type": "array", "items": {"type": "string"}},
        "factual_consistency": {
            "type": "object",
            "properties": {
                "passed": {"type": "boolean"},
                "contradictions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["passed", "contradictions"],
            "additionalProperties": False,
        },
        "task_outcomes": {"type": "object"},
    },
    "required": [
        "final_reward",
        "completion_percentage",
        "trainable",
        "criteria",
        "process_deductions",
        "commentary",
        "integrity_issues",
        "factual_consistency",
        "task_outcomes",
    ],
    "additionalProperties": True,
}
