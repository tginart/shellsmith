"""Remove private host vat paths from data sent to the Game Master.

Architecture map:
    Pi event/action containing the real temporary cwd
      -> recursively replace trusted host prefixes with stable virtual paths
      -> persist and send only the redacted world-facing value

Important invariant:
    Redaction changes only strings and preserves the surrounding JSON shape. The
    private path mapping comes from the trusted orchestrator, never model input.

Why this exists:
    Pi must run in a real temporary cwd, but the simulated world should know only
    a canonical `/workspace`. Prompt instructions were not sufficient to prevent
    a simulator from replaying the real path in host commands after boot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from shellsmith.types import JsonValue


def redact_private_paths(
    value: JsonValue,
    *,
    replacements: Sequence[tuple[str, str]],
) -> JsonValue:
    """Recursively replace private path prefixes in one JSON-compatible value.

    Main callers:
        ``WorldBridge.step`` for per-action requests and
        ``EpisodeOrchestrator.judge_episode`` for terminal evidence.

    Longer private prefixes are applied first so a workspace nested under a vat
    root becomes `/workspace`, not `/vat/workspace`.
    """
    ordered = sorted(replacements, key=lambda pair: len(pair[0]), reverse=True)
    for private, virtual in ordered:
        if not private or not virtual:
            raise ValueError("path redaction entries must be non-empty strings")
    return _redact(value, ordered)


def _redact(value: JsonValue, replacements: Sequence[tuple[str, str]]) -> JsonValue:
    if isinstance(value, str):
        redacted = value
        for private, virtual in replacements:
            redacted = redacted.replace(private, virtual)
        return redacted
    if isinstance(value, list):
        return [_redact(item, replacements) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _redact(child, replacements)
            for key, child in value.items()
        }
    return value
