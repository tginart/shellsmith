"""Resolve effort labels to the binary reasoning mode recorded on the wire.

Effort and mode are different fields. Keep episode setup, auditing and sampling
on the same mapping; an unknown label must never silently disable reasoning.
"""
from __future__ import annotations

ENABLED_EFFORTS = frozenset({
    "thinking", "deep", "high", "low", "medium", "xhigh", "on", "enable_thinking",
})
DISABLED_EFFORTS = frozenset({"none", "off", "minimal", "disable_thinking"})


def thinking_mode_for_effort(effort: object, *, default: str = "thinking") -> str:
    if effort is None:
        return default
    if not isinstance(effort, str) or not effort.strip():
        raise ValueError("Reasoning effort must be a nonempty supported string.")
    normalized = effort.strip().lower()
    if normalized in ENABLED_EFFORTS:
        return "thinking"
    if normalized in DISABLED_EFFORTS:
        return "disable_thinking"
    raise ValueError(f"Unknown reasoning effort {effort!r}.")
