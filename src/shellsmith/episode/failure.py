"""Structured failure telemetry for shellsmith episodes.

Architecture map
----------------
    timeout/error at its owning boundary
        -> ``serialize_failure`` captures type, stage, component, cause chain,
           traceback, deadline, and explicit provider attribution
        -> ``episode.json.failure_detail`` keeps the earliest durable diagnosis
        -> campaign and Tinker outcomes copy that diagnosis unchanged
        -> provider health consumes only ``provider_attributable=True`` failures

The short ``error`` string remains for human-readable summaries. The structured
record exists because ``TimeoutError`` normally has an empty message and cannot
identify which of several independent deadlines expired.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any


class AttributedExecutionError(RuntimeError):
    """A non-timeout failure attributed at a concrete execution boundary."""

    def __init__(
        self,
        message: str,
        *,
        component: str,
        stage: str,
        provider_attributable: bool,
    ) -> None:
        self.component = component
        self.stage = stage
        self.provider_attributable = provider_attributable
        super().__init__(message)


class ExecutionTimeoutError(TimeoutError):
    """A deadline failure annotated at the boundary that owns the deadline."""

    def __init__(
        self,
        *,
        component: str,
        stage: str,
        timeout_seconds: float,
        provider_attributable: bool,
    ) -> None:
        self.component = component
        self.stage = stage
        self.timeout_seconds = timeout_seconds
        self.provider_attributable = provider_attributable
        super().__init__(
            f"{component} stage {stage!r} exceeded {timeout_seconds:g} seconds"
        )


def _annotated_error(error: BaseException) -> BaseException | None:
    """Find the owning annotated boundary in an exception cause chain.

    A provider-attributable inner boundary wins over a generic outer wrapper.
    This prevents a vat deadline wrapper from erasing a GM failure that
    already identified its actual owner.
    """
    current: BaseException | None = error
    seen: set[int] = set()
    annotated: list[BaseException] = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if all(
            hasattr(current, name)
            for name in ("component", "stage", "provider_attributable")
        ):
            annotated.append(current)
        current = current.__cause__ or current.__context__
    return next(
        (
            item
            for item in annotated
            if getattr(item, "provider_attributable", False) is True
        ),
        annotated[0] if annotated else None,
    )


def serialize_failure(
    error: BaseException,
    *,
    stage: str,
    component: str = "episode_orchestrator",
    provider_attributable: bool = False,
) -> dict[str, Any]:
    """Return one JSON-safe diagnosis while the original traceback still exists.

    Main callers:
        ``EpisodeOrchestrator.run`` and campaign-level fallback catches.

    Boundary annotations win over orchestration defaults. This lets a GM
    timeout nested inside the Player phase remain attributable to the
    GM, while the Player's own prompt deadline remains neutral.
    """
    annotated = _annotated_error(error)
    source = annotated or error
    detail: dict[str, Any] = {
        "error": f"{type(error).__name__}: {error}",
        "error_type": type(error).__name__,
        "failure_stage": str(getattr(source, "stage", stage)),
        "failure_component": str(getattr(source, "component", component)),
        "provider_attributable": bool(
            getattr(source, "provider_attributable", provider_attributable)
        ),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    }
    timeout_seconds = getattr(source, "timeout_seconds", None)
    if isinstance(timeout_seconds, (int, float)):
        detail["timeout_seconds"] = float(timeout_seconds)
    if error.__cause__ is not None:
        detail["cause_type"] = type(error.__cause__).__name__
        detail["cause_message"] = str(error.__cause__)
    return detail


def read_episode_failure_detail(episode_path: Path) -> dict[str, Any] | None:
    """Read a previously persisted diagnosis without guessing around bad data."""
    if not episode_path.is_file():
        return None
    payload = json.loads(episode_path.read_text(encoding="utf-8"))
    detail = payload.get("failure_detail")
    if detail is None:
        return None
    if not isinstance(detail, dict):
        raise RuntimeError(f"Malformed failure_detail in {episode_path}.")
    return {str(key): value for key, value in detail.items()}


def outcome_failure_fields(detail: dict[str, Any] | None) -> dict[str, Any]:
    """Select structured fields that campaign outcomes should preserve."""
    if detail is None:
        return {}
    names = (
        "error_type",
        "failure_stage",
        "failure_component",
        "timeout_seconds",
        "provider_attributable",
        "traceback",
        "cause_type",
        "cause_message",
    )
    return {name: detail[name] for name in names if name in detail}
