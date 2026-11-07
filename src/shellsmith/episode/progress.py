"""Shared live-progress events for shellsmith episodes.

Architecture map
----------------
    orchestrator / bridge / campaign emit ProgressEvent
            |
            v
    progress.emit(...)
            |
            +-- stdout listener (default for live smoke / campaign)
            +-- future: viewer / file tailer / SSE (same events)

Why this exists
---------------
Campaigns used to stay silent until a giant final JSON dump. Operators need
turn-by-turn signal while parallel jobs run. One event stream keeps stdout and
a later live viewer from inventing two different progress APIs.

Main callers:
    ``episode.orchestrator``, ``episode.bridge``, ``campaign.rollout_campaign``,
    ``episode.smoke`` (installs the default stdout listener).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import sys
import threading
import time
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class ProgressEvent:
    """One follow-along signal for humans or a live dashboard.

    ``phase`` values used today:
        episode_start, boot_start, boot_done, agent_start, turn_start,
        turn_done, judge_start, judge_done, episode_done, episode_error,
        campaign_job_start, campaign_job_done
    """

    episode_id: str
    phase: str
    message: str
    step_index: int | None = None
    world_id: str | None = None
    action_name: str | None = None
    elapsed_seconds: float | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


ProgressListener = Callable[[ProgressEvent], None]

_lock = threading.Lock()
_listeners: list[ProgressListener] = []
_stdout_installed = False


def add_listener(listener: ProgressListener) -> None:
    """Register a consumer (stdout, viewer, etc.). Safe under parallel jobs."""
    with _lock:
        if listener not in _listeners:
            _listeners.append(listener)


def remove_listener(listener: ProgressListener) -> None:
    with _lock:
        if listener in _listeners:
            _listeners.remove(listener)


def emit(event: ProgressEvent) -> None:
    """Fan out one event to every registered listener. Never raises to callers."""
    with _lock:
        listeners = list(_listeners)
    for listener in listeners:
        try:
            listener(event)
        except Exception as caught:
            # Progress must never abort an episode; surface listener bugs loudly.
            print(
                f"[gw progress] listener {listener!r} failed: "
                f"{type(caught).__name__}: {caught}",
                file=sys.stderr,
                flush=True,
            )


def emit_simple(
    *,
    episode_id: str,
    phase: str,
    message: str,
    step_index: int | None = None,
    world_id: str | None = None,
    action_name: str | None = None,
    elapsed_seconds: float | None = None,
    **extras: Any,
) -> None:
    """Convenience wrapper used at call sites."""
    emit(
        ProgressEvent(
            episode_id=episode_id,
            phase=phase,
            message=message,
            step_index=step_index,
            world_id=world_id,
            action_name=action_name,
            elapsed_seconds=elapsed_seconds,
            extras=extras,
        )
    )


def format_stdout_line(event: ProgressEvent) -> str:
    """Render one compact terminal line for interleaved parallel rollouts."""
    parts = [f"[gw {event.episode_id}]", event.phase]
    if event.step_index is not None:
        parts.append(f"turn={event.step_index}")
    if event.action_name:
        parts.append(f"action={event.action_name}")
    if event.elapsed_seconds is not None:
        parts.append(f"elapsed={event.elapsed_seconds:.1f}s")
    if event.world_id:
        parts.append(f"task={event.world_id}")
    parts.append(event.message)
    return " ".join(parts)


def stdout_listener(event: ProgressEvent) -> None:
    """Default human follow-along sink: one flushed line per event."""
    print(format_stdout_line(event), flush=True)


def install_stdout_progress(*, force: bool = False) -> None:
    """Idempotently attach the default stdout listener for live runs.

    Main callers:
        ``run_live_smoke``, ``run_campaign``, ``checkout_batch.run_batch``.
    """
    global _stdout_installed
    with _lock:
        if _stdout_installed and not force:
            return
        if stdout_listener not in _listeners:
            _listeners.append(stdout_listener)
        _stdout_installed = True


class TurnTimer:
    """Small monotonic timer for turn elapsed fields."""

    def __init__(self) -> None:
        self._started = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self._started
