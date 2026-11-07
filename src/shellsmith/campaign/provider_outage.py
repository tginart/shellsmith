"""Campaign-owned GM provider health: streak → cooldown → down.

Architecture map
----------------
    episode finishes (any world on runtime R)
            |
            v
    ProviderHealthTracker.observe(runtime, success=...)
            |
            +-- success  -> reset consecutive streak for R
            +-- failure  -> consecutive += 1
                    |
                    +-- consecutive < threshold (default 5)
                    |       keep running (one-offs / retries are fine)
                    +-- consecutive >= threshold  (trip)
                            |
                            +-- first trip  -> COOLDOWN (default 5 min)
                            |                     no new work for R until expiry
                            +-- second trip -> DOWN
                                                  halt or fallback per config

Keying
------
Health is tracked per ``gm_runtime`` (claude / codex / pi), matching the
DualCapPool provider bucket. Terra and Sol share the Codex budget and the
Codex health streak; Opus and Sonnet share Claude.

Why consecutive-fail, not error markers
---------------------------------------
Marker lists miss real outages (e.g. Claude "monthly spend limit") and misfire
on unrelated strings. A provider that fails every episode is down regardless of
the error text. Quality rejects (episode finished, no error, not accepted) do
**not** count — the provider still answered.

Main callers:
    ``run_campaign_wave`` / ``run_campaign`` / ``checkout_batch`` observe each
    outcome; ``stream_engine`` waits through cooldown and asks ``is_down``
    before adaptive top-ups; halt vs fallback still comes from config.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


PROVIDER_OUTAGE_MODES = ("halt", "fallback")

# Defaults locked for hero / baby runs (overridable on the tracker for tests).
DEFAULT_CONSECUTIVE_FAILURE_THRESHOLD = 5
DEFAULT_COOLDOWN_SECONDS = 300.0


class ProviderHealthStatus(str, Enum):
    """Lifecycle for one GM runtime inside a campaign."""

    HEALTHY = "healthy"
    COOLDOWN = "cooldown"
    DOWN = "down"


class ProviderHealthEvent(str, Enum):
    """What observe() just did (for logging / tests)."""

    OK = "ok"
    FAILURE_COUNTED = "failure_counted"
    ENTERED_COOLDOWN = "entered_cooldown"
    MARKED_DOWN = "marked_down"
    IGNORED_DURING_COOLDOWN = "ignored_during_cooldown"
    IGNORED_NON_PROVIDER_FAILURE = "ignored_non_provider_failure"
    ALREADY_DOWN = "already_down"


@dataclass
class ProviderRuntimeState:
    """Mutable health bookkeeping for one gm_runtime."""

    status: ProviderHealthStatus = ProviderHealthStatus.HEALTHY
    consecutive_failures: int = 0
    trip_count: int = 0
    cooldown_until_monotonic: float = 0.0
    last_error: str | None = None


@dataclass
class ObserveResult:
    """Return value from ProviderHealthTracker.observe."""

    event: ProviderHealthEvent
    runtime: str
    status: ProviderHealthStatus
    consecutive_failures: int
    trip_count: int
    cooldown_remaining_seconds: float = 0.0


@dataclass
class ProviderHealthTracker:
    """Campaign-scoped consecutive-fail circuit for GM runtimes.

    Main callers:
        Campaign wave / inference queue after each episode; stream engine
        before admitting more work on a runtime.

    Why this exists:
        Replaces marker-based ``provider_infrastructure_failure`` circuit
        opening. One-off errors stay retries; sustained failure cools then
        marks the provider down.
    """

    consecutive_failure_threshold: int = DEFAULT_CONSECUTIVE_FAILURE_THRESHOLD
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
    time_fn: Callable[[], float] = field(default=time.monotonic)
    sleep_fn: Callable[[float], Any] = field(default=asyncio.sleep)
    _states: dict[str, ProviderRuntimeState] = field(default_factory=dict, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.consecutive_failure_threshold < 1:
            raise RuntimeError("consecutive_failure_threshold must be at least one.")
        if self.cooldown_seconds < 0:
            raise RuntimeError("cooldown_seconds must be non-negative.")

    @staticmethod
    def runtime_key(gm_runtime: str) -> str:
        key = gm_runtime.strip().lower()
        if not key:
            raise RuntimeError("gm_runtime is required for provider health.")
        return key

    def state(self, gm_runtime: str) -> ProviderRuntimeState:
        key = self.runtime_key(gm_runtime)
        if key not in self._states:
            self._states[key] = ProviderRuntimeState()
        return self._states[key]

    def _expire_cooldown_if_due(self, state: ProviderRuntimeState) -> None:
        if state.status != ProviderHealthStatus.COOLDOWN:
            return
        if self.time_fn() >= state.cooldown_until_monotonic:
            state.status = ProviderHealthStatus.HEALTHY
            state.consecutive_failures = 0

    def is_down(self, gm_runtime: str) -> bool:
        return self.state(gm_runtime).status == ProviderHealthStatus.DOWN

    def is_in_cooldown(self, gm_runtime: str) -> bool:
        state = self.state(gm_runtime)
        self._expire_cooldown_if_due(state)
        return state.status == ProviderHealthStatus.COOLDOWN

    def cooldown_remaining_seconds(self, gm_runtime: str) -> float:
        state = self.state(gm_runtime)
        self._expire_cooldown_if_due(state)
        if state.status != ProviderHealthStatus.COOLDOWN:
            return 0.0
        return max(0.0, state.cooldown_until_monotonic - self.time_fn())

    def snapshot(self) -> dict[str, Any]:
        """JSON-friendly view of every runtime seen so far."""
        out: dict[str, Any] = {}
        now = self.time_fn()
        for key, state in sorted(self._states.items()):
            self._expire_cooldown_if_due(state)
            remaining = 0.0
            if state.status == ProviderHealthStatus.COOLDOWN:
                remaining = max(0.0, state.cooldown_until_monotonic - now)
            out[key] = {
                "status": state.status.value,
                "consecutive_failures": state.consecutive_failures,
                "trip_count": state.trip_count,
                "cooldown_remaining_seconds": round(remaining, 3),
                "last_error": state.last_error,
            }
        return out

    def observe_outcome(self, outcome: dict[str, Any]) -> ObserveResult:
        """Update provider health, ignoring explicitly local episode failures."""
        runtime = str(outcome.get("gm_runtime") or "")
        if _outcome_is_neutral(outcome):
            return self._neutral_result(runtime)
        success = outcome_provider_success(outcome)
        error = outcome.get("error")
        error_text = error if isinstance(error, str) else None
        return self.observe(runtime, success=success, error=error_text)

    def observe(
        self,
        gm_runtime: str,
        *,
        success: bool,
        error: str | None = None,
    ) -> ObserveResult:
        """Record one episode result for a runtime (sync; call under lock in async).

        Success (provider answered without hard error) clears the streak.
        Failure increments; crossing the threshold trips cooldown or DOWN.
        """
        return self._observe_unlocked(gm_runtime, success=success, error=error)

    async def observe_async(
        self,
        gm_runtime: str,
        *,
        success: bool,
        error: str | None = None,
    ) -> ObserveResult:
        """Async-safe observe for concurrent campaign waves."""
        async with self._lock:
            return self._observe_unlocked(
                gm_runtime, success=success, error=error
            )

    async def observe_outcome_async(self, outcome: dict[str, Any]) -> ObserveResult:
        runtime = str(outcome.get("gm_runtime") or "")
        if _outcome_is_neutral(outcome):
            async with self._lock:
                return self._neutral_result(runtime)
        success = outcome_provider_success(outcome)
        error = outcome.get("error")
        error_text = error if isinstance(error, str) else None
        return await self.observe_async(
            runtime, success=success, error=error_text
        )

    def _neutral_result(self, gm_runtime: str) -> ObserveResult:
        """Report a local failure without changing the provider's streak."""
        key = self.runtime_key(gm_runtime)
        state = self.state(key)
        self._expire_cooldown_if_due(state)
        remaining = self.cooldown_remaining_seconds(key)
        return ObserveResult(
            event=ProviderHealthEvent.IGNORED_NON_PROVIDER_FAILURE,
            runtime=key,
            status=state.status,
            consecutive_failures=state.consecutive_failures,
            trip_count=state.trip_count,
            cooldown_remaining_seconds=remaining,
        )

    def _observe_unlocked(
        self,
        gm_runtime: str,
        *,
        success: bool,
        error: str | None = None,
    ) -> ObserveResult:
        key = self.runtime_key(gm_runtime)
        state = self.state(key)
        if state.status == ProviderHealthStatus.DOWN:
            return ObserveResult(
                event=ProviderHealthEvent.ALREADY_DOWN,
                runtime=key,
                status=state.status,
                consecutive_failures=state.consecutive_failures,
                trip_count=state.trip_count,
            )

        self._expire_cooldown_if_due(state)
        if state.status == ProviderHealthStatus.COOLDOWN:
            # In-flight completions during cooldown do not start a second trip.
            if error is not None:
                state.last_error = error
            return ObserveResult(
                event=ProviderHealthEvent.IGNORED_DURING_COOLDOWN,
                runtime=key,
                status=state.status,
                consecutive_failures=state.consecutive_failures,
                trip_count=state.trip_count,
                cooldown_remaining_seconds=max(
                    0.0, state.cooldown_until_monotonic - self.time_fn()
                ),
            )

        if success:
            state.consecutive_failures = 0
            state.last_error = None
            return ObserveResult(
                event=ProviderHealthEvent.OK,
                runtime=key,
                status=state.status,
                consecutive_failures=0,
                trip_count=state.trip_count,
            )

        state.consecutive_failures += 1
        if error is not None:
            state.last_error = error
        if state.consecutive_failures < self.consecutive_failure_threshold:
            return ObserveResult(
                event=ProviderHealthEvent.FAILURE_COUNTED,
                runtime=key,
                status=state.status,
                consecutive_failures=state.consecutive_failures,
                trip_count=state.trip_count,
            )

        # Trip: first → cooldown; second → down.
        state.consecutive_failures = 0
        state.trip_count += 1
        if state.trip_count >= 2:
            state.status = ProviderHealthStatus.DOWN
            return ObserveResult(
                event=ProviderHealthEvent.MARKED_DOWN,
                runtime=key,
                status=state.status,
                consecutive_failures=0,
                trip_count=state.trip_count,
            )

        state.status = ProviderHealthStatus.COOLDOWN
        state.cooldown_until_monotonic = self.time_fn() + self.cooldown_seconds
        return ObserveResult(
            event=ProviderHealthEvent.ENTERED_COOLDOWN,
            runtime=key,
            status=state.status,
            consecutive_failures=0,
            trip_count=state.trip_count,
            cooldown_remaining_seconds=self.cooldown_seconds,
        )

    async def wait_until_available(self, gm_runtime: str) -> None:
        """Block until runtime is healthy enough to start a new episode.

        Raises:
            RuntimeError if the runtime is already marked DOWN.
        """
        key = self.runtime_key(gm_runtime)
        while True:
            async with self._lock:
                state = self.state(key)
                if state.status == ProviderHealthStatus.DOWN:
                    raise RuntimeError(
                        f"GM provider {key!r} is marked down after repeated "
                        f"failure streaks (trip_count={state.trip_count}). "
                        f"last_error={state.last_error!r}."
                    )
                self._expire_cooldown_if_due(state)
                if state.status == ProviderHealthStatus.HEALTHY:
                    return
                remaining = max(
                    0.0, state.cooldown_until_monotonic - self.time_fn()
                )
            if remaining <= 0:
                continue
            await self.sleep_fn(remaining)


def _outcome_is_neutral(outcome: dict[str, Any]) -> bool:
    """True for a failed episode explicitly attributed outside GM."""
    error = outcome.get("error")
    return (
        isinstance(error, str)
        and bool(error.strip())
        and outcome.get("provider_attributable") is not True
    )


def outcome_provider_success(outcome: dict[str, Any]) -> bool:
    """True when an outcome contains no provider-attributable failure."""
    return not episode_counts_as_provider_failure(outcome)


def episode_counts_as_provider_failure(outcome: dict[str, Any]) -> bool:
    """Only explicitly attributed GM errors feed the failure streak."""
    error = outcome.get("error")
    return (
        isinstance(error, str)
        and bool(error.strip())
        and outcome.get("provider_attributable") is True
    )


# --- Backup remap (unchanged policy: Sol↔Opus, terra↔Sonnet) -----------------


@dataclass(frozen=True)
class GMTarget:
    """One concrete GM harness triple used by smoke / campaign jobs."""

    runtime: str
    model: str
    provider: str


_SOL = GMTarget(runtime="codex", model="gpt-6.1-sol", provider="codex-cli")
_OPUS = GMTarget(runtime="claude", model="opus", provider="claude-cli")
_TERRA = GMTarget(runtime="codex", model="gpt-5.6-terra", provider="codex-cli")
_SONNET = GMTarget(runtime="claude", model="sonnet", provider="claude-cli")

_BACKUP_BY_MODEL: dict[str, GMTarget] = {
    "gpt-6.1-sol": _OPUS,
    # Immutable imported hero slates still name this exact Sol version.
    "gpt-5.6-sol": _OPUS,
    "sol": _OPUS,
    "opus": _SOL,
    "gpt-5.6-terra": _SONNET,
    "terra": _SONNET,
    "sonnet": _TERRA,
}


def resolve_backup_world(
    *,
    runtime: str,
    model: str,
    provider: str,
) -> GMTarget:
    """Return the bidirectional backup GM for one failing primary.

    Main callers:
        Campaign fallback remap of remaining jobs; trainer new-group fallback.

    Raises:
        RuntimeError when ``model`` is not in the Sol/Opus or terra/Sonnet table.
    """
    del runtime, provider  # Partner triple is authoritative; model selects the pair.
    key = model.strip()
    backup = _BACKUP_BY_MODEL.get(key)
    if backup is None:
        backup = _BACKUP_BY_MODEL.get(key.casefold())
    if backup is None:
        raise RuntimeError(
            "No provider-outage backup GM is configured for model "
            f"{model!r}. Known models: gpt-6.1-sol, opus, gpt-5.6-terra, sonnet "
            "(plus aliases sol/terra)."
        )
    return backup


def apply_backup_to_job_fields(
    *,
    gm_runtime: str,
    gm_model: str,
    gm_provider: str,
    gm_fallback_applied: bool,
) -> tuple[GMTarget, str]:
    """Resolve backup fields for one remaining job, refusing a second hop.

    Returns:
        (backup_target, original_gm_model)

    Raises:
        RuntimeError if fallback was already applied (no ping-pong) or unknown model.
    """
    if gm_fallback_applied:
        raise RuntimeError(
            "Provider-outage fallback already applied for this job; refusing a "
            f"second remap from {gm_model!r}."
        )
    backup = resolve_backup_world(
        runtime=gm_runtime,
        model=gm_model,
        provider=gm_provider,
    )
    return backup, gm_model


# --- Deprecated marker helpers (kept as thin wrappers for old call sites) ----


def provider_infrastructure_failure(error: str | None) -> bool:
    """Deprecated alias: any non-empty error counts as a provider-health failure.

    Prefer ``episode_counts_as_provider_failure(outcome)`` on full outcomes.
    Marker substring matching is intentionally gone — see module docstring.
    """
    if error is None:
        return False
    return bool(str(error).strip())


def provider_circuit_should_open(outcomes: list[dict[str, Any]]) -> bool:
    """Deprecated: old global circuit (≥2 marker-infra, 0 accepts).

    New code must use ``ProviderHealthTracker``. This shim approximates the
    old boolean for callers not yet migrated: True when any runtime in the
    outcome list has ≥ threshold consecutive hard errors with zero successes
    interrupting that suffix — without cooldown. Prefer the tracker.
    """
    # Build a throwaway tracker and replay; open only if something is DOWN
    # (two full trips). For single-shot tests that expected open after 2 infra
    # markers, this no longer matches — callers should migrate.
    tracker = ProviderHealthTracker()
    for outcome in outcomes:
        runtime = str(outcome.get("gm_runtime") or "unknown")
        tracker.observe_outcome({**outcome, "gm_runtime": runtime})
    return any(tracker.is_down(key) for key in tracker.snapshot())
