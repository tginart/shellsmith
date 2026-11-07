"""Harness-neutral representation of context compaction inside RL rollouts.

Architecture map:
    harness records + harness evidence
        -> one explicit adapter (Pi)
        -> ordered ``CompactionTurn`` values and validated boundaries
        -> optimizer-specific datum construction

Product rules:
    A summary generation is a policy action and trains once. Its later use is
    prompt context, not a second action. Every successful compaction must have
    explicit generation evidence and, unless terminal, continuation evidence;
    prompt shortening alone is never accepted as proof. All turns retain one
    rollout identity even though compaction starts a new trainable sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence


RequestKind = Literal[
    "world_action",
    "ordinary_action",
    "unscored_model_turn",
    "compaction_generation",
    "compaction_replay",
    "final_text",
]


@dataclass(frozen=True)
class CompactionTurn:
    """One sampled request plus its role in the logical rollout."""

    record: object
    kind: RequestKind
    world_turn_indices: tuple[int, ...] = ()
    sequence_index: int = 0
    boundary_id: str | None = None
    summary: str | None = None
    retained_context_evidence: tuple[str, ...] = ()

    @property
    def request_id(self) -> str:
        return str(getattr(self.record, "request_id", ""))

    @property
    def trains_sampled_tokens(self) -> bool:
        return self.kind != "compaction_replay"


@dataclass(frozen=True)
class CompactionBoundary:
    """Auditable link from one summary action to an optional continuation."""

    boundary_id: str
    generation_request_id: str
    next_request_id: str | None
    sequence_before: int
    sequence_after: int | None
    summary: str
    retained_context_evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExcludedRequest:
    """A sampled request the adapter kept out of the trajectory, and why.

    Why this exists:
        Some harness requests are not agent turns at all (BladeRunner's
        background "session title" call is UI metadata). They must not train,
        but dropping them silently would hide that a sampled record was thrown
        away. Recording the exclusion keeps the audit trail honest.
    """

    request_id: str
    reason: str


@dataclass(frozen=True)
class CompactionAwareTrajectory:
    """One rollout after harness-specific compaction interpretation."""

    rollout_id: str
    harness_id: str
    turns: tuple[CompactionTurn, ...]
    boundaries: tuple[CompactionBoundary, ...] = ()
    excluded: tuple[ExcludedRequest, ...] = ()

    @property
    def trainable_sampled_token_count(self) -> int:
        """Count sampled policy tokens once, excluding deterministic closes."""
        total = 0
        for turn in self.turns:
            if not turn.trains_sampled_tokens:
                continue
            tokens = tuple(getattr(turn.record, "sampled_tokens", ()))
            forced = getattr(turn.record, "forced_close_token_span", None)
            forced_count = 0 if forced is None else forced[1] - forced[0]
            total += len(tokens) - forced_count
        return total

    @property
    def sequence_count(self) -> int:
        return 0 if not self.turns else 1 + max(turn.sequence_index for turn in self.turns)

    def validate(self) -> "CompactionAwareTrajectory":
        request_ids = [turn.request_id for turn in self.turns]
        if any(not value for value in request_ids):
            raise RuntimeError("Compaction trajectory contains an empty request_id.")
        if len(set(request_ids)) != len(request_ids):
            raise RuntimeError("Compaction trajectory request_id values must be unique.")
        if self.trainable_sampled_token_count <= 0:
            raise RuntimeError("Compaction trajectory has no trainable sampled tokens.")
        excluded_ids = [item.request_id for item in self.excluded]
        if any(not value for value in excluded_ids):
            raise RuntimeError("Excluded requests must carry a request_id.")
        overlap = set(excluded_ids) & set(request_ids)
        if overlap:
            raise RuntimeError(
                f"Requests {sorted(overlap)} are both excluded and present as turns."
            )
        boundary_ids = {boundary.boundary_id for boundary in self.boundaries}
        if len(boundary_ids) != len(self.boundaries):
            raise RuntimeError("Compaction boundary IDs must be unique.")
        known = set(request_ids)
        for boundary in self.boundaries:
            if boundary.generation_request_id not in known:
                raise RuntimeError(
                    f"Compaction boundary {boundary.boundary_id!r} references an unknown request."
                )
            if boundary.next_request_id is not None and boundary.next_request_id not in known:
                raise RuntimeError(
                    f"Compaction boundary {boundary.boundary_id!r} references an unknown continuation."
                )
            if not boundary.summary.strip():
                raise RuntimeError(
                    f"Compaction boundary {boundary.boundary_id!r} has an empty summary."
                )
            if boundary.next_request_id is None and boundary.sequence_after is not None:
                raise RuntimeError("Terminal compaction boundaries cannot name a later sequence.")
            if boundary.next_request_id is not None and (
                boundary.sequence_after != boundary.sequence_before + 1
            ):
                raise RuntimeError("Compaction boundaries must advance exactly one sequence.")
        return self


class CompactionAdapter(Protocol):
    """Convert ordered live records plus harness evidence into one trajectory."""

    harness_id: str

    def adapt(
        self,
        *,
        rollout_id: str,
        records: Sequence[object],
        evidence: Path | Sequence[dict[str, Any]] | None = None,
    ) -> CompactionAwareTrajectory: ...


def adapter_for_harness(harness_id: str) -> CompactionAdapter:
    """Resolve adapters explicitly; unsupported harnesses never guess."""
    if harness_id == "pi":
        from .pi import PiCompactionAdapter

        return PiCompactionAdapter()
    raise RuntimeError(f"No compaction adapter is registered for harness {harness_id!r}.")
