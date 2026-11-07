"""In-memory stash of accepted Tinker captures for one campaign stream.

Architecture map
----------------
    rollout runner  --put(capture, episode_id)-->  CaptureStore
    stream engine   --discard_episodes(ids)----->  drop RAM entries + capture_index rows
    trainer         --captures_for_outcomes()--->  accepted captures for one GRPO group
                                                   (rejects mixed group identity keys)
    resume          --load_from_run_dir()------->  rehydrate from durable capture_index.json

Invariant: one GRPO group never mixes group identity keys (for example, two Game
Master models after a provider fallback).

Why this exists
---------------
Ported from the deprecated Tinker collector: only the
store survives. Its proxy-based rollout runner is replaced by Shellsmith's
policy endpoint.

Main callers:
    ``train.stream_engine`` (discard on fallback) and the trainer (put, bank).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shellsmith.credit.to_tinker_bridge import CapturedRollout
from shellsmith.train.durable_capture import (
    load_captures_from_index,
    remove_capture_index_entries,
)

@dataclass
class CaptureStore:
    """Trainer-side stash of accepted Tinker captures for one campaign stream.

    Main callers:
        Tinker rollout runner (put); stream engine on fallback discard;
        the trainer when banking a GROUP_READY world for GRPO.
    """

    _by_episode: dict[str, CapturedRollout] = field(default_factory=dict)
    _run_dir: Path | None = None

    def put(self, capture: CapturedRollout, *, episode_id: str) -> None:
        if not episode_id:
            raise RuntimeError("CaptureStore requires a non-empty episode_id.")
        if not capture.group_identity_key:
            raise RuntimeError("CaptureStore requires a non-empty group_identity_key.")
        self._by_episode[episode_id] = capture

    def discard_episodes(self, episode_ids: list[str]) -> None:
        for episode_id in episode_ids:
            self._by_episode.pop(episode_id, None)
        if self._run_dir is not None:
            remove_capture_index_entries(self._run_dir, episode_ids)

    def get(self, episode_id: str) -> CapturedRollout | None:
        return self._by_episode.get(episode_id)

    def captures_for_outcomes(
        self, outcomes: list[dict[str, Any]]
    ) -> tuple[CapturedRollout, ...]:
        """Return accepted captures for outcomes; reject mixed identity keys."""
        captures: list[CapturedRollout] = []
        for outcome in outcomes:
            if outcome.get("accepted_count") != 1:
                continue
            episode_id = str(outcome["episode_id"])
            capture = self._by_episode.get(episode_id)
            if capture is None:
                raise RuntimeError(
                    f"Accepted outcome {episode_id!r} has no capture in the store."
                )
            captures.append(capture)
        return self.require_single_identity(tuple(captures))

    @staticmethod
    def require_single_identity(
        captures: tuple[CapturedRollout, ...],
    ) -> tuple[CapturedRollout, ...]:
        keys = {item.group_identity_key for item in captures}
        if len(keys) > 1:
            raise RuntimeError(
                "CaptureStore cannot build a Tinker group from mixed identity keys: "
                + ", ".join(sorted(keys))
            )
        return captures

    @classmethod
    def load_from_run_dir(cls, run_dir: Path, *, deps: Any) -> "CaptureStore":
        """Rehydrate accepted captures from capture_index.json.

        Main callers:
            the trainer when resuming a non-empty campaign root.
        """
        store = cls(_run_dir=Path(run_dir))
        loaded = load_captures_from_index(run_dir, deps=deps)
        for episode_id, capture in loaded.items():
            store.put(capture, episode_id=episode_id)
        return store

