"""Local POSIX filename budgets, before any training SDK or sampler work.

Use a conservative 255-byte component allowance for macOS/Linux, even if a
volume permits more. A smaller PC_NAME_MAX on the destination filesystem wins.
For not-yet-created directories, query the nearest existing ancestor without
creating anything; this also follows an existing destination mount/symlink.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shellsmith.train.config import TrainConfig
    from shellsmith.train.world_selection import SelectedWorld

MAX_COMPONENT_BYTES = 255


def _component_limit(directory: Path) -> int:
    candidate = directory
    while True:
        try:
            limit = os.pathconf(candidate, "PC_NAME_MAX")
        except FileNotFoundError:
            if candidate == candidate.parent:
                raise ValueError(f"Cannot determine filename byte limit for {directory}.") from None
            candidate = candidate.parent
        except OSError as error:
            raise ValueError(f"Cannot determine filename byte limit for {directory}: {error}") from error
        else:
            # Indeterminate/unlimited pathconf results do not relax our allowance.
            return min(MAX_COMPONENT_BYTES, limit) if limit > 0 else MAX_COMPONENT_BYTES


def validate_component_name(name: str, *, label: str, directory: Path | None = None) -> None:
    """Check encoded bytes, not characters; no path is created or opened for writing."""
    limit = MAX_COMPONENT_BYTES if directory is None else _component_limit(directory)
    size = len(os.fsencode(name))
    if size > limit:
        raise ValueError(f"{label} requires {size} bytes; filename component limit is {limit} bytes. "
                         "Shorten run.run_id or the selected world ID.")


def validate_training_names(*, config: TrainConfig, worlds: tuple[SelectedWorld, ...],
                            campaign_id: str, segment_root: Path) -> None:
    """Budget this invocation's restored campaign identity and largest attempt.

    Reuse the collector's job builder instead of maintaining a parallel episode
    ID/rollout-directory formatter. Only one job per world is needed: decimal
    attempt widths never shrink, and the builder supplies its minimum padding.
    """
    from shellsmith.campaign.stream_campaign import build_task_attempt_jobs

    validate_component_name(segment_root.name, label="Campaign segment directory", directory=segment_root.parent)
    for world in worlds:
        job = build_task_attempt_jobs(
            campaign_id=campaign_id, campaign_root=segment_root,
            world_file=world.world_file, world_id=world.world_id, flavor="",
            attempt_start=config.world_overrides.get(world.world_id, {}).get(
                "maximum_attempts", config.adaptive.maximum_attempts), attempt_count=1,
            gm_runtime=world.gm_runtime, gm_model=world.gm_model,
            gm_provider=world.gm_provider, gm_reasoning_effort=world.gm_reasoning_effort,
        )[0]
        validate_component_name(job.output_dir.parent.name, label="Selected world ID directory", directory=segment_root)
        validate_component_name(job.output_dir.name, label="World attempt directory", directory=job.output_dir.parent)
        validate_component_name(f"{job.episode_id}.jsonl", label="Player record filename",
                                directory=segment_root / "player_records")
        # smoke._create_staging_root uses this prefix plus tempfile's eight-byte
        # random suffix. Tests measure the real staging name and exercise its
        # exact boundary, so changes to that naming contract cannot go unnoticed.
        validate_component_name(f"{job.episode_id}-world-xxxxxxxx", label="Episode staging directory",
                                directory=Path(tempfile.gettempdir()))
