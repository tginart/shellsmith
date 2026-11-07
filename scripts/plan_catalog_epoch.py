"""
Purpose: Print a reproducible catalog epoch plan without a model or service call.
Input: A pinned epoch TOML file and optional duration assumptions.
Output: JSON counts, memory controls, hashes, and an explicitly estimated duration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tomllib

from shellsmith.train.config import load_train_config
from shellsmith.train.world_selection import freeze_worlds, select_world_stream


def plan(config_path: Path, *, collection_minutes_per_world: float = 30.0,
         update_seconds: float = 1214.22) -> dict:
    """
    Purpose: Check pinned inputs and calculate the complete single-pass schedule.
    Input: The config path and explicit duration assumptions.
    Output: A JSON-compatible plan. This function performs read-only local I/O.
    """
    config = load_train_config(config_path)
    worlds = freeze_worlds(select_world_stream(config).worlds)
    count = len(worlds)
    updates = math.ceil(count / config.train.groups_per_update)
    assert config.train.steps >= updates, "The target omits the final partial world group."
    assert config.adaptive.initial_attempts == config.adaptive.maximum_attempts
    service_path = Path("examples/studio-catalog-epoch-service.toml")
    service = tomllib.loads(service_path.read_text())["service"]
    return {
        "status": "prepared_only", "worlds": count,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "slate_sha256": hashlib.sha256(Path(config.worlds.hero_slate).read_bytes()).hexdigest(),
        "manifest_sha256": worlds[0].selected.expected_manifest_sha256,
        "seed": config.worlds.seed, "logical_attempts": count * config.adaptive.initial_attempts,
        "attempts_per_world": config.adaptive.initial_attempts,
        "accepted_episodes": "measured at runtime; not guaranteed by attempt count",
        "maximum_optimizer_updates": updates,
        "final_group_worlds": count % config.train.groups_per_update or config.train.groups_per_update,
        "rollout_gm_slots": config.adaptive.parallelism,
        "compatible_gpu_batch_limit": 4, "service_memory_settings": service,
        "estimated_seconds": count * collection_minutes_per_world * 60 + updates * update_seconds,
        "estimate_assumptions": {
            "collection_minutes_per_world": collection_minutes_per_world,
            "update_seconds": update_seconds,
            "update_source": "prior merged full-batch BF16 median; not a new timing qualification",
            "collection_source": "planning assumption, eight attempts share eight slots",
            "excludes": "outages, manual stops, initialization, publication and checkpoint overhead",
        },
        "update_groups": [[world.selected.world_id for world in worlds[start:start + config.train.groups_per_update]]
                          for start in range(0, count, config.train.groups_per_update)],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("examples/studio-catalog-epoch.toml"),
                        help="Pinned epoch config, relative to the repository root.")
    parser.add_argument("--collection-minutes-per-world", type=float, default=30.0,
                        help="Estimated collection time for eight attempts of one world.")
    parser.add_argument("--update-seconds", type=float, default=1214.22,
                        help="Estimated accumulated update time from the prior merged benchmark.")
    args = parser.parse_args()
    print(json.dumps(plan(args.config, collection_minutes_per_world=args.collection_minutes_per_world,
                          update_seconds=args.update_seconds), indent=2))
