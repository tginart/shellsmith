"""A fresh-process consumer of the existing canonical world-file loader.

This intentionally imports the public loader directly, not a substitute parser.
It checks the on-disk index's expected identities/hashes and emits only the
public Player projection (plus identity metadata), never private world prose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from shellsmith.world_file import load_world_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    index_path = args.index.resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    worlds = []
    for row in index["worlds"]:
        spec = load_world_file(index_path.parent / row["path"])
        assert spec.id == row["id"], row["id"]
        assert spec.content_hash() == row["content_sha256"], spec.id
        player = spec.player_projection()
        assert player == spec.player, spec.id
        digest = hashlib.sha256(player.encode("utf-8")).hexdigest()
        assert digest == row["player_sha256"], spec.id
        for use in row["seed_uses"]:
            assert use["commit_sha"] not in player, spec.id
            assert use["source_url"] not in player, spec.id
            assert use["role_evidence"] not in player, spec.id
            assert " ".join(use["source_scope"].split()) not in " ".join(player.split()), spec.id
        assert "## World" not in player and "## Reward" not in player, spec.id
        worlds.append({"id": spec.id, "content_sha256": spec.content_hash(),
                       "player_sha256": digest, "player": player})
    assert [world["id"] for world in worlds] == [row["id"] for row in index["worlds"]]
    print(json.dumps({"consumer_pid": os.getpid(), "worlds": worlds}, indent=2))


if __name__ == "__main__":
    main()
