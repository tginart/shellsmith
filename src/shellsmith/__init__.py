"""Shellsmith: train terminal agents in a vat answered by a Game Master.

Layout:
    world_file.py / prompts / ../worlds  — World file authoring boundary
    episode/                             — one Player <-> Game Master episode
    campaign/                            — many worlds and episodes
    credit/ / train/                     — per-turn credit, Tinker datums, adaptive policy

See README.md for the architecture map.
"""

from shellsmith.world_file import (
    WorldFileSpec,
    load_world_file,
)

__all__ = [
    "WorldFileSpec",
    "load_world_file",
]
