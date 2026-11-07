"""Parse one shellsmith task Markdown into a typed spec.

What this is (plain English)
----------------------------
A "world file" is a single ``.md`` file that describes one episode. This module
only checks **section shape** (and that sections are non-empty). It does not
judge whether the world is solvable, realistic, or fair — that is empirical.

Required Markdown shape
-----------------------
::

    # Short title

    ## World
    <private brief for GM — full document is also kept as markdown>

    ## Player
    <the ONLY section Player/vat is allowed to see>

    ## Reward
    <how GM should score at the end>

    ## Termination conditions
    <success / fail / neutral ending rules>

Who sees what
-------------
::

    full .md  -----------------+---> GM (boot + judge)
                               |
    ## Player only ----+---> Player (vat prompt)

Main API
--------
    load_world_file(path)  -> WorldFileSpec
    WorldFileSpec.from_markdown(...)
    .player_projection()         -> vat-safe text
    .content_hash()                      -> sha256 of full markdown

Where this sits
---------------
::

    worlds/drafts/*.md
            |
            v
    contracts.py          (this file — one task)
            |
            +---> smoke / orchestrator (run one episode)
            +---> draft_catalog.py    (audit many drafts)
            +---> hero_slate.py       (pick a fixed campaign set)

Why keep the schema tiny
------------------------
The experiment asks whether a strong model can *simulate* a world. Encoding
repos/APIs/tools in a huge JSON schema would pre-solve that. So we only enforce
ordered H2 sections + privacy projection.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


REQUIRED_SECTIONS = (
    "World",
    "Player",
    "Reward",
    "Termination conditions",
)


@dataclass(frozen=True)
class WorldFileSpec:
    """One lightweight world prompt and its Player projection."""

    id: str
    title: str
    markdown: str
    world: str
    player: str
    reward: str
    termination_conditions: str

    @classmethod
    def from_markdown(cls, *, task_id: str, markdown: str) -> "WorldFileSpec":
        """Parse the three required Markdown sections without interpreting them.

        Main callers:
            `load_world_file` and contract tests.

        The parser deliberately does not validate whether the described world is
        complete, solvable, realistic, or internally consistent. Those are
        empirical properties of the world-simulator rollout, not schema properties.
        """
        lines = markdown.splitlines()
        title_indexes = [
            index
            for index, line in enumerate(lines)
            if line.startswith("# ") and not line.startswith("## ")
        ]
        if len(title_indexes) != 1:
            raise RuntimeError("World file Markdown must contain exactly one H1 title.")
        title_index = title_indexes[0]
        title = lines[title_index][2:].strip()
        if not title:
            raise RuntimeError("World file H1 title must not be empty.")

        section_indexes: list[tuple[str, int]] = []
        for index, line in enumerate(lines):
            if line.startswith("## "):
                section_indexes.append((line[3:].strip(), index))
        names = tuple(name for name, _ in section_indexes)
        if names != REQUIRED_SECTIONS:
            raise RuntimeError(
                "World file Markdown must contain exactly these ordered H2 sections: "
                f"{list(REQUIRED_SECTIONS)}; found {list(names)}."
            )
        if title_index > section_indexes[0][1]:
            raise RuntimeError("World file H1 title must precede all H2 sections.")

        sections: dict[str, str] = {}
        for position, (name, start) in enumerate(section_indexes):
            end = (
                section_indexes[position + 1][1]
                if position + 1 < len(section_indexes)
                else len(lines)
            )
            content = "\n".join(lines[start + 1 : end]).strip()
            if not content:
                raise RuntimeError(f"World file section {name!r} must not be empty.")
            sections[name] = content

        normalized = markdown.strip() + "\n"
        return cls(
            id=task_id,
            title=title,
            markdown=normalized,
            world=sections["World"],
            player=sections["Player"],
            reward=sections["Reward"],
            termination_conditions=sections["Termination conditions"],
        )

    def player_projection(self) -> str:
        """Return the only world file content authorized to enter the vat."""
        return self.player

    def content_hash(self) -> str:
        """Return the SHA-256 identity of the complete private Markdown prompt."""
        return hashlib.sha256(self.markdown.encode("utf-8")).hexdigest()


def load_world_file(path: Path) -> WorldFileSpec:
    """Load one Markdown world file, deriving its stable ID from the filename."""
    if path.suffix.lower() != ".md":
        raise RuntimeError("World files must use the .md extension.")
    return WorldFileSpec.from_markdown(
        task_id=path.stem,
        markdown=path.read_text(encoding="utf-8"),
    )
