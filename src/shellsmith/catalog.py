"""Approved Shellsmith catalog manifest loader.

Architecture map:
    worlds/manifest.json
        -> validate schema, ids, paths, and duplicate entries
        -> parse each Markdown task through the canonical contract loader
        -> immutable ``CatalogEntry`` rows for manifest-based selection

Explicit world-file and draft-catalog campaign selection can bypass this loader;
the manifest is a curated selection, not a universal runtime security boundary.

Product rules:
    - The manifest is the promotion boundary. Markdown under ``drafts/`` is not
      runnable merely because it exists.
    - Every path must remain under the manifest's ``worlds/`` directory and name
      a real Markdown file, including after symlink resolution.
    - The manifest id must exactly match the id declared inside the Markdown.
    - Duplicate ids or paths fail loudly.

Why this exists:
    The authoring tree contains meta-specs, reports, and unpromoted drafts. A
    directory glob cannot distinguish approved tasks from authoring material.
    One explicit manifest gives runtime code and the UI the same allowlist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from shellsmith.paths import CATALOG_MANIFEST
from shellsmith.world_file import (
    WorldFileSpec,
    load_world_file,
)


CATALOG_SCHEMA = "shellsmith.catalog.v1"

REASONING_EFFORTS = ("low", "medium", "high", "xhigh")


def claimed_reasoning_effort(raw: Mapping[str, Any] | None, *, source: str) -> str | None:
    """Return a manifest entry's claimed thinking default, or None if omitted.

    An entry that names ``reasoning_effort`` must supply a known value; empty or
    unknown values raise so a typo is never silently ignored.

    Main callers:
        ``load_catalog_entries`` for each manifest task.
    """

    if raw is None or "reasoning_effort" not in raw:
        return None
    value = raw["reasoning_effort"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: reasoning_effort is missing or malformed; got {value!r}")
    effort = value.strip().lower()
    if effort not in REASONING_EFFORTS:
        raise ValueError(
            f"{source}: unknown reasoning_effort {value!r}; "
            f"expected one of {', '.join(REASONING_EFFORTS)}"
        )
    return effort


@dataclass(frozen=True)
class CatalogEntry:
    """One promoted task and its already-validated Markdown contract."""

    id: str
    path: Path
    spec: WorldFileSpec
    # Optional task-spec thinking override. None means inherit the taskset.
    # A claimed key that is missing or unknown fails catalog load.
    reasoning_effort: str | None = None


def load_catalog_entries(
    manifest_path: Path = CATALOG_MANIFEST,
) -> tuple[CatalogEntry, ...]:
    """Load the promoted catalog, refusing malformed or unsafe entries.

    Main callers:
        Campaign world selection and the trainer's world stream.
    """

    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing Shellsmith catalog manifest: {manifest_path}")
    raw: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != CATALOG_SCHEMA:
        raise ValueError(
            f"Shellsmith catalog schema must be {CATALOG_SCHEMA!r}"
        )
    tasks = raw.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Shellsmith catalog tasks must be a non-empty list")

    allowed_root = manifest_path.parent
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    entries: list[CatalogEntry] = []
    for index, item in enumerate(tasks):
        if not isinstance(item, dict):
            raise ValueError(f"Shellsmith catalog task {index} must be an object")
        task_id = str(item.get("id") or "").strip()
        relative_path = str(item.get("path") or "").strip()
        if not task_id or not relative_path:
            raise ValueError(
                f"Shellsmith catalog task {index} needs non-empty id and path"
            )
        path = (manifest_path.parent / relative_path).resolve()
        try:
            path.relative_to(allowed_root)
        except ValueError as error:
            raise ValueError(
                f"Shellsmith catalog path escapes worlds/: {relative_path}"
            ) from error
        if path.suffix.lower() != ".md" or not path.is_file():
            raise FileNotFoundError(
                f"Shellsmith catalog task {task_id!r} has no Markdown file: {path}"
            )
        if task_id in seen_ids:
            raise ValueError(f"duplicate Shellsmith catalog id: {task_id}")
        if path in seen_paths:
            raise ValueError(f"duplicate Shellsmith catalog path: {relative_path}")

        spec = load_world_file(path)
        if spec.id != task_id:
            raise ValueError(
                f"Shellsmith catalog id {task_id!r} does not match "
                f"Markdown id {spec.id!r} in {path.name}"
            )
        seen_ids.add(task_id)
        seen_paths.add(path)
        entries.append(
            CatalogEntry(
                id=task_id,
                path=path,
                spec=spec,
                reasoning_effort=claimed_reasoning_effort(
                    item, source=f"Shellsmith catalog task {task_id}"
                ),
            )
        )
    return tuple(entries)


__all__ = [
    "CATALOG_MANIFEST",
    "CATALOG_SCHEMA",
    "CatalogEntry",
    "load_catalog_entries",
]
