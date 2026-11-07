"""Load a checked-in hero-run slate: which tasks × which GM.

What this is (plain English)
----------------------------
A hero slate is a small JSON manifest that freezes a **reproducible campaign**:

    - which world file files to run
    - which GM harness each task gets (Codex / Claude / Pi, model, effort)
    - deterministic shuffle + batch grouping for the trainer

Without this, "run the good worlds on Sol low" is tribal knowledge. With it,
interruption/resume and cross-run comparisons stay auditable (manifest is hashed).

Pipeline
--------
::

    hero_slate.json  (schema shellsmith.hero_slate.v1)
            |
            v
    load_hero_slate(path)
            |
            |  resolve task paths from draft catalog rules
            |  attach SimulatorSpec per world (or default_simulator)
            |  fail loud on unknown / duplicate / overlapping config
            v
    HeroSlate
      worlds: (HeroWorld(world_file, simulator), ...)
            |
            v
    trainer consumes fixed-size rollout groups

Where this sits
---------------
::

    contracts.py       parse one task .md
    draft_catalog.py   classify / QA the draft pile
    hero_slate.py      pick the immutable run queue + simulator map  (this file)
    trainer consumes fixed-size rollout groups (training-run scripting layer)

What it does NOT do
-------------------
    - execute episodes (smoke / orchestrator do that)
    - acceptance_audit or Tinker conversion
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from shellsmith.paths import CATALOG_MANIFEST, PROJECT_ROOT
from shellsmith.world_file import (
    load_world_file,
)
from shellsmith.campaign.draft_catalog import (
    DEFAULT_DRAFT_ROOT,
    NON_RUNNABLE,
    classify,
)


SCHEMA = "shellsmith.hero_slate.v1"


@dataclass(frozen=True)
class SimulatorSpec:
    name: str
    runtime: str
    model: str
    provider: str
    reasoning_effort: str


@dataclass(frozen=True)
class HeroWorld:
    world_file: Path
    simulator: SimulatorSpec

    @property
    def world_id(self) -> str:
        return load_world_file(self.world_file).id


@dataclass(frozen=True)
class HeroSlate:
    path: Path
    sha256: str
    groups_per_update: int
    shuffle_seed: int
    worlds: tuple[HeroWorld, ...]
    # Named simulator blocks from the slate JSON (`opus`, `terra`, …).
    # Trainer `--override-simulator` looks names up here after load.
    simulators: dict[str, SimulatorSpec]
    pinned_assets: dict[str, Any] | None = None


def _object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Hero slate field {field!r} must be an object.")
    return value


def _strings(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeError(f"Hero slate field {field!r} must be a string list.")
    return value


def _validate_pinned_assets(payload: dict[str, Any], selected: dict[str, Path]) -> dict[str, Any] | None:
    """Qualification slates freeze the approved catalog and every selected byte."""
    if "pinned_assets" not in payload:
        return
    pins = _object(payload["pinned_assets"], field="pinned_assets")
    if set(pins) != {"manifest_sha256", "world_sha256"}:
        raise RuntimeError("Hero slate pinned_assets requires only manifest_sha256 and world_sha256.")
    worlds = _object(pins["world_sha256"], field="pinned_assets.world_sha256")
    if set(worlds) != set(selected):
        raise RuntimeError("Hero slate pinned world IDs must exactly match its selected world IDs.")
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in [pins["manifest_sha256"], *worlds.values()]):
        raise RuntimeError("Hero slate pinned assets must contain lowercase SHA256 digests.")
    if hashlib.sha256(CATALOG_MANIFEST.read_bytes()).hexdigest() != pins["manifest_sha256"]:
        raise RuntimeError("Hero slate pinned catalog manifest changed.")
    from shellsmith.catalog import load_catalog_entries

    approved = {entry.id: entry.path for entry in load_catalog_entries(CATALOG_MANIFEST)}
    for world_id, path in selected.items():
        if approved.get(world_id) != path:
            raise RuntimeError(f"Pinned hero world is not an approved manifest path: {world_id}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != worlds[world_id]:
            raise RuntimeError(f"Hero slate pinned world changed: {world_id}")
    return {"manifest_sha256": pins["manifest_sha256"], "world_sha256": dict(worlds)}


def load_hero_slate(path: Path) -> HeroSlate:
    """Resolve one checked-in manifest into the trainer's exact world queue.

    Main callers:
        The live Tinker trainer before it creates a sampling or training client.

    Why this exists:
        Model assignment and world order affect reward comparability. Keeping
        both in one hashed artifact makes interruption and resume auditable.
    """
    resolved = path.expanduser().resolve()
    raw_bytes = resolved.read_bytes()
    payload = json.loads(raw_bytes)
    if payload.get("schema") != SCHEMA:
        raise RuntimeError(f"Unsupported hero slate schema in {resolved}.")
    groups_per_update = int(payload.get("groups_per_update", 0))
    if groups_per_update != 2:
        raise RuntimeError("Hero slate requires exactly two groups per update.")
    catalog = _object(payload.get("catalog"), field="catalog")
    flavors = set(
        _strings(catalog.get("include_flavors"), field="catalog.include_flavors")
    )
    if not flavors or not flavors <= {"agentic", "public_repo", "paper"}:
        raise RuntimeError("Hero slate catalog flavors are invalid.")
    excluded = set(_strings(catalog.get("exclude", []), field="catalog.exclude"))
    # Optional allowlist for baby / sliced slates. When set, flavor selection
    # only keeps these IDs (paper_worlds + additional_worlds still add).
    include_ids: set[str] | None = None
    if "include" in catalog:
        include_ids = set(_strings(catalog.get("include"), field="catalog.include"))
        if not include_ids:
            raise RuntimeError("Hero slate catalog.include must be non-empty when set.")

    selected: dict[str, Path] = {}
    for candidate in sorted(DEFAULT_DRAFT_ROOT.glob("*.md")):
        if candidate.name in NON_RUNNABLE:
            continue
        spec = load_world_file(candidate)
        flavor = classify(spec, path=candidate)
        if flavor in flavors and spec.id not in excluded:
            if include_ids is None or spec.id in include_ids:
                selected[spec.id] = candidate.resolve()
    unknown_exclusions = excluded - {
        candidate.stem
        for candidate in DEFAULT_DRAFT_ROOT.glob("*.md")
        if candidate.name not in NON_RUNNABLE
    }
    if unknown_exclusions:
        raise RuntimeError(
            "Hero slate excludes unknown catalog worlds: "
            + ", ".join(sorted(unknown_exclusions))
        )
    if include_ids is not None:
        unknown_includes = include_ids - {
            candidate.stem
            for candidate in DEFAULT_DRAFT_ROOT.glob("*.md")
            if candidate.name not in NON_RUNNABLE
        }
        if unknown_includes:
            raise RuntimeError(
                "Hero slate catalog.include names unknown draft worlds: "
                + ", ".join(sorted(unknown_includes))
            )
        missing_includes = include_ids - set(selected)
        if missing_includes:
            raise RuntimeError(
                "Hero slate catalog.include IDs were not selected "
                "(wrong flavor, excluded, or non-runnable): "
                + ", ".join(sorted(missing_includes))
            )
    for raw in _strings(
        catalog.get("additional_worlds", []),
        field="catalog.additional_worlds",
    ):
        candidate = PROJECT_ROOT.joinpath(raw).resolve()
        spec = load_world_file(candidate)
        if spec.id in selected:
            raise RuntimeError(f"Duplicate hero world ID: {spec.id}")
        selected[spec.id] = candidate
    for task_id in _strings(payload.get("paper_worlds"), field="paper_worlds"):
        candidate = DEFAULT_DRAFT_ROOT.joinpath(f"{task_id}.md").resolve()
        spec = load_world_file(candidate)
        if classify(spec, path=candidate) != "paper":
            raise RuntimeError(f"Hero paper entry is not a paper world: {task_id}")
        if spec.id in selected:
            raise RuntimeError(f"Duplicate hero world ID: {spec.id}")
        selected[spec.id] = candidate

    pinned_assets = _validate_pinned_assets(payload, selected)

    simulator_rows = _object(payload.get("simulators"), field="simulators")
    simulators: dict[str, SimulatorSpec] = {}
    for name, raw in simulator_rows.items():
        row = _object(raw, field=f"simulators.{name}")
        simulators[name] = SimulatorSpec(
            name=name,
            runtime=str(row["runtime"]),
            model=str(row["model"]),
            provider=str(row["provider"]),
            reasoning_effort=str(row["reasoning_effort"]),
        )
        required_cli_provider = {
            "claude": "claude-cli",
            "codex": "codex-cli",
        }.get(simulators[name].runtime)
        if (
            required_cli_provider is not None
            and simulators[name].provider != required_cli_provider
        ):
            raise RuntimeError(
                f"Simulator {name!r} uses runtime {simulators[name].runtime!r} "
                f"and must declare provider {required_cli_provider!r}, not "
                f"{simulators[name].provider!r}."
            )
    default_name = str(payload.get("default_simulator"))
    if default_name not in simulators:
        raise RuntimeError("Hero slate default_simulator is undefined.")
    assignments = _object(payload.get("assignments"), field="assignments")
    assigned_model: dict[str, str] = {}
    for simulator_name, raw_ids in assignments.items():
        if simulator_name not in simulators:
            raise RuntimeError(f"Assignment names undefined simulator: {simulator_name}")
        for task_id in _strings(raw_ids, field=f"assignments.{simulator_name}"):
            if task_id not in selected:
                raise RuntimeError(f"Assignment names unselected world: {task_id}")
            if task_id in assigned_model:
                raise RuntimeError(f"World has multiple simulator assignments: {task_id}")
            assigned_model[task_id] = simulator_name

    world_order = _strings(payload.get("world_order"), field="world_order")
    if len(world_order) != len(set(world_order)):
        raise RuntimeError("Hero slate world_order contains duplicate IDs.")
    missing_from_order = set(selected) - set(world_order)
    extra_in_order = set(world_order) - set(selected)
    if missing_from_order or extra_in_order:
        raise RuntimeError(
            "Hero slate world_order is not an exact permutation of selected worlds; "
            f"missing={sorted(missing_from_order)}, extra={sorted(extra_in_order)}."
        )
    worlds = [
        HeroWorld(
            world_file=selected[task_id],
            simulator=simulators[assigned_model.get(task_id, default_name)],
        )
        for task_id in world_order
    ]
    expected_counts = _object(
        payload.get("expected_simulator_counts"),
        field="expected_simulator_counts",
    )
    actual_counts = {name: 0 for name in simulators}
    for world in worlds:
        actual_counts[world.simulator.name] += 1
    normalized_expected = {
        str(name): int(count) for name, count in expected_counts.items()
    }
    if actual_counts != normalized_expected:
        raise RuntimeError(
            "Hero slate simulator counts changed: "
            f"expected {normalized_expected}, got {actual_counts}."
        )
    return HeroSlate(
        path=resolved,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        groups_per_update=groups_per_update,
        shuffle_seed=int(payload["shuffle_seed"]),
        worlds=tuple(worlds),
        simulators=dict(simulators),
        pinned_assets=pinned_assets,
    )
