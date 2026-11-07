"""Regression coverage for the standalone catalog's world-file boundary."""

import json
from pathlib import Path

import pytest

from shellsmith.catalog import CATALOG_SCHEMA, load_catalog_entries
from shellsmith.paths import DEFAULT_WORLD


def _manifest(root: Path, entry: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    path.write_text(json.dumps({
        "schema": CATALOG_SCHEMA,
        "tasks": [{"id": DEFAULT_WORLD.stem, "path": entry}],
    }))
    return path


@pytest.mark.parametrize("escape", ["parent", "absolute", "symlink"])
def test_catalog_refuses_worlds_outside_manifest_directory(tmp_path: Path, escape: str) -> None:
    outside = tmp_path / DEFAULT_WORLD.name
    outside.write_text(DEFAULT_WORLD.read_text())
    root = tmp_path / "worlds"
    root.mkdir()
    if escape == "symlink":
        (root / DEFAULT_WORLD.name).symlink_to(outside)
        entry = DEFAULT_WORLD.name
    elif escape == "absolute":
        entry = str(outside)
    else:
        entry = "../" + DEFAULT_WORLD.name
    manifest = _manifest(root, entry)
    with pytest.raises(ValueError, match="escapes worlds/"):
        load_catalog_entries(manifest)


def test_catalog_accepts_nested_promoted_world(tmp_path: Path) -> None:
    root = tmp_path / "worlds"
    world = root / "drafts" / DEFAULT_WORLD.name
    world.parent.mkdir(parents=True)
    world.write_text(DEFAULT_WORLD.read_text())
    entries = load_catalog_entries(_manifest(root, "drafts/" + DEFAULT_WORLD.name))
    assert len(entries) == 1
    assert entries[0].path == world.resolve()
    assert entries[0].spec.id == DEFAULT_WORLD.stem


def test_explicit_campaign_selection_can_include_unpromoted_world(tmp_path: Path) -> None:
    from shellsmith.campaign.rollout_campaign import _parser, select_worlds

    world = tmp_path / "unpromoted.md"
    world.write_text(DEFAULT_WORLD.read_text())
    args = _parser().parse_args(["--world-file", str(world)])
    selected = select_worlds(args)
    assert [path for path, _ in selected] == [world.resolve()]
    guide = (DEFAULT_WORLD.parent / "README.md").read_text()
    assert "not a security boundary" in guide


def test_shipped_catalog_loads_all_promoted_worlds() -> None:
    entries = load_catalog_entries()
    assert len(entries) == 177
    assert DEFAULT_WORLD.resolve() in {entry.path for entry in entries}


def test_seed_and_simulator_worlds_are_core_members_with_context_variants_excluded(
) -> None:
    """Purpose: Check the new catalog boundary against the actual world files.
    Input: The shipped manifest, seed index, and simulator-operation contracts.
    Output: Exact promoted paths with no context-qualification duplicate.
    """
    entries = load_catalog_entries()
    promoted = {entry.id: entry.path for entry in entries}
    worlds = DEFAULT_WORLD.parent
    seeds = json.loads((worlds / "seeds" / "coverage.json").read_text())["worlds"]
    assert len(seeds) == 23
    for row in seeds:
        assert promoted[row["id"]] == (worlds / "seeds" / row["path"]).resolve()
    meta = sorted((worlds / "drafts").glob("meta_world_simulator_*.md"))
    assert len(meta) == 3
    for path in meta:
        assert promoted[path.stem] == path.resolve()
    assert not any("mlx_context" in path.parts for path in promoted.values())


def test_explicit_core_ids_select_seed_and_simulator_worlds() -> None:
    from shellsmith.train.config import RunSection, TrainConfig, WorldsSection
    from shellsmith.train.world_selection import select_world_stream

    ids = ("seed_constraint_schedule_solver_001", "meta_world_simulator_gcs_checksum_001")
    config = TrainConfig(
        run=RunSection(run_id="promoted-id-selection"),
        worlds=WorldsSection(select="ids", count=len(ids), ids=ids),
    )
    stream = select_world_stream(config)
    assert tuple(world.world_file.stem for world in stream.worlds) == ids


def test_current_qualification_slates_keep_pins_and_historical_slates_reject_new_catalog(
) -> None:
    from shellsmith.campaign.hero_slate import load_hero_slate

    root = DEFAULT_WORLD.parent.parent
    for kind in ("train", "eval"):
        current = load_hero_slate(root / "hero_runs" / f"mlx_qualification_{kind}_v2.json")
        assert len(current.worlds) == 4
        assert current.pinned_assets is not None
        with pytest.raises(RuntimeError, match="catalog manifest changed"):
            load_hero_slate(root / "hero_runs" / f"mlx_qualification_{kind}_v1.json")


def test_current_qualification_plan_does_not_inherit_historical_execution_claims(
) -> None:
    """Purpose: Keep new qualification plans separate from historical execution.
    Input: The current plan, slate files, and selected world files.
    Output: Exact artifact hashes and an explicit unexecuted status.
    """
    import hashlib

    root = DEFAULT_WORLD.parent.parent
    plan = json.loads((root / "hero_runs/mlx_qualification_evidence_v2.json").read_text())
    assert plan["execution_status"] == "not_executed"
    assert plan["execution_history"] == []
    assert "live_execution" not in plan
    assert plan["manifest_sha256"] == hashlib.sha256(
        (root / "worlds/manifest.json").read_bytes()
    ).hexdigest()
    for kind in ("train", "eval"):
        details = plan[kind]
        assert details["slate_path"] == f"hero_runs/mlx_qualification_{kind}_v2.json"
        assert details["slate_sha256"] == hashlib.sha256(
            (root / details["slate_path"]).read_bytes()
        ).hexdigest()
        for world in details["worlds"]:
            assert world["sha256"] == hashlib.sha256(
                (root / world["path"]).read_bytes()
            ).hexdigest()
