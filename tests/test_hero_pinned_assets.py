"""Qualification inputs stay pinned through selection and private freezing."""

from dataclasses import replace
import hashlib
import json

import pytest

from shellsmith.campaign.hero_slate import load_hero_slate
from shellsmith.paths import CATALOG_MANIFEST, PROJECT_ROOT
from shellsmith.train.config import RunSection, TrainConfig
from shellsmith.train.world_selection import freeze_worlds, select_world_stream


@pytest.fixture
def pinned_slate(tmp_path):
    source = PROJECT_ROOT / "hero_runs/first_hero_baby_terra_4.json"
    slate = load_hero_slate(source)
    document = json.loads(source.read_text())
    document["pinned_assets"] = {
        "manifest_sha256": hashlib.sha256(CATALOG_MANIFEST.read_bytes()).hexdigest(),
        "world_sha256": {world.world_id: hashlib.sha256(world.world_file.read_bytes()).hexdigest()
                         for world in slate.worlds},
    }
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(document))
    return path


def _config(path):
    config = TrainConfig(run=RunSection(run_id="qualification"))
    return replace(config, worlds=replace(config.worlds, select="hero", hero_slate=str(path)))


def test_pinned_slate_preserves_approved_hashes_until_freezing(pinned_slate):
    stream = select_world_stream(_config(pinned_slate))
    pins = json.loads(pinned_slate.read_text())["pinned_assets"]
    assert stream.provenance["pinned_assets"] == pins
    frozen = freeze_worlds(stream.worlds)
    assert {world.spec.id: world.sha256 for world in frozen} == pins["world_sha256"]


@pytest.mark.parametrize("corruption, message", [
    ("manifest", "catalog manifest changed"),
    ("world", "pinned world changed"),
    ("missing", "exactly match"),
    ("malformed", "lowercase SHA256"),
    ("unknown_setting", "requires only"),
])
def test_pinned_slate_rejects_changed_or_incomplete_assets(pinned_slate, corruption, message):
    document = json.loads(pinned_slate.read_text())
    pins = document["pinned_assets"]
    world_id = next(iter(pins["world_sha256"]))
    if corruption == "manifest":
        pins["manifest_sha256"] = "0" * 64
    elif corruption == "world":
        pins["world_sha256"][world_id] = "0" * 64
    elif corruption == "missing":
        del pins["world_sha256"][world_id]
    elif corruption == "malformed":
        pins["world_sha256"][world_id] = "X" * 64
    else:
        pins["extra"] = True
    pinned_slate.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match=message):
        select_world_stream(_config(pinned_slate))


@pytest.mark.parametrize("asset", ["world", "manifest"])
def test_pin_is_checked_again_at_frozen_boundary(pinned_slate, asset):
    stream = select_world_stream(_config(pinned_slate))
    world = stream.worlds[0]
    changed = replace(world, **{"expected_sha256" if asset == "world" else "expected_manifest_sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="changed before freezing"):
        freeze_worlds((changed, *stream.worlds[1:]))
