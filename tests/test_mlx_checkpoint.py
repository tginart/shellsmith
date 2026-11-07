"""Portable recovery/integrity tests, including abrupt incomplete publication."""

import json
from pathlib import Path

import pytest

from shellsmith.mlx.checkpoint import publish_checkpoint, verify_checkpoint

IDENTITY = {"backend": "mlx", "model": {"asset_sha256": "frozen-model"}, "adapter": {"rank": 16, "scale": 2.0}}


def _sampler(root, version=1, adapter=b"BF16 adapters"):
    def writer(folder):
        (folder / "adapters.safetensors").write_bytes(adapter)
    return publish_checkpoint(root / f"sampler-{version}", writer=writer, kind="sampler", policy_identity=IDENTITY,
                              policy_version=version)


def _state(root, sampler, *, version=1, adapter=b"BF16 adapters", name="state"):
    def writer(folder):
        for filename, data in {"adapters.safetensors": adapter, "optimizer.safetensors": b"FP32 master and moments",
                               "rng.safetensors": b"explicit RNG"}.items():
            (folder / filename).write_bytes(data)
    return publish_checkpoint(root / name, writer=writer, kind="state", policy_identity=IDENTITY, policy_version=version,
                              sampler_path=sampler, optimizer={"steps": version}, metadata={"step": version, "next_world_offset": 7})


def test_atomic_complete_state_covers_adapter_optimizer_rng_and_cursor(tmp_path):
    sampler = _sampler(tmp_path)
    path = _state(tmp_path, sampler)
    doc = verify_checkpoint(path, expected_identity=IDENTITY)
    assert doc["complete"] is True
    assert set(doc["files"]) == {"adapters.safetensors", "optimizer.safetensors", "rng.safetensors"}
    assert doc["trainer_position"]["next_world_offset"] == 7
    assert not list(tmp_path.glob(".pending-*"))


def test_corrupt_tensor_and_wrong_identity_are_rejected(tmp_path):
    sampler = _sampler(tmp_path)
    path = Path(_state(tmp_path, sampler))
    with pytest.raises(ValueError, match="frozen"):
        verify_checkpoint(path, expected_identity={"different": "model"})
    (path / "optimizer.safetensors").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="integrity"):
        verify_checkpoint(path)


def test_state_requires_exact_matching_sampler_weights(tmp_path):
    sampler = _sampler(tmp_path)
    with pytest.raises(ValueError, match="different adapter"):
        _state(tmp_path, sampler, adapter=b"other weights")
    assert not (tmp_path / "state").exists()


def test_writer_failure_never_advertises_recoverable_checkpoint(tmp_path):
    def writer(folder):
        (folder / "adapters.safetensors").write_bytes(b"partial")
        raise OSError("simulated interrupted write")
    with pytest.raises(OSError, match="interrupted"):
        publish_checkpoint(tmp_path / "state", writer=writer, kind="state", policy_identity=IDENTITY, policy_version=1)
    assert not (tmp_path / "state").exists()
    assert not list(tmp_path.glob(".pending-*"))


def test_manifest_cannot_drop_rng_or_disagree_with_cursor(tmp_path):
    sampler = _sampler(tmp_path)
    path = Path(_state(tmp_path, sampler))
    manifest = path / "manifest.json"
    document = json.loads(manifest.read_text())
    document["trainer_position"]["step"] = 0
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="cursor"):
        verify_checkpoint(path)
    document["trainer_position"]["step"] = 1
    document["files"].pop("rng.safetensors")
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="inventory"):
        verify_checkpoint(path)


def test_symlink_components_rejected(tmp_path):
    sampler = Path(_sampler(tmp_path))
    original = tmp_path / "outside"
    original.write_bytes((sampler / "adapters.safetensors").read_bytes())
    (sampler / "adapters.safetensors").unlink()
    (sampler / "adapters.safetensors").symlink_to(original)
    with pytest.raises(ValueError, match="integrity"):
        verify_checkpoint(sampler, kind="sampler")
