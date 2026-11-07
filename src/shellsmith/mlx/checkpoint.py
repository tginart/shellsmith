"""Atomic, integrity-checked native MLX sampler and full recovery artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

from .identity import canonical, digest, file_digest

FORMAT = "shellsmith-mlx-checkpoint-v1"
COMPONENTS = {"sampler": {"adapters.safetensors"},
              "state": {"adapters.safetensors", "optimizer.safetensors", "rng.safetensors"}}


def verify_checkpoint(path: str | Path, *, kind: str = "state", expected_identity: dict[str, Any] | None = None,
                      verify_sampler: bool = True) -> dict[str, Any]:
    """Verify all files and recovery metadata before numerical deserialization."""
    folder = Path(path)
    if kind not in COMPONENTS or folder.is_symlink() or not folder.is_dir():
        raise ValueError("Invalid MLX checkpoint directory or kind.")
    manifest = folder / "manifest.json"
    if manifest.is_symlink():
        raise ValueError("MLX checkpoint manifest cannot be a symlink.")
    try:
        doc = json.loads(manifest.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("MLX checkpoint has no valid manifest.") from exc
    if (not isinstance(doc, dict) or doc.get("format") != FORMAT or doc.get("complete") is not True
            or doc.get("kind") != kind):
        raise ValueError("Incomplete or incompatible MLX checkpoint manifest.")
    identity = doc.get("policy_identity")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("MLX checkpoint lacks immutable policy identity.")
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("MLX checkpoint differs from frozen model, tokenizer, adapter, runtime, or source identity.")
    if "identity_sha256" in identity and identity["identity_sha256"] != digest(
            {key: value for key, value in identity.items() if key != "identity_sha256"}):
        raise ValueError("MLX checkpoint policy identity digest is inconsistent.")
    version = doc.get("policy_version")
    if type(version) is not int or version < 0:
        raise ValueError("MLX checkpoint policy version is invalid.")
    files = doc.get("files")
    if not isinstance(files, dict) or set(files) != COMPONENTS[kind]:
        raise ValueError("MLX checkpoint component inventory is incomplete or unexpected.")
    if {entry.name for entry in folder.iterdir()} != COMPONENTS[kind] | {"manifest.json"}:
        raise ValueError("MLX checkpoint directory contains unexpected components.")
    for name, expected_sha in files.items():
        component = folder / name
        if (not isinstance(expected_sha, str) or len(expected_sha) != 64 or component.is_symlink()
                or not component.is_file() or file_digest(component) != expected_sha):
            raise ValueError(f"MLX checkpoint integrity failure: {name}")
    position = doc.get("trainer_position")
    if position is not None and (not isinstance(position, dict) or
                                 ("step" in position and position["step"] != version)):
        raise ValueError("MLX checkpoint trainer cursor does not match the policy version.")
    if kind == "state":
        metadata = doc.get("optimizer")
        if not isinstance(metadata, dict) or metadata.get("steps") != version or type(metadata.get("steps")) is not int:
            raise ValueError("Full MLX checkpoint optimizer does not match the policy version.")
        sampler = doc.get("sampler_path")
        if not isinstance(sampler, str) or not sampler or not Path(sampler).is_absolute():
            raise ValueError("Full MLX checkpoint lacks its corresponding sampler artifact.")
        if verify_sampler:
            sampler_doc = verify_checkpoint(sampler, kind="sampler", expected_identity=identity, verify_sampler=False)
            if sampler_doc["policy_version"] != version:
                raise ValueError("Full MLX checkpoint references a different sampler policy version.")
            if sampler_doc["files"]["adapters.safetensors"] != files["adapters.safetensors"]:
                raise ValueError("Full MLX checkpoint and matching sampler contain different adapter weights.")
    return doc


def publish_checkpoint(target: str | Path, *, writer: Callable[[Path], None], kind: str,
                       policy_identity: dict[str, Any], policy_version: int, sampler_path: str | None = None,
                       optimizer: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None) -> str:
    """Fsync all components and metadata before one atomic directory rename."""
    if kind not in COMPONENTS or type(policy_version) is not int or policy_version < 0:
        raise ValueError("Invalid MLX checkpoint kind/version.")
    target = Path(target).absolute()
    if target.name in {"", ".", ".."} or target.exists():
        raise FileExistsError(f"MLX checkpoint destination already exists or is invalid: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=target.parent))
    try:
        writer(staging)
        if {entry.name for entry in staging.iterdir()} != COMPONENTS[kind]:
            raise RuntimeError("MLX checkpoint writer did not produce the exact component inventory.")
        doc = {"format": FORMAT, "complete": True, "kind": kind, "policy_identity": policy_identity,
               "policy_version": policy_version, "sampler_path": sampler_path,
               "optimizer": optimizer, "trainer_position": metadata,
               "files": {name: file_digest(staging / name) for name in sorted(COMPONENTS[kind])}}
        manifest = staging / "manifest.json"
        with manifest.open("wb") as stream:
            stream.write(canonical(doc) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        for name in COMPONENTS[kind]:
            path = staging / name
            if path.is_symlink():
                raise ValueError("Checkpoint writers cannot publish symlinked components.")
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        verify_checkpoint(staging, kind=kind, expected_identity=policy_identity)
        os.rename(staging, target)
        fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return str(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
