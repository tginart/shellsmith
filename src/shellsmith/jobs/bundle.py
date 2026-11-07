"""Content-addressed source staging; credentials and run directories never enter it."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

from .storage import canonical, digest, private_dir, read, write

MAX_BUNDLE_BYTES = 32 * 1024 * 1024
ROOTS = ("src/shellsmith", "src/studio_compute", "worlds", "hero_runs")
SKIP = {"node_modules", "__pycache__", ".git", ".venv", "credentials", "auth", "secrets", "runs",
        "credentials.json", "auth.json", "secrets.json"}
SUFFIXES = {".py", ".json", ".jsonl", ".md", ".mjs", ".js", ".cjs", ".toml", ".lock", ".jinja", ".txt"}


def pack(source_root: Path) -> dict:
    root = source_root.expanduser().resolve()
    files = {}
    for name in ("pyproject.toml", "README.md"):
        path = root / name
        if path.is_file() and not path.is_symlink():
            files[name] = path.read_bytes()
    for name in ROOTS:
        folder = root / name
        if not folder.is_dir() or folder.is_symlink():
            continue
        for path in sorted(folder.rglob("*")):
            relative = path.relative_to(root)
            if any(part in SKIP or part.startswith(".") for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"Source bundles refuse symlinks: {relative}")
            if path.is_file() and path.suffix in SUFFIXES:
                files[relative.as_posix()] = path.read_bytes()
    if "src/shellsmith/jobs/worker.py" not in files:
        raise ValueError("Select the Shellsmith checkout containing the detached job worker")
    manifest = {name: digest(value) for name, value in sorted(files.items())}
    source_sha256 = digest(canonical(manifest))
    files["runtime_manifest.json"] = canonical({"source_sha256": source_sha256, "files": manifest})
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        for name, value in sorted(files.items()):
            item = tarfile.TarInfo(name)
            item.size, item.mode, item.mtime = len(value), 0o600, 0
            archive.addfile(item, io.BytesIO(value))
    raw = data.getvalue()
    if len(raw) > MAX_BUNDLE_BYTES or sum(map(len, files.values())) > MAX_BUNDLE_BYTES:
        raise ValueError("Frozen source exceeds the 32 MiB staging limit")
    return {"source_sha256": source_sha256, "archive": base64.b64encode(raw).decode()}


def verify(root: Path, expected: str) -> dict:
    manifest = read(root / "runtime_manifest.json")
    if manifest.get("source_sha256") != expected or digest(canonical(manifest["files"])) != expected:
        raise ValueError("Frozen runtime source identity mismatch")
    for name, sha in manifest["files"].items():
        parts = PurePosixPath(name)
        if parts.is_absolute() or ".." in parts.parts:
            raise ValueError("Unsafe runtime manifest path")
        path = root / name
        if any((root / Path(*Path(name).parts[:index])).is_symlink() for index in range(1, len(Path(name).parts) + 1)):
            raise ValueError("Frozen runtime contains a symlinked source path")
        if path.is_symlink() or not path.is_file() or digest(path.read_bytes()) != sha:
            raise ValueError(f"Frozen runtime source changed: {name}")
    for name in ROOTS:
        for path in (root / name).rglob("*"):
            relative = path.relative_to(root)
            if any(part in SKIP or part.startswith(".") for part in relative.parts):
                continue
            if path.is_file() and path.suffix in SUFFIXES and relative.as_posix() not in manifest["files"]:
                raise ValueError("Unexpected source file in frozen runtime: " + relative.as_posix())
    return manifest


def unpack(home: Path, payload: dict) -> Path:
    expected = payload["source_sha256"]
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("Invalid source identity")
    root = private_dir(home / "runtime" / expected)
    if (root / "runtime_manifest.json").exists():
        verify(root, expected)
        return root
    raw = base64.b64decode(payload["archive"], validate=True)
    if len(raw) > MAX_BUNDLE_BYTES:
        raise ValueError("Source bundle exceeds limit")
    total = 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        names = set()
        for item in archive:
            name = PurePosixPath(item.name)
            total += item.size
            if (not item.isfile() or name.is_absolute() or ".." in name.parts or
                    item.name in names or total > MAX_BUNDLE_BYTES):
                raise ValueError("Unsafe or oversized source bundle")
            names.add(item.name)
            target = root / item.name
            private_dir(target.parent)
            if target.exists() or target.is_symlink():
                raise ValueError("Partial source staging exists; use a new verified source revision")
            stream = archive.extractfile(item)
            if stream is None:
                raise ValueError("Incomplete source bundle")
            with target.open("xb") as output:
                output.write(stream.read())
            target.chmod(0o600)
    verify(root, expected)
    return root
