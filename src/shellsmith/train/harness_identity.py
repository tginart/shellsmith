"""Local provenance for the Pi SDK and Codex GM selected by a training run.

The entry's canonical path and actual bytes are authoritative, not its version
or the configured string. A bounded package-runtime fingerprint also catches
changed local imports; it deliberately does not traverse dependency installs,
source maps, declarations, or directory symlinks. This is not a hash of the
entire transitive Node dependency graph.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from shellsmith.episode import smoke

_RUNTIME_SUFFIXES = frozenset({".js", ".mjs", ".cjs", ".json", ".ts", ".mts", ".cts", ".node", ".wasm"})
_DIRECT_HELPERS = ("core/auth-storage.js", "core/http-dispatcher.js")


def _file_identity(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _runtime_identity(entry: Path) -> dict[str, Any]:
    # These are explicit deep imports in vat_runner/pi-session.js, even for a
    # standalone custom entry that has no enclosing package.json.
    helpers = {name: _file_identity(entry.parent / name) if (entry.parent / name).is_file() else None
               for name in _DIRECT_HELPERS}
    package = None
    for parent in entry.parents:
        if parent.name == "node_modules":
            break
        candidate = parent / "package.json"
        if candidate.is_file():
            package = candidate
            break
    if package is None:
        return {"direct_helpers": helpers}

    # For the pinned dist/index.js this is only dist/, not its enormous nested
    # node_modules. Entries in other layouts use their package-local subtree.
    relative = entry.relative_to(package.parent)
    root = package.parent / relative.parts[0] if len(relative.parts) > 1 else package.parent
    records = []

    def unreadable(error: OSError) -> None:
        raise error

    for directory, directories, filenames in os.walk(root, onerror=unreadable, followlinks=False):
        current = Path(directory)
        kept = []
        for name in sorted(directories):
            if name == "node_modules" or name.startswith("."):
                continue
            child = current / name
            if child.is_symlink():
                records.append({"name": child.relative_to(root).as_posix(), "symlink": os.readlink(child)})
            else:
                kept.append(name)
        directories[:] = kept
        for name in sorted(filenames):
            child = current / name
            if child.suffix not in _RUNTIME_SUFFIXES or name.endswith((".d.ts", ".d.mts", ".d.cts")):
                continue
            records.append({"name": child.relative_to(root).as_posix(), **_file_identity(child)})
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "package_json": _file_identity(package), "root": str(root),
        "sha256": hashlib.sha256(encoded).hexdigest(), "file_count": len(records),
        "direct_helpers": helpers,
    }


def pi_sdk_identity(configured_module_path: str) -> dict[str, Any]:
    """Resolve and fingerprint the SDK without importing it or installing npm.

    Custom overrides must name an existing file. A missing default installation
    is represented by its expected canonical path and a null actual-byte hash,
    so pure snapshot/resume checks also work in Python-only CI. Live episode
    preflight still requires the file. The runner must receive ``module_path``
    from this result, never resolve the original config string again.
    """
    selected = Path(configured_module_path).expanduser() if configured_module_path else smoke.PINNED_PI_SDK
    try:
        entry = selected.resolve()
        installed = entry.is_file()
        if not installed and (configured_module_path or entry.exists()):
            raise RuntimeError(f"Pi SDK module must exist and be a regular file: {entry}")
        return {
            "module_path": str(entry),
            "module_sha256": hashlib.sha256(entry.read_bytes()).hexdigest() if installed else None,
            "runtime": _runtime_identity(entry) if installed else None,
            "lockfile": _file_identity(smoke.VAT_RUNNER / "package-lock.json") if not configured_module_path else None,
        }
    except OSError as error:
        raise RuntimeError(f"Cannot snapshot Pi SDK file {selected}: {error}") from error


# Use Node's actual platform/arch and createRequire resolution, not Python's
# architecture (which can differ under emulation) or a guessed node_modules.
# This probe only resolves package metadata; it never imports/runs the launcher
# or native Codex, invokes npm, or contacts a model service.
_CODEX_NPM_RESOLVE = r"""
const {createRequire} = require('node:module');
const path = require('node:path');
const [entry, layout] = process.argv.slice(1);
const targets = {
  'darwin:x64': 'x86_64-apple-darwin', 'darwin:arm64': 'aarch64-apple-darwin',
  'linux:x64': 'x86_64-unknown-linux-musl', 'linux:arm64': 'aarch64-unknown-linux-musl',
};
const target = targets[`${process.platform}:${process.arch}`];
if (!target) throw new Error('Unsupported Codex npm platform/architecture');
let packageJson = null;
try {
  packageJson = createRequire(entry).resolve(`@openai/codex-${process.platform}-${process.arch}/package.json`);
} catch {}
const root = packageJson ? path.dirname(packageJson) : path.dirname(path.dirname(entry));
console.log(JSON.stringify({package_json: packageJson,
  native: path.join(root, 'vendor', target, layout, 'codex')}));
"""


def _codex_executable_identity(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f"Codex executable must be an executable regular file: {path}")
    # Native payloads can be hundreds of MB; do not load them all into memory.
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def _codex_npm_identity(entry: Path) -> dict[str, Any] | None:
    if entry.name != "codex.js" or entry.parent.name != "bin":
        return None
    package = entry.parent.parent / "package.json"
    if not package.is_file():
        return None
    metadata = json.loads(package.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("name") != "@openai/codex":
        return None
    source = entry.read_text(encoding="utf-8")
    # Bound support to the npm launcher's two known POSIX native layouts. An
    # unknown layout must not silently record only the unchanged JS wrapper.
    layouts = set(re.findall(
        r"path\.join\(\s*(?:vendorRoot\s*,\s*targetTriple|archRoot)\s*,\s*['\"](bin|codex)['\"]\s*,",
        source,
    ))
    if not source.startswith("#!/usr/bin/env node\n") or len(layouts) != 1:
        raise RuntimeError(f"Unsupported Codex npm launcher layout: {entry}")
    node = shutil.which("node")
    if node is None:
        raise RuntimeError(f"Node executable not found for Codex npm launcher: {entry}")
    resolved = subprocess.run(
        [str(Path(node).resolve(strict=True)), "-e", _CODEX_NPM_RESOLVE, str(entry), layouts.pop()],
        capture_output=True, text=True, check=True, timeout=10,
    )
    payload = json.loads(resolved.stdout)
    native_package = payload["package_json"]
    return {
        "package_json": _file_identity(package),
        "platform_package_json": _file_identity(Path(native_package)) if native_package else None,
        "native_executable": _codex_executable_identity(Path(payload["native"])),
    }


def codex_identity() -> dict[str, Any]:
    """Snapshot the executable selected by smoke's default Codex PATH lookup.

    Canonical launcher path/bytes are authoritative, not a version label. For
    supported @openai/codex npm layouts, also hash the Node-resolved platform
    package (or bundled vendor fallback), its native binary and both manifests.
    No Codex code or --version command is executed. Generic wrappers cover only
    their entry file: this does NOT pin interpreters, arbitrary wrapper imports,
    helper programs, shared libraries, environment/config, or transitive deps.

    A missing default has null path/hash so pure snapshots work without Codex;
    live training must reject it before authentication. Bind the runner to this
    exact ``executable_path``, never a subsequent PATH/symlink lookup. This pins
    launcher selection, not immutable bytes during a running training session.
    """
    selected = shutil.which("codex")  # Same default selection as smoke._parser().
    if selected is None:
        return {"executable_path": None, "executable_sha256": None, "npm": None}
    try:
        executable = _codex_executable_identity(Path(selected))
        return {
            "executable_path": executable["path"], "executable_sha256": executable["sha256"],
            "npm": _codex_npm_identity(Path(executable["path"])),
        }
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"Cannot snapshot Codex executable {selected}: {error}") from error
