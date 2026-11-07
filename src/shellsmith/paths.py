"""Filesystem anchors for Shellsmith: where worlds live and where runs write.

Architecture map
----------------
    PACKAGE_ROOT   src/shellsmith/            prompts/, episode/vat_runner/
    PROJECT_ROOT   repo checkout root         worlds/, hero_runs/, docker/
        WORLDS_DIR          worlds/                 promoted world files
        CATALOG_MANIFEST    worlds/manifest.json    explicit runtime allowlist
        DRAFTS_DIR          worlds/drafts/          campaign drafts (some promoted)
        DEFAULT_WORLD       worlds/checkout_idempotency_001.md
    DEFAULT_LOG_ROOT   <system temp>/shellsmith-<uid>  per-user run output

Invariants
----------
- World files are data in the source checkout, not package data. A checkout
  (editable install) has ``worlds/``; the Docker sandbox image deliberately does
  not, because the Player shares that container. World anchors do not check
  existence at import time; missing worlds raise at the call site. The default
  run root is created and verified private before any caller can use it.
- ``SHELLSMITH_ROOT`` overrides the checkout location.
- Run output defaults to the system temp directory. A configured ``log_dir``
  (config file or ``--log-dir``) replaces ``DEFAULT_LOG_ROOT`` for a run.

Why this exists
---------------
In the original monorepo every module derived paths from the monorepo root.
This module is the one place that knows the standalone layout.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile

PACKAGE_ROOT = Path(__file__).resolve().parent


_ROOT_OVERRIDE = os.environ.get("SHELLSMITH_ROOT", "").strip()
PROJECT_ROOT = (
    Path(_ROOT_OVERRIDE).expanduser().resolve() if _ROOT_OVERRIDE else PACKAGE_ROOT.parents[1]
)
WORLDS_DIR = PROJECT_ROOT / "worlds"
CATALOG_MANIFEST = WORLDS_DIR / "manifest.json"
DRAFTS_DIR = WORLDS_DIR / "drafts"
DEFAULT_WORLD = WORLDS_DIR / "checkout_idempotency_001.md"
HERO_RUNS_DIR = PROJECT_ROOT / "hero_runs"
PROMPTS_DIR = PACKAGE_ROOT / "prompts"
def _default_log_root() -> Path:
    # POSIX temporary directories may be shared by all accounts. Do not let
    # the first account's output directory block every other account's runs.
    # Platforms without getuid (Windows) already have a per-user temp tree.
    name = f"shellsmith-{os.getuid()}" if hasattr(os, "getuid") else "shellsmith"
    return Path(tempfile.gettempdir()) / name


def _prepare_default_log_root(root: Path) -> Path:
    """Create an account-private root; never follow a pre-planted symlink."""
    try:
        root.mkdir(mode=0o700)
    except FileExistsError:
        pass
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise PermissionError(f"Default run root is not a real directory: {root}")
    if hasattr(os, "getuid"):
        if info.st_uid != os.getuid():
            raise PermissionError(f"Default run root is owned by another account: {root}")
        # Verify and chmod the same directory inode, without following a link
        # swapped into the pathname between the initial check and open.
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        directory = os.open(root, flags)
        try:
            if os.fstat(directory).st_uid != os.getuid():
                raise PermissionError(f"Default run root changed owner: {root}")
            os.fchmod(directory, 0o700)
        finally:
            os.close(directory)
    return root


DEFAULT_LOG_ROOT = _prepare_default_log_root(_default_log_root())
