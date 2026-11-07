"""Exercise default-root initialization through the shipped paths module."""

import os
from pathlib import Path
import runpy
import stat

import pytest

from shellsmith import paths

pytestmark = pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership boundary")


def _initialize(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(paths.tempfile, "gettempdir", lambda: str(tmp_path))
    return runpy.run_path(paths.__file__)["DEFAULT_LOG_ROOT"]


def test_default_root_is_created_private_on_import(tmp_path, monkeypatch) -> None:
    root = _initialize(tmp_path, monkeypatch)
    assert root.is_dir()
    assert root.stat().st_uid == os.getuid()
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert _initialize(tmp_path, monkeypatch) == root


def test_default_root_rejects_preplanted_symlink(tmp_path, monkeypatch) -> None:
    outside = tmp_path / "attacker"
    outside.mkdir()
    root = tmp_path / f"shellsmith-{os.getuid()}"
    root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(PermissionError, match="not a real directory"):
        _initialize(tmp_path, monkeypatch)
    assert list(outside.iterdir()) == []


def test_default_root_rejects_other_account_owner(tmp_path, monkeypatch) -> None:
    other_uid = os.getuid() + 1
    root = tmp_path / f"shellsmith-{other_uid}"
    root.mkdir()
    monkeypatch.setattr(paths.os, "getuid", lambda: other_uid)
    with pytest.raises(PermissionError, match="another account"):
        _initialize(tmp_path, monkeypatch)


def test_existing_owned_root_becomes_private_without_losing_data(tmp_path, monkeypatch) -> None:
    root = tmp_path / f"shellsmith-{os.getuid()}"
    root.mkdir(mode=0o755)
    (root / "previous-run").write_text("preserve")
    assert _initialize(tmp_path, monkeypatch) == root
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert (root / "previous-run").read_text() == "preserve"
