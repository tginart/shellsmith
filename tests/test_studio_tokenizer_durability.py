"""Real fsync barriers plus injected syscall failures; no power-loss claim."""

import errno
import os
from pathlib import Path
import stat

import pytest

from studio_compute.tokenizer import materialize
from test_studio_setup_assets import bundle


def fd_target(fd, parent):
    info = os.fstat(fd)
    if stat.S_ISDIR(info.st_mode):
        if (info.st_dev, info.st_ino) == (parent.stat().st_dev, parent.stat().st_ino):
            return "parent"
        return "staging"
    for path in parent.glob(".pending-*/*"):
        if (path.stat().st_dev, path.stat().st_ino) == (info.st_dev, info.st_ino):
            return path.name
    pytest.fail("Unexpected synced descriptor")


def test_assets_and_staging_are_synced_before_rename_then_parent(tmp_path, monkeypatch):
    assets = bundle(**{"tokenizer.json": b"vocabulary", "chat_template.jinja": b"template"})
    events = []
    original_sync, original_rename = os.fsync, Path.rename
    def fsync(fd):
        events.append(fd_target(fd, tmp_path))
        return original_sync(fd)
    def rename(path, target):
        events.append("rename")
        return original_rename(path, target)
    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(Path, "rename", rename)
    visible = materialize(assets, tmp_path)
    assert events == ["tokenizer.json", "chat_template.jinja", "staging", "rename", "parent"]
    assert (visible / "tokenizer.json").read_bytes() == b"vocabulary"
    events.clear()
    assert materialize(assets, tmp_path) == visible
    assert events == ["parent"]  # Cache hit completes an uncertain parent barrier.


@pytest.mark.parametrize("failed", ["tokenizer.json", "staging", "parent"])
def test_failed_barrier_cannot_report_success_and_retry_is_safe(tmp_path, monkeypatch, failed):
    assets = bundle(**{"tokenizer.json": b"vocabulary", "chat_template.jinja": b"template"})
    original = os.fsync
    failed_once = []
    def fsync(fd):
        if fd_target(fd, tmp_path) == failed and not failed_once:
            failed_once.append(True)
            raise OSError(errno.EIO, "injected durability barrier failure")
        return original(fd)
    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(OSError, match="injected durability barrier"):
        materialize(assets, tmp_path)
    visible = tmp_path / assets["hash"]
    assert visible.exists() == (failed == "parent")
    assert not list(tmp_path.glob(".pending-*"))
    if visible.exists():
        assert (visible / "tokenizer.json").read_bytes() == b"vocabulary"
        assert (visible / "chat_template.jinja").read_bytes() == b"template"
    monkeypatch.setattr(os, "fsync", original)
    assert materialize(assets, tmp_path) == visible
