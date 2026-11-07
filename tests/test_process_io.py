"""Real filesystem regression tests for atomic, durable JSON publication."""

from concurrent.futures import ThreadPoolExecutor
import errno
import json
import os
from pathlib import Path
import stat
import threading

import pytest

from shellsmith import process_io
from shellsmith.train.durable_capture import (
    clear_pending_update, load_durable_outcome, load_pending_update,
    write_durable_outcome, write_pending_update,
)


@pytest.mark.parametrize("entry", ["request", "outcome", "pending"])
def test_file_sync_precedes_publish_and_directory_sync(tmp_path, monkeypatch, entry) -> None:
    events = []
    real_fsync, real_replace = os.fsync, Path.replace

    def fsync(fd):
        events.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        real_fsync(fd)

    def replace(source, target):
        assert json.loads(source.read_text())
        events.append("publish")
        return real_replace(source, target)

    monkeypatch.setattr(process_io.os, "fsync", fsync)
    monkeypatch.setattr(Path, "replace", replace)
    if entry == "request":
        path = process_io.write_json(tmp_path / "request.json", {"prompt": "hello"})
        assert json.loads(path.read_text()) == {"prompt": "hello"}
    elif entry == "outcome":
        write_durable_outcome(tmp_path, {"episode_id": "e1", "accepted_count": 1})
        assert load_durable_outcome(tmp_path)["accepted_count"] == 1
    else:
        write_pending_update(tmp_path, {"step": 4})
        assert load_pending_update(tmp_path)["step"] == 4
    assert events[-3:] == ["file", "publish", "directory"]
    assert not list(tmp_path.rglob("*.tmp"))


def test_pending_update_deletion_is_synced(tmp_path, monkeypatch) -> None:
    write_pending_update(tmp_path, {"step": 4})
    real_fsync = os.fsync
    observed = []

    def fsync(fd):
        observed.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        assert load_pending_update(tmp_path) is None
        real_fsync(fd)

    monkeypatch.setattr(process_io.os, "fsync", fsync)
    clear_pending_update(tmp_path)
    assert observed == [True]


def test_windows_request_logging_keeps_file_sync_without_directory_open(tmp_path, monkeypatch) -> None:
    real_open, real_fsync = os.open, os.fsync
    synced = []

    def open_file(path, flags, *args, **kwargs):
        assert Path(path) != tmp_path, "Windows CRT cannot open the directory"
        return real_open(path, flags, *args, **kwargs)

    def fsync(fd):
        assert stat.S_ISREG(os.fstat(fd).st_mode)
        synced.append(True)
        real_fsync(fd)

    monkeypatch.setattr(process_io.sys, "platform", "win32")
    monkeypatch.setattr(process_io.os, "open", open_file)
    monkeypatch.setattr(process_io.os, "fsync", fsync)
    path = process_io.write_json(tmp_path / "request.json", {"stage": "boot"})
    assert json.loads(path.read_text()) == {"stage": "boot"}
    assert synced == [True]


@pytest.mark.parametrize("error_number", [errno.EINVAL, errno.EIO])
def test_directory_sync_only_tolerates_unsupported_filesystems(tmp_path, monkeypatch, error_number) -> None:
    real_fsync = os.fsync

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(error_number, "directory sync")
        real_fsync(fd)

    monkeypatch.setattr(process_io.os, "fsync", fsync)
    path = tmp_path / "record.json"
    if error_number == errno.EIO:
        with pytest.raises(OSError) as raised:
            process_io.write_json(path, {"value": 1})
        assert raised.value.errno == errno.EIO
    else:
        process_io.write_json(path, {"value": 1})
    assert json.loads(path.read_text()) == {"value": 1}


def test_failed_file_sync_preserves_previous_record_and_removes_temporary(tmp_path, monkeypatch) -> None:
    path = process_io.write_json(tmp_path / "record.json", {"value": "old"})

    def fail_sync(fd):
        raise OSError(errno.EIO, "disk failure")

    monkeypatch.setattr(process_io.os, "fsync", fail_sync)
    with pytest.raises(OSError):
        process_io.write_json(path, {"value": "new"})
    assert json.loads(path.read_text()) == {"value": "old"}
    assert list(tmp_path.iterdir()) == [path]


def test_concurrent_writers_do_not_share_a_temporary_file(tmp_path, monkeypatch) -> None:
    real_fsync = os.fsync
    both_written = threading.Barrier(2)

    def fsync(fd):
        real_fsync(fd)
        if stat.S_ISREG(os.fstat(fd).st_mode):
            both_written.wait(timeout=5)

    monkeypatch.setattr(process_io.os, "fsync", fsync)
    path = tmp_path / "record.json"
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(process_io.write_json, path, {"writer": n}) for n in range(2)]
        for future in futures:
            assert future.result(timeout=10) == path
    assert json.loads(path.read_text()) in [{"writer": 0}, {"writer": 1}]
    assert list(tmp_path.iterdir()) == [path]


def test_new_parent_directory_entries_are_synced(tmp_path, monkeypatch) -> None:
    real_fsync = os.fsync
    synced_inodes = []

    def fsync(fd):
        info = os.fstat(fd)
        if stat.S_ISDIR(info.st_mode):
            synced_inodes.append(info.st_ino)
        real_fsync(fd)

    monkeypatch.setattr(process_io.os, "fsync", fsync)
    parent = tmp_path / "campaign" / "logs"
    process_io.write_json(parent / "record.json", {"value": 1})
    assert synced_inodes == [tmp_path.stat().st_ino, parent.parent.stat().st_ino, parent.stat().st_ino]
