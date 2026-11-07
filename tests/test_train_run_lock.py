"""Real POSIX filesystem and separate-interpreter training run lock tests."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import sys
import time
import traceback

import pytest

from shellsmith.train.run_lock import (
    TrainRunLockError,
    TrainRunLockedError,
    lock_train_run,
)

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
_TRY_LOCK = """
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, sys.argv[1])
from shellsmith.train.run_lock import TrainRunLockedError, lock_train_run
started = time.monotonic()
try:
    with lock_train_run(Path(sys.argv[2])):
        Path(sys.argv[3]).write_text("entered\\n", encoding="utf-8")
except TrainRunLockedError as error:
    print(json.dumps({"status": "locked", "message": str(error),
                      "elapsed": time.monotonic() - started}))
else:
    print(json.dumps({"status": "entered"}))
"""
_HOLD_LOCK = """
import os
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from shellsmith.train.run_lock import lock_train_run
with lock_train_run(Path(sys.argv[2])):
    print("held", flush=True)
    if sys.stdin.readline().strip() == "exit":
        os._exit(23)  # Skip context-manager cleanup; the kernel must release it.
"""


def _sidecar(log_dir: Path) -> Path:
    run = log_dir.resolve()
    return run.parent / f".{run.name}.trainer.lock"


def _attempt(log_dir: Path, marker: Path, *, cwd: Path | None = None) -> dict:
    # A separate Python interpreter gets independent descriptors. The timeout
    # fails a mistakenly blocking implementation, rather than hanging pytest.
    result = subprocess.run(
        [sys.executable, "-c", _TRY_LOCK, str(_SOURCE_ROOT), str(log_dir), str(marker)],
        cwd=cwd, capture_output=True, text=True, timeout=10, check=True,
    )
    assert result.stderr == ""
    return json.loads(result.stdout)


def _assert_locked(result: dict, marker: Path) -> None:
    assert result["status"] == "locked"
    assert result["elapsed"] < 2
    assert "already locked" in result["message"]
    assert "different run.log_dir" in result["message"]
    assert "Do not delete" in result["message"]
    assert not marker.exists(), "the losing process must never enter its body"


@pytest.mark.parametrize("existing", ["missing", "empty", "state"])
def test_lock_is_private_stable_and_outside_untouched_run(tmp_path, capsys, existing):
    log_dir = tmp_path / "run"
    state = log_dir / "trainer_state.json"
    if existing != "missing":
        log_dir.mkdir()
    if existing == "state":
        state.write_bytes(b"synthetic-state-secret")
        state.chmod(0o600)
        before = state.stat()
    lock = _sidecar(log_dir)
    with lock_train_run(log_dir) as value:
        assert value is None
        assert lock.is_file() and not lock.is_symlink()
        info = lock.stat()
        assert info.st_uid == os.geteuid()
        assert info.st_nlink == 1
        assert stat.S_IMODE(info.st_mode) == 0o600
        assert lock.read_bytes() == b""
        if existing == "missing":
            assert not log_dir.exists()
        elif existing == "empty":
            assert list(log_dir.iterdir()) == []
        else:
            assert state.stat() == before
            assert state.read_bytes() == b"synthetic-state-secret"
            assert list(log_dir.iterdir()) == [state]
    assert lock.exists(), "release must never unlink the shared lock inode"
    with lock_train_run(log_dir):
        assert (lock.stat().st_dev, lock.stat().st_ino) == (info.st_dev, info.st_ino)
    assert capsys.readouterr() == ("", "")


def test_missing_parent_is_created_without_creating_run(tmp_path):
    log_dir = tmp_path / "missing" / "parent" / "run"
    with lock_train_run(log_dir):
        assert log_dir.parent.is_dir()
        assert stat.S_IMODE(log_dir.parent.stat().st_mode) == 0o700
        assert not log_dir.exists()
        assert list(log_dir.parent.iterdir()) == [_sidecar(log_dir)]


@pytest.mark.parametrize("mask", [0, 0o077, 0o777])
def test_new_lock_mode_is_0600_under_real_umask(tmp_path, mask):
    log_dir = tmp_path / "run"
    previous = os.umask(mask)
    try:
        with lock_train_run(log_dir):
            assert stat.S_IMODE(_sidecar(log_dir).stat().st_mode) == 0o600
    finally:
        os.umask(previous)


def test_separate_python_cannot_enter_until_whole_context_has_exited(tmp_path):
    log_dir = tmp_path / "run"
    marker = tmp_path / "entered"
    with lock_train_run(log_dir):
        _assert_locked(_attempt(log_dir, marker), marker)
        assert not log_dir.exists(), "even the fresh-directory check must be protected"
        log_dir.mkdir()
        state = log_dir / "trainer_state.json"
        state.write_text('{"status":"running"}', encoding="utf-8")
        _assert_locked(_attempt(log_dir, marker), marker)
        state.write_text('{"status":"complete"}', encoding="utf-8")
        _assert_locked(_attempt(log_dir, marker), marker)
        inode = _sidecar(log_dir).stat().st_ino
    assert _attempt(log_dir, marker) == {"status": "entered"}
    assert marker.read_text(encoding="utf-8") == "entered\n"
    assert _sidecar(log_dir).stat().st_ino == inode
    assert state.read_text(encoding="utf-8") == '{"status":"complete"}'


def test_same_process_reentry_also_fails_fast(tmp_path):
    log_dir = tmp_path / "run"
    with lock_train_run(log_dir):
        with pytest.raises(TrainRunLockedError, match="already locked"):
            with lock_train_run(log_dir):
                pytest.fail("a second run in the same process must not enter")
        _assert_locked(_attempt(log_dir, tmp_path / "entered"), tmp_path / "entered")


@pytest.mark.parametrize("failure_type", [RuntimeError, OSError, KeyboardInterrupt, asyncio.CancelledError])
def test_exceptions_propagate_unchanged_and_release_for_another_process(tmp_path, failure_type):
    log_dir = tmp_path / "run"
    failure = failure_type("synthetic caller failure")
    with pytest.raises(failure_type) as caught:
        with lock_train_run(log_dir):
            raise failure
    assert caught.value is failure
    assert _attempt(log_dir, tmp_path / "entered") == {"status": "entered"}


@pytest.mark.parametrize("other_path", ["other", "different-parent/run"])
def test_other_runs_are_independent(tmp_path, other_path):
    with lock_train_run(tmp_path / "run"):
        assert _attempt(tmp_path / other_path, tmp_path / "entered") == {"status": "entered"}


@pytest.mark.parametrize("exit_kind", ["exit", "kill"])
def test_subprocess_exit_releases_kernel_lock_without_removing_sidecar(tmp_path, exit_kind):
    log_dir = tmp_path / "run"
    marker = tmp_path / "entered"
    with subprocess.Popen(
        [sys.executable, "-c", _HOLD_LOCK, str(_SOURCE_ROOT), str(log_dir)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) as process:
        try:
            with selectors.DefaultSelector() as ready:
                ready.register(process.stdout, selectors.EVENT_READ)
                assert ready.select(timeout=10), "lock holder did not become ready"
            assert process.stdout.readline() == "held\n"
            inode = _sidecar(log_dir).stat().st_ino
            _assert_locked(_attempt(log_dir, marker), marker)
            if exit_kind == "kill":
                process.kill()
                output, error = process.communicate(timeout=10)
                assert process.returncode == -signal.SIGKILL
            else:
                output, error = process.communicate("exit\n", timeout=10)
                assert process.returncode == 23
            assert (output, error) == ("", "")
            assert _attempt(log_dir, marker) == {"status": "entered"}
            assert _sidecar(log_dir).stat().st_ino == inode
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)


@pytest.mark.parametrize("alias_kind", [
    "run-symlink", "dangling-run-symlink", "parent-symlink", "relative", "dotdot", "symlink-dotdot",
])
def test_path_aliases_share_one_cross_process_lock(tmp_path, alias_kind):
    log_dir = tmp_path / "runs" / "run"
    log_dir.parent.mkdir()
    if alias_kind in {"run-symlink", "dangling-run-symlink"}:
        if alias_kind == "run-symlink":
            log_dir.mkdir()
        alias = tmp_path / "alias"
        alias.symlink_to(log_dir, target_is_directory=True)
    elif alias_kind == "parent-symlink":
        linked_parent = tmp_path / "linked-parent"
        linked_parent.symlink_to(log_dir.parent, target_is_directory=True)
        alias = linked_parent / log_dir.name
    elif alias_kind == "relative":
        alias = Path("runs/run")
    else:
        child = log_dir.parent / "child"
        child.mkdir()
        if alias_kind == "symlink-dotdot":
            linked_child = tmp_path / "linked-child"
            linked_child.symlink_to(child, target_is_directory=True)
            child = linked_child
        alias = child / ".." / log_dir.name
    marker = tmp_path / "entered"
    with lock_train_run(log_dir):
        _assert_locked(_attempt(alias, marker, cwd=tmp_path), marker)
        inode = _sidecar(log_dir).stat().st_ino
    assert _attempt(alias, marker, cwd=tmp_path) == {"status": "entered"}
    assert _sidecar(log_dir).stat().st_ino == inode
    assert list(tmp_path.rglob("*.trainer.lock")) == [_sidecar(log_dir)]


@pytest.mark.parametrize("bad", [
    "symlink", "dangling-symlink", "hardlink", "fifo", "directory", "public", "readonly", "executable",
])
def test_unsafe_lock_files_fail_closed_without_mutation_or_secret_output(tmp_path, capsys, bad):
    log_dir = tmp_path / "run"
    lock = _sidecar(log_dir)
    target = tmp_path / "target"
    target.write_bytes(b"synthetic-lock-secret")
    target.chmod(0o600)
    if bad == "symlink":
        lock.symlink_to(target)
    elif bad == "dangling-symlink":
        lock.symlink_to(tmp_path / "absent-target")
    elif bad == "hardlink":
        os.link(target, lock)
    elif bad == "fifo":
        os.mkfifo(lock, 0o600)
    elif bad == "directory":
        lock.mkdir()
    else:
        lock.write_bytes(b"synthetic-lock-secret")
        lock.chmod({"public": 0o644, "readonly": 0o400, "executable": 0o700}[bad])
    before = lock.lstat()
    started = time.monotonic()
    with pytest.raises(TrainRunLockError) as caught:
        with lock_train_run(log_dir):
            pytest.fail("unsafe lock storage must not enter the run")
    assert time.monotonic() - started < 2
    assert lock.lstat() == before, "unsafe files must not be repaired, replaced, or removed"
    assert not log_dir.exists()
    assert target.read_bytes() == b"synthetic-lock-secret"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert "synthetic-lock-secret" not in "".join(traceback.format_exception(caught.value))
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("mode", [0o770, 0o707, 0o777])
def test_writable_shared_parent_is_rejected_without_repair(tmp_path, mode):
    directory = tmp_path / "shared"
    directory.mkdir(mode=0o700)
    directory.chmod(mode)
    log_dir = directory / "run"
    try:
        with pytest.raises(TrainRunLockError, match="not group/world-writable"):
            with lock_train_run(log_dir):
                pytest.fail("other users must not be able to replace the sidecar")
        assert stat.S_IMODE(directory.stat().st_mode) == mode
        assert list(directory.iterdir()) == []
    finally:
        directory.chmod(0o700)


@pytest.mark.parametrize("target_kind", ["lock", "parent"])
def test_foreign_ownership_is_rejected_with_real_chown(tmp_path, target_kind):
    log_dir = tmp_path / "private-parent" / "run"
    with lock_train_run(log_dir):
        pass
    target = _sidecar(log_dir) if target_kind == "lock" else log_dir.parent
    uid = os.geteuid()
    try:
        os.chown(target, uid + 1, -1)
    except PermissionError:
        pytest.skip("Changing actual ownership requires privilege; no UID/stat mocking.")
    try:
        with pytest.raises(TrainRunLockError, match="owned by the current user"):
            with lock_train_run(log_dir):
                pytest.fail("foreign-owned storage must not enter the run")
        assert target.stat().st_uid == uid + 1
    finally:
        os.chown(target, uid, -1)
