"""Exclusive, nonblocking training-run locks for POSIX hosts.

Call after config validation, before inspecting run state or authenticating, and
hold through the entire run, including finalization. The permanent sidecar lives
outside the run directory, so locking does not claim or populate that directory.
This is an advisory lock: cooperating writers must never remove its sidecar or
rename its parent while a run is active.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
import fcntl
import os
from pathlib import Path
import stat

_FILE_FLAGS = os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


class TrainRunLockError(RuntimeError):
    """The training run cannot be locked safely; no run contents are reported."""


class TrainRunLockedError(TrainRunLockError):
    """Another trainer already holds this run's lock."""


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _check_directory(directory: Path, fd: int) -> None:
    info = os.fstat(fd)
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o022:
        raise TrainRunLockError(
            "Training run parent directory must be owned by the current user and "
            "not group/world-writable; choose a private parent for run.log_dir."
        )
    current = directory.lstat()
    if not stat.S_ISDIR(current.st_mode) or _identity(current) != _identity(info):
        raise TrainRunLockError(
            "Training run parent directory was replaced or became a symlink; "
            "retry with a stable run.log_dir."
        )


def _check_lock(directory_fd: int, name: str, fd: int) -> None:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise TrainRunLockError(
            "Training run lock must be a regular file, not a symlink or hard link."
        )
    if info.st_uid != os.geteuid():
        raise TrainRunLockError("Training run lock must be owned by the current user.")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise TrainRunLockError("Training run lock must have private permissions 0600.")
    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(current.st_mode) or _identity(current) != _identity(info):
        raise TrainRunLockError(
            "Training run lock was replaced; refusing to enter an unlocked run. "
            "Do not delete or replace trainer lock files."
        )


@contextmanager
def lock_train_run(log_dir: Path) -> Iterator[None]:
    """Hold ``.<run name>.trainer.lock`` in the canonical run parent.

    Only missing parent directories and the sidecar may be created; ``log_dir``
    itself and its contents are untouched. Existing lock files must be private,
    singly linked, regular files owned by this user, in a user-owned parent that
    is not group/world-writable. Unsafe files are rejected, never repaired.
    Symlink/relative aliases of the run resolve to the same sidecar. Contention
    raises TrainRunLockedError immediately, without entering the caller's body.
    """
    with ExitStack() as cleanup:
        try:
            # Resolve the whole path, not just its parent: an existing or dangling
            # symlink to the run must not introduce a second independent lock.
            run = log_dir.expanduser().resolve()
            if run == run.parent:
                raise TrainRunLockError("run.log_dir must be a dedicated directory, not the filesystem root.")
            directory = run.parent
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | _FILE_FLAGS)
            cleanup.callback(os.close, directory_fd)
            _check_directory(directory, directory_fd)
            name = f".{run.name}.trainer.lock"
            try:
                fd = os.open(
                    name, os.O_RDWR | os.O_CREAT | os.O_EXCL | _FILE_FLAGS,
                    0o600, dir_fd=directory_fd,
                )
            except FileExistsError:
                fd = os.open(name, os.O_RDWR | _FILE_FLAGS, dir_fd=directory_fd)
                cleanup.callback(os.close, fd)
            else:
                cleanup.callback(os.close, fd)
                # Set an exact private mode even under a restrictive umask, but
                # only on the newly created inode, never on an existing file.
                os.fchmod(fd, 0o600)
            _check_lock(directory_fd, name, fd)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise TrainRunLockedError(
                    f"Training run directory {run} is already locked by another trainer. "
                    "Wait for that run to finish or choose a different run.log_dir. "
                    f"Do not delete the lock file {directory / name}."
                ) from None
            # Reject a pathname swap between open and flock; all sidecar I/O is
            # anchored to the directory descriptor rather than a mutable path.
            _check_directory(directory, directory_fd)
            _check_lock(directory_fd, name, fd)
        except OSError:
            raise TrainRunLockError(
                "Cannot safely open or acquire the training run lock. Check that "
                "run.log_dir has an accessible, user-owned parent and its sidecar "
                "is a regular, non-symlink file with permissions 0600. "
                "Do not delete a lock file while a trainer may be running."
            ) from None
        # Do not translate exceptions from the run. Closing releases flock on
        # success, failure, or cancellation; never unlink the stable sidecar.
        yield None
