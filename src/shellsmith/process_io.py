"""Durable subprocess capture and durable JSON writes for Game Master adapters.

Architecture map
----------------
    run_captured_process(command, cwd, stdout_path, stderr_path, input_data, timeout)
        dedicated POSIX session -> capture_process -> concurrent pipe readers
        -> every chunk appended + fsynced to its file -> bounded in-memory tail
        deadline / cancellation -> TERM group, bounded grace, KILL, drain, re-raise
    write_json(path, value)
        strict JSON (no NaN, no default=str) -> temp file -> fsync -> rename
        -> fsync the directory

Invariants
----------
- A capture never substitutes empty logs: interrupted output is kept on disk.
- Capture errors propagate; the caller still interprets the exit code.

Why this exists
---------------
Vendored from the original monorepo's process capture and ``write_json``
helpers. Shellsmith adds owned POSIX subprocess lifetime
management; it does not need the campaign replay journal.

Main subprocess caller:
    ``episode.gm_process.codex`` (other GM execution is disabled).
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from typing import Any

# Upper bound on captured subprocess output kept in memory per stream. The full
# stream is always written to disk; only this tail is returned for error
# reporting (successful callers parse the persisted files).
CAPTURE_TAIL_BYTES = 4 * 1024 * 1024
PROCESS_TERMINATE_GRACE_SECONDS = 3.0
PROCESS_DRAIN_GRACE_SECONDS = 3.0


async def _finish_cleanup(cleanup: asyncio.Task[Any], original: BaseException) -> None:
    """Keep owned cleanup alive through repeated cancellation; preserve its cause."""
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            continue
        except BaseException:
            break
    try:
        cleanup.result()
    except BaseException as error:
        original.add_note(f"Subprocess cleanup also failed: {type(error).__name__}: {error}")


async def run_captured_process(
    command: list[str], *, cwd: Path, stdout_path: Path, stderr_path: Path,
    input_data: bytes | None = None, timeout: float,
) -> tuple[int, bytes, bytes]:
    """Own a fresh POSIX session from spawn through capture and interruption.

    Retain the spawn task if cancellation races process creation, so its eventual
    PID/group cannot be abandoned before cleanup acquires the process handle.
    Descendants inherit this group; this is not containment of deliberate setsid
    or setpgid escapes (the GM's separate sandbox supplies its confinement).
    """
    spawn = asyncio.create_task(asyncio.create_subprocess_exec(
        *command, cwd=str(cwd), stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    ))
    try:
        process = await asyncio.shield(spawn)
    except BaseException as original:
        async def finish_spawn() -> None:
            process = await spawn
            try:
                await capture_process(process, stdout_path=stdout_path, stderr_path=stderr_path,
                                      timeout=0, process_group_id=process.pid)
            except TimeoutError:
                pass  # Immediate deadline deliberately enters group cleanup.
        await _finish_cleanup(asyncio.create_task(finish_spawn()), original)
        raise
    stdout, stderr = await capture_process(
        process, stdout_path=stdout_path, stderr_path=stderr_path,
        input_data=input_data, timeout=timeout, process_group_id=process.pid,
    )
    assert process.returncode is not None
    return process.returncode, stdout, stderr


async def capture_process(process: Any, *, stdout_path: Path, stderr_path: Path,
                          input_data: bytes | None = None, timeout: float,
                          process_group_id: int | None = None) -> tuple[bytes, bytes]:
    """Drain both child pipes concurrently and fsync every received chunk.

    Steps:
        Start one reader per pipe; each streams the full output to its file and
        keeps only a bounded tail in memory.
        Feed stdin (when given), wait for exit, and return both tails.
        On timeout/cancellation/error, TERM the explicitly owned process group,
        give the group 3 seconds, KILL survivors even if its leader/pipes exited,
        and drain/reap within a final bound. Re-raise the initiating exception.

    ``process_group_id`` is an ownership assertion from a start_new_session
    spawn, never a group inferred from an arbitrary child. Without it the legacy
    single-process capture contract remains, but Codex always uses owned groups.
    """
    if process_group_id is not None and (
        process_group_id <= 1 or process_group_id != process.pid or process_group_id == os.getpgrp()
    ):
        raise ValueError("Capture must own a dedicated child process group, never the caller's group.")

    async def drain(reader: Any, path: Path) -> bytes:
        # Stream to disk in full, but retain only a bounded tail in memory: a
        # verbose Codex/Pi turn can emit a large JSONL log, and the success paths
        # parse the persisted file while failure handling needs only a tail, so
        # keeping every byte could exhaust the worker container. Trim at a
        # newline so the retained bytes never split a multi-byte character.
        path.parent.mkdir(parents=True, exist_ok=True)
        tail = bytearray()
        with path.open("xb") as handle:
            while chunk := await reader.read(65536):
                handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
                tail.extend(chunk)
                if len(tail) > CAPTURE_TAIL_BYTES:
                    newline = tail.find(b"\n", len(tail) - CAPTURE_TAIL_BYTES)
                    if newline != -1:
                        del tail[: newline + 1]
                    else:
                        # A single record larger than the limit has no newline in
                        # the window; still cap it. Drop leading UTF-8
                        # continuation bytes so the prefix starts on a char.
                        del tail[: len(tail) - CAPTURE_TAIL_BYTES]
                        while tail and (tail[0] & 0xC0) == 0x80:
                            del tail[0]
        return bytes(tail)

    readers = [asyncio.create_task(drain(process.stdout, stdout_path)),
               asyncio.create_task(drain(process.stderr, stderr_path))]
    reaper = asyncio.create_task(process.wait())

    async def feed() -> None:
        if input_data is not None:
            process.stdin.write(input_data)
            await process.stdin.drain()
            process.stdin.close()

    feeder = asyncio.create_task(feed())

    async def wait() -> tuple[bytes, bytes]:
        # Observe all I/O concurrently, even when stdin is backpressured. A local
        # log failure must not be relabeled as a later provider timeout.
        _, _, stdout, stderr = await asyncio.gather(feeder, reaper, *readers)
        return stdout, stderr

    waiter = asyncio.create_task(wait())

    def send_signal(sig: signal.Signals) -> None:
        try:
            if process_group_id is not None:
                os.killpg(process_group_id, sig)
            elif process.returncode is None:
                process.send_signal(sig)
        except ProcessLookupError:
            pass  # The owned target exited between observation and signaling.

    def target_exists() -> bool:
        if process_group_id is None:
            return process.returncode is None
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return False
        return True

    async def terminate_and_drain() -> None:
        if process.stdin is not None:
            process.stdin.close()
        try:
            send_signal(signal.SIGTERM)
            deadline = asyncio.get_running_loop().time() + PROCESS_TERMINATE_GRACE_SECONDS
            # A leader returncode or pipe EOF says nothing about descendants.
            # Only group disappearance ends this grace period early.
            while target_exists() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(.02)
        finally:
            try:
                if target_exists():
                    send_signal(signal.SIGKILL)
            finally:
                # The waiter may have failed while feeding stdin or writing a
                # log. Reaping must not depend on that combined task succeeding.
                tasks = [feeder, reaper, waiter, *readers]
                _, pending = await asyncio.wait(tasks, timeout=PROCESS_DRAIN_GRACE_SECONDS)
                if pending:
                    # asyncio.Process exposes no public close method. At the
                    # bounded cutoff, close its transport/pipes as well as tasks;
                    # cancelling StreamReader.read alone leaves pipe FDs open.
                    process._transport.close()
                for task in pending:
                    task.cancel()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                errors = [value for value in results if isinstance(value, Exception)]
                if errors:
                    raise ExceptionGroup("subprocess capture cleanup failed", errors)
                if pending:
                    raise TimeoutError("Subprocess output did not drain within the cleanup deadline.")

    try:
        return await asyncio.wait_for(asyncio.shield(waiter), timeout)
    except BaseException as original:
        await _finish_cleanup(asyncio.create_task(terminate_and_drain()), original)
        raise


def fsync_directory(path: Path) -> None:
    """Persist directory entries where supported; never hide real I/O errors.

    Windows' CRT cannot open a directory with os.open. Some filesystems also
    explicitly reject directory fsync; file contents are still fsynced there.
    """
    if sys.platform == "win32":
        return
    unsupported = {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}
    try:
        directory = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as error:
        if error.errno in unsupported:
            return
        raise
    try:
        try:
            os.fsync(directory)
        except OSError as error:
            if error.errno not in unsupported:
                raise
    finally:
        os.close(directory)


def _mkdir_durable(path: Path) -> None:
    missing = []
    current = path
    while not current.is_dir():
        missing.append(current)
        parent = current.parent
        if parent == current:
            raise NotADirectoryError(path)
        current = parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        fsync_directory(directory.parent)


def write_json(path: Path, value: Any) -> Path:
    """Publish strict JSON after file fsync, then sync the containing directory.

    Each writer owns a unique private temporary file. Unsupported directory
    syncing is explicitly best-effort; other I/O failures propagate.
    """
    data = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    _mkdir_durable(path.parent)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
