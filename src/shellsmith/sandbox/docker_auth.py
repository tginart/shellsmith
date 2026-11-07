"""File-backed Codex OAuth lifecycle for the Docker launcher (POSIX hosts).

Integration::

    with stage_codex_auth() as auth:
        # docker cp auth.codex_home -> container:/tmp/shellsmith-codex
        # Run; in finally, STOP the container, including on failure/cancellation.
        # docker cp container:/tmp/shellsmith-codex/auth.json -> auth.returned_auth_path
        restore_codex_auth(auth)  # before docker rm; never export auth as an artifact

The context also reconciles an existing return file on exceptional exit. It
holds the source lock through reconciliation and removes private staging on
exit. Recovery errors are attached to an existing exception, not substituted
for it. No Docker processes, login commands, or keyring access occur here.

The stable sidecar flock serializes cooperating Shellsmith launches, including
across atomic auth.json replacements. It is advisory: stop other Codex clients
using this source during an episode. Full bytes AND file identity/metadata are
compared immediately before replacement, but POSIX has no portable atomic
compare-and-rename against writers that ignore this lock.

Schema checked against Codex rust-v0.159.2, commit
ff6aec96948b70d94983af2641a6b67c94faeff5, codex-rs/login/src/auth/{storage,manager}.rs
and codex-rs/config/src/types.rs. The external ChatGPT-token mode is ephemeral
and requires externally managed refresh; it is not a file-lifecycle substitute.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
from typing import Any
import uuid

CODEX_AUTH_PATH_ENV = "SHELLSMITH_CODEX_AUTH_PATH"
DEFAULT_LOCK_TIMEOUT_SECONDS = 30.0
_MAX_AUTH_BYTES = 1024 * 1024
_TOKEN_FIELDS = ("id_token", "access_token", "refresh_token", "account_id")
_ROTATING_FIELDS = ("id_token", "access_token", "refresh_token")
_FILE_FLAGS = os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


class CodexAuthError(RuntimeError):
    """A credential-safe diagnostic: messages never contain auth contents."""


class CodexAuthLockTimeout(CodexAuthError, TimeoutError):
    """Another cooperating process still owns this source's lock."""


class CodexAuthConflict(CodexAuthError):
    """Reconciliation needs attention; returned OAuth is in recovery_path."""

    def __init__(self, reason: str, recovery_path: Path, *, host_replaced: bool = False):
        self.recovery_path = recovery_path
        self.host_replaced = host_replaced
        status = "" if host_replaced else " Host auth was not replaced."
        super().__init__(
            f"{reason}{status} Returned OAuth credentials were "
            f"saved privately at {recovery_path}. Reconcile this credential file "
            "manually before another run; do not archive or share it."
        )


@dataclass(repr=False)
class CodexAuthSession:
    """Paths for Docker transport; secret state is deliberately absent from repr.

    returned_auth_path initially does not exist. Copy only the stopped
    container's auth.json there, without docker cp -a or -L. Call restore once
    after a successful copy, while inside stage_codex_auth. No path here belongs
    in retained run output. The lock file is permanent and must not be unlinked.
    """

    source_path: Path
    codex_home: Path
    returned_auth_path: Path
    lock_path: Path
    _directory_fd: int
    _lock_fd: int
    _snapshot: tuple[bytes, os.stat_result]
    _original: dict[str, Any]
    _outbound: dict[str, Any]
    _active: bool = True
    _attempted: bool = False
    _restored: bool = False
    _error: CodexAuthError | None = None
    _keep_temporary: bool = False


def resolve_codex_auth_path(*, environ: Mapping[str, str] | None = None) -> Path:
    """Explicit auth path > CODEX_HOME/auth.json > ~/.codex/auth.json.

    environ is injectable for path-only tests; this function never changes
    HOME, CODEX_HOME, configuration, or any process environment.
    """
    env = os.environ if environ is None else environ
    override = env.get(CODEX_AUTH_PATH_ENV, "")
    if override.strip():
        return Path(override).expanduser().absolute()
    home = env.get("CODEX_HOME", "")
    base = Path(home).expanduser() if home.strip() else Path.home() / ".codex"
    return (base / "auth.json").absolute()


def _missing_auth(path: Path) -> CodexAuthError:
    return CodexAuthError(
        f"Codex auth file is missing: {path}. Codex supports keyring and auto "
        "credential stores, so an existing login may be in the OS keyring. "
        "This Docker integration supports file-backed ChatGPT OAuth only; it "
        "does not read the keyring. To use file storage, run "
        "`codex -c 'cli_auth_credentials_store=\"file\"' login` in your normal "
        "persistent Codex home (respecting CODEX_HOME), not a temporary or run "
        "directory. Use file mode for subsequent Codex invocations in that home "
        "too. This is a new login, not a keyring export. Point CODEX_HOME or "
        "SHELLSMITH_CODEX_AUTH_PATH at the resulting persistent credentials."
    )


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _version(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
        info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
    )


def _check_file(info: os.stat_result, label: str) -> None:
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise CodexAuthError(f"{label} must be a regular file, not a symlink or hard link.")
    if info.st_uid != os.geteuid():
        raise CodexAuthError(f"{label} must be owned by the current user.")
    if stat.S_IMODE(info.st_mode) not in (0o600, 0o400):
        raise CodexAuthError(f"{label} must be private (0600 or owner-read-only 0400).")


def _check_directory(directory: Path, fd: int) -> None:
    info = os.fstat(fd)
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o022:
        raise CodexAuthError("Codex auth directory must be user-owned and not group/world-writable.")
    try:
        current = directory.lstat()
    except OSError:
        raise CodexAuthError("Codex auth directory disappeared or became inaccessible.") from None
    if not stat.S_ISDIR(current.st_mode) or _identity(current) != _identity(info):
        raise CodexAuthError("Codex auth directory was replaced or is a symlink.")


@contextmanager
def _source_directory(source: Path) -> Iterator[tuple[Path, int]]:
    # Canonicalize ancestors (e.g. macOS /var), but reject a symlink for the
    # credentials directory itself. All source I/O is anchored to its open fd.
    try:
        if stat.S_ISLNK(source.parent.lstat().st_mode):
            raise CodexAuthError("Codex auth directory must not be a symlink.")
        directory = source.parent.resolve(strict=True)
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | _FILE_FLAGS)
    except FileNotFoundError:
        raise _missing_auth(source) from None
    except OSError:
        raise CodexAuthError("Cannot safely open the Codex auth directory.") from None
    try:
        _check_directory(directory, fd)
        yield directory / source.name, fd
    finally:
        os.close(fd)


def _read_private(directory_fd: int, name: str, label: str) -> tuple[bytes, os.stat_result]:
    fd = os.open(name, os.O_RDONLY | _FILE_FLAGS, dir_fd=directory_fd)
    try:
        before = os.fstat(fd)
        _check_file(before, label)
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(_MAX_AUTH_BYTES + 1)
        after = os.fstat(fd)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if _version(before) != _version(after) or _version(after) != _version(current):
            raise CodexAuthError(f"{label} changed while being read.")
        if len(raw) > _MAX_AUTH_BYTES:
            raise CodexAuthError(f"{label} exceeds the credential file size limit.")
        return raw, after
    finally:
        os.close(fd)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(_: str) -> None:
    raise ValueError("non-finite JSON number")


def _parse_auth(raw: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        document = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (ValueError, UnicodeError, RecursionError):
        # Unicode/JSON parser exceptions can retain the offending secret bytes.
        raise CodexAuthError("Codex auth is not a valid, unambiguous JSON document.") from None
    if not isinstance(document, dict) or document.get("auth_mode") != "chatgpt":
        raise CodexAuthError("The Docker Codex GM requires file-backed ChatGPT OAuth, not API-key or external auth.")
    tokens = document.get("tokens")
    if not isinstance(tokens, dict):
        raise CodexAuthError("Codex ChatGPT OAuth tokens must be an object.")
    for key in ("access_token", "refresh_token"):
        if not isinstance(tokens.get(key), str) or not tokens[key].strip():
            raise CodexAuthError("Codex ChatGPT access and refresh tokens must be nonempty strings.")
    # Keep the launcher's access/refresh minimum; validate optional ID/account
    # fields without decoding JWTs or copying arbitrary additional secrets.
    if "id_token" in tokens and (not isinstance(tokens["id_token"], str) or not tokens["id_token"].strip()):
        raise CodexAuthError("Codex ChatGPT ID token must be a nonempty string when present.")
    if tokens.get("account_id") is not None and (
        not isinstance(tokens["account_id"], str) or not tokens["account_id"].strip()
    ):
        raise CodexAuthError("Codex ChatGPT account ID must be a nonempty string or null.")
    minimal: dict[str, Any] = {
        "auth_mode": "chatgpt",
        "tokens": {key: tokens[key] for key in _TOKEN_FIELDS if key in tokens},
    }
    if document.get("last_refresh") is not None:
        if not isinstance(document["last_refresh"], str) or not document["last_refresh"].strip():
            raise CodexAuthError("Codex last_refresh must be a nonempty string when present.")
        minimal["last_refresh"] = document["last_refresh"]
    return document, minimal


def _encode(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _write_private(directory_fd: int, name: str, raw: bytes) -> None:
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FILE_FLAGS, 0o600, dir_fd=directory_fd)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(fd)
    except BaseException:
        os.unlink(name, dir_fd=directory_fd)
        raise
    finally:
        os.close(fd)


def _check_lock(directory_fd: int, name: str, fd: int) -> None:
    info = os.fstat(fd)
    _check_file(info, "Codex auth lock")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise CodexAuthError("Codex auth lock must have permissions 0600.")
    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if _identity(info) != _identity(current):
        raise CodexAuthError("Codex auth lock was replaced; refusing an unlocked snapshot.")


@contextmanager
def _source_lock(directory_fd: int, name: str, timeout: float) -> Iterator[int]:
    try:
        try:
            fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | _FILE_FLAGS, 0o600, dir_fd=directory_fd)
        except FileExistsError:
            fd = os.open(name, os.O_RDWR | _FILE_FLAGS, dir_fd=directory_fd)
    except OSError:
        raise CodexAuthError("Cannot safely open the Codex auth lock file.") from None
    try:
        deadline = time.monotonic() + timeout
        try:
            _check_lock(directory_fd, name, fd)
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except (BlockingIOError, InterruptedError):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise CodexAuthLockTimeout(
                            f"Timed out after {timeout:g}s waiting for the Codex auth lock. "
                            "Another episode may be refreshing this source; do not delete the lock file."
                        ) from None
                    time.sleep(min(0.05, remaining))
            _check_lock(directory_fd, name, fd)
        except CodexAuthError:
            raise
        except OSError:
            raise CodexAuthError("Cannot safely lock the Codex auth lock file.") from None
        yield fd  # Do not translate OSError raised by the caller's episode.
    finally:
        # Closing releases flock even on cancellation. Never unlink a lock:
        # a waiter could already hold the old inode while another opens new.
        os.close(fd)


@contextmanager
def stage_codex_auth(
    auth_path: Path | str | None = None,
    *,
    lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> Iterator[CodexAuthSession]:
    """Lock, then snapshot and stage OAuth outside retained run directories.

    Missing file storage does not imply missing login: diagnostics explain
    keyring/auto and the supported explicit file-mode login migration. Existing
    auth/lock permissions are checked, never silently repaired. Staging is 0700
    and auth is 0600; no process environment is modified.
    """
    if not math.isfinite(lock_timeout_seconds) or lock_timeout_seconds < 0:
        raise ValueError("Codex auth lock timeout must be finite and nonnegative.")
    source = resolve_codex_auth_path() if auth_path is None else Path(auth_path).expanduser().absolute()
    with _source_directory(source) as (source, directory_fd):
        digest = hashlib.sha256(os.fsencode(source.name)).hexdigest()[:24]
        lock_name = f".shellsmith-codex-{digest}.lock"
        with _source_lock(directory_fd, lock_name, lock_timeout_seconds) as lock_fd:
            _check_directory(source.parent, directory_fd)
            try:
                snapshot = _read_private(directory_fd, source.name, "Codex auth file")
            except FileNotFoundError:
                raise _missing_auth(source) from None
            except OSError:
                raise CodexAuthError("Cannot safely read the Codex auth file (symlinks are not supported).") from None
            original, outbound = _parse_auth(snapshot[0])
            temporary = Path(tempfile.mkdtemp(prefix="shellsmith-codex-auth-"))
            session: CodexAuthSession | None = None
            try:
                codex_home = temporary / "home"
                codex_home.mkdir(mode=0o700)
                home_fd = os.open(codex_home, os.O_RDONLY | os.O_DIRECTORY | _FILE_FLAGS)
                try:
                    _write_private(home_fd, "auth.json", _encode(outbound))
                finally:
                    os.close(home_fd)
                session = CodexAuthSession(
                    source, codex_home, temporary / "returned-auth.json", source.parent / lock_name,
                    directory_fd, lock_fd, snapshot, original, outbound,
                )
                failure: BaseException | None = None
                try:
                    yield session
                except BaseException as error:
                    failure = error
                    raise
                finally:
                    try:
                        if not session._attempted and os.path.lexists(session.returned_auth_path):
                            restore_codex_auth(session)
                    except CodexAuthError as error:
                        if failure is None:
                            raise
                        failure.add_note(str(error))
                    finally:
                        session._active = False
            finally:
                if session is None or not session._keep_temporary:
                    shutil.rmtree(temporary)


def _recover(
    session: CodexAuthSession, returned: dict[str, Any], reason: str, *, host_replaced: bool = False,
) -> None:
    name = f".shellsmith-codex-recovery-{uuid.uuid4().hex}.json"
    try:
        _check_directory(session.source_path.parent, session._directory_fd)
        _write_private(session._directory_fd, name, _encode(returned))
        os.fsync(session._directory_fd)
    except (OSError, CodexAuthError):
        # A read-only/replaced/full source directory cannot hold a durable
        # recovery file. Do not destroy the only remaining rotated credentials.
        session._keep_temporary = True
        raise CodexAuthError(
            f"{reason} Could not durably save credential recovery in the original "
            "auth directory. Returned credentials are retained in private staging "
            f"at {session.returned_auth_path}; recover them before removing it. "
            "Do not archive or share this credential file."
        ) from None
    raise CodexAuthConflict(reason, session.source_path.parent / name, host_replaced=host_replaced)


def _restore(session: CodexAuthSession) -> bool:
    returned_fd = os.open(session.returned_auth_path.parent, os.O_RDONLY | os.O_DIRECTORY | _FILE_FLAGS)
    try:
        raw, _ = _read_private(returned_fd, session.returned_auth_path.name, "Returned Codex auth")
    finally:
        os.close(returned_fd)
    _, returned = _parse_auth(raw)
    before_tokens = session._outbound["tokens"]
    after_tokens = returned["tokens"]
    if all(before_tokens.get(key) == after_tokens.get(key) for key in _ROTATING_FIELDS):
        return False  # formatting, unrelated fields and timestamp-only changes aren't rotation
    if before_tokens.get("account_id") != after_tokens.get("account_id"):
        _recover(session, returned, "Returned Codex credentials changed account identity.")
    # An optional ID token missing from the returned document must not silently
    # leave the old account's ID alongside new access/refresh tokens.
    if "id_token" in before_tokens and "id_token" not in after_tokens:
        _recover(session, returned, "Returned Codex credentials lost the ID token.")
    merged = dict(session._original)
    merged["tokens"] = {**session._original["tokens"], **after_tokens}
    if "last_refresh" in returned:
        merged["last_refresh"] = returned["last_refresh"]
    name = f".shellsmith-codex-write-{uuid.uuid4().hex}.tmp"
    written = False
    replaced = False
    try:
        _check_directory(session.source_path.parent, session._directory_fd)
        _check_lock(session._directory_fd, session.lock_path.name, session._lock_fd)
        _write_private(session._directory_fd, name, _encode(merged))
        written = True
        # Do the expensive write/fsync first, then compare immediately before
        # rename. A replaced file is a conflict even if its bytes are identical.
        current = _read_private(session._directory_fd, session.source_path.name, "Host Codex auth")
        if current[0] != session._snapshot[0] or _version(current[1]) != _version(session._snapshot[1]):
            raise CodexAuthError("Host Codex auth changed since the locked snapshot.")
        os.replace(name, session.source_path.name, src_dir_fd=session._directory_fd, dst_dir_fd=session._directory_fd)
        written = False
        replaced = True
        os.fsync(session._directory_fd)
    except (OSError, CodexAuthError):
        reason = (
            "Codex auth was replaced, but directory durability could not be confirmed."
            if replaced else "Host Codex auth changed or could not be safely replaced."
        )
        _recover(session, returned, reason, host_replaced=replaced)
    finally:
        if written:
            os.unlink(name, dir_fd=session._directory_fd)
    return True


def restore_codex_auth(session: CodexAuthSession) -> bool:
    """Reconcile a stopped container's returned auth while still holding the lock.

    True means tokens were atomically restored (file and directory fsynced);
    False means tokens did not rotate and host auth was not touched. Repeated
    calls return the same result/error, without duplicate writes or recovery
    files. A missing/invalid/unsafe return file is an error, never a no-op.
    Conflicts preserve returned OAuth in a unique private file in the original
    auth directory and raise CodexAuthConflict with its recovery_path. Never
    print auth contents or put these paths into an artifact manifest.
    """
    if not session._active:
        raise CodexAuthError("Codex auth restoration requires the active staging context and source lock.")
    if session._attempted:
        if session._error is not None:
            raise session._error
        return session._restored
    session._attempted = True
    try:
        session._restored = _restore(session)
    except (OSError, CodexAuthError) as error:
        if isinstance(error, CodexAuthError):
            session._error = error
        else:
            session._error = CodexAuthError(
                "Cannot safely read returned Codex auth. Copy the stopped container's "
                "auth.json to returned_auth_path before restoring; no symlinks, "
                "hard links or non-private files are accepted."
            )
        raise session._error from None
    return session._restored
