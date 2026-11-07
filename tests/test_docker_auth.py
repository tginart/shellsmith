"""Real filesystem and spawned-process OAuth lifecycle tests; no Codex or mocks."""

from __future__ import annotations

from contextlib import contextmanager
import json
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import shutil
import stat
import sys
import time
import traceback

import pytest

from shellsmith.sandbox.docker_auth import (
    CODEX_AUTH_PATH_ENV,
    CodexAuthConflict,
    CodexAuthError,
    CodexAuthLockTimeout,
    resolve_codex_auth_path,
    restore_codex_auth,
    stage_codex_auth,
)


def _document(generation: str = "original") -> dict:
    # Synthetic credentials only. No tests read an actual Codex home.
    return {
        "auth_mode": "chatgpt",
        "OPENAI_API_KEY": "synthetic-host-api-key",
        "tokens": {
            "id_token": f"synthetic-id-{generation}",
            "access_token": f"synthetic-access-{generation}",
            "refresh_token": f"synthetic-refresh-{generation}",
            "account_id": "synthetic-account",
            "host_extension": {"keep": "synthetic-host-only"},
        },
        "last_refresh": "2026-10-05T00:00:00Z",
        "host_extension": {"nested": [1, True, None]},
    }


def _write(path: Path, document: dict | bytes) -> bytes:
    raw = document if isinstance(document, bytes) else json.dumps(document, indent=2).encode()
    path.write_bytes(raw)
    path.chmod(0o600)
    return raw


def _replace(path: Path, document: dict | bytes) -> bytes:
    temporary = path.with_name("external-replacement.json")
    raw = _write(temporary, document)
    os.replace(temporary, path)
    return raw


@pytest.fixture
def auth_path(tmp_path: Path) -> Path:
    directory = tmp_path / "persistent-codex"
    directory.mkdir(mode=0o700)
    path = directory / "auth.json"
    _write(path, _document())
    return path


def _returned(session, generation: str = "rotated") -> dict:
    document = _document(generation)
    # Unrelated values in an untrusted returned file must never replace host values.
    document["OPENAI_API_KEY"] = "synthetic-container-injection"
    document["tokens"]["host_extension"] = "synthetic-container-injection"
    document["host_extension"] = {"container": True}
    document["last_refresh"] = "2026-10-05T01:00:00Z"
    _write(session.returned_auth_path, document)
    return document


def _recoveries(auth_path: Path) -> list[Path]:
    return sorted(auth_path.parent.glob(".shellsmith-codex-recovery-*.json"))


def _assert_recovery(path: Path, generation: str = "rotated") -> None:
    assert path.is_file() and not path.is_symlink()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.stat().st_uid == os.geteuid()
    document = json.loads(path.read_bytes())
    assert document["auth_mode"] == "chatgpt"
    assert document["tokens"]["refresh_token"] == f"synthetic-refresh-{generation}"
    assert set(document) == {"auth_mode", "tokens", "last_refresh"}
    assert set(document["tokens"]) == {"id_token", "access_token", "refresh_token", "account_id"}
    assert b"synthetic-host-api-key" not in path.read_bytes()
    assert b"synthetic-container-injection" not in path.read_bytes()


def test_resolution_respects_explicit_path_and_codex_home_without_changing_environment(tmp_path):
    environment = dict(os.environ)
    assert resolve_codex_auth_path(environ={}) == Path.home() / ".codex/auth.json"
    assert resolve_codex_auth_path(environ={"CODEX_HOME": str(tmp_path)}) == tmp_path / "auth.json"
    assert resolve_codex_auth_path(environ={
        "CODEX_HOME": str(tmp_path / "not-selected"),
        CODEX_AUTH_PATH_ENV: str(tmp_path / "selected.json"),
    }) == tmp_path / "selected.json"
    assert resolve_codex_auth_path(environ={"CODEX_HOME": "", CODEX_AUTH_PATH_ENV: " "}) == Path.home() / ".codex/auth.json"
    assert resolve_codex_auth_path(environ={"CODEX_HOME": "~/persistent-codex"}) == Path.home() / "persistent-codex/auth.json"
    assert dict(os.environ) == environment


@pytest.mark.parametrize("missing_parent", [False, True])
def test_missing_auth_explains_existing_keyring_login_and_supported_migration(tmp_path, missing_parent):
    path = tmp_path / "absent/auth.json" if missing_parent else tmp_path / "auth.json"
    with pytest.raises(CodexAuthError) as caught:
        with stage_codex_auth(path):
            pytest.fail("missing file must not stage")
    message = str(caught.value)
    for text in ("keyring", "auto", "existing login", "persistent", "CODEX_HOME", "file-backed"):
        assert text in message
    assert "codex -c 'cli_auth_credentials_store=\"file\"' login" in message
    assert "run `codex login`" not in message
    assert not path.exists()


def test_staging_is_private_sanitized_disposable_and_does_not_modify_source(auth_path, tmp_path, capsys):
    before = auth_path.read_bytes(), auth_path.stat()
    with stage_codex_auth(auth_path) as session:
        copied = session.codex_home / "auth.json"
        assert not session.codex_home.is_relative_to(tmp_path)
        assert session.returned_auth_path.parent == session.codex_home.parent
        assert not session.returned_auth_path.exists()
        assert stat.S_IMODE(session.codex_home.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(session.codex_home.stat().st_mode) == 0o700
        assert stat.S_IMODE(copied.stat().st_mode) == 0o600
        assert stat.S_IMODE(session.lock_path.stat().st_mode) == 0o600
        assert session.lock_path.stat().st_uid == os.geteuid()
        staged = json.loads(copied.read_bytes())
        assert set(staged) == {"auth_mode", "tokens", "last_refresh"}
        assert set(staged["tokens"]) == {"id_token", "access_token", "refresh_token", "account_id"}
        assert "synthetic" not in repr(session)
    assert not session.codex_home.parent.exists()
    assert session.lock_path.exists(), "stable sidecar must never be unlinked"
    assert auth_path.read_bytes() == before[0]
    assert auth_path.stat() == before[1]
    assert capsys.readouterr() == ("", "")


def test_rotation_is_atomic_fsynced_and_preserves_unrelated_host_fields(auth_path):
    original = json.loads(auth_path.read_bytes())
    with stage_codex_auth(auth_path) as session, auth_path.open("rb") as old_descriptor:
        old_inode = auth_path.stat().st_ino
        _returned(session)
        calls = []
        old_profile = sys.getprofile()

        def profile(frame, event, function):
            # Observe real C calls, without replacing fsync/replace or the unit.
            if event == "c_call" and function in (os.fsync, os.replace):
                calls.append(function)

        try:
            sys.setprofile(profile)
            assert restore_codex_auth(session) is True
        finally:
            sys.setprofile(old_profile)
        assert calls == [os.fsync, os.replace, os.fsync]
        assert json.loads(old_descriptor.read()) == original, "replace, not in-place truncation"
        assert auth_path.stat().st_ino != old_inode
        assert stat.S_IMODE(auth_path.stat().st_mode) == 0o600
        after = json.loads(auth_path.read_bytes())
        assert after["OPENAI_API_KEY"] == original["OPENAI_API_KEY"]
        assert after["host_extension"] == original["host_extension"]
        assert after["tokens"]["host_extension"] == original["tokens"]["host_extension"]
        assert after["tokens"]["refresh_token"] == "synthetic-refresh-rotated"
        assert after["last_refresh"] == "2026-10-05T01:00:00Z"
        inode = auth_path.stat().st_ino
        assert restore_codex_auth(session) is True
        assert auth_path.stat().st_ino == inode
    assert not _recoveries(auth_path)
    assert not list(auth_path.parent.glob("*.tmp"))
    with pytest.raises(CodexAuthError, match="active staging context"):
        restore_codex_auth(session)


@pytest.mark.parametrize("token", ["access_token", "refresh_token", "id_token"])
def test_change_to_any_real_token_is_rotation(auth_path, token):
    with stage_codex_auth(auth_path) as session:
        returned = json.loads((session.codex_home / "auth.json").read_bytes())
        returned["tokens"][token] = "synthetic-single-rotation"
        _write(session.returned_auth_path, returned)
        assert restore_codex_auth(session) is True
    assert json.loads(auth_path.read_bytes())["tokens"][token] == "synthetic-single-rotation"


@pytest.mark.parametrize("edit_host", [False, True])
def test_unchanged_tokens_never_rewrite_host_even_when_timestamp_changes(auth_path, edit_host):
    with stage_codex_auth(auth_path) as session:
        _returned(session, "original")
        if edit_host:
            _replace(auth_path, _document("external"))
        before = auth_path.read_bytes(), auth_path.stat()
        assert restore_codex_auth(session) is False
    assert (auth_path.read_bytes(), auth_path.stat()) == before
    assert not _recoveries(auth_path)


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt, FileNotFoundError])
def test_exceptional_exit_restores_before_cleanup_and_preserves_original_failure(auth_path, failure_type):
    failure = failure_type("episode failed")
    with pytest.raises(failure_type) as caught:
        with stage_codex_auth(auth_path) as session:
            _returned(session)
            raise failure
    assert caught.value is failure
    assert json.loads(auth_path.read_bytes())["tokens"]["refresh_token"] == "synthetic-refresh-rotated"
    assert not session.codex_home.parent.exists()
    with stage_codex_auth(auth_path, lock_timeout_seconds=0):
        pass  # failure released the real lock


def test_conflict_during_exception_is_reported_as_note_without_losing_tokens(auth_path):
    failure = KeyboardInterrupt("episode interrupted")
    with pytest.raises(KeyboardInterrupt) as caught:
        with stage_codex_auth(auth_path) as session:
            _returned(session)
            external = _replace(auth_path, _document("external"))
            raise failure
    assert caught.value is failure
    recovery, = _recoveries(auth_path)
    _assert_recovery(recovery)
    assert auth_path.read_bytes() == external
    note = "\n".join(failure.__notes__)
    assert str(recovery) in note
    assert "Host auth was not replaced" in note
    assert "synthetic" not in note
    assert not session.codex_home.parent.exists()


@pytest.mark.parametrize("change", ["content", "same-bytes-replacement", "metadata", "unrelated-field", "invalid-json"])
def test_cas_conflicts_preserve_every_host_replacement_and_create_private_recovery(auth_path, change, tmp_path, capsys):
    run_output = tmp_path / "run/artifacts"
    run_output.mkdir(parents=True)
    with stage_codex_auth(auth_path) as session:
        returned = _returned(session)
        if change == "content":
            _replace(auth_path, _document("external"))
        elif change == "same-bytes-replacement":
            _replace(auth_path, auth_path.read_bytes())
        elif change == "metadata":
            auth_path.chmod(0o400)
        elif change == "unrelated-field":
            document = _document()
            document["host_extension"]["new"] = True
            _write(auth_path, document)
        else:
            _write(auth_path, b"external incomplete JSON")
        before = auth_path.read_bytes(), auth_path.stat()
        with pytest.raises(CodexAuthConflict) as caught:
            restore_codex_auth(session)
        assert caught.value.host_replaced is False
        recovery = caught.value.recovery_path
        assert recovery.parent == auth_path.parent
        _assert_recovery(recovery)
        assert json.loads(recovery.read_bytes())["tokens"]["refresh_token"] == returned["tokens"]["refresh_token"]
        with pytest.raises(CodexAuthConflict) as repeated:
            restore_codex_auth(session)
        assert repeated.value is caught.value
        assert len(_recoveries(auth_path)) == 1
        assert "synthetic" not in str(caught.value) + repr(caught.value)
    assert (auth_path.read_bytes(), auth_path.stat()) == before
    assert not list(run_output.rglob("*"))
    assert not session.codex_home.parent.exists()
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("replacement", ["missing", "symlink", "fifo", "directory", "public"])
def test_unsafe_host_changes_are_not_followed_or_overwritten(auth_path, tmp_path, replacement):
    other = tmp_path / "other-secret.json"
    external = _write(other, _document("external"))
    with stage_codex_auth(auth_path) as session:
        _returned(session)
        if replacement == "public":
            auth_path.chmod(0o644)
        else:
            auth_path.unlink()
            if replacement == "symlink":
                auth_path.symlink_to(other)
            elif replacement == "fifo":
                os.mkfifo(auth_path, 0o600)
            elif replacement == "directory":
                auth_path.mkdir()
        before = auth_path.lstat() if replacement != "missing" else None
        with pytest.raises(CodexAuthConflict) as caught:
            restore_codex_auth(session)
        _assert_recovery(caught.value.recovery_path)
    assert other.read_bytes() == external
    if before is None:
        assert not auth_path.exists()
    else:
        after = auth_path.lstat()
        assert (after.st_ino, after.st_mode) == (before.st_ino, before.st_mode)


@pytest.mark.parametrize("bad", [
    b"not-json-synthetic-secret", b"\xffsynthetic-secret", b"[]",
    b'{"auth_mode":"chatgpt","auth_mode":"apikey"}',
    b'{"auth_mode":"chatgpt","tokens":NaN}',
    {"auth_mode": "apikey"}, {"auth_mode": "chatgptAuthTokens"},
    {"auth_mode": "chatgpt", "tokens": []},
    {"auth_mode": "chatgpt", "tokens": {"access_token": "synthetic-secret"}},
    {"auth_mode": "chatgpt", "tokens": {"access_token": 123, "refresh_token": "synthetic-secret"}},
    {"auth_mode": "chatgpt", "tokens": {"access_token": " ", "refresh_token": "synthetic-secret"}},
])
def test_invalid_source_auth_is_rejected_without_echoing_contents(auth_path, bad):
    original = _write(auth_path, bad)
    with pytest.raises(CodexAuthError) as caught:
        with stage_codex_auth(auth_path):
            pytest.fail("invalid auth must not stage")
    assert auth_path.read_bytes() == original
    rendered = "".join(traceback.format_exception(caught.value))
    assert "synthetic-secret" not in rendered
    _write(auth_path, _document())
    with stage_codex_auth(auth_path, lock_timeout_seconds=0):
        pass  # validation failures also close the lock


@pytest.mark.parametrize("bad", ["symlink", "hardlink", "fifo", "directory", "public", "executable", "oversized"])
def test_unsafe_source_files_fail_closed_without_repairing_them(auth_path, tmp_path, bad):
    if bad == "symlink":
        target = tmp_path / "symlink-target.json"
        auth_path.rename(target)
        auth_path.symlink_to(target)
    elif bad == "hardlink":
        os.link(auth_path, tmp_path / "hardlink.json")
    elif bad in {"fifo", "directory"}:
        auth_path.unlink()
        if bad == "fifo":
            os.mkfifo(auth_path, 0o600)
        else:
            auth_path.mkdir()
    elif bad == "public":
        auth_path.chmod(0o644)
    elif bad == "executable":
        auth_path.chmod(0o700)
    else:
        _write(auth_path, b" " * (1024 * 1024 + 1))
    before = auth_path.lstat()
    started = time.monotonic()
    with pytest.raises(CodexAuthError):
        with stage_codex_auth(auth_path):
            pytest.fail("unsafe auth must not stage")
    assert time.monotonic() - started < 2
    after = auth_path.lstat()
    assert (before.st_mode, before.st_ino) == (after.st_mode, after.st_ino)


@pytest.mark.parametrize("bad", ["symlink", "hardlink", "fifo", "public", "readonly"])
def test_existing_lock_validation_uses_real_files_and_never_unlinks_lock(auth_path, tmp_path, bad):
    with stage_codex_auth(auth_path) as session:
        lock = session.lock_path
    if bad == "symlink":
        lock.unlink()
        lock.symlink_to(auth_path)
    elif bad == "hardlink":
        os.link(lock, tmp_path / "lock-hardlink")
    elif bad == "fifo":
        lock.unlink()
        os.mkfifo(lock, 0o600)
    elif bad == "public":
        lock.chmod(0o666)
    else:
        lock.chmod(0o400)
    before = lock.lstat()
    with pytest.raises(CodexAuthError):
        with stage_codex_auth(auth_path, lock_timeout_seconds=0.2):
            pytest.fail("unsafe lock must not stage")
    assert (lock.lstat().st_mode, lock.lstat().st_ino) == (before.st_mode, before.st_ino)


def test_symlinked_or_shared_credential_directory_is_rejected(auth_path, tmp_path):
    alias = tmp_path / "linked-home"
    alias.symlink_to(auth_path.parent, target_is_directory=True)
    with pytest.raises(CodexAuthError, match="symlink"):
        with stage_codex_auth(alias / "auth.json"):
            pytest.fail("symlinked credential directory must fail")
    auth_path.parent.chmod(0o770)
    try:
        with pytest.raises(CodexAuthError, match="group/world-writable"):
            with stage_codex_auth(auth_path):
                pytest.fail("shared writable credential directory must fail")
    finally:
        auth_path.parent.chmod(0o700)


@pytest.mark.parametrize("target", ["auth", "lock", "directory"])
def test_foreign_ownership_is_rejected_using_real_chown(auth_path, target):
    with stage_codex_auth(auth_path) as session:
        path = {"auth": auth_path, "lock": session.lock_path, "directory": auth_path.parent}[target]
    uid = os.geteuid()
    try:
        os.chown(path, uid + 1, -1)
    except PermissionError:
        pytest.skip("Changing actual file ownership requires privilege; no stat/UID mocking.")
    try:
        with pytest.raises(CodexAuthError, match="owned"):
            with stage_codex_auth(auth_path):
                pytest.fail("foreign-owned auth storage must fail")
    finally:
        os.chown(path, uid, -1)


@pytest.mark.parametrize("bad", ["missing", "symlink", "hardlink", "fifo", "directory", "public", "invalid"])
def test_invalid_return_is_an_error_not_a_silent_noop(auth_path, tmp_path, bad):
    original = auth_path.read_bytes()
    with stage_codex_auth(auth_path) as session:
        returned = session.returned_auth_path
        if bad == "symlink":
            returned.symlink_to(auth_path)
        elif bad == "hardlink":
            other = tmp_path / "outside-return.json"
            _write(other, _document("rotated"))
            os.link(other, returned)
        elif bad == "fifo":
            os.mkfifo(returned, 0o600)
        elif bad == "directory":
            returned.mkdir()
        elif bad == "public":
            _returned(session)
            returned.chmod(0o644)
        elif bad == "invalid":
            _write(returned, b"\xffsynthetic-secret")
        with pytest.raises(CodexAuthError) as caught:
            restore_codex_auth(session)
        assert "synthetic" not in str(caught.value)
    assert auth_path.read_bytes() == original
    assert not session.codex_home.parent.exists()
    assert not _recoveries(auth_path)


@pytest.mark.parametrize("change", ["account", "missing-id", "lock-replaced"])
def test_nonrotation_identity_changes_require_recovery_not_a_blind_merge(auth_path, change):
    original = auth_path.read_bytes()
    with stage_codex_auth(auth_path) as session:
        document = _returned(session)
        if change == "account":
            document["tokens"]["account_id"] = "different-synthetic-account"
        elif change == "missing-id":
            del document["tokens"]["id_token"]
        else:
            session.lock_path.unlink()
            _write(session.lock_path, b"")
        _write(session.returned_auth_path, document)
        with pytest.raises(CodexAuthConflict) as caught:
            restore_codex_auth(session)
        assert json.loads(caught.value.recovery_path.read_bytes())["tokens"] == {
            key: value for key, value in document["tokens"].items() if key != "host_extension"
        }
    assert auth_path.read_bytes() == original


def test_multiple_conflicts_keep_distinct_recovery_files(auth_path):
    paths = []
    for generation in ("first", "second"):
        with stage_codex_auth(auth_path) as session:
            _returned(session, generation)
            _replace(auth_path, _document("external"))
            with pytest.raises(CodexAuthConflict) as caught:
                restore_codex_auth(session)
            paths.append(caught.value.recovery_path)
    assert paths[0] != paths[1]
    for path, generation in zip(paths, ("first", "second")):
        _assert_recovery(path, generation)


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses the real permission failure exercised here")
def test_recovery_write_failure_retains_only_private_staging_instead_of_losing_tokens(auth_path):
    original = auth_path.read_bytes()
    session = None
    try:
        with pytest.raises(CodexAuthError, match="retained in private staging") as caught:
            with stage_codex_auth(auth_path) as session:
                _returned(session)
                auth_path.parent.chmod(0o500)
                restore_codex_auth(session)
        assert session is not None
        assert str(session.returned_auth_path) in str(caught.value)
        assert "synthetic" not in str(caught.value)
        assert stat.S_IMODE(session.returned_auth_path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(session.returned_auth_path.stat().st_mode) == 0o600
        assert json.loads(session.returned_auth_path.read_bytes())["tokens"]["refresh_token"] == "synthetic-refresh-rotated"
        assert auth_path.read_bytes() == original
    finally:
        auth_path.parent.chmod(0o700)
        if session is not None:
            shutil.rmtree(session.codex_home.parent, ignore_errors=True)


def test_replaced_auth_directory_does_not_redirect_restore_or_recovery(auth_path, tmp_path):
    moved = tmp_path / "moved-original-home"
    session = None
    try:
        with pytest.raises(CodexAuthError, match="retained in private staging"):
            with stage_codex_auth(auth_path) as session:
                _returned(session)
                auth_path.parent.rename(moved)
                auth_path.parent.mkdir(mode=0o700)
                replacement = _write(auth_path, _document("external"))
                restore_codex_auth(session)
        assert auth_path.read_bytes() == replacement
        assert not _recoveries(auth_path)
        assert json.loads((moved / "auth.json").read_bytes())["tokens"]["refresh_token"] == "synthetic-refresh-original"
        assert json.loads(session.returned_auth_path.read_bytes())["tokens"]["refresh_token"] == "synthetic-refresh-rotated"
    finally:
        if session is not None:
            shutil.rmtree(session.codex_home.parent, ignore_errors=True)


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan")])
def test_wait_deadline_must_be_finite_and_nonnegative(auth_path, timeout):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        with stage_codex_auth(auth_path, lock_timeout_seconds=timeout):
            pytest.fail("unbounded wait must fail")


# Spawn, rather than fork: independent descriptors/locks, portable to macOS, and
# no inherited session fds that might accidentally keep a parent's flock alive.
def _snapshot_worker(path, attempted, results, timeout):
    attempted.set()
    started = time.monotonic()
    try:
        with stage_codex_auth(Path(path), lock_timeout_seconds=timeout) as session:
            tokens = json.loads((session.codex_home / "auth.json").read_bytes())["tokens"]
            results.put(("acquired", tokens["refresh_token"]))
    except CodexAuthLockTimeout:
        results.put(("timeout", time.monotonic() - started))
    except BaseException as error:
        results.put((type(error).__name__, str(error)))


def _hold_worker(path, results, release):
    with stage_codex_auth(Path(path), lock_timeout_seconds=5) as session:
        results.put((str(session.codex_home.parent), str(session.lock_path)))
        release.wait(10)


def _external_writer(path, results):
    _replace(Path(path), _document("external-process"))
    results.put("replaced")


@contextmanager
def _process(target, *args):
    process = multiprocessing.get_context("spawn").Process(target=target, args=args)
    process.start()
    try:
        yield process
    finally:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        assert not process.is_alive()


def test_multiprocess_waiter_snapshots_after_refresh_and_lock_is_held_until_context_exit(auth_path):
    ctx = multiprocessing.get_context("spawn")
    attempted, results = ctx.Event(), ctx.Queue()
    with stage_codex_auth(auth_path) as session:
        worker = ctx.Process(target=_snapshot_worker, args=(str(auth_path), attempted, results, 5))
        worker.start()
        try:
            assert attempted.wait(5)
            with pytest.raises(Empty):
                results.get(timeout=0.2)
            _returned(session)
            assert restore_codex_auth(session)
            with pytest.raises(Empty):
                results.get(timeout=0.2)  # atomic source replacement must not release the sidecar lock
        except BaseException:
            worker.terminate()
            worker.join(timeout=5)
            raise
    try:
        assert results.get(timeout=5) == ("acquired", "synthetic-refresh-rotated")
        worker.join(timeout=5)
        assert worker.exitcode == 0
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
        results.close()


def test_multiprocess_wait_is_bounded_and_different_source_is_not_blocked(auth_path):
    ctx = multiprocessing.get_context("spawn")
    with stage_codex_auth(auth_path):
        attempted, results = ctx.Event(), ctx.Queue()
        with _process(_snapshot_worker, str(auth_path), attempted, results, 0.2):
            assert attempted.wait(5)
            status, elapsed = results.get(timeout=5)
            assert status == "timeout"
            assert 0.15 <= elapsed < 3
        results.close()
        other = auth_path.with_name("different-auth.json")
        _write(other, _document("different"))
        attempted, results = ctx.Event(), ctx.Queue()
        with _process(_snapshot_worker, str(other), attempted, results, 0.2):
            assert results.get(timeout=5) == ("acquired", "synthetic-refresh-different")
        results.close()


def test_real_noncooperating_process_edit_is_preserved_with_recovery(auth_path):
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    with stage_codex_auth(auth_path) as session:
        _returned(session)
        with _process(_external_writer, str(auth_path), results):
            assert results.get(timeout=5) == "replaced"
        with pytest.raises(CodexAuthConflict) as caught:
            restore_codex_auth(session)
        _assert_recovery(caught.value.recovery_path)
    assert json.loads(auth_path.read_bytes())["tokens"]["refresh_token"] == "synthetic-refresh-external-process"
    results.close()


def test_killed_process_releases_kernel_lock_without_unlinking_sidecar(auth_path):
    ctx = multiprocessing.get_context("spawn")
    results, release = ctx.Queue(), ctx.Event()
    temporary = None
    try:
        with _process(_hold_worker, str(auth_path), results, release) as worker:
            temporary, lock_path = results.get(timeout=5)
            inode = Path(lock_path).stat().st_ino
            worker.terminate()
            worker.join(timeout=5)
            with stage_codex_auth(auth_path, lock_timeout_seconds=1) as session:
                assert session.lock_path.stat().st_ino == inode
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        results.close()
