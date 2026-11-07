"""Execution-home operations; durable IDs survive a dropped SSH response."""
from __future__ import annotations

import base64
import errno
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .bundle import verify
from .readiness import installed_dependencies, probe
from .storage import alive, boot_identity, canonical, digest, identifier, lock, now, owner_identity, private_dir, process_identity, read, write

ACTIVE = {"queued", "starting", "running", "stopping"}
ARTIFACT_PAGE_BYTES = 4 * 1024 * 1024
ARTIFACT_PAGE_FILES = 256


def job_folder(home: Path, job_id: str) -> Path:
    return private_dir(private_dir(home / "jobs") / identifier(job_id))


def update(folder: Path, **changes) -> dict:
    with lock(folder / "record.lock"):
        record = read(folder / "job.json")
        if record.get("owner_identity") is not None and record["owner_identity"] != owner_identity():
            raise ValueError("This job belongs to a different physical machine or account")
        record.update(changes)
        write(folder / "job.json", record)
        return record


def status(home: Path, job_id: str) -> dict:
    folder = job_folder(home, job_id)
    with lock(folder / "record.lock"):
        record = read(folder / "job.json")
        _same_owner(record)
        if record["status"] in ACTIVE and not _unlaunched_queue(record) and not alive(record.get("worker")):
            if record.get("stop_requested") and not alive(record.get("child")):
                record.update(status="stopped", updated=now(), reason="Stop reconciled after worker exit")
                write(folder / "job.json", record)
            elif now() - record["updated"] > 10:
                record.update(status="interrupted", updated=now(),
                              reason="Recorded worker no longer exists; explicit resume is required")
                write(folder / "job.json", record)
        return record


def _unlaunched_queue(record: dict) -> bool:
    """Only the current boot's intent without a process can retry its launch."""
    return (record["status"] == "queued" and record.get("worker") is None and record.get("child") is None
            and not record.get("stop_requested")
            and record.get("queued_boot", record.get("submitted_boot")) == boot_identity())


def _same_owner(record: dict) -> None:
    if record.get("owner_identity") != owner_identity():
        raise ValueError("This job belongs to a different physical machine or account")


def _signal_stop(folder: Path, record: dict) -> dict:
    """Reconcile a durable stop, including a surviving trainer after worker loss."""
    if alive(record.get("worker")):
        os.kill(record["worker"]["pid"], signal.SIGTERM)
    elif alive(record.get("child")):
        identity = record["child"]
        os.killpg(identity["pid"], signal.SIGINT)
        deadline = time.monotonic() + 3
        while alive(identity) and time.monotonic() < deadline:
            time.sleep(.1)
        if alive(identity):
            os.killpg(identity["pid"], signal.SIGKILL)
        record.update(status="stopped", updated=now(), reason="Stopped orphaned trainer after worker loss")
        write(folder / "job.json", record)
    elif record["status"] in ACTIVE:
        record.update(status="stopped", updated=now(), reason="Stop reconciled after worker exit")
        write(folder / "job.json", record)
    return record


def _launch(folder: Path, record: dict) -> dict:
    # Never inherit SSH stdin/stdout/stderr or let the transport own the process.
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(record["runtime"]) / "src")
    environment["PATH"] = str(Path(record["python"]).parent) + os.pathsep + environment.get("PATH", "")
    environment["SHELLSMITH_ROOT"] = record["runtime"]
    environment["MLX_ENABLE_TF32"] = "0"
    with (folder / "worker.log").open("ab", buffering=0) as log:
        # Cross a durable launch boundary before creating a worker. Once Popen
        # returns, missing identity/publication must never make this retryable.
        record.update(status="starting", worker=None, child=None, updated=now())
        try:
            write(folder / "job.json", record)
            child = subprocess.Popen([record["python"], "-m", "shellsmith.jobs.worker", "--home", record["home"],
                                     "--job-id", record["job_id"]], cwd=record["runtime"], env=environment,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                    start_new_session=True, close_fds=True)
        except OSError:
            # A pre-spawn write or Popen failed before returning a child. This
            # intent may retry; later errors keep the starting state.
            record.update(status="queued", updated=now())
            write(folder / "job.json", record)
            raise
    # Worker initialization takes record.lock after this launch operation releases it.
    record.update(worker=process_identity(child.pid), status="starting", updated=now())
    write(folder / "job.json", record)
    return record


def _reconcile_request(folder: Path, record: dict, action: str, request_id: str) -> dict:
    """Finish side effects only after the request's durable mutation exists."""
    if (action in {"submit", "resume"} and _unlaunched_queue(record)
            and record.get("queued_request_id", record.get("request_id")) == request_id):
        return _launch(folder, record)
    if action == "stop" and record.get("stop_requested"):
        return _signal_stop(folder, record)
    return record


def _full_checkpoint(folder: Path) -> dict:
    state_path = folder / "run" / "trainer_state.json"
    if not state_path.is_file():
        raise ValueError("Resume requires an existing full optimizer checkpoint")
    state = read(state_path)
    rows = [row for row in state.get("checkpoints", []) if row.get("kind") == "state"]
    if not rows:
        raise ValueError("Resume requires a full state checkpoint; adapter exports are insufficient")
    row = rows[-1]
    path = row.get("path") or row.get("state_path")
    if not isinstance(path, str) or not Path(path).is_dir() or not row.get("sampler_path"):
        raise ValueError("Full checkpoint and matching sampler weights are missing")
    # The numerical backend validates all tensor hashes before restore.
    manifest = Path(path) / "manifest.json"
    if not manifest.is_file():
        raise ValueError("Full checkpoint integrity manifest is missing")
    from shellsmith.mlx.checkpoint import verify_checkpoint
    identity = row.get("policy_identity")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("Full checkpoint row lacks its frozen policy identity")
    checked = verify_checkpoint(Path(path), kind="state", expected_identity=identity)
    if (checked["sampler_path"] != row["sampler_path"] or
            checked["policy_version"] != row.get("step")):
        raise ValueError("Full checkpoint differs from the committed trainer recovery row")
    return row


def _adapter_checkpoint(config: Path, path: str, expected_sha256: str | None = None) -> dict:
    """Pin a campaign's actual sampler against its frozen model and runtime."""
    from shellsmith.train.config import load_train_config
    from shellsmith.mlx.checkpoint import verify_checkpoint
    from shellsmith.mlx.identity import resolve_identity
    settings = load_train_config(config, campaign_mode="inference")
    if settings.compute.backend != "mlx" or not Path(path).is_absolute():
        raise ValueError("Campaign adapter must be an absolute MLX sampler checkpoint")
    manifest = verify_checkpoint(Path(path), kind="sampler", expected_identity=resolve_identity(settings.mlx))
    sha = digest((Path(path) / "manifest.json").read_bytes())
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError("Frozen campaign adapter manifest changed before execution")
    return {"path": path, "manifest_sha256": sha, "policy_identity": manifest["policy_identity"],
            "policy_version": manifest["policy_version"]}


def dispatch(home: Path, runtime: Path, envelope: dict) -> dict:
    private_dir(home)
    action = envelope["action"]
    if action == "check":
        result = probe()
        if envelope.get("config_document") is not None:
            from shellsmith.train.preflight import probe_document
            run_probe = probe_document(envelope["config_document"])
            result["run"] = run_probe
            result["ready"] = result["ready"] and run_probe.get("ready", False)
            result["blockers"].extend(run_probe.get("blockers", []))
        return result
    job_id = identifier(envelope["job_id"])
    folder = job_folder(home, job_id)
    if action == "status":
        return status(home, job_id)
    if action == "logs":
        _same_owner(read(folder / "job.json"))
        offset, limit = envelope.get("offset", 0), envelope.get("limit", 65536)
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 1024 * 1024:
            raise ValueError("Invalid log offset or size")
        path = folder / "worker.log"
        if path.is_symlink():
            raise ValueError("Refusing symlinked logs")
        if not path.exists():
            return {"text": "", "offset": offset, "eof": True}
        with path.open("rb") as stream:
            stream.seek(offset)
            chunk = stream.read(limit)
            return {"text": chunk.decode("utf-8", errors="replace"), "offset": offset + len(chunk),
                    "eof": stream.read(1) == b""}
    if action == "artifacts":
        _same_owner(read(folder / "job.json"))
        return artifacts(folder, envelope=envelope)
    if action not in {"submit", "stop", "resume"}:
        raise ValueError("Unknown job operation")
    request_id = identifier(envelope["request_id"])
    receipts = private_dir(home / "receipts")
    receipt = receipts / f"{request_id}.json"
    signature = digest(canonical({key: value for key, value in envelope.items() if key != "archive"}))
    with lock(home / "requests.lock"):
        if receipt.exists():
            saved = read(receipt)
            if saved["signature"] != signature:
                raise ValueError("Request ID was already used with different inputs")
            previous_folder = job_folder(home, saved["job_id"])
            with lock(previous_folder / "record.lock"):
                previous = read(previous_folder / "job.json")
                _same_owner(previous)
                _reconcile_request(previous_folder, previous, saved["action"], request_id)
            return status(home, saved["job_id"])
        with lock(folder / "record.lock"):
            transaction = {"signature": signature, "action": action}
            if (folder / "job.json").exists():
                existing = read(folder / "job.json")
                _same_owner(existing)
                transactions = existing.get("request_transactions", {})
                if not isinstance(transactions, dict):
                    raise ValueError("Job request transaction history is invalid")
                if request_id in transactions:
                    if transactions[request_id] != transaction:
                        raise ValueError("Request ID was already used with different inputs")
                    # The job mutation and request identity are one atomic write.
                    # A crash before the separate receipt must not repeat a
                    # resume's attempt increment or strand a queued submission.
                    write(receipt, {"signature": signature, "job_id": job_id, "action": action})
                    return _reconcile_request(folder, existing, action, request_id)
            if action == "submit":
                if (folder / "job.json").exists():
                    raise ValueError("Job ID already exists under another request")
                config = base64.b64decode(envelope["config"], validate=True)
                if digest(config) != envelope["config_sha256"]:
                    raise ValueError("Frozen configuration hash mismatch")
                verify(runtime, envelope["source_sha256"])
                command = envelope.get("command", "train")
                adapter_path = envelope.get("adapter_path")
                if command not in {"train", "campaign"} or (adapter_path is not None and command != "campaign"):
                    raise ValueError("Invalid job command or adapter selection")
                (folder / "run.toml").write_bytes(config)
                (folder / "run.toml").chmod(0o600)
                adapter = _adapter_checkpoint(folder / "run.toml", adapter_path) if adapter_path is not None else None
                queue_boot = boot_identity()
                record = {"format": "shellsmith.studio-job.v1", "job_id": job_id, "home": str(home),
                          "command_kind": command, "adapter": adapter,
                          "execution_policy": {"MLX_ENABLE_TF32": "0"},
                          "runtime": str(runtime), "source_sha256": envelope["source_sha256"],
                          "config_sha256": envelope["config_sha256"], "python": sys.executable,
                          "dependencies": installed_dependencies(), "destination": envelope["destination"],
                          "owner_identity": owner_identity(),
                          "submitted_boot": queue_boot, "queued_boot": queue_boot, "queued_request_id": request_id,
                          "created": now(), "updated": now(), "attempt": 1, "status": "queued",
                          "request_id": request_id, "worker": None, "child": None, "stop_requested": False,
                          "request_transactions": {request_id: transaction}}
                write(folder / "job.json", record)
            else:
                record = read(folder / "job.json")
                if record["owner_identity"] != owner_identity():
                    raise ValueError("This job belongs to a different physical machine or account")
                if record["source_sha256"] != envelope["source_sha256"]:
                    raise ValueError("Control must use the original frozen runtime")
                record.setdefault("request_transactions", {})[request_id] = transaction
                if action == "stop":
                    record.update(stop_requested=True, updated=now())
                    if record["status"] in ACTIVE:
                        record["status"] = "stopping"
                    write(folder / "job.json", record)
                else:
                    if record.get("command_kind", "train") != "train":
                        raise ValueError("Campaign jobs do not resume; submit a new campaign request")
                    if alive(record.get("worker")) or alive(record.get("child")):
                        raise ValueError("The original worker or trainer is still active")
                    if record["status"] == "completed":
                        raise ValueError("Completed jobs do not resume automatically")
                    verify(Path(record["runtime"]), record["source_sha256"])
                    if installed_dependencies() != record["dependencies"] or sys.executable != record["python"]:
                        raise ValueError("Resume must use the original interpreter and dependency identity")
                    _full_checkpoint(folder)
                    record.update(attempt=record["attempt"] + 1, stop_requested=False, status="queued", updated=now(),
                                  worker=None, child=None, queued_boot=boot_identity(), queued_request_id=request_id)
                    write(folder / "job.json", record)
            # Publish the durable receipt before launching. A retry can reconcile queued work.
            write(receipt, {"signature": signature, "job_id": job_id, "action": action})
            return _reconcile_request(folder, record, action, request_id)


def artifacts(folder: Path, *, envelope: dict | None = None) -> dict:
    """Read a bounded metadata snapshot; tensor checkpoints stay on the Studio."""
    if envelope is not None and envelope.get("artifacts_version") == 2:
        return _artifact_page(folder, envelope)
    run = folder / "run"
    files, size = {}, 0
    if run.exists():
        for path in sorted(run.rglob("*")):
            relative = path.relative_to(run)
            if (any(part.startswith(".") or part.lower() in {"credentials", "auth", "secrets", "node_modules"}
                    for part in relative.parts) or path.name.lower() in {"auth.json", "credentials.json", "secrets.json"}
                    or path.suffix not in {".json", ".jsonl", ".md", ".txt"}):
                continue
            if any((run / Path(*relative.parts[:index])).is_symlink() for index in range(1, len(relative.parts) + 1)):
                raise ValueError("Artifact sync refuses symlinked paths")
            if path.is_symlink():
                raise ValueError("Artifact sync refuses symlinks")
            if not path.is_file():
                continue
            data = path.read_bytes()
            size += len(data)
            if size > 32 * 1024 * 1024:
                raise ValueError("Metadata snapshot exceeds 32 MiB; request smaller artifacts")
            files[relative.as_posix()] = {"data": base64.b64encode(data).decode(), "sha256": digest(data)}
    return {"job": read(folder / "job.json"), "files": files}


def _artifact_inventory(run: Path) -> tuple[list, str]:
    entries = []
    if run.is_symlink():
        raise ValueError("Artifact sync refuses symlinked paths")
    if run.exists():
        for path in sorted(run.rglob("*")):
            relative = path.relative_to(run)
            if (any(part.startswith(".") or part.lower() in {"credentials", "auth", "secrets", "node_modules"}
                    for part in relative.parts) or path.name.lower() in {"auth.json", "credentials.json", "secrets.json"}
                    or path.suffix not in {".json", ".jsonl", ".md", ".txt"}):
                continue
            if any((run / Path(*relative.parts[:index])).is_symlink() for index in range(1, len(relative.parts) + 1)):
                raise ValueError("Artifact sync refuses symlinked paths")
            if path.is_file():
                entries.append((relative.as_posix(), path, _artifact_stamp(path.stat())))
    return entries, digest(canonical({name: stamp for name, _, stamp in entries}))


def _artifact_stamp(stat) -> list[int]:
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def _artifact_open(path: Path, stamp: list[int], run: Path):
    # Hold each parent directory by descriptor so a concurrent symlink swap
    # cannot turn an allowed run-relative name into an outside credential read.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    parent = None
    try:
        parent = os.open(run, flags | os.O_DIRECTORY)
        parts = path.relative_to(run).parts
        for component in parts[:-1]:
            child = os.open(component, flags | os.O_DIRECTORY, dir_fd=parent)
            os.close(parent)
            parent = child
        descriptor = os.open(parts[-1], flags, dir_fd=parent)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError("Artifact sync refuses symlinked paths") from error
        if error.errno == errno.ENOENT:
            raise ValueError("Artifacts changed during sync; retry after the job settles") from error
        raise
    finally:
        if parent is not None:
            os.close(parent)
    stream = os.fdopen(descriptor, "rb")
    if _artifact_stamp(os.fstat(stream.fileno())) != stamp:
        stream.close()
        raise ValueError("Artifacts changed during sync; retry after the job settles")
    return stream


def _artifact_page(folder: Path, envelope: dict) -> dict:
    """Stream bounded chunks with a stable inventory; never load whole captures."""
    limit = envelope.get("limit", ARTIFACT_PAGE_BYTES)
    cursor = envelope.get("cursor")
    if type(limit) is not int or not 1 <= limit <= ARTIFACT_PAGE_BYTES:
        raise ValueError("Invalid artifact page size")
    entries, inventory = _artifact_inventory(folder / "run")
    index = offset = 0
    file_sha = None
    if cursor is not None:
        if (not isinstance(cursor, dict) or set(cursor) != {"inventory_sha256", "path", "offset", "file_sha256"}
                or type(cursor["offset"]) is not int or cursor["offset"] < 0):
            raise ValueError("Invalid artifact continuation cursor")
        if cursor["inventory_sha256"] != inventory:
            raise ValueError("Artifacts changed during sync; retry after the job settles")
        names = [entry[0] for entry in entries]
        if cursor["path"] not in names:
            raise ValueError("Invalid artifact continuation path")
        index, offset, file_sha = names.index(cursor["path"]), cursor["offset"], cursor["file_sha256"]
        if offset > entries[index][2][2] or (offset and (
                not isinstance(file_sha, str) or len(file_sha) != 64
                or any(character not in "0123456789abcdef" for character in file_sha))):
            raise ValueError("Invalid artifact continuation offset or checksum")
        if offset == 0 and file_sha is not None:
            raise ValueError("Invalid artifact continuation checksum")
    chunks, remaining = [], limit
    while index < len(entries) and remaining > 0 and len(chunks) < ARTIFACT_PAGE_FILES:
        name, path, stamp = entries[index]
        with _artifact_open(path, stamp, folder / "run") as stream:
            if file_sha is None:
                checksum = hashlib.sha256()
                while data := stream.read(128 * 1024):
                    checksum.update(data)
                file_sha = checksum.hexdigest()
            stream.seek(offset)
            data = stream.read(min(remaining, stamp[2] - offset))
            if (_artifact_stamp(os.fstat(stream.fileno())) != stamp
                    or len(data) != min(remaining, stamp[2] - offset)):
                raise ValueError("Artifacts changed during sync; retry after the job settles")
        end = offset + len(data)
        chunks.append({"path": name, "offset": offset, "size": stamp[2],
                       "sha256": digest(data), "file_sha256": file_sha,
                       "data": base64.b64encode(data).decode(), "eof": end == stamp[2]})
        remaining -= len(data)
        if end == stamp[2]:
            index, offset, file_sha = index + 1, 0, None
        else:
            offset = end
    if _artifact_inventory(folder / "run")[1] != inventory:
        raise ValueError("Artifacts changed during sync; retry after the job settles")
    next_cursor = ({"inventory_sha256": inventory, "path": entries[index][0], "offset": offset,
                    "file_sha256": file_sha} if index < len(entries) else None)
    return {"format": "shellsmith.job-artifacts.v2", "job": read(folder / "job.json"),
            "inventory_sha256": inventory, "chunks": chunks, "next_cursor": next_cursor,
            "done": next_cursor is None}
