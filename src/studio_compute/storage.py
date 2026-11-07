"""Crash-safe state and receipts. Full-state commits precede successful receipts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import uuid

from .protocol import Conflict, canonical, digest


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("wb") as file:
        file.write(canonical(value))
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def file_hash(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_file = (self.root / "owner.lock").open("a")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.lock_file.close()
            raise RuntimeError(f"Another service owns Studio state root: {self.root}") from error
        except BaseException:
            self.lock_file.close()
            raise
        try:
            self._open_database()
        except BaseException:
            if hasattr(self, "db"):
                self.db.close()
            self.lock_file.close()
            raise

    def _open_database(self):
        self.live_progress = {}
        self.checkpoints = self.root / "checkpoints"
        self.checkpoints.mkdir(exist_ok=True)
        self.db = sqlite3.connect(
            self.root / "journal.sqlite3", check_same_thread=False
        )
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS operations (
                id TEXT PRIMARY KEY, client TEXT NOT NULL, kind TEXT NOT NULL,
                hash TEXT NOT NULL, body TEXT NOT NULL, status TEXT NOT NULL,
                result TEXT, error TEXT, progress TEXT);
            CREATE TABLE IF NOT EXISTS clients (id TEXT PRIMARY KEY, state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS policies (id TEXT PRIMARY KEY, state TEXT NOT NULL);
        """)
        self.db.commit()

    def submit(self, request_id, client, kind, body):
        fingerprint = digest({"client": client, "kind": kind, "body": body})
        row = self.db.execute(
            "SELECT * FROM operations WHERE id=?", (request_id,)
        ).fetchone()
        if row:
            if row["hash"] != fingerprint:
                raise Conflict("Request ID was already used with different inputs.")
            if row["status"] == "interrupted" and kind == "create" and not self.db.execute(
                "SELECT 1 FROM clients WHERE id=?", (client,)
            ).fetchone():
                # Recovery already reconstructed every renamed checkpoint. An
                # interrupted create with no client has no committed effect;
                # safely retry its original durable intent, never an optimizer.
                with self.db:
                    self.db.execute(
                        "UPDATE operations SET status='queued',error=NULL,progress=NULL,body=? WHERE id=?",
                        (canonical(body).decode(), request_id),
                    )
                return self.operation(request_id), True
            return self.operation(request_id), False
        with self.db:
            self.db.execute(
                "INSERT INTO operations(id,client,kind,hash,body,status) VALUES (?,?,?,?,?,?)",
                (
                    request_id,
                    client,
                    kind,
                    fingerprint,
                    canonical(body).decode(),
                    "queued",
                ),
            )
        return self.operation(request_id), True

    def operation(self, request_id):
        row = self.db.execute(
            "SELECT * FROM operations WHERE id=?", (request_id,)
        ).fetchone()
        if row is None:
            raise KeyError(request_id)
        operation = {
            key: json.loads(row[key]) if row[key] else None
            for key in ("result", "progress")
        } | {key: row[key] for key in ("id", "client", "kind", "status", "error")}
        if request_id in self.live_progress:
            operation["progress"] = self.live_progress[request_id]
        return operation

    def running(self, request_id, progress=None):
        with self.db:
            changed = self.db.execute(
                "UPDATE operations SET status='running',progress=? WHERE id=? AND status IN ('queued','running')",
                (canonical(progress).decode() if progress else None, request_id),
            )
        if changed.rowcount:
            self.live_progress[request_id] = progress

    def observe_progress(self, request_id: str, progress: dict) -> None:
        """
        Purpose: Update transient progress without a disk commit on the GPU owner.
        Input: An admitted operation ID and its latest materialized progress.
        Output: The next operation poll sees this progress until its terminal receipt.
        Note: Admission, RNG assignment, and terminal receipts remain durable.
        """
        if request_id not in self.live_progress:
            raise KeyError("Progress requires durable admission: " + request_id)
        self.live_progress[request_id] = progress

    def assign_rng(self, request_id, client, state, body):
        with self.db:
            self.db.execute(
                "UPDATE operations SET body=? WHERE id=?",
                (canonical(body).decode(), request_id),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO clients VALUES (?,?)",
                (client, canonical(state).decode()),
            )

    def _terminal_body(self, request_id):
        # Only this small linkage is needed to recover consumed backward receipts.
        row = self.db.execute("SELECT kind FROM operations WHERE id=?", (request_id,)).fetchone()
        if row is None:
            raise KeyError(request_id)
        if row[0] == "optim_step":
            body = json.loads(self.db.execute("SELECT body FROM operations WHERE id=?", (request_id,)).fetchone()[0])
            return canonical({"backward_operation": body.get("backward_operation")}).decode()
        return "{}"

    def finish(self, request_id, result, *, client_state=None, policy=None):
        with self.db:
            if client_state is not None:
                client = self.operation(request_id)["client"]
                self.db.execute(
                    "INSERT OR REPLACE INTO clients VALUES (?,?)",
                    (client, canonical(client_state).decode()),
                )
            if policy is not None:
                self.db.execute(
                    "INSERT OR REPLACE INTO policies VALUES (?,?)",
                    (policy["id"], canonical(policy).decode()),
                )
            self.db.execute(
                "UPDATE operations SET status='succeeded',result=?,error=NULL,body=?,progress=NULL WHERE id=?",
                (canonical(result).decode(), self._terminal_body(request_id), request_id),
            )
        self.live_progress.pop(request_id, None)

    def fail(self, request_id, error, status="failed"):
        with self.db:
            self.db.execute(
                "UPDATE operations SET status=?,error=?,body=?,progress=NULL WHERE id=?",
                (status, str(error), self._terminal_body(request_id), request_id),
            )
        self.live_progress.pop(request_id, None)

    def states(self, table):
        if table not in ("clients", "policies"):
            raise ValueError(table)
        return {
            row["id"]: json.loads(row["state"])
            for row in self.db.execute(f"SELECT * FROM {table}")
        }

    def checkpoint_intent(self, request_id):
        row = self.db.execute("SELECT hash,client FROM operations WHERE id=?", (request_id,)).fetchone()
        if row is None:
            raise KeyError(request_id)
        return {"hash": row["hash"], "client": row["client"]}

    def checkpoint(self, request_id, writer, *, identity, state, result, intent=None):
        # Snapshot journal metadata before slow serialization. A service caller
        # supplies this under its journal lock; the physical writer uses no DB.
        intent = self.checkpoint_intent(request_id) if intent is None else intent
        reference = digest({"operation": request_id, "identity": identity})
        destination = self.checkpoints / reference
        temporary = self.checkpoints / (".pending-" + uuid.uuid4().hex)
        temporary.mkdir()
        try:
            writer(temporary)
            files = {}
            for path in sorted(temporary.iterdir()):
                if not path.is_file() or path.is_symlink():
                    raise ValueError(
                        "Checkpoint writers must produce regular flat files."
                    )
                with path.open("rb") as file:
                    os.fsync(file.fileno())
                files[path.name] = file_hash(path)
            state = {**state, "checkpoint": "studio-state:" + reference}
            result = {**result, "checkpoint": state["checkpoint"]}
            document = {
                "format": 1,
                "identity": identity,
                "operation": request_id,
                "request_hash": intent["hash"],
                "client": intent["client"],
                "state": state,
                "result": result,
                "files": files,
            }
            atomic_json(
                temporary / "manifest.json",
                {**document, "manifest_sha256": digest(document)},
            )
            sync_directory(temporary)
            os.rename(temporary, destination)
            sync_directory(self.checkpoints)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return state, result

    def manifest(self, reference, identity=None):
        if not isinstance(reference, str) or not reference.startswith("studio-state:"):
            raise ValueError("Expected an opaque Studio state reference.")
        key = reference.removeprefix("studio-state:")
        if len(key) != 64 or any(x not in "0123456789abcdef" for x in key):
            raise ValueError("Malformed state reference.")
        folder = self.checkpoints / key
        if folder.is_symlink() or (folder / "manifest.json").is_symlink():
            raise ValueError("Unsafe checkpoint manifest path.")
        document = json.loads((folder / "manifest.json").read_text())
        sealed = dict(document)
        expected_manifest = sealed.pop("manifest_sha256", None)
        if expected_manifest != digest(sealed):
            raise ValueError("Checkpoint manifest corruption.")
        document = sealed
        if document.get("format") != 1 or (
            identity is not None and document["identity"] != identity
        ):
            raise ValueError("Checkpoint compute identity differs.")
        return folder, document

    def verify(self, reference, identity=None):
        folder, document = self.manifest(reference, identity)
        if (folder / "retired.json").exists():
            from .retention import CheckpointUnavailable
            raise CheckpointUnavailable("Checkpoint weights were pruned: " + reference)
        if set(path.name for path in folder.iterdir()) != set(document["files"]) | {
            "manifest.json"
        }:
            raise ValueError("Checkpoint file inventory differs.")
        for name, expected in document["files"].items():
            if (
                Path(name).name != name
                or (folder / name).is_symlink()
                or file_hash(folder / name) != expected
            ):
                raise ValueError("Checkpoint corruption: " + name)
        return folder, document

    def recover(self, identity):
        from .retention import cleanup
        cleanup(self)
        # A checkpoint rename may have succeeded just before receipt commit.
        # Its embedded receipt completes that exact operation, never re-executes it.
        cleaned = False
        for folder in sorted(self.checkpoints.iterdir()):
            staging_key = folder.name.removeprefix(".pending-")
            if folder.name.startswith(".pending-") and len(staging_key) == 32 and all(
                c in "0123456789abcdef" for c in staging_key
            ):
                if folder.is_symlink() or not folder.is_dir():
                    raise ValueError("Unsafe checkpoint staging entry.")
                shutil.rmtree(folder)
                cleaned = True
                continue
            if folder.name.startswith("."):
                continue
            if (folder / "retired.json").exists():
                self.manifest("studio-state:" + folder.name, identity)
                continue
            _, doc = self.verify("studio-state:" + folder.name, identity)
            row = self.db.execute(
                "SELECT hash,status FROM operations WHERE id=?", (doc["operation"],)
            ).fetchone()
            if row is None or row["hash"] != doc["request_hash"]:
                raise ValueError("Checkpoint has no matching durable intent.")
            if row["status"] in ("running", "queued", "interrupted"):
                self.finish(doc["operation"], doc["result"], client_state=doc["state"])
        if cleaned:
            sync_directory(self.checkpoints)
        with self.db:
            self.db.execute(
                "UPDATE operations SET status='interrupted',error='Service stopped before commit' WHERE status IN ('queued','running')"
            )
            # Backward receipts describe volatile gradient accumulation. They
            # are reusable only when their optimizer receipt already committed.
            consumed = set()
            for row in self.db.execute(
                "SELECT body FROM operations WHERE kind='optim_step' AND status='succeeded'"
            ):
                consumed.add(json.loads(row[0]).get("backward_operation"))
            for row in self.db.execute(
                "SELECT id FROM operations WHERE kind='forward_backward' AND status='succeeded'"
            ).fetchall():
                if row[0] not in consumed:
                    self.db.execute(
                        "UPDATE operations SET status='interrupted',error='Volatile gradients lost on restart' WHERE id=?",
                        (row[0],),
                    )
        with self.db:
            # Compact old terminal rows too. Fetch IDs, never a batch of large bodies.
            ids = self.db.execute("SELECT id FROM operations WHERE status NOT IN ('queued','running')").fetchall()
            for row in ids:
                self.db.execute("UPDATE operations SET body=?,progress=NULL WHERE id=?", (self._terminal_body(row[0]), row[0]))
        # Verify referenced state even if its receipt was committed earlier.
        for state in self.states("clients").values():
            if not state.get("closed"):
                self.verify(state["checkpoint"], identity)

    def close(self):
        try:
            self.db.close()
        finally:
            if not self.lock_file.closed:
                fcntl.flock(self.lock_file, fcntl.LOCK_UN)
                self.lock_file.close()
