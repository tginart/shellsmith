"""Run-owned initialization attempts: uncertain retries versus later sessions."""

import json
import os
import stat
import fcntl
from pathlib import Path
import uuid

from shellsmith.process_io import write_json
from studio_compute.protocol import bounded_identifier

INITIALIZATION_FILE = "studio_initialization.json"


class InitializationJournal:
    def __init__(self, root, namespace, identity):
        self.path = Path(root) / INITIALIZATION_FILE
        self.namespace, self.identity = namespace, identity
        self.fd = None

    def acquire(self):
        if self.fd is not None:
            return
        parent = self.path.parent.parent
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock = parent / ("." + self.path.parent.name + ".studio-client.lock")
        fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600:
                raise RuntimeError("Unsafe Studio coordinator lock.")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(fd)
            raise RuntimeError("Studio coordinator output is owned by another live initialization.") from error
        except BaseException:
            os.close(fd)
            raise
        self.fd = fd

    def release(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __del__(self):
        self.release()

    def read(self):
        if not self.path.exists():
            return None
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("Initialization journal must be a regular file.")
        record = json.loads(self.path.read_text())
        if (record.get("format") != 1 or record.get("namespace") != self.namespace
                or record.get("identity") != self.identity
                or record.get("phase") not in ("pending", "active", "failed")):
            raise ValueError("Initialization journal identity or phase differs.")
        return record

    def write(self, record):
        write_json(self.path, {"format": 1, "namespace": self.namespace,
                               "identity": self.identity, **record})

    def begin(self, checkpoint, previous_owner=None):
        old = self.read()
        if old and old["phase"] == "pending":
            if old["checkpoint"] != checkpoint:
                raise RuntimeError("Resolve the pending initialization before changing checkpoints.")
            return old
        prefix = self.namespace + ":initialize:" + uuid.uuid4().hex
        record = {"phase": "pending", "checkpoint": checkpoint,
                  "request_id": bounded_identifier(prefix + ":restore")}
        if previous_owner:
            record.update(previous_owner=previous_owner,
                          release_id=bounded_identifier(prefix + ":release"))
        self.write(record)  # Durable before release or restore submission.
        return record

    def owned(self, request_id, result, checkpoint=None):
        self.write({"phase": "active", "checkpoint": checkpoint,
                    "request_id": request_id, "owner_operation": request_id, "result": result})
