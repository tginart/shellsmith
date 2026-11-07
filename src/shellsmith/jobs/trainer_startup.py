"""Publish a trainer's birth and await durable admission before importing its CLI."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import select
import sys
import time

from .bundle import verify
from .service import job_folder
from .storage import alive, digest, lock, now, owner_identity, process_identity, read, write


def launch_binding(record: dict) -> dict:
    return {key: record[key] for key in (
        "job_id", "attempt", "owner_identity", "worker", "runtime", "python",
        "source_sha256", "config_sha256", "command")}


def admitted(record: dict, binding: dict) -> bool:
    return (all(record.get(key) == value for key, value in binding.items())
            and record["status"] in {"starting", "running"}
            and not record.get("stop_requested"))


def run(home: Path, job_id: str, binding: dict, gate_fd: int) -> int:
    """A closed gate, changed attempt, or durable stop never loads the trainer."""
    try:
        folder = job_folder(home, job_id)
        if binding["owner_identity"] != owner_identity() or binding["job_id"] != job_id:
            raise ValueError("Trainer launch owner or job identity changed")
        runtime = Path(__file__).resolve().parents[3]
        if runtime != Path(binding["runtime"]).resolve():
            raise ValueError("Trainer launch frozen runtime changed")
        if Path(sys.executable).resolve() != Path(binding["python"]).resolve():
            raise ValueError("Trainer launch interpreter changed")
        identity = None
        for attempt in range(3):
            identity = process_identity(os.getpid())
            if identity is not None and identity.get("pid") == os.getpid():
                break
            identity = None
            if attempt < 2:
                time.sleep(.05)
        if identity is None:
            raise RuntimeError("Cannot establish the trainer's process birth identity")
        with lock(folder / "record.lock"):
            record = read(folder / "job.json")
            if not admitted(record, binding):
                return 1
            prior = record.get("child")
            if prior != identity and alive(prior):
                raise ValueError("A different live child already owns this job attempt")
            record.update(child=identity, updated=now())
            write(folder / "job.json", record)
        # The writer is owned only by the supervisor. Its death closes the gate;
        # even a late-starting wrapper cannot begin an unrecorded training run.
        ready, _, _ = select.select([gate_fd], [], [], 30)
        if not ready or os.read(gate_fd, 1) != b"G":
            return 1
        with lock(folder / "record.lock"):
            record = read(folder / "job.json")
            if not admitted(record, binding) or record["status"] != "running" or record.get("child") != identity:
                return 1
        verify(runtime, binding["source_sha256"])
        if digest((folder / "run.toml").read_bytes()) != binding["config_sha256"]:
            raise ValueError("Trainer launch frozen configuration changed")
        # Recheck after source validation, which can take time on a large bundle.
        with lock(folder / "record.lock"):
            record = read(folder / "job.json")
            if not admitted(record, binding) or record.get("child") != identity:
                return 1
        command = binding["command"]
        if command[:3] != [binding["python"], "-m", "shellsmith"]:
            raise ValueError("Trainer launch command changed")
        sys.argv = ["shellsmith", *command[3:]]
        runpy.run_module("shellsmith", run_name="__main__", alter_sys=True)
        return 0
    finally:
        os.close(gate_fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--gate-fd", type=int, required=True)
    args = parser.parse_args()
    return run(args.home, args.job_id, json.loads(args.binding), args.gate_fd)


if __name__ == "__main__":
    raise SystemExit(main())
