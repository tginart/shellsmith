"""Detached supervisor. The Studio owns child lifetime, heartbeats and outcomes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .bundle import verify
from .readiness import installed_dependencies
from .service import _adapter_checkpoint, job_folder, update
from .storage import alive, digest, lock, now, owner_identity, process_identity, read, write
from .trainer_startup import launch_binding


def _self_identity() -> dict:
    pid = os.getpid()
    for attempt in range(3):
        identity = process_identity(pid)
        if identity is not None and identity.get("pid") == pid:
            return identity
        if attempt < 2:
            time.sleep(.05)
    raise RuntimeError("Cannot establish the worker's process birth identity")


def _record_child(folder: Path, identity: dict | None, *, executing: bool,
                  binding: dict | None = None) -> bool:
    with lock(folder / "record.lock"):
        record = read(folder / "job.json")
        if record.get("owner_identity") is not None and record["owner_identity"] != owner_identity():
            raise ValueError("This job belongs to a different physical machine or account")
        if binding is not None and any(record.get(key) != value for key, value in binding.items()):
            return False
        if binding is not None and record["status"] not in {"starting", "running", "stopping"}:
            return False
        if executing and record["status"] not in {"queued", "starting", "running", "stopping"}:
            return False
        if binding is not None and identity is None:
            return False
        record.update(child=identity, updated=now())
        # Running promises a durable trainer identity. A stop may have arrived
        # during Popen or identity lookup; never move that record back to running.
        if executing and identity and record["status"] in {"queued", "starting"} and not record.get("stop_requested"):
            record.update(status="running", phase="executing")
        write(folder / "job.json", record)
        return bool(identity and record["status"] == "running" and not record.get("stop_requested"))


def _update_attempt(folder: Path, binding: dict | None, **changes) -> dict | None:
    if binding is None:
        return update(folder, **changes)
    with lock(folder / "record.lock"):
        record = read(folder / "job.json")
        if record.get("owner_identity") is not None and record["owner_identity"] != owner_identity():
            raise ValueError("This job belongs to a different physical machine or account")
        if (all(record.get(key) == value for key, value in binding.items())
                and record["status"] in {"starting", "running", "stopping"}):
            record.update(changes)
            write(folder / "job.json", record)
            return record
    return None


def _claim_worker(folder: Path, binding: dict, identity: dict) -> dict | None:
    with lock(folder / "record.lock"):
        record = read(folder / "job.json")
        if (any(record.get(key) != value for key, value in binding.items())
                or record["status"] not in {"queued", "starting"} or record.get("stop_requested")
                or record.get("worker") not in (None, identity)):
            return None
        if record["owner_identity"] != owner_identity():
            raise ValueError("Frozen job owner machine/account identity changed")
        record.update(worker=identity, updated=now())
        write(folder / "job.json", record)
        return record


def _wait_child(child, folder: Path, interrupted, *, timeout_seconds: float | None = None,
                executing: bool = False, start_gate: int | None = None,
                binding: dict | None = None) -> tuple[int, bool]:
    def release_gate(allowed):
        nonlocal start_gate
        if start_gate is not None:
            try:
                if allowed and not interrupted():
                    os.write(start_gate, b"G")
            except BrokenPipeError:
                pass  # The wrapper already rejected or exited this attempt.
            finally:
                os.close(start_gate)
                start_gate = None
    try:
        identity = process_identity(child.pid)
        published = _record_child(folder, identity, executing=executing, binding=binding)
        if identity is not None:
            release_gate(published)
        stop_time = None
        started = time.monotonic()
        while child.poll() is None:
            if identity is None:
                identity = process_identity(child.pid)
                if identity is not None:
                    published = _record_child(folder, identity, executing=executing, binding=binding)
                    release_gate(published)
            latest = read(folder / "job.json")
            requested = interrupted() or latest.get("stop_requested", False)
            expired = timeout_seconds is not None and time.monotonic() - started > timeout_seconds
            if requested or expired:
                release_gate(False)
                if stop_time is None:
                    if alive(identity):
                        os.killpg(child.pid, signal.SIGINT)
                        stop_time = time.monotonic()
                    _update_attempt(folder, binding, status="stopping", stop_requested=requested, updated=now())
                elif time.monotonic() - stop_time > 30 and alive(identity):
                    os.killpg(child.pid, signal.SIGKILL)
            _update_attempt(folder, binding, heartbeat=now(), updated=now())
            time.sleep(.5)
        code = child.wait()
        return code, interrupted() or read(folder / "job.json").get("stop_requested", False)
    finally:
        release_gate(False)


def run(home: Path, job_id: str) -> int:
    folder = job_folder(home, job_id)
    with lock(folder / "worker.lock", nonblocking=True):
        record = read(folder / "job.json")
        if record["status"] not in {"queued", "starting"}:
            return 0
        child = None
        start_gate = None
        binding = {key: record[key] for key in ("job_id", "attempt", "owner_identity", "runtime", "python",
                                               "source_sha256", "config_sha256")}
        interrupted = False
        def stop(signum, frame):
            nonlocal interrupted
            interrupted = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            if record["owner_identity"] != owner_identity():
                raise ValueError("Frozen job owner machine/account identity changed")
            # The dispatcher can disappear after Popen but before publication.
            # Claim our exact birth before any slow preflight/dependency work.
            record = _claim_worker(folder, binding, _self_identity())
            if record is None:
                return 0
            binding["worker"] = record["worker"]
            verify(Path(record["runtime"]), record["source_sha256"])
            if installed_dependencies() != record["dependencies"]:
                raise ValueError("Frozen dependency identity changed before execution")
            config = folder / "run.toml"
            if digest(config.read_bytes()) != record["config_sha256"]:
                raise ValueError("Frozen configuration changed before execution")
            kind = record.get("command_kind", "train")
            if kind not in {"train", "campaign"}:
                raise ValueError("Frozen job command is invalid")
            adapter = record.get("adapter")
            if adapter is not None:
                if kind != "campaign":
                    raise ValueError("Only campaign jobs can select an adapter")
                _adapter_checkpoint(config, adapter["path"], adapter["manifest_sha256"])
            env = dict(os.environ)
            env["SHELLSMITH_ROOT"] = record["runtime"]
            env["PYTHONPATH"] = str(Path(record["runtime"]) / "src")
            env["PATH"] = str(Path(record["python"]).parent) + os.pathsep + env.get("PATH", "")
            env["SHELLSMITH_GPU_LEASE_DIR"] = str(Path("~/.local/state/shellsmith/gpu-lease").expanduser())
            env["HF_HUB_OFFLINE"] = "1"
            env["TOKENIZERS_PARALLELISM"] = "false"
            env["MLX_ENABLE_TF32"] = "0"
            if record.get("execution_policy", {"MLX_ENABLE_TF32": "0"}) != {"MLX_ENABLE_TF32": "0"}:
                raise ValueError("Frozen MLX execution policy is incompatible")
            # A source revision owns its locked Pi dependency tree. Package locks
            # are frozen source; numerical/model dependencies are never installed here.
            vat = Path(record["runtime"]) / "src/shellsmith/episode/vat_runner"
            if (vat / "package-lock.json").is_file():
                with lock(home / "runtime-dependencies.lock"):
                    marker = vat / "node_modules/.shellsmith-lock-sha256"
                    wanted = digest((vat / "package-lock.json").read_bytes())
                    if not marker.exists() or marker.read_text() != wanted:
                        if interrupted or read(folder / "job.json").get("stop_requested"):
                            _update_attempt(folder, binding, status="stopped", updated=now(), finished=now())
                            return 0
                        if _update_attempt(folder, binding, phase="preparing_dependencies", updated=now()) is None:
                            return 0
                        prepare = ["npm", "ci", "--ignore-scripts", "--no-audit", "--no-fund"]
                        child = subprocess.Popen(prepare, cwd=vat, env=env, stdin=subprocess.DEVNULL,
                                                 close_fds=True, start_new_session=True)
                        code, stopped = _wait_child(child, folder, lambda: interrupted, timeout_seconds=600,
                                                    binding=binding)
                        if stopped:
                            _update_attempt(folder, binding, status="stopped", exit_code=code, updated=now(), finished=now())
                            return code
                        if code:
                            raise subprocess.CalledProcessError(code, prepare)
                        marker.write_text(wanted)
            if interrupted or read(folder / "job.json").get("stop_requested"):
                _update_attempt(folder, binding, status="stopped", updated=now(), finished=now())
                return 0
            command = [record["python"], "-m", "shellsmith", kind, "--config", str(config),
                       "--log-dir", str(folder / "run")]
            if adapter is not None:
                command.extend(["--adapter-path", adapter["path"]])
            record = _update_attempt(folder, binding, phase="launching", updated=now(), command=command)
            if record is None:
                return 0
            binding = launch_binding(record)
            gate_reader, start_gate = os.pipe()
            try:
                launcher = [record["python"], "-m", "shellsmith.jobs.trainer_startup",
                            "--home", str(home), "--job-id", job_id, "--binding", json.dumps(binding),
                            "--gate-fd", str(gate_reader)]
                child = subprocess.Popen(launcher, cwd=record["runtime"], env=env, stdin=subprocess.DEVNULL,
                                         close_fds=True, pass_fds=(gate_reader,), start_new_session=True)
            finally:
                os.close(gate_reader)
            gate = start_gate
            start_gate = None  # _wait_child owns and closes the writer from here.
            code, stopped = _wait_child(child, folder, lambda: interrupted, executing=True,
                                        start_gate=gate, binding=binding)
            _update_attempt(folder, binding, status="stopped" if stopped else ("completed" if code == 0 else "failed"),
                            exit_code=code, updated=now(), finished=now())
            return code
        except BaseException as error:
            if start_gate is not None:
                os.close(start_gate)
            if child and child.poll() is None:
                identity = process_identity(child.pid)
                if alive(identity):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            _update_attempt(folder, binding, status="failed", reason=f"{type(error).__name__}: {error}", updated=now())
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    return run(args.home, args.job_id)


if __name__ == "__main__":
    raise SystemExit(main())
