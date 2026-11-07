"""
Purpose: Run a bounded full-lifecycle smoke test in one prepared smoke root.
Input: A root from prepare_root.py with train steps = 2, and the interrupt point.
Output: smoke-result.json with process identities, phase times, and checks.
Method:
    1. Start the service and the trainer as the epoch supervisor does.
    2. Wait for update 1 (or only the initial checkpoint) and for new rollouts.
    3. Interrupt the owned trainer and service process groups.
    4. Restart both. Wait for a restore and for rollouts to resume.
    5. Check that completed work is not repeated, then stop both.
Note: Only process groups that this script creates are signaled.
"""

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PYTHON = str(Path.home() / ".local/share/shellsmith/studio-v1/env/bin/python")


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    plan = json.loads((root / "epoch-plan.json").read_text())
    env = {**os.environ, "PYTHONPATH": str(root / "source/src"),
           "CODEX_HOME": str(Path.home() / ".codex"),
           "SHELLSMITH_GM_AUTH_GUARD": str(root / "codex-account.json")}
    result = {"root": str(root), "started": time.time(), "phases": {}}
    try:
        service, trainer = _start(root, plan, env, result, label="first")
        updates = 1 if args.interrupt_after == "update" else 0
        _wait(lambda: _state(root).get("rft_updates_completed", 0) >= updates
              and _count(root, "publish") >= updates + 1, timeout_s=4 * 3600,
              label=f"update {updates}")
        result["phases"][f"update_{updates}_committed"] = time.time()
        published = _max_id(root)
        _wait(lambda: _count(root, "sample", after=published) >= args.samples,
              timeout_s=3600, label="rollouts after update 1")
        result["phases"]["second_collection_started"] = time.time()
        before = _snapshot(root)
        _stop(trainer), _stop(service)
        result["phases"]["interrupted"] = time.time()
        service, trainer = _start(root, plan, env, result, label="second")
        restarted = before["max_id"]
        _wait(lambda: _count(root, "restore", after=restarted) >= 1,
              timeout_s=1800, label="restore")
        result["phases"]["restored"] = time.time()
        _wait(lambda: _count(root, "sample", after=restarted) >= args.samples,
              timeout_s=3600, label="rollouts after restore")
        result["phases"]["resumed"] = time.time()
        after = _snapshot(root)
        result["checks"] = {
            "before": before, "after": after,
            "create_not_repeated": after["kinds"].get("create", 0) == 1,
            "updates_not_repeated": (after["kinds"].get("forward_backward", 0)
                                     == before["kinds"].get("forward_backward", 0)
                                     == updates),
            "optimizer_not_repeated": (after["kinds"].get("optim_step", 0)
                                       == before["kinds"].get("optim_step", 0)
                                       == updates),
            "restored": after["kinds"].get("restore", 0) >= 1,
            "state_kept": (after["state"].get("rft_updates_completed")
                           == before["state"].get("rft_updates_completed") == updates),
            "offset_kept": (after["state"].get("next_world_offset")
                            == before["state"].get("next_world_offset")),
        }
        result["passed"] = all(result["checks"][key] for key in result["checks"]
                               if key not in ("before", "after"))
        _stop(trainer), _stop(service)
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for process in result.pop("_owned", []):
            _stop(process)
        result["finished"] = time.time()
        (root / "smoke-result.json").write_text(json.dumps(result, indent=1, default=str))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Prepared smoke root.")
    parser.add_argument("--samples", type=int, default=16,
                        help="New sample operations that show resumed rollouts.")
    parser.add_argument("--interrupt-after", choices=("update", "rollouts"),
                        default="update",
                        help="Interrupt after update 1, or during the first rollouts.")
    return parser.parse_args()


def _start(root: Path, plan: dict, env: dict, result: dict, *, label: str) -> tuple:
    """Start the service, check its identity, then start the trainer."""
    source = root / "source"
    log = (root / "service.log").open("a")
    service = subprocess.Popen(
        [PYTHON, "-u", "-m", "studio_compute.cli", "serve", "--config",
         str(root / "service.toml")], cwd=source, env=env, stdout=log,
        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    result.setdefault("_owned", []).append(service)
    result["phases"][f"{label}_service_started"] = time.time()
    result[f"{label}_service"] = _identity(service.pid)
    capabilities = _wait(lambda: _capabilities(plan["endpoint"]), timeout_s=1800,
                         label="service capabilities", process=service)
    result["phases"][f"{label}_service_ready"] = time.time()
    sys.path.insert(0, str(source / "scripts"))
    from run_studio_epoch import verify_runtime_sources
    verify_runtime_sources(capabilities, source)
    assert capabilities["max_inference_batch"] == plan["inference_concurrency"]
    (root / f"capabilities-{label}.json").write_text(json.dumps(capabilities))
    command = [PYTHON, "-u", "-m", "shellsmith", "train", "--config",
               str(root / "train.toml")]
    with (root / "trainer.log").open("a") as trainer_log:
        subprocess.run([*command, "--dry-run"], cwd=source, env=env, stdout=trainer_log,
                       stderr=subprocess.STDOUT, timeout=300, check=True)
        trainer = subprocess.Popen(command, cwd=source, env=env, stdout=trainer_log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
    result["_owned"].append(trainer)
    result[f"{label}_trainer"] = _identity(trainer.pid)
    result["phases"][f"{label}_trainer_started"] = time.time()
    return service, trainer


def _wait(condition, *, timeout_s: float, label: str, process=None):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Owned process exited while waiting for {label}.")
        value = condition()
        if value:
            print(json.dumps({"reached": label, "at": time.time()}), flush=True)
            return value
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for {label}.")


def _capabilities(endpoint: str):
    try:
        with urllib.request.urlopen(endpoint + "/v1/capabilities", timeout=5) as response:
            return json.loads(response.read())
    except OSError:
        return None


def _journal(root: Path):
    path = root / "service/journal.sqlite3"
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True) if path.exists() else None


def _count(root: Path, kind: str, *, after: int = 0) -> int:
    db = _journal(root)
    if db is None:
        return 0
    with db:
        return db.execute("select count(*) from operations where kind=? and "
                          "status='succeeded' and rowid>?", (kind, after)).fetchone()[0]


def _max_id(root: Path) -> int:
    with _journal(root) as db:
        return db.execute("select coalesce(max(rowid), 0) from operations").fetchone()[0]


def _snapshot(root: Path) -> dict:
    with _journal(root) as db:
        kinds = dict(db.execute("select kind, count(*) from operations where "
                                "status='succeeded' group by kind").fetchall())
    return {"kinds": kinds, "max_id": _max_id(root), "state": _state(root)}


def _state(root: Path) -> dict:
    path = root / "train/trainer_state.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _identity(pid: int) -> dict:
    line = subprocess.check_output(["/bin/ps", "-p", str(pid), "-o", "lstart=",
                                    "-o", "command="], text=True).strip()
    return {"pid": pid, "ps_start_and_command": line}


def _stop(process) -> None:
    """Stop one owned process group as run_studio_epoch.stop_owned does."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=120)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


if __name__ == "__main__":
    main()
