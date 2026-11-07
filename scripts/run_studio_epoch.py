"""
Purpose: Own one fresh Studio service and normal trainer for one finite hero epoch.
Input: A frozen source snapshot and a verified epoch-plan.json in a fresh run root.
Output: Durable process identities, launch receipts, local logs, and terminal receipts.
Note: A dispatch receipt and an exclusive launch lock prevent automatic duplicate starts.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from studio_compute.storage import atomic_json, file_hash

PINNED_CODEX = (
    str(Path.home() / ".local/share/shellsmith/studio-v1/tools/node_modules/.bin/codex")
)


def process_identity(pid: int) -> dict:
    result = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
        text=True,
        capture_output=True,
        check=True,
    )
    return {"pid": pid, "ps_start_and_command": result.stdout.strip()}


def validate_completion(state: dict, returncode: int) -> None:
    if returncode != 0 or state.get("status") != "complete":
        raise RuntimeError("The normal trainer did not complete successfully.")
    if state.get("next_world_offset") != 177:
        raise RuntimeError("The normal trainer did not exhaust all 177 worlds.")
    if not 0 <= state.get("rft_updates_completed", -1) <= 89:
        raise RuntimeError("The epoch optimizer commit count is invalid.")


def stop_owned(process: subprocess.Popen | None) -> None:
    """
    Purpose: Stop only a process group created by this supervisor and reap its leader.
    Input: Its Popen handle. Output: The owned group has exited or received SIGKILL.
    Note: Other Studio owners and the GPU lease holder are never selected by name.
    """
    if process is None:
        return
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def verify_runtime_sources(capabilities: dict, source: Path) -> None:
    """Require the live service to report the frozen snapshot's runtime files."""
    reported = capabilities["runtime"]["source_sha256"]
    expected = {
        name: file_hash(source / "src/studio_compute" / name) for name in reported
    }
    if reported != expected:
        raise RuntimeError("The live service does not run the frozen source snapshot.")


def preflight(root: Path, plan: dict) -> None:
    """
    Purpose: Verify the frozen source, finite world stream, authentication, and settings.
    Input: The prepared epoch root and plan. Output: An exception if an identity differs.
    Note: This function creates no training client and acquires no GPU allocations.
    """
    from shellsmith.episode.gm_process.auth_identity import command_token, verify_guard
    from shellsmith.jobs.bundle import verify
    from shellsmith.train.config import load_train_config
    from shellsmith.train.trainer import select_world_stream

    source = Path(plan["runtime_root"])
    verify(source, plan["source_sha256"])
    os.chdir(source)
    for filename, key in (
        ("train.toml", "train_config_sha256"),
        ("service.toml", "service_config_sha256"),
    ):
        if file_hash(root / filename) != plan[key]:
            raise ValueError("Prepared config identity changed: " + filename)
    config = load_train_config(root / "train.toml")
    stream = select_world_stream(config)
    ids = [world.world_id for world in stream.worlds]
    if ids != plan["selected_world_ids"] or len(ids) != 177 or len(set(ids)) != 177:
        raise ValueError("The canonical 177-world stream changed.")
    if file_hash(source / config.worlds.hero_slate) != plan["hero_slate_sha256"]:
        raise ValueError("The hero slate identity changed.")
    service = tomllib.loads((root / "service.toml").read_text())["service"]
    if service["checkpoint_retention"] != 2 or service["recovery_state"] != str(
        root / "train/trainer_state.json"
    ):
        raise ValueError("The approved checkpoint recovery policy differs.")
    if config.train.steps != 89 or config.train.groups_per_update != 2:
        raise ValueError("The finite epoch update bounds changed.")
    # Episode workers and GPU decode slots are scheduling settings. The plan
    # declares both; attempts per world are part of the logical work.
    if (
        config.adaptive.initial_attempts,
        config.adaptive.maximum_attempts,
        config.adaptive.parallelism,
        service["inference_concurrency"],
    ) != (8, 8, plan["episode_parallelism"], plan["inference_concurrency"]):
        raise ValueError("The eight-attempt profile or planned slot counts changed.")
    if (
        config.studio.cache_precision != "8bit"
        or service["model_profile"] != "flash-experimental"
    ):
        raise ValueError("The experimental Flash inference profile changed.")
    verify_guard(PINNED_CODEX)
    if not command_token(root / "codex-account.json"):
        raise RuntimeError("The pinned Codex account has no access token.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="The fresh prepared epoch root."
    )
    parser.add_argument(
        "--days",
        type=float,
        default=30,
        help="The maximum wall time for this finite epoch.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify inputs without any process dispatch.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / "epoch-plan.json").read_text())
    preflight(root, plan)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    "source_sha256": plan["source_sha256"],
                    "worlds": 177,
                }
            )
        )
        return
    lock = (root / "launch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (
        (root / "launch.json").exists()
        or (root / "service").exists()
        or (root / "train").exists()
    ):
        raise RuntimeError(
            "This epoch already has launch or state evidence; do not dispatch again."
        )
    receipt = {
        "status": "service_loading",
        "run_id": plan["run_id"],
        "source_sha256": plan["source_sha256"],
        "train_config_sha256": plan["train_config_sha256"],
        "service_config_sha256": plan["service_config_sha256"],
        "supervisor": process_identity(os.getpid()),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "backprop_url": plan["backprop_url"],
        "deadline_days": args.days,
        "checkpoint_retention": 2,
    }
    atomic_json(root / "launch.json", receipt)
    service = trainer = None
    started = time.monotonic()
    # Children must import the frozen snapshot, never the editable release.
    child_env = {**os.environ, "PYTHONPATH": str(Path(plan["runtime_root"]) / "src")}

    def interrupted(*_: object) -> None:
        raise InterruptedError("The epoch supervisor received a stop signal.")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with (root / "service.log").open("a") as log:
            service = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "studio_compute.cli",
                    "serve",
                    "--config",
                    str(root / "service.toml"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=child_env,
            )
        receipt["service"] = process_identity(service.pid)
        atomic_json(root / "launch.json", receipt)
        endpoint = plan["endpoint"]
        while True:
            if service.poll() is not None:
                raise RuntimeError("The owned service exited before readiness.")
            if time.monotonic() - started > 600:
                raise TimeoutError(
                    "The owned service did not become ready in 600 seconds."
                )
            try:
                with urllib.request.urlopen(
                    endpoint + "/v1/capabilities", timeout=5
                ) as response:
                    capabilities = json.load(response)
                break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1)
        if (
            capabilities["max_inference_batch"] != plan["inference_concurrency"]
            or "8bit" not in capabilities["cache_precisions"]
        ):
            raise RuntimeError(
                "Service capabilities do not match the planned 8-bit slot profile."
            )
        verify_runtime_sources(capabilities, Path(plan["runtime_root"]))
        atomic_json(root / "capabilities.json", capabilities)
        command = [
            sys.executable,
            "-u",
            "-m",
            "shellsmith",
            "train",
            "--config",
            str(root / "train.toml"),
        ]
        with (root / "readiness-live.json").open("w") as output:
            subprocess.run(
                [*command, "--dry-run"],
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=300,
                check=True,
                env=child_env,
            )
        with (root / "trainer.log").open("a") as log:
            trainer = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=child_env,
            )
        receipt.update(status="trainer_started", trainer=process_identity(trainer.pid))
        atomic_json(root / "launch.json", receipt)
        while trainer.poll() is None:
            if service.poll() is not None:
                raise RuntimeError("The owned service exited during the epoch.")
            if time.monotonic() - started > args.days * 86400:
                raise TimeoutError("The multi-day epoch deadline expired.")
            state_path = root / "train/trainer_state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                receipt.update(
                    status="running",
                    trainer_progress={
                        key: state.get(key)
                        for key in (
                            "status",
                            "step",
                            "rft_updates_completed",
                            "next_world_offset",
                        )
                    },
                )
                atomic_json(root / "launch.json", receipt)
            time.sleep(5)
        state = json.loads((root / "train/trainer_state.json").read_text())
        validate_completion(state, trainer.returncode)
        receipt.update(
            status="complete",
            trainer_returncode=trainer.returncode,
            trainer_progress={
                key: state.get(key)
                for key in (
                    "status",
                    "step",
                    "rft_updates_completed",
                    "next_world_offset",
                )
            },
        )
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            stop_owned(trainer)
        finally:
            stop_owned(service)
            receipt.update(
                owned_processes_stopped=True, elapsed_seconds=time.monotonic() - started
            )
            atomic_json(root / "launch.json", receipt)
            atomic_json(root / "result.json", receipt)
            lock.close()


if __name__ == "__main__":
    main()
