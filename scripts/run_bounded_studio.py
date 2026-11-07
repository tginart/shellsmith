#!/usr/bin/env python3
"""Own one diagnostic process group with separate startup/execution deadlines."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    ready_option = parser.add_mutually_exclusive_group(required=True)
    ready_option.add_argument("--ready-file")
    ready_option.add_argument("--ready-immediate", action="store_true")
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--startup-seconds", type=float, default=300)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--graceful-signal", choices=("term", "int"), default="term")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    ready, receipt = Path(args.ready_file) if args.ready_file else None, Path(args.receipt)
    if (ready is not None and ready.exists()) or receipt.exists():
        raise RuntimeError("Use fresh diagnostic receipt paths; an existing run is never restarted.")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    child = subprocess.Popen(command, start_new_session=True)
    started = time.monotonic();activated = started if args.ready_immediate else None;reason = None
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        while child.poll() is None:
            now = time.monotonic()
            if activated is None and ready is not None and ready.exists():
                activated = now
            if (activated is None and now - started > args.startup_seconds) or (
                activated is not None and now - activated > args.seconds
            ):
                reason = "startup_deadline" if activated is None else "execution_deadline"
                break
            time.sleep(.25)
    except KeyboardInterrupt:
        reason = "interrupted"
    finally:
        if reason is None and activated is None:
            if ready is not None and ready.exists():
                activated = time.monotonic()
            else:
                reason = "startup_exit"
        if child.poll() is None:
            try:os.killpg(child.pid, signal.SIGINT if args.graceful_signal == "int" else signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL);child.wait()
        # The leader may exit while a descendant still owns the session. Reap
        # that explicitly owned group on successful exits as well as deadlines.
        try:os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:pass
        receipt.parent.mkdir(parents=True, exist_ok=True)
        value = {"pid": child.pid, "returncode": child.returncode, "stop_reason": reason,
                 "elapsed_seconds": time.monotonic() - started,
                 "execution_seconds": None if activated is None else time.monotonic() - activated,
                 "execution_limit_seconds": args.seconds}
        temporary = receipt.with_suffix(".tmp")
        with temporary.open("w") as file:json.dump(value, file, indent=2);file.flush();os.fsync(file.fileno())
        os.replace(temporary, receipt)
        print(json.dumps(value), flush=True)
    raise SystemExit(0 if child.returncode == 0 and reason is None else 1)


if __name__ == "__main__":
    main()
