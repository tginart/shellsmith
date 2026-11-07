"""
Purpose: Capture real service counters and require measured backward progress.
Input: A loopback Studio metrics endpoint and a fresh output file.
Output: CPU counter samples for the Backprop qualification validator.
Note: This command submits no model operation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib.error import URLError
from urllib.request import urlopen


BACKWARD_PHASES = (
    "backward", "head_backward", "replay_branch_backward", "replay_shared_backward",
)


def has_backward_progress(row: dict) -> bool:
    """
    Purpose: Require a numerical backward phase or its measured elapsed counter.
    Input: A service metrics sample with operation progress receipts.
    Output: True only if a forward_backward operation has executed backward work.
    """
    for operation in row.get("training_operations", {}).values():
        if operation["kind"] != "forward_backward":
            continue
        progress = operation.get("progress") or {}
        if progress.get("phase") in BACKWARD_PHASES:
            return True
        if any(progress.get("phase_seconds", {}).get(phase, 0) > 0
               for phase in BACKWARD_PHASES):
            return True
    return False


def observe(endpoint: str, *, output: Path, timeout_seconds: float = 480) -> dict:
    """
    Purpose: Observe one complete backward operation through read-only HTTP calls.
    Input: The endpoint, fresh output path, and maximum observation duration.
    Output: Persisted samples, or an error if no backward progress appears.
    """
    if output.exists():
        raise FileExistsError(output)
    samples = []
    saw_backward = False
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urlopen(endpoint.rstrip("/") + "/v1/metrics", timeout=2) as response:
                row = json.load(response)
            samples.append(row)
            saw_backward = saw_backward or has_backward_progress(row)
            if saw_backward and not row.get("training_operations"):
                break
        except (URLError, TimeoutError, ConnectionError):
            # Startup and restart windows are expected within this bounded probe.
            pass
        time.sleep(.2)
    result = {"scope": "CPU observation; no model operation submitted",
              "samples": samples, "saw_backward": bool(saw_backward)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    assert saw_backward, "No live backward progress was captured before the deadline."
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8773",
                        help="The qualification service loopback endpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Fresh sample JSON file.")
    parser.add_argument("--timeout-seconds", type=float, default=480,
                        help="Maximum CPU observation duration, in seconds.")
    args = parser.parse_args()
    result = observe(args.endpoint, output=args.output, timeout_seconds=args.timeout_seconds)
    print(json.dumps({"samples": len(result["samples"]), "saw_backward": result["saw_backward"]}))
