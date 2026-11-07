"""
Purpose: Run the native BF16 derivative and lifecycle suite under the Studio GPU lease.
Input: A pinned MLX checkout and a fresh private evidence folder.
Output: A pytest report and a bounded worker receipt.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from mlx_hardware_roofline import supervise

TESTS = (
    "test_saved_norm.py",
    "test_bf16_token_decode.py",
    "test_bf16_rotary.py",
    "test_bf16_inference_cache.py",
    "test_gdn_sequential.py",
    "test_saved_layers.py",
    "test_attention_metal.py",
    "test_gdn_metal.py",
    "test_bf16_profile.py",
    "test_studio_numerics.py",
)


def run(args: argparse.Namespace) -> None:
    """Run native qualification without model assets or persistent optimizer changes."""
    started = time.time()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *["tests/" + name for name in TESTS],
            "--junitxml",
            str(args.output / "tests.xml"),
        ]
    )
    report = {
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "tests": TESTS,
        "started_at": started,
        "finished_at": time.time(),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh private evidence directory."
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=600, help="Hard limit, at most one hour."
    )
    parser.add_argument(
        "--lease",
        default=str(Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"),
        help="Existing exclusive Studio GPU lease.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    assert 0 < args.timeout_seconds <= 3600
    if args.worker:
        run(args)
    else:
        supervise(args, script=Path(__file__))
