"""
Purpose: Build a read-only graph reader outside the qualified MLX environment.
Input: A new build folder and the graph-reader C++ source.
Output: A native module and source, compiler, package, and binary identities.
Note: Only the temporary target receives the pinned nanobind build package.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import sysconfig
import time
from pathlib import Path


def build(root: Path, source: Path) -> None:
    """
    Purpose: Compile graph metadata access against the installed MLX headers.
    Input: New build directory and source. Output: An isolated native module.
    """
    root.mkdir(exist_ok=False)
    started = time.time()
    assert importlib.metadata.version("mlx") == "0.32.3"
    packages = root / "packages"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--disable-pip-version-check",
            "install",
            "--target",
            str(packages),
            "--no-deps",
            "nanobind==3.0.1",
        ],
        check=True,
        timeout=120,
    )
    nanobind = packages / "nanobind"
    mlx = Path(importlib.metadata.distribution("mlx-metal").locate_file("mlx"))
    extension = root / ("_mlx_graph_reader" + sysconfig.get_config_var("EXT_SUFFIX"))
    compiler = subprocess.check_output(
        ["xcrun", "--find", "clang++"], text=True
    ).strip()
    sdk = subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
    ).strip()
    command = [
        compiler,
        "-isysroot",
        sdk,
        "-O2",
        "-std=c++20",
        "-shared",
        "-fPIC",
        "-fvisibility=hidden",
        "-undefined",
        "dynamic_lookup",
        "-DNB_DOMAIN=mlx",
        "-DNB_COMPACT_ASSERTIONS",
        "-I" + sysconfig.get_path("include"),
        "-I" + str(nanobind / "include"),
        "-I" + str(nanobind / "ext/robin_map/include"),
        "-I" + str(mlx / "include"),
        str(source),
        str(nanobind / "src/nb_combined.cpp"),
        "-L" + str(mlx / "lib"),
        "-lmlx",
        "-Wl,-rpath," + str(mlx / "lib"),
        "-o",
        str(extension),
    ]
    subprocess.run(command, check=True, timeout=120)
    receipt = {
        "status": "passed",
        "started_at": started,
        "finished_at": time.time(),
        "compiler": subprocess.check_output([compiler, "--version"], text=True),
        "command": command,
        "python": sys.version,
        "mlx_version": importlib.metadata.version("mlx"),
        "nanobind_version": "3.0.1",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "extension": str(extension),
    }
    (root / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="New isolated build folder."
    )
    parser.add_argument(
        "--source", type=Path, required=True, help="Read-only graph-reader source."
    )
    args = parser.parse_args()
    build(args.root, args.source)
