"""
Purpose: Build a read-only graph auditor against the pinned local MLX runtime.
Input: A nanobind v3.0.1 checkout and a fresh output directory.
Output: A local extension and reproducible compiler/source identity records.
Note: This command installs no package and changes no runtime file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

NANOBIND_COMMIT = "db4827f06f6f1680e5d4004c95fc8d69299dba8b"


def build(args: argparse.Namespace) -> None:
    """Compile the pinned binding ABI into a fresh, external build directory."""
    if sys.platform != "darwin" or importlib.metadata.version("mlx") != "0.32.3":
        raise RuntimeError(
            "The audit extension requires the pinned Mac MLX 0.32.3 runtime."
        )
    commit = subprocess.check_output(
        ["git", "-C", str(args.nanobind_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != NANOBIND_COMMIT:
        raise ValueError("The auditor requires the recorded nanobind v3.0.1 commit.")
    spec = importlib.util.find_spec("mlx")
    mlx = Path(next(iter(spec.submodule_search_locations)))
    source = (
        Path(__file__).resolve().parents[1]
        / "src/studio_compute/numerics/audit/audit.cpp"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    flags = [
        "/usr/bin/clang++",
        "-std=c++20",
        "-O2",
        "-fPIC",
        "-fvisibility=hidden",
        "-fno-strict-aliasing",
        "-DNB_COMPACT_ASSERTIONS",
        "-DNB_DOMAIN=mlx",
        "-I" + sysconfig.get_paths()["include"],
        "-I" + str(args.nanobind_source / "include"),
        "-I" + str(args.nanobind_source / "ext/robin_map/include"),
        "-I" + str(mlx / "include"),
    ]
    module = args.output / (
        "_shellsmith_graph_audit" + sysconfig.get_config_var("EXT_SUFFIX")
    )
    commands = [
        [*flags, "-c", str(source), "-o", str(args.output / "audit.o")],
        [
            *flags,
            "-DNB_BUILD",
            "-c",
            str(args.nanobind_source / "src/nb_combined.cpp"),
            "-o",
            str(args.output / "nanobind.o"),
        ],
        [
            "/usr/bin/clang++",
            "-shared",
            "-undefined",
            "dynamic_lookup",
            str(args.output / "audit.o"),
            str(args.output / "nanobind.o"),
            "-L" + str(mlx / "lib"),
            "-lmlx",
            "-Wl,-rpath," + str(mlx / "lib"),
            "-o",
            str(module),
        ],
    ]
    identity = {
        "mlx": "0.32.3",
        "nanobind_commit": commit,
        "commands": commands,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "runtime_library_sha256": hashlib.sha256(
            (mlx / "lib/libmlx.dylib").read_bytes()
        ).hexdigest(),
    }
    for command in commands:
        subprocess.run(command, check=True, timeout=120)
    identity["extension_sha256"] = hashlib.sha256(module.read_bytes()).hexdigest()
    (args.output / "build-identity.json").write_text(
        json.dumps(identity, indent=2) + "\n"
    )
    print(module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nanobind-source",
        type=Path,
        required=True,
        help="Pinned v3.0.1 checkout with its robin-map submodule.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh directory outside the installed runtime.",
    )
    build(parser.parse_args())
