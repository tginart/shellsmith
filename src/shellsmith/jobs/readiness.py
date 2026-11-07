"""Bounded Studio probes without loading model weights or changing installations."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

PINS = {"mlx": "0.32.3", "mlx-lm": "0.32.0", "mlx-vlm": "0.7.4", "transformers": "5.17.0"}


def installed_dependencies() -> dict:
    return {item.metadata["Name"].lower(): item.version for item in importlib.metadata.distributions()}


def executable_version(executable: str) -> str | None:
    resolved = shutil.which(executable)
    if not resolved:
        return None
    try:
        result = subprocess.run([resolved, "--version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def probe() -> dict:
    from shellsmith.mlx.readiness import MIN_PHYSICAL_MEMORY_BYTES

    blockers = []
    gm_python_runtime = None
    try:
        from shellsmith.episode.gm_process.python_runtime import current_python_runtime
        runtime = current_python_runtime()
        runtime.probe()
        gm_python_runtime = {"executable": str(runtime.executable), "prefix": str(runtime.prefix)}
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        blockers.append(f"Confined GM Python runtime is unavailable: {error}")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        blockers.append("MLX Studio jobs require an Apple Silicon Mac")
    macos = platform.mac_ver()[0]
    if not macos or int(macos.split(".")[0]) < 26:
        blockers.append("Pinned MLX wheels require macOS 26 or later")
    if sys.version_info[:2] != (3, 12):
        blockers.append("Use the dedicated Python 3.12 training environment")
    deps = installed_dependencies()
    for name, wanted in PINS.items():
        if deps.get(name) != wanted:
            blockers.append(f"{name} must be {wanted}; found {deps.get(name, 'missing')}")
    metal = False
    if deps.get("mlx") == PINS["mlx"]:
        try:
            import mlx.core as mx
            metal = mx.metal.is_available()
        except (ImportError, RuntimeError):
            pass
    if not metal:
        blockers.append("MLX Metal is unavailable")
    node, codex = executable_version("node"), executable_version("codex")
    if node != "v22.22.0":
        blockers.append(f"Node must be v22.22.0; found {node or 'missing'}")
    if codex != "codex-cli 0.159.2":
        blockers.append(f"Codex must be 0.159.2; found {codex or 'missing'}")
    memory = None
    if platform.system() == "Darwin":
        try:
            memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=5))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    if type(memory) is not int or memory < MIN_PHYSICAL_MEMORY_BYTES:
        blockers.append("MLX v1 requires at least 256 GiB physical unified memory (274877906944 bytes); "
                        f"detected {memory if memory is not None else 'unknown'}. Use a qualified Studio.")
    return {"ready": not blockers, "blockers": blockers, "python": sys.executable,
            "python_version": platform.python_version(), "system": platform.system(),
            "machine": platform.machine(), "macos": macos, "metal": metal,
            "memory_bytes": memory, "node": node, "codex": codex,
            "dependencies": deps, "uid": os.getuid(), "gm_python_runtime": gm_python_runtime}
