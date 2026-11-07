"""Concrete MLX readiness, with no weight loading or run-directory mutation."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any


def probe_document(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("compute", {}).get("backend", "tinker") != "mlx":
        return {"ready": False, "blockers": [
            'Studio readiness requires compute.backend = "mlx"; use train/campaign for Tinker.']}

    from shellsmith.jobs.readiness import probe
    from shellsmith.mlx.readiness import readiness
    from shellsmith.train.harness_identity import pi_sdk_identity

    report = probe()
    numerical = readiness(document.get("mlx", {}))
    blockers = list(dict.fromkeys(report["blockers"] + numerical["blockers"]))
    sdk = pi_sdk_identity(document.get("player", {}).get("pi_sdk_module_path", ""))
    if sdk["module_sha256"] is None:
        blockers.append("Install the lockfile-pinned Pi SDK before starting live episodes.")
    executable = shutil.which("codex")
    authenticated = False
    if executable:
        try:
            login = subprocess.run([executable, "login", "status"], capture_output=True, timeout=15)
            authenticated = login.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            pass
    if not authenticated:
        blockers.append("Authorize Codex on the Studio with 'codex login --device-auth'.")
    return report | {"ready": not blockers, "blockers": blockers, "model": numerical.get("model", {}),
                     "profile": numerical["profile"], "codex_authenticated": authenticated,
                     "pi_sdk": sdk}
