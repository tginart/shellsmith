"""Standalone Studio readiness uses the same capacity floor as MLX admission."""
import subprocess
import sys
from types import SimpleNamespace

import pytest

from shellsmith.jobs import readiness
from shellsmith.mlx.readiness import MIN_PHYSICAL_MEMORY_BYTES


@pytest.mark.parametrize("capacity", [MIN_PHYSICAL_MEMORY_BYTES, MIN_PHYSICAL_MEMORY_BYTES + 1,
                                      MIN_PHYSICAL_MEMORY_BYTES - 1, 128 * 1024**3,
                                      "invalid", None])
def test_standalone_probe_requires_qualified_memory(monkeypatch, capacity):
    from shellsmith.episode.gm_process import python_runtime
    monkeypatch.setattr(python_runtime, "current_python_runtime", lambda: SimpleNamespace(
        probe=lambda: None, executable="/fixture/python", prefix="/fixture"))
    monkeypatch.setattr(readiness.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(readiness.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(readiness.platform, "mac_ver", lambda: ("27.0", (), ""))
    monkeypatch.setattr(readiness, "sys", SimpleNamespace(version_info=(3, 12), executable=sys.executable))
    monkeypatch.setattr(readiness, "installed_dependencies", lambda: dict(readiness.PINS))
    monkeypatch.setattr(readiness, "executable_version", lambda name: {
        "node": "v22.22.0", "codex": "codex-cli 0.159.2"}[name])
    core = SimpleNamespace(metal=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=core))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    def memory(*args, **kwargs):
        if capacity is None:
            raise subprocess.TimeoutExpired("sysctl", 5)
        return str(capacity)
    monkeypatch.setattr(readiness.subprocess, "check_output", memory)
    report = readiness.probe()
    qualified = type(capacity) is int and capacity >= MIN_PHYSICAL_MEMORY_BYTES
    assert report["ready"] is qualified, report["blockers"]
    if not qualified:
        assert any("256 GiB physical unified memory" in item for item in report["blockers"])
