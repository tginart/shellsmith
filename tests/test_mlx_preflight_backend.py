"""A config-aware Studio check admits only documents it can submit."""
import subprocess
from types import SimpleNamespace

import pytest

from shellsmith.jobs import readiness as studio
from shellsmith.mlx import readiness as native
from shellsmith.train import harness_identity, preflight


@pytest.mark.parametrize("backend", [None, "tinker", "typo"])
def test_preflight_rejects_non_mlx_before_host_or_model_probes(monkeypatch, backend):
    def forbidden(*args, **kwargs):
        pytest.fail("Non-MLX Studio config reached host/model work")
    monkeypatch.setattr(studio, "probe", forbidden)
    monkeypatch.setattr(native, "readiness", forbidden)
    document = {} if backend is None else {"compute": {"backend": backend}}
    report = preflight.probe_document(document)
    assert report["ready"] is False
    assert any('compute.backend = "mlx"' in item for item in report["blockers"])


def test_preflight_accepts_explicit_mlx_and_keeps_all_readiness_checks(monkeypatch):
    effects = []
    monkeypatch.setattr(studio, "probe", lambda: effects.append("host") or {"ready": True, "blockers": []})
    monkeypatch.setattr(native, "readiness", lambda settings: effects.append("model") or {
        "blockers": [], "profile": "qwen38_dense_8bit_32k", "model": {"frozen": True}})
    monkeypatch.setattr(harness_identity, "pi_sdk_identity", lambda path: {"module_sha256": "sdk"})
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/fixture/codex")
    monkeypatch.setattr(preflight, "subprocess", SimpleNamespace(
        run=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0), TimeoutExpired=subprocess.TimeoutExpired))
    report = preflight.probe_document({"compute": {"backend": "mlx"}})
    assert report["ready"] and report["codex_authenticated"] and report["model"]["frozen"]
    assert effects == ["host", "model"]
