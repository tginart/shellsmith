"""Incompatible service capabilities cannot poison an immutable run snapshot."""

import asyncio
from dataclasses import replace
import threading

import pytest

from shellsmith.train import studio
from shellsmith.train.trainer import run_train, SNAPSHOT_FILE
from studio_compute.http import server
from studio_compute.service import Service
from test_studio_service import Engine
from test_trainer_resume import _runtime


@pytest.mark.parametrize("mismatch", ["model", "context", "cache"])
def test_training_rejects_capability_mismatch_before_snapshot(tmp_path, monkeypatch, mismatch):
    class CapabilitiesEngine(Engine):
        capabilities = {**Engine.capabilities, "model": "Qwen/Qwen3.8-27B", "context_window": 262144}
    service = Service(tmp_path / "service", CapabilitiesEngine)
    http = server(service, ("127.0.0.1", 0))
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        config, _ = _runtime(tmp_path, monkeypatch)
        settings = replace(config.studio, endpoint=f"http://127.0.0.1:{http.server_port}")
        settings = replace(settings, **({"model_id": "mistyped-model"} if mismatch == "model" else {"context_window": 262145} if mismatch == "context" else {"cache_precision": "8bit"}))
        config = replace(config, compute=replace(config.compute, backend="studio"), studio=settings)
        monkeypatch.setattr(studio, "StudioBackend", lambda *a, **k: pytest.fail("No backend setup before compatibility"))
        with pytest.raises(ValueError, match="Configured (model differs|context exceeds|cache precision is unsupported)"):
            asyncio.run(run_train(config))
        assert not (config.resolved_log_dir() / SNAPSHOT_FILE).exists()
        assert not service.clients
        assert service.store.db.execute("SELECT COUNT(*) FROM operations").fetchone()[0] == 0
    finally:
        http.shutdown(); http.server_close(); thread.join(); service.close()
