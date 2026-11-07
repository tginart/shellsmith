"""Restore acquisition must not roll back another live client's gradients."""

import asyncio
from dataclasses import replace
import threading

import pytest

from shellsmith.train import studio
from shellsmith.train.config import TrainConfig, RunSection, ComputeSection
from studio_compute.client import OperationError
from studio_compute.http import server
from studio_compute.service import Service
from test_studio_service import Engine, call, batch


def test_live_restore_conflict_preserves_owner_and_exact_retry(tmp_path):
    service = Service(tmp_path, Engine)
    try:
        created = call(service, "a", "create", {}, "create")
        call(service, "a", "forward_backward", batch(), "owner-back")
        previous = dict(service.clients["a"])
        service.submit("intruder", "a", "restore", {"checkpoint": created["checkpoint"]})
        assert service.wait("intruder")["status"] == "failed"
        assert "client is live" in service.operation("intruder")["error"]
        assert service.engine.clients["a"]["gradient"] == "owner-back"
        assert service.clients["a"] == previous and service.live_clients == {"a"}
        result = call(service, "a", "optim_step", {"learning_rate": 1.0, "backward_operation": "owner-back"}, "owner-step")
        assert result["optim_metrics"]["optimizer_steps"] == 1
        call(service, "a", "suspend", {}, "release")
        restored = call(service, "a", "restore", {"checkpoint": created["checkpoint"]}, "restore")
        assert call(service, "a", "restore", {"checkpoint": created["checkpoint"]}, "restore") == restored
    finally:
        service.close()


@pytest.mark.parametrize("mode", ["new", "closed", "suspended", "restart"])
def test_restore_can_acquire_unowned_or_released_client(tmp_path, mode):
    service = Service(tmp_path, Engine)
    try:
        created = call(service, "a", "create", {"seed": 7}, "create")
        if mode in ("closed", "suspended"):
            call(service, "a", "close" if mode == "closed" else "suspend", {}, "release")
        if mode == "restart":
            service.close(); service = Service(tmp_path, Engine)
            assert not service.live_clients
        target = "b" if mode == "new" else "a"
        result = call(service, target, "restore", {"checkpoint": created["checkpoint"]}, "restore")
        assert result["revision"] == 0 and target in service.live_clients
        assert service.engine.clients[target]["weight"] == 7
        call(service, target, "forward_backward", batch(), "back")
        assert call(service, target, "optim_step", {"learning_rate": 1.0, "backward_operation": "back"}, "step")["optim_metrics"]["optimizer_steps"] == 1
    finally:
        service.close()


def test_backend_only_releases_its_acquired_client_for_controlled_restore(tmp_path, monkeypatch):
    class CapabilitiesEngine(Engine):
        capabilities = {**Engine.capabilities, "model": "Qwen/Qwen3.8-27B", "context_window": 262144}
    service = Service(tmp_path, CapabilitiesEngine)
    http = server(service, ("127.0.0.1", 0))
    thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
    monkeypatch.setattr(studio, "load_renderer", lambda *a, **k: None)
    config = TrainConfig(run=RunSection(run_id="owner", log_dir=str(tmp_path / "run")), compute=ComputeSection(backend="studio"))
    config = replace(config, studio=replace(config.studio, endpoint=f"http://127.0.0.1:{http.server_port}"))
    try:
        owner, intruder = studio.StudioBackend(config), studio.StudioBackend(config)
        original = asyncio.run(owner.initialize())
        call(service, "owner", "forward_backward", batch(), "back")
        with pytest.raises(ValueError, match="1–200"):
            asyncio.run(owner.restore_state(original["checkpoint"], request_id=""))
        assert service.engine.clients["owner"]["gradient"] == "back"
        with pytest.raises(RuntimeError, match="owned by another live initialization"):
            asyncio.run(intruder.initialize({"path": original["checkpoint"]}))
        intruder.close()
        assert service.store.db.execute("SELECT COUNT(*) FROM operations WHERE kind='suspend'").fetchone()[0] == 0
        assert service.engine.clients["owner"]["gradient"] == "back"
        call(service, "owner", "optim_step", {"learning_rate": 1.0, "backward_operation": "back"}, "step")
        assert asyncio.run(owner.restore_state(original["checkpoint"], request_id="controlled-restore"))["revision"] == 0
        assert service.store.db.execute("SELECT COUNT(*) FROM operations WHERE kind='suspend'").fetchone()[0] == 1
        assert asyncio.run(owner.restore_state(original["checkpoint"], request_id="controlled-restore"))["revision"] == 0
        assert service.store.db.execute("SELECT COUNT(*) FROM operations WHERE kind='suspend'").fetchone()[0] == 1
        assert service.live_clients == {"owner"}
        owner.close()
        assert not service.live_clients
    finally:
        http.shutdown(); http.server_close(); thread.join(); service.close()
