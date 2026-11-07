"""Stable resume receipts and canonical-output acquisition identity over HTTP."""

import asyncio
from dataclasses import replace
import threading

import pytest

from shellsmith.train import studio, trainer
from shellsmith.train.run_lock import lock_train_run, TrainRunLockedError
from studio_compute.client import OperationError
from studio_compute.http import server
from studio_compute.service import Service
from test_studio_service import Engine, call, batch
from test_trainer_resume import _runtime


@pytest.fixture
def endpoint(tmp_path):
    class TrackedEngine(Engine):
        capabilities = {**Engine.capabilities, "model": "Qwen/Qwen3.8-27B", "context_window": 262144}
        def restore(self, client, folder, state):
            self.events.append(("restore", client))
            return super().restore(client, folder, state)
    service = Service(tmp_path / "service", TrackedEngine)
    http = server(service, ("127.0.0.1", 0))
    thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
    try: yield service, f"http://127.0.0.1:{http.server_port}"
    finally:
        http.shutdown(); http.server_close(); thread.join(); service.close()


def config_for(tmp_path, monkeypatch, url):
    config, _ = _runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(studio, "load_renderer", lambda *a, **k: None)
    return replace(config, compute=replace(config.compute, backend="studio"),
                   studio=replace(config.studio, endpoint=url, training_client="shared"))


def test_lost_resume_response_retries_original_restore_without_daemon_restart(tmp_path, monkeypatch, endpoint):
    service, url = endpoint
    config = config_for(tmp_path, monkeypatch, url)
    original = call(service, "source", "create", {}, "source-create")
    first = studio.StudioBackend(config)
    acall = first.client.acall
    ids = []
    async def lost(kind, body, *, request_id):
        result = await acall(kind, body, request_id=request_id)
        ids.append(request_id)
        raise TimeoutError("injected lost committed restore response")
    monkeypatch.setattr(first.client, "acall", lost)
    checkpoint = {"path": original["checkpoint"]}
    with pytest.raises(TimeoutError, match="lost committed"):
        asyncio.run(first.initialize(checkpoint))
    assert service.operation(ids[0])["status"] == "succeeded"
    assert not first._acquired and "shared" in service.live_clients
    first.close()
    second = studio.StudioBackend(config)
    result = asyncio.run(second.initialize(checkpoint))
    assert result == service.operation(ids[0])["result"]
    assert second._acquired
    assert [event for event in service.engine.events if event == ("restore", "shared")] == [("restore", "shared")]
    assert service.store.db.execute("SELECT COUNT(*) FROM operations WHERE kind='restore' AND client='shared'").fetchone()[0] == 1
    assert service.store.db.execute("SELECT COUNT(*) FROM operations WHERE kind='suspend'").fetchone()[0] == 0
    second.close()
    assert "shared" not in service.live_clients


def test_other_output_cannot_reuse_create_receipt_or_release_owner(tmp_path, monkeypatch, endpoint):
    service, url = endpoint
    config = config_for(tmp_path, monkeypatch, url)
    owner = studio.StudioBackend(config)
    asyncio.run(owner.initialize())
    call(service, "shared", "forward_backward", batch(), "owner-back")
    other = replace(config, run=replace(config.run, log_dir=str(tmp_path / "different-output")))
    intruder = studio.StudioBackend(other)
    assert intruder.operation_prefix != owner.operation_prefix
    with pytest.raises(OperationError, match="Training client already exists"):
        asyncio.run(intruder.initialize())
    assert not intruder._acquired
    intruder.close()
    # The actual trainer path also fails before registering cleanup/baseline work.
    async def forbidden(**kwargs): pytest.fail("Competing coordinator reached baseline publication")
    monkeypatch.setattr(trainer, "save_checkpoint", forbidden)
    with pytest.raises(OperationError, match="Training client already exists"):
        asyncio.run(trainer.run_train(other))
    assert service.store.db.execute("SELECT COUNT(*) FROM operations WHERE kind='suspend'").fetchone()[0] == 0
    assert service.engine.clients["shared"]["gradient"] == "owner-back"
    assert call(service, "shared", "optim_step", {"learning_rate": 1.0, "backward_operation": "owner-back"}, "owner-step")["optim_metrics"]["optimizer_steps"] == 1
    creates = service.store.db.execute("SELECT id,status FROM operations WHERE kind='create'").fetchall()
    assert len(creates) == 2 and len({row["id"] for row in creates}) == 2
    assert {row["status"] for row in creates} == {"succeeded", "failed"}
    owner.close()


def test_canonical_output_alias_is_same_coordinator_and_local_lock_blocks_overlap(tmp_path, monkeypatch, endpoint):
    _, url = endpoint
    config = config_for(tmp_path, monkeypatch, url)
    alias = tmp_path / "alias"; alias.symlink_to(tmp_path, target_is_directory=True)
    alternate = replace(config, run=replace(config.run, log_dir=str(alias / "run")))
    first, second = studio.StudioBackend(config), studio.StudioBackend(alternate)
    assert first.operation_prefix == second.operation_prefix
    with lock_train_run(config.resolved_log_dir()):
        with pytest.raises(TrainRunLockedError):
            asyncio.run(trainer.run_train(alternate))
