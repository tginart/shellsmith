"""Real RPC preflight and cleanup ownership; no GM/model execution."""

import asyncio
from dataclasses import replace
import threading

import pytest

from shellsmith.campaign import inference
from shellsmith.train import studio, trainer
from studio_compute.client import Client, OperationError
from studio_compute.http import server
from studio_compute.protocol import digest
from studio_compute.service import Service
from test_studio_service import Engine, call, batch
from test_trainer_resume import _runtime


@pytest.fixture
def endpoint(tmp_path):
    class CapabilitiesEngine(Engine):
        capabilities = {**Engine.capabilities, "model": "Qwen/Qwen3.8-27B", "context_window": 262144}
    service = Service(tmp_path / "service", CapabilitiesEngine)
    http = server(service, ("127.0.0.1", 0))
    thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
    try:
        yield service, f"http://127.0.0.1:{http.server_port}"
    finally:
        http.shutdown(); http.server_close(); thread.join(); service.close()


def config_for(tmp_path, monkeypatch, url):
    config, _ = _runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(studio, "load_renderer", lambda *a, **k: None)
    return replace(config, compute=replace(config.compute, backend="studio"),
                   studio=replace(config.studio, endpoint=url))


def test_unknown_policy_leaves_campaign_reusable_and_lookup_is_read_only(tmp_path, monkeypatch, endpoint):
    service, url = endpoint
    config = config_for(tmp_path, monkeypatch, url)
    client = Client(url)
    unknown = "studio-policy:" + "f" * 64
    with pytest.raises(ValueError, match="Unknown unpublished policy"):
        asyncio.run(inference.run_inference_campaign(config, sampler_path=unknown))
    assert not config.resolved_log_dir(kind="campaigns").exists()
    assert not service.clients and not service.engine.events
    assert service.store.db.execute("SELECT COUNT(*) FROM operations").fetchone()[0] == 0
    with pytest.raises(ValueError, match="Malformed policy"):
        client.policy_info("studio-policy:bad")
    call(service, "source", "create", {}, "source-create")
    policy = call(service, "source", "publish", {"revision": 0}, "source-publish")["policy"]
    before = service.store.db.execute("SELECT COUNT(*) FROM operations").fetchone()[0]
    allocations = dict(service.engine.clients)
    assert client.policy_info(policy) == {"policy": policy, "revision": 0, "identity": service.engine.identity}
    assert service.store.db.execute("SELECT COUNT(*) FROM operations").fetchone()[0] == before
    assert service.engine.clients == allocations
    class Ready(Exception): pass
    monkeypatch.setattr(inference, "open_rollout_stack", lambda *a, **k: (_ for _ in ()).throw(Ready()))
    with pytest.raises(Ready):
        asyncio.run(inference.run_inference_campaign(config, sampler_path=policy))
    assert (config.resolved_log_dir(kind="campaigns") / inference.SNAPSHOT_FILE).exists()
    assert len(service.clients) == 2


def test_policy_lookup_identity_race_rejects_before_mutation(tmp_path, monkeypatch, endpoint):
    service, url = endpoint
    config = config_for(tmp_path, monkeypatch, url)
    call(service, "source", "create", {}, "source-create")
    policy = call(service, "source", "publish", {"revision": 0}, "source-publish")["policy"]
    original = Client.policy_info
    def changed(client, reference):
        value = original(client, reference)
        return {**value, "identity": {**value["identity"], "numerics": "restarted"}}
    monkeypatch.setattr(Client, "policy_info", changed)
    with pytest.raises(RuntimeError, match="identity changed during policy lookup"):
        asyncio.run(inference.run_inference_campaign(config, sampler_path=policy))
    assert set(service.clients) == {"source"}
    assert not config.resolved_log_dir(kind="campaigns").exists()
    assert service.store.db.execute("SELECT COUNT(*) FROM operations").fetchone()[0] == 2


@pytest.mark.parametrize("mode", ["train", "inference"])
def test_unsupported_cache_rejects_real_entry_before_snapshot_or_mutation(tmp_path, monkeypatch, endpoint, mode):
    service, url = endpoint
    config = config_for(tmp_path, monkeypatch, url)
    config = replace(config, studio=replace(config.studio, cache_precision="8bit"))
    monkeypatch.setattr(studio, "StudioBackend", lambda *a, **k: pytest.fail("No backend allocation before preflight"))
    with pytest.raises(ValueError, match="cache precision is unsupported"):
        asyncio.run(trainer.run_train(config) if mode == "train" else inference.run_inference_campaign(config))
    assert not config.resolved_log_dir(kind="campaigns" if mode == "inference" else "runs").exists()
    assert not service.clients and not service.engine.events
    assert service.store.db.execute("SELECT COUNT(*) FROM operations").fetchone()[0] == 0


@pytest.mark.parametrize("mode", ["train", "inference"])
def test_failed_acquisition_does_not_suspend_live_owner(tmp_path, monkeypatch, endpoint, mode):
    service, url = endpoint
    config = config_for(tmp_path, monkeypatch, url)
    owner = "owner" if mode == "train" else "inference:" + digest({
        "run_id": config.run.run_id, "output": str(config.resolved_log_dir(kind="campaigns").resolve())})
    config = replace(config, studio=replace(config.studio, training_client=owner))
    call(service, owner, "create", {}, "original-owner-create")
    policy = call(service, owner, "publish", {"revision": 0}, "owner-publish")["policy"]
    call(service, owner, "forward_backward", batch(), "owner-back")
    previous = dict(service.clients[owner])
    with pytest.raises(OperationError, match="Training client already exists"):
        asyncio.run(trainer.run_train(config) if mode == "train" else inference.run_inference_campaign(config, sampler_path=policy))
    assert service.clients[owner] == previous
    assert service.engine.clients[owner]["gradient"] == "owner-back"
    assert service.store.db.execute("SELECT COUNT(*) FROM operations WHERE kind='suspend'").fetchone()[0] == 0
    result = call(service, owner, "optim_step", {"learning_rate": 0.001, "backward_operation": "owner-back"}, "owner-step")
    assert result["optim_metrics"]["optimizer_steps"] == 1


def test_successful_acquisition_is_suspended_on_later_training_failure(tmp_path, monkeypatch, endpoint):
    service, url = endpoint
    config = config_for(tmp_path, monkeypatch, url)
    async def fail(**kwargs):
        raise RuntimeError("injected post-acquisition baseline failure")
    monkeypatch.setattr(trainer, "save_checkpoint", fail)
    with pytest.raises(RuntimeError, match="injected post-acquisition"):
        asyncio.run(trainer.run_train(config))
    assert len(service.clients) == 1 and not service.resident_clients and not service.engine.clients
    assert service.store.db.execute("SELECT COUNT(*) FROM operations WHERE kind='suspend' AND status='succeeded'").fetchone()[0] == 1
