"""A daemon identity change between snapshot and discovery fails before mutation."""

import asyncio
from dataclasses import replace
import threading

import pytest

from shellsmith.campaign import inference
from shellsmith.train import studio
from studio_compute.client import Client
from studio_compute.http import server
from studio_compute.service import Service
from test_studio_service import Engine
from test_trainer_resume import _runtime


@pytest.mark.parametrize("changed", ["tokenizer", "numerics"])
def test_studio_inference_rejects_preflight_identity_race(tmp_path, monkeypatch, changed):
    class InferenceEngine(Engine):
        capabilities = {**Engine.capabilities, "model": "Qwen/Qwen3.8-27B", "context_window": 262144}
    service = Service(tmp_path / "service", InferenceEngine)
    http = server(service, ("127.0.0.1", 0))
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        config, _ = _runtime(tmp_path, monkeypatch)
        config = replace(config, compute=replace(config.compute, backend="studio"),
                         studio=replace(config.studio, endpoint=f"http://127.0.0.1:{http.server_port}"))
        original = Client.capabilities
        calls = []
        def capabilities(client):
            value = original(client)
            calls.append(value)
            if len(calls) > 1:
                value = {**value, "identity": {**value["identity"], changed: "changed-after-snapshot"}}
            return value
        monkeypatch.setattr(Client, "capabilities", capabilities)
        monkeypatch.setattr(studio, "load_renderer", lambda *a, **k: None)
        monkeypatch.setattr(inference, "open_rollout_stack", lambda *a, **k: pytest.fail("No rollouts"))
        with pytest.raises(RuntimeError, match="Studio.*identity changed after campaign preflight"):
            asyncio.run(inference.run_inference_campaign(config, deps=None, sampler_path=None))
        assert len(calls) == 2
        assert not service.clients and not service.engine.events
        assert service.store.db.execute("SELECT COUNT(*) FROM operations").fetchone()[0] == 0
        assert not (config.resolved_log_dir() / inference.SNAPSHOT_FILE).exists()
    finally:
        http.shutdown(); http.server_close(); thread.join(); service.close()


@pytest.mark.parametrize("changed", ["tokenizer", "numerics"])
def test_studio_training_rejects_identity_race_before_snapshot_or_mutation(tmp_path, monkeypatch, changed):
    from shellsmith.train import trainer
    class TrainingEngine(Engine):
        capabilities = {**Engine.capabilities, "model": "Qwen/Qwen3.8-27B", "context_window": 262144}
    service = Service(tmp_path / "service", TrainingEngine)
    http = server(service, ("127.0.0.1", 0))
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        config, _ = _runtime(tmp_path, monkeypatch)
        config = replace(config, compute=replace(config.compute, backend="studio"),
                         studio=replace(config.studio, endpoint=f"http://127.0.0.1:{http.server_port}"))
        original = Client.capabilities
        calls = []
        def capabilities(client):
            value = original(client)
            calls.append(value)
            if len(calls) > 1:
                value = {**value, "identity": {**value["identity"], changed: "changed-after-snapshot"}}
            return value
        monkeypatch.setattr(Client, "capabilities", capabilities)
        monkeypatch.setattr(studio, "load_renderer", lambda *a, **k: None)
        async def forbidden_initialize(*a, **k):
            pytest.fail("No initialization before identity comparison")
        monkeypatch.setattr(studio.StudioBackend, "initialize", forbidden_initialize)
        monkeypatch.setattr(studio.StudioBackend, "close", lambda *a: pytest.fail("No mutating suspend cleanup"))
        with pytest.raises(RuntimeError, match="Studio.*identity changed between snapshot and backend discovery"):
            asyncio.run(trainer.run_train(config))
        assert len(calls) == 2
        assert not service.clients and not service.engine.events
        assert service.store.db.execute("SELECT COUNT(*) FROM operations").fetchone()[0] == 0
        assert not (config.resolved_log_dir() / trainer.SNAPSHOT_FILE).exists()
        assert not (config.resolved_log_dir() / trainer.STATE_FILE).exists()
    finally:
        http.shutdown(); http.server_close(); thread.join(); service.close()
