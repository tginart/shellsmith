"""Late endpoint changes and cancelled HTTP handlers cannot cross acquisitions."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import importlib.util
from pathlib import Path
import threading

import pytest

from shellsmith.policy.server import PlayerEndpoint
from shellsmith.train import studio
from shellsmith.train.config import TrainConfig, RunSection, ComputeSection
from studio_compute.http import server
from studio_compute.protocol import Conflict
from studio_compute.service import Service
from test_studio_service import Engine, call, batch
from test_player_endpoint import _request
from test_mlx_sampler import FakeSession


@pytest.fixture
def services(tmp_path, monkeypatch):
    resources = []
    def start(identity):
        class Tracked(Engine):
            capabilities = {**Engine.capabilities, "model": "Qwen/Qwen3.8-27B", "context_window": 262144,
                            "vocabulary_size": 128}
            def work(self, client, kind, body):
                if kind == "sample":
                    self.events.append((client, "sample"))
                    return {"sampled_tokens": [10, 99], "behavior_logprobs": [-1., -1.],
                            "finish_reason": "stop", "policy_revision": body["revision"]}
                return (yield from super().work(client, kind, body))
        Tracked.identity = {**Engine.identity, "runtime": identity}
        service = Service(tmp_path / identity, Tracked)
        http = server(service, ("127.0.0.1", 0))
        thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
        resources.append((service, http, thread))
        return service, f"http://127.0.0.1:{http.server_port}"
    monkeypatch.setattr(studio, "load_renderer", lambda *a, **k: FakeSession([]).renderer)
    try:
        yield start
    finally:
        for service, http, thread in reversed(resources):
            http.shutdown(); http.server_close(); thread.join(); service.close()


def config_for(tmp_path, endpoint):
    config = TrainConfig(run=RunSection(run_id="owner", log_dir=str(tmp_path / "run")),
                         compute=ComputeSection(backend="studio"))
    return replace(config, studio=replace(config.studio, endpoint=endpoint))


@pytest.mark.parametrize("operation", ["create", "restore", "release"])
def test_initialization_and_release_reject_retargeted_service_before_intent(tmp_path, services, operation):
    original, endpoint = services("original")
    replacement, other_endpoint = services("replacement")
    backend = studio.StudioBackend(config_for(tmp_path, endpoint))
    source = call(original, "source", "create", {}, "source")
    if operation == "release":
        asyncio.run(backend.initialize())
        # Equal acquisition ID/body cannot defeat the distinct discovered identity.
        call(replacement, "owner", "create", {}, backend._owner_operation)
        call(replacement, "owner", "forward_backward", batch(), "replacement-back")
    backend.client.endpoint = other_endpoint
    before = replacement.store.db.execute("SELECT COUNT(*) FROM operations").fetchone()[0]
    with pytest.raises(Conflict, match="identity changed"):
        if operation == "release": backend.close()
        else: asyncio.run(backend.initialize(None if operation == "create" else {"path": source["checkpoint"]}))
    assert replacement.store.db.execute("SELECT COUNT(*) FROM operations").fetchone()[0] == before
    if operation == "release":
        assert replacement.engine.clients["owner"]["gradient"] == "replacement-back"
        assert replacement.live_clients == {"owner"}
    else:
        assert replacement.clients == {} and replacement.engine.clients == {}
    backend.close()


@pytest.mark.parametrize("resume", [False, True])
def test_cancelled_two_pass_http_handler_cannot_sample_after_backend_cleanup(tmp_path, monkeypatch, services, resume):
    service, address = services("daemon")
    backend = studio.StudioBackend(config_for(tmp_path, address))
    created = asyncio.run(backend.initialize())
    asyncio.run(backend.publish_policy(revision=0))
    entered, proceed, ended = threading.Event(), threading.Event(), threading.Event()
    rpc = backend.client.call
    def pause_after_thinking(kind, body, *, request_id):
        result = rpc(kind, body, request_id=request_id)
        if kind == "sample" and not entered.is_set():
            entered.set()
            assert proceed.wait(10)
        return result
    monkeypatch.setattr(backend.client, "call", pause_after_thinking)
    endpoint = PlayerEndpoint(sampler=backend.sampler, records_dir=tmp_path / "records").start()
    original_end = endpoint._end_episode_request
    def end(*args):
        original_end(*args); ended.set()
    monkeypatch.setattr(endpoint, "_end_episode_request", end)
    endpoint.open_episode("cancelled")
    url = endpoint.base_url("cancelled") + "/chat/completions"
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_request, url, key=endpoint.api_key,
                body={"messages": [{"role": "user", "content": "continue"}], "reasoning_effort": "medium"})
            try:
                assert entered.wait(5)
                endpoint.close_episode("cancelled")
                endpoint.stop()  # The daemon handler remains blocked in its first pass.
                backend.close()
                assert "owner" not in service.resident_clients and "owner" not in service.live_clients
                if resume:
                    asyncio.run(backend.initialize({"path": created["checkpoint"]}))
                state = dict(service.clients["owner"])
                residents = set(service.resident_clients)
            finally:
                proceed.set()
            assert future.result(timeout=10)[0] == 503
        assert ended.wait(5)
        assert service.clients["owner"] == state and service.resident_clients == residents
        assert [e for e in service.engine.events if e == ("owner", "sample")] == [("owner", "sample")]
        assert endpoint._episodes == {} and endpoint._records == {} and backend.sampler.pins == {}
        assert not (tmp_path / "records" / "cancelled.jsonl").exists()
    finally:
        proceed.set(); endpoint.stop(); backend.close()


def test_smoke_probe_control_restore_works_for_its_live_isolation_client(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/smoke_studio_runtime.py"
    spec = importlib.util.spec_from_file_location("smoke_studio_probe", path)
    probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
    service = Service(tmp_path, Engine)
    try:
        call(service, "a", "create", {"seed": 7}, "create-a")
        call(service, "b", "create", {"seed": 23}, "create-b")
        call(service, "a", "forward_backward", batch(), "backward")
        checkpoint = call(service, "a", "optim_step", {"learning_rate": .1, "backward_operation": "backward"}, "optimizer")["checkpoint"]
        assert "b" in service.live_clients and service.engine.clients["b"]["weight"] == 23
        control = probe.restore_control("b", checkpoint, lambda client, kind, body, name: call(service, client, kind, body, name))
        assert control["revision"] == 1 and service.engine.clients["b"]["weight"] == 9
        assert service.engine.clients["a"]["weight"] == 9 and service.live_clients == {"a", "b"}
    finally:
        service.close()
