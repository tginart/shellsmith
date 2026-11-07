"""Crash/recovery, cold-client ownership, and terminal batch receipt regressions."""

import asyncio
from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from shellsmith.train.config import TrainConfig, RunSection, ComputeSection, validate_train_config
from shellsmith.train.studio import StudioBackend
from studio_compute.client import Client
from studio_compute.engine import ResidentEngine
from studio_compute.http import loopback_host, server
from studio_compute.cli import parse_args
from studio_compute.service import Service
from studio_compute.storage import Store
from studio_compute.protocol import Conflict
from test_studio_service import Engine, call, batch


def test_cold_clients_release_and_reload_only_on_demand(tmp_path):
    service = Service(tmp_path, Engine)
    try:
        for i in range(20):
            call(service, f"c{i}", "create", {"seed": i}, f"create-{i}")
            call(service, f"c{i}", "suspend", {}, f"suspend-{i}")
            assert not service.engine.clients and not service.resident_clients
        assert len(service.clients) == 20
        assert not any(s["closed"] for s in service.clients.values())
        assert call(service, "c7", "inspect", {}, "inspect-7") == {"weight": 7, "step": 0}
        assert set(service.engine.clients) == {"c7"}
    finally:
        service.close()
    service = Service(tmp_path, Engine)
    try:
        assert not service.engine.clients
        assert call(service, "c7", "inspect", {}, "inspect-7-restarted")["weight"] == 7
        assert set(service.engine.clients) == {"c7"}
    finally:
        service.close()


def test_direct_restore_clones_committed_optimizer_without_create(tmp_path):
    service = Service(tmp_path, Engine)
    try:
        call(service, "original", "create", {"seed": 9}, "create")
        call(service, "original", "forward_backward", batch(), "back")
        committed = call(service, "original", "optim_step", {"learning_rate": 1, "backward_operation": "back"}, "step")
        cloned = call(service, "clone", "restore", {"checkpoint": committed["checkpoint"]}, "clone")
        assert cloned["revision"] == 1
        assert call(service, "clone", "inspect", {}, "inspect") == {"weight": 11, "step": 1}
        call(service, "clone", "close", {}, "close")
        call(service, "clone", "restore", {"checkpoint": committed["checkpoint"]}, "reopen")
        assert not service.clients["clone"]["closed"]
    finally:
        service.close()


def test_interrupted_create_can_retry_original_intent_but_optimizer_cannot(tmp_path):
    store = Store(tmp_path)
    store.submit("create", "a", "create", {"seed": 5})
    store.running("create")  # no checkpoint rename before crash
    store.submit("optimizer", "missing", "optim_step", {"learning_rate": 1})
    store.close()
    service = Service(tmp_path, Engine)
    try:
        assert service.operation("create")["status"] == "interrupted"
        created = call(service, "a", "create", {"seed": 5}, "create")
        assert service.engine.clients["a"]["weight"] == 5
        assert service.submit("create", "a", "create", {"seed": 5})["result"] == created
        with pytest.raises(Conflict):
            service.submit("create", "a", "create", {"seed": 6})
        assert service.submit("optimizer", "missing", "optim_step", {"learning_rate": 1})["status"] == "interrupted"
    finally:
        service.close()


def test_checkpoint_rename_before_crash_deduplicates_create_instead_of_retry(tmp_path):
    store = Store(tmp_path)
    store.submit("create", "a", "create", {})
    state, result = store.checkpoint("create", lambda folder: (folder / "state.json").write_text('{"step":0,"weight":4}'),
        identity=Engine.identity, state={"revision": 0, "seed": 4, "closed": False}, result={"revision": 0})
    store.close()
    service = Service(tmp_path, Engine)
    try:
        assert service.submit("create", "a", "create", {})["result"] == result
        assert not service.engine.clients
        assert call(service, "a", "inspect", {}, "inspect")["weight"] == 4
    finally:
        service.close()


def test_backend_resume_restores_directly_and_close_suspends(tmp_path):
    calls = []
    class RPC:
        async def acall(self, kind, body, *, request_id):
            calls.append((kind, body, request_id))
            return {"revision": 3, "checkpoint": "studio-state:fixture"}
        def call(self, kind, body, *, request_id):
            calls.append((kind, body, request_id))
        def client_info(self):
            return {"live": False, "owner_operation": None}
    backend = StudioBackend.__new__(StudioBackend)
    backend.config, backend.operation_prefix, backend.client = TrainConfig(), "run", RPC()
    from shellsmith.train.studio_initialization import InitializationJournal
    backend.capabilities = {"identity": {"model": "fixture"}}
    backend._acquired = False
    backend._initialization_target = backend._initialization_result = None
    backend.initializations = InitializationJournal(tmp_path / "run", "run", {})
    assert asyncio.run(backend.initialize({"path": "studio-state:fixture"}))["revision"] == 3
    backend.close()
    assert calls[0][:2] == ("restore", {"checkpoint": "studio-state:fixture", "expected_identity": backend.capabilities["identity"]})
    assert calls[1][:2] == ("suspend", {"expected_owner": calls[0][2], "expected_identity": backend.capabilities["identity"]})


@pytest.mark.parametrize("client", ["unknown", "closed"])
def test_cleanup_of_missing_or_closed_client_is_harmless(tmp_path, client):
    service = Service(tmp_path, Engine)
    try:
        if client == "closed":
            call(service, client, "create", {}, "create")
            call(service, client, "close", {}, "close")
        call(service, client, "suspend", {}, "cleanup")
        assert not service.engine.clients
        assert (client in service.clients) == (client == "closed")
    finally:
        service.close()


@pytest.mark.parametrize("bad_client", ["unknown", "closed"])
def test_failed_batch_leader_stays_terminal_while_valid_member_completes(tmp_path, bad_client):
    started, release = threading.Event(), threading.Event()
    class Batched(Engine):
        def sample_batch(self, bodies):
            started.set()
            yield {"phase": "prefill"}
            assert release.wait(5)
            return [{"sampled_tokens": [2], "policy_revision": 0} for body in bodies]
    service = Service(tmp_path, Batched)
    try:
        call(service, "good", "create", {}, "create")
        policy = call(service, "good", "publish", {"revision": 0}, "publish")["policy"]
        if bad_client == "closed":
            call(service, bad_client, "create", {}, "create-bad")
            call(service, bad_client, "close", {}, "close-bad")
        with service.condition:
            for name in (bad_client, "good"):
                service.submit(name + "-sample", name, "sample", {
                    "policy": policy, "prompt_token_ids": [1], "max_tokens": 1,
                })
        assert started.wait(5)
        assert service.wait(bad_client + "-sample", 1)["status"] == "failed"
        release.set()
        assert service.wait("good-sample")["status"] == "succeeded"
        assert service.operation(bad_client + "-sample")["status"] == "failed"
    finally:
        release.set()
        service.close()


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.3", "8.8.8.8", "::ffff:192.168.1.3", "example.com"])
def test_public_unauthenticated_binds_rejected_before_service_construction(host, tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        server(None, (host, 0))
    with pytest.raises(SystemExit):
        parse_args(["serve", "--model", "tiny", "--host", host])
    config = tmp_path / "public.toml"
    config.write_text(f'[service]\nmodel="tiny"\nhost="{host}"\n')
    with pytest.raises(SystemExit):
        parse_args(["serve", "--config", str(config)])


@pytest.mark.parametrize("host,expected", [("127.0.0.1", "127.0.0.1"), ("localhost", "127.0.0.1"), ("::1", "::1")])
def test_loopback_binds_accepted(host, expected):
    assert loopback_host(host) == expected
    assert parse_args(["serve", "--model", "tiny", "--host", host]).host == expected


@pytest.mark.parametrize("kind", ["forward_backward", "optim_step", "save", "restore", "sample", "logprobs"])
def test_compute_and_sampling_deadlines_are_independent(monkeypatch, kind):
    import studio_compute.client as module
    ticks = iter([0, 2, 3, 4])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    client = Client(timeout=100, sampling_timeout=1)
    calls = []
    client.submit = lambda *a, **k: {"status": "running"}
    def operation(*args, **kwargs):
        calls.append(True)
        return {"status": "succeeded", "result": {"ok": True}}
    client.operation = operation
    if kind in ("sample", "logprobs"):
        with pytest.raises(TimeoutError):
            client.call(kind, {}, request_id="request")
        assert not calls
    else:
        assert client.call(kind, {}, request_id="request") == {"ok": True}
        assert calls == [True]


def test_backend_passes_distinct_compute_and_player_timeouts(monkeypatch):
    from shellsmith.train import studio
    config = TrainConfig(run=RunSection(run_id="deadlines"), compute=ComputeSection(backend="studio"))
    config = replace(config, studio=replace(config.studio, operation_timeout_seconds=12000))
    supplied = []
    def rpc(*args, **kwargs):
        supplied.append(kwargs)
        return SimpleNamespace(capabilities=lambda: {"model": config.studio.model_id, "context_window":262144,"identity":{}, "cache_precisions": ["bf16"]})
    monkeypatch.setattr(studio, "Client", rpc)
    monkeypatch.setattr(studio, "load_renderer", lambda *a, **k: None)
    studio.StudioBackend(config)
    assert supplied[0]["timeout"] == 12000
    assert supplied[0]["sampling_timeout"] == config.player.prompt_timeout_seconds
    with pytest.raises(ValueError, match="operation_timeout_seconds"):
        validate_train_config(replace(config, studio=replace(config.studio, operation_timeout_seconds=0)))


def test_execution_identity_rejects_numerically_different_recovery(monkeypatch, tmp_path):
    import studio_compute.telemetry as telemetry
    monkeypatch.setattr(telemetry, "memory", lambda: {})
    monkeypatch.setattr(telemetry, "process_identity", lambda: {})
    base = {"model": "fixture", "numerics": "same-source"}
    def factory(*args, **kwargs):
        return SimpleNamespace(mx=SimpleNamespace(set_cache_limit=lambda n: None, float32="fp32", bfloat16="bf16"),
            lm=SimpleNamespace(layers=[object(), object()]), identity=base,
            capabilities={"precision": {}}, bundle={}, adapter=SimpleNamespace(state=lambda: {}))
    original = ResidentEngine("tiny", checkpoint_layers=1, model_factory=factory, execution_profile="legacy")
    store = Store(tmp_path)
    try:
        store.submit("create", "a", "create", {})
        _, receipt = store.checkpoint("create", lambda folder: (folder / "factors").write_bytes(b"fixture"),
            identity=original.identity, state={}, result={})
        for options in ({"segment_tokens": 1024}, {"head_rows": 64}, {"checkpoint_layers": 2}, {"prefill_tokens": 1024}, {"execution_profile": "bf16-fast"}):
            changed = ResidentEngine("tiny", model_factory=factory, **({"checkpoint_layers":1,"execution_profile":"legacy"} | options))
            assert changed.identity != original.identity
            with pytest.raises(ValueError, match="compute identity"):
                store.verify(receipt["checkpoint"], changed.identity)
        identical = ResidentEngine("tiny", checkpoint_layers=1, model_factory=factory, metal_cache_gib=0, execution_profile="legacy")
        assert identical.identity == original.identity  # allocator cache cap is not layer math
        store.verify(receipt["checkpoint"], identical.identity)
        assert base == {"model": "fixture", "numerics": "same-source"}
    finally:
        store.close()


def test_rng_cursor_survives_suspension_and_fresh_process(tmp_path):
    class RNGEngine(Engine):
        def create(self, client, body):
            super().create(client, body)
            self.clients[client]["rng"] = [body.get("seed", 0), 0]
            return {"rng_key": self.clients[client]["rng"], "rng_counter": 0}
        def restore(self, client, folder, state):
            super().restore(client, folder, state)
            self.clients[client]["rng"] = state["rng_key"]
        def next_rng(self, client):
            key = self.clients[client]["rng"]
            self.clients[client]["rng"] = [key[0], key[1] + 1]
            return self.clients[client]["rng"], key
        def work(self, client, kind, body):
            if kind == "sample":
                yield {"phase": "decode"}
                return {"sampled_tokens": [body["rng_key"][1]], "policy_revision": 0}
            return (yield from super().work(client, kind, body))
    service = Service(tmp_path, RNGEngine)
    try:
        call(service, "a", "create", {"seed": 11}, "create")
        policy = call(service, "a", "publish", {"revision": 0}, "publish")["policy"]
        body = {"policy": policy, "prompt_token_ids": [1], "max_tokens": 1}
        assert call(service, "a", "sample", body, "sample0")["sampled_tokens"] == [0]
        cursor = dict(service.clients["a"])
        call(service, "a", "suspend", {}, "suspend")
        assert service.clients["a"] == cursor and not service.engine.clients
        assert call(service, "a", "sample", body, "sample1")["sampled_tokens"] == [1]
    finally:
        service.close()
    service = Service(tmp_path, RNGEngine)
    try:
        assert not service.engine.clients
        assert call(service, "a", "sample", body, "sample2")["sampled_tokens"] == [2]
        assert service.clients["a"]["rng_key"] == [11, 3]
    finally:
        service.close()


@pytest.mark.parametrize("fail", [False, True])
def test_inference_coordinator_suspends_session_even_on_setup_failure(monkeypatch, tmp_path, fail):
    from shellsmith.campaign import inference
    closed = []
    session = SimpleNamespace(close=lambda: closed.append(True))
    async def campaign(config, **kwargs):
        kwargs["session_holder"]["session"] = session
        if fail:
            raise RuntimeError("injected setup failure")
        return {"status": "complete"}
    monkeypatch.setattr(inference, "_run_inference_campaign_locked", campaign)
    config = TrainConfig(run=RunSection(run_id="cleanup", log_dir=str(tmp_path / "run")), compute=ComputeSection(backend="studio"))
    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            asyncio.run(inference.run_inference_campaign(config))
    else:
        assert asyncio.run(inference.run_inference_campaign(config))["status"] == "complete"
    assert closed == [True]
