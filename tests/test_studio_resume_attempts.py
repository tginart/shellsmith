"""Actual coordinator process exits against a surviving daemon."""

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading

import pytest

from shellsmith.train import studio, trainer
from shellsmith.train.studio_initialization import INITIALIZATION_FILE
from studio_compute.client import OperationError
from studio_compute.http import server
from studio_compute.service import Service
from test_studio_service import Engine, call, batch
from test_trainer_resume import _runtime


@pytest.fixture
def setup(tmp_path, monkeypatch):
    class Tracked(Engine):
        capabilities = {**Engine.capabilities, "model": "Qwen/Qwen3.8-27B", "context_window": 262144}
        def restore(self, client, folder, state):
            self.events.append(("restore", client))
            return super().restore(client, folder, state)
    service = Service(tmp_path / "service", Tracked)
    http = server(service, ("127.0.0.1", 0))
    thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
    config, _ = _runtime(tmp_path, monkeypatch)
    config = replace(config, compute=replace(config.compute, backend="studio"),
                     studio=replace(config.studio, endpoint=f"http://127.0.0.1:{http.server_port}", training_client="resumed"))
    monkeypatch.setattr(studio, "load_renderer", lambda *a, **k: None)
    source = call(service, "source", "create", {"seed": 9}, "source")
    try: yield service, config, source["checkpoint"]
    finally:
        http.shutdown(); http.server_close(); thread.join(); service.close()


def crash_coordinator(tmp_path, config, checkpoint, phase):
    # Pickle is a private test fixture, never an application input format.
    fixture = tmp_path / "config.pickle"; fixture.write_bytes(pickle.dumps(config))
    code = r"""
import asyncio,json,os,pickle,sys
from shellsmith.train import studio
studio.load_renderer=lambda *a,**k:None
config=pickle.load(open(sys.argv[1],'rb'))
backend=studio.StudioBackend(config)
original=backend.client.acall
async def uncertain(kind,body,*,request_id):
 result=await original(kind,body,request_id=request_id)
 if kind=='restore':os._exit(23)
 return result
if sys.argv[3]=='pending':backend.client.acall=uncertain
asyncio.run(backend.initialize({'path':sys.argv[2]}))
policy=asyncio.run(backend.publish_policy(revision=0))
backend.client.call('sample',{'policy':policy,'prompt_token_ids':[1],'max_tokens':1},request_id='crash-sample')
backend.client.call('forward_backward',{'sequences':[{'prompt_token_ids':[1],'sampled_tokens':[2],
 'behavior_logprobs':[-1.0],'advantages':[0.1],'loss_mask':[1.0]}]},request_id='crash-update:backward')
os._exit(23)
"""
    result = subprocess.run([sys.executable, "-c", code, str(fixture), checkpoint, phase], capture_output=True, text=True, timeout=30)
    assert result.returncode == 23, result.stderr


@pytest.mark.parametrize("phase", ["pending", "active"])
def test_uncertain_attempt_retries_but_used_session_rotates_and_rolls_back(tmp_path, monkeypatch, setup, phase):
    service, config, checkpoint = setup
    crash_coordinator(tmp_path, config, checkpoint, phase)
    path = config.resolved_log_dir() / INITIALIZATION_FILE
    old = json.loads(path.read_text())
    assert old["phase"] == phase
    if phase == "active":
        assert service.engine.clients["resumed"]["gradient"] == "crash-update:backward"
        assert service.clients["resumed"]["rng_counter"] == 1
    backend = studio.StudioBackend(config)
    asyncio.run(backend.initialize({"path": checkpoint}))
    current = json.loads(path.read_text())
    assert current["phase"] == "active"
    assert (current["request_id"] == old["request_id"]) is (phase == "pending")
    assert len([event for event in service.engine.events if event == ("restore", "resumed")]) == (1 if phase == "pending" else 2)
    assert service.engine.clients["resumed"]["gradient"] is None
    assert service.clients["resumed"].get("rng_counter", 0) == 0
    if phase == "active":
        state = {"pending_studio_update": {"id": "crash-update"}}
        asyncio.run(trainer._recover_studio_pending(backend=backend, state=state,
                    state_path=config.resolved_log_dir() / "recovery-test.json", journal=None))
        assert "pending_studio_update" not in state
    result = asyncio.run(backend.client.update(batch()["sequences"], learning_rate=0.1, request_id="fresh-update"))
    assert result["optim_metrics"]["optimizer_steps"] == 1
    backend.close()


def test_pending_journal_failure_precedes_release_and_restore(tmp_path, monkeypatch, setup):
    service, config, checkpoint = setup
    crash_coordinator(tmp_path, config, checkpoint, "active")
    from shellsmith.train import studio_initialization
    original = studio_initialization.write_json
    def fail(path, record):
        if record["phase"] == "pending": raise OSError("injected pending durability failure")
        return original(path, record)
    monkeypatch.setattr(studio_initialization, "write_json", fail)
    backend = studio.StudioBackend(config)
    with pytest.raises(OSError, match="pending durability"):
        asyncio.run(backend.initialize({"path": checkpoint}))
    assert service.engine.clients["resumed"]["gradient"] == "crash-update:backward"
    assert service.store.db.execute("SELECT COUNT(*) FROM operations WHERE kind='suspend'").fetchone()[0] == 0
    assert len([event for event in service.engine.events if event == ("restore", "resumed")]) == 1
    monkeypatch.setattr(studio_initialization, "write_json", original)
    backend = studio.StudioBackend(config)
    asyncio.run(backend.initialize({"path": checkpoint})); backend.close()


def test_release_fence_preserves_an_intervening_new_owner(tmp_path, monkeypatch, setup):
    service, config, checkpoint = setup
    crash_coordinator(tmp_path, config, checkpoint, "active")
    backend = studio.StudioBackend(config)
    original = backend.client.client_info
    def race():
        old = original()
        call(service, "resumed", "suspend", {"expected_owner": old["owner_operation"]}, "external-release")
        call(service, "resumed", "restore", {"checkpoint": checkpoint}, "different-owner")
        call(service, "resumed", "forward_backward", batch(), "different-back")
        return old
    monkeypatch.setattr(backend.client, "client_info", race)
    with pytest.raises(OperationError, match="acquisition changed"):
        asyncio.run(backend.initialize({"path": checkpoint}))
    backend.close()
    assert service.engine.clients["resumed"]["gradient"] == "different-back"
    assert service.clients["resumed"]["owner_operation"] == "different-owner"


def test_baseline_retry_without_trainer_state_reuses_uncertain_attempt(tmp_path, monkeypatch, setup):
    service, config, _ = setup
    first = studio.StudioBackend(config)
    created = asyncio.run(first.initialize())
    first.close()
    retry = studio.StudioBackend(config)
    original = retry.client.acall
    async def uncertain(kind, body, *, request_id):
        result = await original(kind, body, request_id=request_id)
        if kind == "restore":
            raise TimeoutError("receipt lost after committed restore")
        return result
    monkeypatch.setattr(retry.client, "acall", uncertain)
    with pytest.raises(TimeoutError, match="receipt lost"):
        asyncio.run(retry.initialize())
    path = config.resolved_log_dir() / INITIALIZATION_FILE
    pending = json.loads(path.read_text())
    assert pending["phase"] == "pending" and pending["checkpoint"] == created["checkpoint"]
    recovered = studio.StudioBackend(config)
    assert asyncio.run(recovered.initialize())["revision"] == 0
    assert json.loads(path.read_text())["request_id"] == pending["request_id"]
    assert len([event for event in service.engine.events if event == ("restore", "resumed")]) == 1
    recovered.close()


def test_snapshot_and_initialization_journal_retry_reaches_first_state_publication(tmp_path, monkeypatch, setup):
    service, config, _ = setup
    original = trainer.write_json
    attempts = []
    def fail_state(path, *args, **kwargs):
        if Path(path).name == trainer.STATE_FILE:
            attempts.append(True)
            raise OSError("injected initial state publication failure")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(trainer, "write_json", fail_state)
    for _ in range(2):
        with pytest.raises(OSError, match="initial state publication"):
            asyncio.run(trainer.run_train(config))
    assert len(attempts) == 2
    assert {p.name for p in config.resolved_log_dir().iterdir()} == {trainer.SNAPSHOT_FILE, INITIALIZATION_FILE}
    assert "resumed" not in service.live_clients
