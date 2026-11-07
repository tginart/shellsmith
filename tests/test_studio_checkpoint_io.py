"""Responsive checkpoint polling and shared immutable snapshot payloads."""

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from shellsmith.train.config import TrainConfig, RunSection, ComputeSection
from studio_compute.client import Client
from studio_compute.engine import ResidentEngine
from studio_compute.http import server
from studio_compute.service import Service
from test_studio_service import Engine, call, batch


@pytest.mark.parametrize("kind", ["create", "optim_step", "restore"])
def test_checkpoint_serialization_does_not_block_http_poll_or_submission(tmp_path, kind):
    writing, release, responded = threading.Event(), threading.Event(), threading.Event()
    class SlowWriter(Engine):
        block = False
        def write_state(self, client, folder):
            if self.block:
                writing.set()
                assert release.wait(5)
            return super().write_state(client, folder)
    service = Service(tmp_path, SlowWriter)
    http = server(service, ("127.0.0.1", 0))
    thread = threading.Thread(target=http.serve_forever, daemon=True);thread.start()
    reader = None; responses = []
    try:
        created = call(service, "a", "create", {}, "create-a")
        if kind == "optim_step":
            call(service, "a", "forward_backward", batch(), "back")
        elif kind == "restore":
            call(service, "a", "suspend", {}, "release-before-restore")
        service.engine.block = True
        body = ({"learning_rate": 1.0, "backward_operation": "back"} if kind == "optim_step"
                else {"checkpoint": created["checkpoint"]} if kind == "restore" else {})
        client_id = "new" if kind == "create" else "a"
        service.submit("slow-checkpoint", client_id, kind, body)
        assert writing.wait(3)
        rpc = Client(f"http://127.0.0.1:{http.server_port}", "a")
        def poll():
            try:
                responses.append(rpc.operation("slow-checkpoint", wait=0.05))
                responses.append(rpc.submit("inspect", {}, request_id="queued-inspect"))
            finally:
                responded.set()
        reader = threading.Thread(target=poll, daemon=True);reader.start()
        assert responded.wait(1), "Global poll lock held during checkpoint I/O"
        assert responses[0]["status"] == "running"
        assert responses[1]["status"] == "queued"
        release.set()
        assert service.wait("slow-checkpoint")["status"] == "succeeded"
        assert service.wait("queued-inspect")["status"] == "succeeded"
    finally:
        release.set()
        if reader: reader.join(3)
        http.shutdown();http.server_close();thread.join();service.close()


class SharedSnapshots(Engine):
    def __init__(self):
        super().__init__()
        self.full_writes = 0

    def create(self, client, body):
        super().create(client, body)
        key = [body.get("seed", 0), 0]
        self.clients[client]["rng"] = key
        return {"rng_key": key, "rng_counter": 0}

    def next_rng(self, client):
        previous = self.clients[client]["rng"]
        self.clients[client]["rng"] = [previous[0], previous[1] + 1]
        return self.clients[client]["rng"], previous

    def restore(self, client, folder, state):
        super().restore(client, folder, state)
        self.clients[client]["rng"] = state["rng_key"]

    def write_state(self, client, folder):
        super().write_state(client, folder)
        entry = self.clients[client]
        (folder / "adapters.safetensors").write_text(json.dumps({"weight": entry["weight"]}))
        (folder / "optimizer.safetensors").write_bytes((str(entry["step"]) + "moments").encode() * 2048)
        (folder / "rng.safetensors").write_text(json.dumps({"key": entry["rng"]}))
        self.full_writes += 1

    def write_snapshot(self, client, folder, source):
        # Exercise the production native snapshot writer with CPU scalar state.
        entry = self.clients[client]
        def boundary():
            if entry["gradient"] is not None:
                raise RuntimeError("Unconsumed gradients")
            return {}, {}
        native = ResidentEngine.__new__(ResidentEngine)
        native.clients = {client: {"optimizer": SimpleNamespace(state=boundary), "rng": entry["rng"]}}
        native.mx = SimpleNamespace(save_safetensors=lambda path, value: Path(path).write_text(json.dumps(value)))
        native.write_snapshot(client, folder, source)


def test_sampler_saves_reuse_checkpoint_and_states_share_optimizer_payloads(tmp_path):
    service = Service(tmp_path, SharedSnapshots)
    try:
        call(service, "a", "create", {"seed": 11}, "create")
        call(service, "a", "forward_backward", batch(), "back")
        optimizer = call(service, "a", "optim_step", {"learning_rate": 1, "backward_operation": "back"}, "step")
        source, _ = service.store.verify(optimizer["checkpoint"])
        writes = service.engine.full_writes
        directory_count = len(list(service.store.checkpoints.iterdir()))
        sampler = call(service, "a", "save", {"kind": "sampler"}, "sampler")
        assert sampler["checkpoint"] == optimizer["checkpoint"]
        assert len(list(service.store.checkpoints.iterdir())) == directory_count
        policy = call(service, "a", "publish", {"revision": 1}, "publish")["policy"]
        assert policy == sampler["policy"]
        snapshots = []
        for i in range(3):
            call(service, "a", "sample", {"policy": policy, "prompt_token_ids": [1], "max_tokens": 1}, f"sample-{i}")
            saved = call(service, "a", "save", {"kind": "state", "metadata": {"cursor": i}}, f"save-{i}")
            folder, document = service.store.verify(saved["checkpoint"])
            snapshots.append(saved["checkpoint"])
            assert saved["policy"] == sampler["policy"]
            assert document["result"]["metadata"] == {"cursor": i}
            assert json.loads((folder / "rng.safetensors").read_text())["key"] == [11, i + 1]
            for name in ("adapters.safetensors", "optimizer.safetensors"):
                assert (folder / name).stat().st_ino == (source / name).stat().st_ino
            assert (folder / "rng.safetensors").stat().st_ino != (source / "rng.safetensors").stat().st_ino
        assert service.engine.full_writes == writes
        optimizer_files = list(service.store.checkpoints.glob("*/optimizer.safetensors"))
        assert len(optimizer_files) == 5 and len({p.stat().st_ino for p in optimizer_files}) == 2
        call(service, "a", "forward_backward", batch(), "pending-grad")
        service.submit("bad-state", "a", "save", {"kind": "state"})
        assert service.wait("bad-state")["status"] == "failed"
        assert service.engine.clients["a"]["gradient"] == "pending-grad"
    finally:
        service.close()
    service = Service(tmp_path, SharedSnapshots)
    try:
        result = call(service, "clone", "restore", {"checkpoint": snapshots[-1]}, "restore-clone")
        assert result["revision"] == 1
        assert service.engine.clients["clone"]["rng"] == [11, 3]
        assert call(service, "clone", "inspect", {}, "inspect-clone") == {"weight": 13, "step": 1}
        assert service.submit("sampler", "a", "save", {"kind": "sampler"})["result"] == sampler
        service.store.verify(optimizer["checkpoint"])
    finally:
        service.close()


@pytest.mark.parametrize("reference", ["tinker://policy", "typo", "studio-policy:short", "studio-policy:" + "G" * 64, "studio-policy:" + "a" * 65])
def test_bad_policy_reference_rejected_before_setup_or_lock(monkeypatch, tmp_path, reference):
    from shellsmith.campaign import inference
    from shellsmith.train import studio
    config = TrainConfig(run=RunSection(run_id="invalid-policy", log_dir=str(tmp_path / "run")), compute=ComputeSection(backend="studio"))
    monkeypatch.setattr(studio, "StudioBackend", lambda *a, **k: pytest.fail("remote setup must not run"))
    async def preflight(*args, **kwargs):
        pytest.fail("campaign preflight or service discovery must not run")
    monkeypatch.setattr(inference, "_run_inference_campaign_locked", preflight)
    before = list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="opaque published policy reference"):
        asyncio.run(inference.run_inference_campaign(config, sampler_path=reference))
    assert list(tmp_path.iterdir()) == before
