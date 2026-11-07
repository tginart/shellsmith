"""Inference setup, concurrent immutable assets, durable ID and staging recovery."""

import asyncio
import base64
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import uuid

import pytest

from shellsmith.train.config import TrainConfig, RunSection, ComputeSection, TrainSection, validate_train_config
from shellsmith.train.studio import StudioBackend, RemoteSamplingSession
from studio_compute.http import server
from studio_compute.protocol import bounded_identifier, digest, identifier
from studio_compute.service import Service
from studio_compute.storage import Store
from studio_compute.tokenizer import materialize
from test_studio_service import Engine, call, batch


def bundle(**files):
    assets = {name: {"base64": base64.b64encode(data).decode(), "sha256": hashlib.sha256(data).hexdigest()}
              for name, data in files.items()}
    return {"files": assets, "hash": digest({name: item["sha256"] for name, item in assets.items()})}


def test_visible_tokenizer_assets_are_never_rewritten(tmp_path, monkeypatch):
    assets = bundle(**{"tokenizer.json": b'{"vocab": "complete"}', "chat_template.jinja": b"template"})
    visible = materialize(assets, tmp_path)
    def rewrite(*args):
        pytest.fail("A visible tokenizer asset was opened for rewriting")
    monkeypatch.setattr(Path, "write_bytes", rewrite)
    assert materialize(assets, tmp_path) == visible
    assert (visible / "tokenizer.json").read_bytes() == b'{"vocab": "complete"}'


def test_two_cold_processes_publish_one_complete_tokenizer_directory(tmp_path):
    assets = bundle(**{"tokenizer.json": b"vocab" * 100000, "chat_template.jinja": b"template" * 10000})
    source = tmp_path / "bundle.json"; source.write_text(json.dumps(assets))
    cache = tmp_path / "cache"
    code = 'import json,sys;from studio_compute.tokenizer import materialize;print(materialize(json.load(open(sys.argv[1])),sys.argv[2]))'
    processes = [subprocess.Popen([sys.executable, "-c", code, str(source), str(cache)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    try:
        paths = []
        for process in processes:
            out, err = process.communicate(timeout=15)
            assert process.returncode == 0, err
            paths.append(Path(out.strip()))
        assert paths[0] == paths[1] == cache / assets["hash"]
        assert {p.name for p in cache.iterdir()} == {assets["hash"]}
        for name, item in assets["files"].items():
            assert hashlib.sha256((paths[0] / name).read_bytes()).hexdigest() == item["sha256"]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill(); process.wait()


def test_failed_tokenizer_staging_is_invisible_and_can_retry(tmp_path, monkeypatch):
    assets = bundle(**{"tokenizer.json": b"vocab", "chat_template.jinja": b"template"})
    original = Path.write_bytes
    def write(path, data):
        if path.name == "chat_template.jinja":
            raise OSError("injected incomplete asset write")
        return original(path, data)
    monkeypatch.setattr(Path, "write_bytes", write)
    with pytest.raises(OSError, match="injected"):
        materialize(assets, tmp_path)
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(Path, "write_bytes", original)
    assert (materialize(assets, tmp_path) / "chat_template.jinja").read_bytes() == b"template"


def test_corrupt_cached_assets_fail_closed_without_overwriting(tmp_path):
    assets = bundle(**{"tokenizer.json": b"valid"})
    visible = materialize(assets, tmp_path)
    (visible / "tokenizer.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Cached tokenizer asset hash"):
        materialize(assets, tmp_path)
    assert (visible / "tokenizer.json").read_bytes() == b"corrupt"


@pytest.mark.parametrize("rate", [0, -1, None])
def test_campaign_evaluates_published_policy_without_training_client_or_rate(monkeypatch, tmp_path, rate):
    from shellsmith.campaign import inference
    from shellsmith.train import trainer, studio
    class InferenceEngine(Engine):
        capabilities = {**Engine.capabilities, "model": "Qwen/Qwen3.8-27B", "context_window": 262144}
        def create(self, client, body):
            assert body["learning_rate"] > 0
            self.events.append(("create", client, body["learning_rate"]))
            return super().create(client, body)
    service = Service(tmp_path / "service", InferenceEngine)
    http = server(service, ("127.0.0.1", 0))
    thread = threading.Thread(target=http.serve_forever, daemon=True);thread.start()
    try:
        training = "retained-training-client"
        call(service, training, "create", {"seed": 9, "learning_rate": 0.001}, "training:create")
        call(service, training, "forward_backward", batch(), "training:back")
        call(service, training, "optim_step", {"learning_rate": 0.001, "backward_operation": "training:back"}, "training:step")
        policy = call(service, training, "publish", {"revision": 1}, "training:publish")["policy"]
        call(service, training, "suspend", {}, "training:suspend")
        previous = dict(service.clients[training])
        config = TrainConfig(run=RunSection(run_id="evaluation", log_dir=str(tmp_path / "evaluation")), compute=ComputeSection(backend="studio"))
        config = replace(config, studio=replace(config.studio, endpoint=f"http://127.0.0.1:{http.server_port}", training_client=training),
                         train=replace(config.train, learning_rate=rate), worlds=replace(config.worlds, count=1))
        validate_train_config(config, campaign_mode="inference")
        monkeypatch.setattr(studio, "load_renderer", lambda *a, **k: None)
        monkeypatch.setattr(trainer, "snapshot", lambda *a, **k: {
            "pi_sdk": {"module_sha256": "fixture", "module_path": str(tmp_path)},
            "policy_identity": {**service.engine.identity, "backend": "studio", "identity_sha256": digest(service.engine.identity)},
            "codex": {"executable_sha256": "fixture"}})
        class ReadyToCollect(Exception): pass
        prepared = []
        def stop_before_gms(*args, **kwargs):
            prepared.append(kwargs)
            raise ReadyToCollect()
        monkeypatch.setattr(inference, "open_rollout_stack", stop_before_gms)
        with pytest.raises(ReadyToCollect):
            asyncio.run(inference.run_inference_campaign(config, sampler_path=policy))
        backend = prepared[0]["prepared_sampler"].backend
        assert backend.client.client_id != training
        assert prepared[0]["pinned_sampler_path"] == backend.policy_ref == policy
        assert service.clients[training] == previous
        assert not service.engine.clients
        assert service.engine.events[-1][0] == "create"
        assert service.engine.events[-1][2] == TrainSection().learning_rate
    finally:
        http.shutdown();http.server_close();thread.join();service.close()


def test_generated_operation_ids_fit_for_long_valid_run_labels(monkeypatch, tmp_path):
    from shellsmith.train import studio
    config = TrainConfig(run=RunSection(run_id="r" * 100, log_dir=str(tmp_path / "run")), compute=ComputeSection(backend="studio"))
    validate_train_config(config)
    requests = []
    class RPC:
        def __init__(self, *args, **kwargs): pass
        def capabilities(self): return {"model": config.studio.model_id, "context_window": 262144, "identity": {}, "cache_precisions": ["bf16"]}
        async def acall(self, kind, body, *, request_id):
            identifier(request_id); requests.append((kind, request_id))
            return {"revision": 0, "policy": "policy", "checkpoint": "checkpoint"}
        def call(self, kind, body, *, request_id):
            identifier(request_id); requests.append((kind, request_id))
            return {"sampled_tokens": [], "behavior_logprobs": [], "finish_reason": "stop", "logprobs": []}
    monkeypatch.setattr(studio, "Client", RPC)
    monkeypatch.setattr(studio, "load_renderer", lambda *a, **k: None)
    backend = studio.StudioBackend(config)
    name = config.run.run_id + "-initial-step00000-sampler"
    asyncio.run(backend.initialize())
    asyncio.run(backend.save_checkpoint(kind="sampler", name=name))
    asyncio.run(backend.save_checkpoint(kind="sampler", name=name))
    asyncio.run(backend.publish_policy(revision=0))
    backend.close()
    session = RemoteSamplingSession(backend, "policy", "request" * 50)
    session.force_tokens([1], [2])
    assert requests[1][1] == requests[2][1]  # stable checkpoint retry
    assert len(requests[1][1]) == 74
    assert all(len(request_id) <= 200 for _, request_id in requests)
    assert bounded_identifier("label" * 100) != bounded_identifier("other" * 100)
    update = bounded_identifier(config.run.run_id * 3 + ":segment:100:update:200", limit=160)
    identifier(update + ":backward");identifier(update + ":optimizer");identifier(update + ":recover:200")


def test_abandoned_checkpoint_staging_removed_after_real_process_exit(tmp_path):
    service = Service(tmp_path, Engine)
    committed = call(service, "a", "create", {}, "committed")["checkpoint"]
    service.close()
    code = '''import os,sys
from studio_compute.storage import Store
s=Store(sys.argv[1]);s.submit("crashed-create","b","create",{})
def writer(folder):
 (folder/"optimizer.safetensors").write_bytes(b"partial"*1000)
 os._exit(17)
s.checkpoint("crashed-create",writer,identity={},state={},result={})
'''
    assert subprocess.run([sys.executable, "-c", code, str(tmp_path)]).returncode == 17
    checkpoints = tmp_path / "checkpoints"
    assert len(list(checkpoints.glob(".pending-*"))) == 1
    unrelated = checkpoints / ".pending-not-a-generated-uuid";unrelated.mkdir()
    store = Store(tmp_path)
    try:
        store.recover(Engine.identity)
        assert list(checkpoints.glob(".pending-*")) == [unrelated]
        assert store.operation("crashed-create")["status"] == "interrupted"
        store.verify(committed, Engine.identity)
    finally:
        store.close()


def test_staging_cleanup_rejects_symlinks_without_touching_target(tmp_path):
    outside = tmp_path / "outside";outside.mkdir();(outside / "keep").write_bytes(b"safe")
    store = Store(tmp_path / "state")
    link = store.checkpoints / (".pending-" + uuid.uuid4().hex);link.symlink_to(outside, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match="Unsafe checkpoint staging"):
            store.recover(Engine.identity)
        assert (outside / "keep").read_bytes() == b"safe" and link.is_symlink()
    finally:
        store.close()
