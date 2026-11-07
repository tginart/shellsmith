"""Independent output directories cannot share durable sampling receipts/RNG."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading

import pytest

from shellsmith.campaign import inference
from shellsmith.train import studio, trainer
from shellsmith.train.config import TrainConfig, RunSection, ComputeSection
from studio_compute.http import server
from studio_compute.protocol import digest
from studio_compute.service import Service
from test_studio_service import Engine, call


@pytest.mark.parametrize("second", ["identical", "prompt", "policy"])
def test_campaign_output_namespaces_clients_receipts_and_rng(tmp_path, monkeypatch, second):
    class SamplingEngine(Engine):
        capabilities = {**Engine.capabilities, "model": "Qwen/Qwen3.8-27B", "context_window": 262144}
        def work(self, client, kind, body):
            if kind in ("sample", "logprobs"):
                self.events.append((client, kind, body["seed"], tuple(body["prompt_token_ids"]), body["policy"]))
                if kind == "logprobs":
                    return {"logprobs": [-1.0] * len(body["target_token_ids"])}
                return {"sampled_tokens": [body["seed"] % 16], "behavior_logprobs": [-1.0],
                        "finish_reason": "length", "policy_revision": body["revision"]}
            return (yield from super().work(client, kind, body))
    service = Service(tmp_path / "service", SamplingEngine)
    http = server(service, ("127.0.0.1", 0))
    thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
    try:
        policies = []
        for index in range(2):
            call(service, f"source-{index}", "create", {"seed": index}, f"source-create-{index}")
            policies.append(call(service, f"source-{index}", "publish", {"revision": 0}, f"source-publish-{index}")["policy"])
        config = TrainConfig(run=RunSection(run_id="same-label", log_dir=str(tmp_path / "first")),
                             compute=ComputeSection(backend="studio"))
        config = replace(config, studio=replace(config.studio, endpoint=f"http://127.0.0.1:{http.server_port}"),
                         worlds=replace(config.worlds, count=1, seed=7))
        monkeypatch.setattr(studio, "load_renderer", lambda *a, **k: None)
        monkeypatch.setattr(trainer, "snapshot", lambda *a, **k: {
            "pi_sdk": {"module_sha256": "fixture", "module_path": str(tmp_path)},
            "policy_identity": {**service.engine.identity, "backend": "studio", "identity_sha256": digest(service.engine.identity)},
            "codex": {"executable_sha256": "fixture"}})
        class CollectedProbe(Exception): pass
        backends, results = [], []
        def collect(*args, **kwargs):
            backend = kwargs["prepared_sampler"].backend
            prompt = [2] if backends and second == "prompt" else [1]
            backends.append(backend)
            def draw():
                session = studio.RemoteSamplingSession(backend, backend.policy_ref, "same-request")
                drawn = session.draw_pass(prompt, maximum=1, stop_ids=set())
                forced = session.force_tokens(drawn[-1], [3])
                return drawn, forced
            result = draw()
            count = len(service.engine.events)
            assert draw() == result  # Recreated cursor retries the original durable IDs.
            assert len(service.engine.events) == count
            results.append(result)
            # Exercise generic token calls while this campaign still owns its
            # acquisition; cleanup now deliberately fences later calls.
            async def token_calls():
                direct = await backend.sample_tokens(policy=policies[0], prompt_token_ids=[1],
                                                     max_tokens=1, request_id="same-direct")
                assert await backend.sample_tokens(policy=policies[0], prompt_token_ids=[1],
                                                   max_tokens=1, request_id="same-direct") == direct
                assert (await backend.token_logprobs(policy=policies[0], prompt_token_ids=[1],
                                                     target_token_ids=[2], request_id="same-score"))["logprobs"] == [-1.0]
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(asyncio.run, token_calls()).result(timeout=10)
            raise CollectedProbe
        monkeypatch.setattr(inference, "open_rollout_stack", collect)
        for index, output in enumerate(("first", "second")):
            current = replace(config, run=replace(config.run, log_dir=str(tmp_path / output)))
            policy = policies[1] if index and second == "policy" else policies[0]
            with pytest.raises(CollectedProbe):
                asyncio.run(inference.run_inference_campaign(current, sampler_path=policy))
        clients = [backend.client.client_id for backend in backends]
        assert clients[0] != clients[1]
        assert backends[0].operation_prefix != backends[1].operation_prefix
        events = [event for event in service.engine.events if event[1] == "sample"][::2]
        assert [event[0] for event in events] == clients
        assert events[0][2] == events[1][2]  # Both start from the configured seed/cursor.
        assert [service.clients[client]["rng_counter"] for client in clients] == [2, 2]
        rows = service.store.db.execute("SELECT id,client FROM operations WHERE kind IN ('sample','logprobs')").fetchall()
        assert len(rows) == 8 and len({row["id"] for row in rows}) == 8
        if second == "identical":
            assert results[0] == results[1]
    finally:
        http.shutdown(); http.server_close(); thread.join(); service.close()


def test_campaign_canonical_output_alias_preserves_namespace(tmp_path, monkeypatch):
    # Derive IDs through the actual campaign setup while stopping before mutation.
    from test_trainer_resume import _runtime
    config, _ = _runtime(tmp_path, monkeypatch)
    target = tmp_path / "outputs"; target.mkdir()
    alias = tmp_path / "alias"; alias.symlink_to(target, target_is_directory=True)
    configs = [replace(config, run=replace(config.run, log_dir=str(base / "campaign")),
                       compute=replace(config.compute, backend="studio")) for base in (target, alias)]
    monkeypatch.setattr(trainer, "snapshot", lambda *a, **k: {
        "pi_sdk": {"module_sha256": "fixture", "module_path": str(tmp_path)},
        "policy_identity": {}, "codex": {"executable_sha256": "fixture"}})
    clients = []
    class DiscoveryProbe(Exception): pass
    def discover(config, *, client_id):
        clients.append(client_id)
        raise DiscoveryProbe
    monkeypatch.setattr(studio, "StudioBackend", discover)
    for config in configs:
        with pytest.raises(DiscoveryProbe):
            asyncio.run(inference.run_inference_campaign(config))
    assert clients[0] == clients[1]
