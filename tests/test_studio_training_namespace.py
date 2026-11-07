"""Selected training clients isolate globally keyed mutation receipts."""

import asyncio
from dataclasses import replace
import threading

from shellsmith.train import studio
from shellsmith.train.backend import TokenSequence, TokenBatch
from shellsmith.train.config import TrainConfig, RunSection, ComputeSection
from studio_compute.http import server
from studio_compute.service import Service
from test_studio_service import Engine


def test_same_run_label_with_different_training_clients_can_both_update(tmp_path, monkeypatch):
    class TrainingEngine(Engine):
        capabilities = {**Engine.capabilities, "model": "Qwen/Qwen3.8-27B", "context_window": 262144}
    service = Service(tmp_path / "service", TrainingEngine)
    http = server(service, ("127.0.0.1", 0))
    thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
    monkeypatch.setattr(studio, "load_renderer", lambda *a, **k: None)
    try:
        backends = []
        for label in ("first", "second"):
            config = TrainConfig(run=RunSection(run_id="same-run", log_dir=str(tmp_path / label)),
                                 compute=ComputeSection(backend="studio"))
            config = replace(config, studio=replace(config.studio, endpoint=f"http://127.0.0.1:{http.server_port}",
                                                   training_client=label))
            backends.append(studio.StudioBackend(config))
        async def update(backend):
            original = await backend.initialize()
            assert await backend.initialize() == original
            sequence = TokenSequence((1, 2), (3,), (-1.0,), (0.1,), (1.0,), "row")
            batch = TokenBatch((sequence,))
            # Exercise the explicit segment ID used by the trainer as well as
            # default IDs used by the convenience update API.
            request = backend.operation_prefix + ":segment:0:update:1"
            result = await backend.update(batch, learning_rate=0.001, clip_low=0.0, clip_high=4.0, request_id=request)
            assert await backend.update(batch, learning_rate=0.001, clip_low=0.0, clip_high=4.0, request_id=request) == result
            await backend.update(batch, learning_rate=0.001, clip_low=0.0, clip_high=4.0)
            return result
        async def run(): return await asyncio.gather(*(update(backend) for backend in backends))
        assert all(result["optim_metrics"]["optimizer_steps"] == 1 for result in asyncio.run(run()))
        assert backends[0].operation_prefix != backends[1].operation_prefix
        assert {client: state["step"] for client, state in service.engine.clients.items()} == {"first": 2, "second": 2}
        rows = service.store.db.execute("SELECT id,client FROM operations WHERE kind IN ('create','optim_step')").fetchall()
        assert len(rows) == len({row["id"] for row in rows}) == 6
    finally:
        http.shutdown(); http.server_close(); thread.join(); service.close()
