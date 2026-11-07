"""Behavioral protocol tests use a deterministic engine with resumable work."""

import asyncio
import json
import subprocess
import sys
import threading
import time

import pytest

from studio_compute.client import Client
from studio_compute.http import server
from studio_compute.protocol import Conflict
from studio_compute.service import Service
from studio_compute.storage import Store


def test_inference_admission_includes_detached_cache_snapshot():
    from types import SimpleNamespace
    from studio_compute.runtime import RuntimeEngine

    engine = RuntimeEngine.__new__(RuntimeEngine)
    engine.resident = SimpleNamespace(estimate_inference=lambda length, **options: 8)
    engine.prefixes = SimpleNamespace(limit=8)
    engine.reserved = {}
    executed = []

    def admit(additional):
        if additional > 12:
            raise MemoryError("Insufficient working and retained cache budget.")

    def sample_units(body, started):
        executed.append(True)
        yield {"phase": "decode"}
        return {"sampled_tokens": [3]}

    engine.admit, engine._sample_units = admit, sample_units
    body = {"prompt_token_ids": [1, 2], "max_tokens": 1}
    with pytest.raises(MemoryError, match="retained cache budget"):
        list(engine._sample(body, 0))
    assert not executed and not engine.reserved
    list(engine._sample({**body, "use_prefix_cache": False}, 0))
    assert executed == [True] and not engine.reserved


class Engine:
    identity = {"model": "test-frozen-base", "numerics": "bf16-fixture"}
    capabilities = {
        "context_window": 32,
        "vocabulary_size": 16,
        "cache_precisions": ["bf16"],
    }
    tokenizer_bundle = {"hash": "test", "files": {}}

    def __init__(self):
        self.base = object()
        self.clients, self.policies, self.events = {}, {}, []
        self.thread = threading.get_ident()
        self.write_failure = False

    def create(self, client, body):
        assert threading.get_ident() == self.thread
        self.clients[client] = {
            "step": 0,
            "weight": body.get("seed", 0),
            "gradient": None,
            "base": self.base,
        }
        return {}

    def restore(self, client, folder, state):
        self.clients[client] = {
            **json.loads((folder / "state.json").read_text()),
            "gradient": None,
            "base": self.base,
        }

    def write_state(self, client, folder):
        if self.write_failure:
            raise OSError("injected disk failure")
        state = {
            k: v
            for k, v in self.clients[client].items()
            if k not in ("base", "gradient")
        }
        (folder / "state.json").write_text(json.dumps(state))

    def policy(self, reference, folder, state, *, adapter_digest):
        self.policies[reference] = json.loads((folder / "state.json").read_text())

    def work(self, client, kind, body):
        assert threading.get_ident() == self.thread
        if kind == "forward_backward":
            for index in range(12):
                self.events.append((client, "backward", index))
                time.sleep(0.003)
                yield {"unit": index}
            self.clients[client]["gradient"] = body["operation_id"]
            return {"forward_metrics": {"loss": 2.0, "reduction": "sum"}}
        if kind == "optim_step":
            if self.clients[client]["gradient"] != body["backward_operation"]:
                raise ValueError("No matching live backward")
            self.clients[client]["step"] += 1
            self.clients[client]["weight"] += 2
            self.clients[client]["gradient"] = None
            return {"optim_metrics": {"optimizer_steps": self.clients[client]["step"]}}
        if kind == "inspect":
            return {"weight": self.clients[client]["weight"], "step": self.clients[client]["step"]}
        if kind == "sample":
            self.events.append((client, "sample"))
            return {
                "sampled_tokens": [self.policies[body["policy"]]["weight"] % 16],
                "policy_revision": body["revision"],
            }
        raise ValueError(kind)

    def abort(self, client, operation_id=None):
        if client in self.clients:
            self.clients[client]["gradient"] = None

    def close_client(self, client):
        self.clients.pop(client)

    def close(self):
        pass


@pytest.fixture
def service(tmp_path):
    instance = Service(tmp_path, Engine)
    yield instance
    instance.close()


def call(service, client, kind, body, request_id):
    service.submit(request_id, client, kind, body)
    op = service.wait(request_id)
    assert op["status"] == "succeeded", op
    return op["result"]


def batch():
    return {
        "sequences": [
            {
                "prompt_token_ids": [1, 2],
                "sampled_tokens": [3, 4],
                "behavior_logprobs": [-1.0, -1.0],
                "advantages": [1.0, 1.0],
                "loss_mask": [1.0, 1.0],
            }
        ]
    }


def test_retry_conflict_and_durable_optimizer(service):
    call(service, "a", "create", {"seed": 3}, "create")
    call(service, "a", "forward_backward", batch(), "back")
    result = call(
        service,
        "a",
        "optim_step",
        {"learning_rate": 1.0, "backward_operation": "back"},
        "step",
    )
    assert (
        service.submit(
            "step",
            "a",
            "optim_step",
            {"learning_rate": 1.0, "backward_operation": "back"},
        )["result"]
        == result
    )
    assert service.engine.clients["a"]["step"] == 1
    folder, document = service.store.verify(result["checkpoint"], Engine.identity)
    assert json.loads((folder / "state.json").read_text())["step"] == 1
    assert document["result"] == result
    with pytest.raises(Conflict):
        service.submit(
            "step",
            "a",
            "optim_step",
            {"learning_rate": 2.0, "backward_operation": "back"},
        )


def test_client_isolation_pinned_publication_and_interleaving(service):
    call(service, "a", "create", {"seed": 2}, "ca")
    call(service, "b", "create", {"seed": 7}, "cb")
    policy = call(service, "a", "publish", {"revision": 0}, "pa")["policy"]
    assert service.engine.clients["a"]["base"] is service.engine.clients["b"]["base"]
    service.submit("long", "a", "forward_backward", batch())
    while not service.engine.events:
        time.sleep(0.001)
    sampled = call(
        service,
        "b",
        "sample",
        {"policy": policy, "prompt_token_ids": [1], "max_tokens": 1},
        "sample",
    )
    assert service.operation("long")["status"] == "running"
    assert sampled == {"sampled_tokens": [2], "policy_revision": 0}
    assert service.wait("long")["status"] == "succeeded"
    call(
        service,
        "a",
        "optim_step",
        {"learning_rate": 1.0, "backward_operation": "long"},
        "step",
    )
    new = call(service, "a", "publish", {"revision": 1}, "pa1")["policy"]
    assert new != policy
    assert call(
        service,
        "b",
        "sample",
        {"policy": policy, "prompt_token_ids": [1], "max_tokens": 1},
        "old",
    )["sampled_tokens"] == [2]
    assert service.engine.clients["b"]["step"] == 0


@pytest.mark.parametrize("limit", [1, 2, 4])
def test_eight_queued_rollouts_obey_memory_batch_limit(tmp_path, limit):
    class BatchEngine(Engine):
        def inference_batch_capacity(self, body, requested):
            return min(limit, requested)

        def sample_batch(self, bodies):
            self.events.append(("batch", len(bodies)))
            yield {"batch_size": len(bodies)}
            return [
                {"sampled_tokens": [self.policies[b["policy"]]["weight"] % 16],
                 "policy_revision": b["revision"]}
                for b in bodies
            ]

    instance = Service(tmp_path, BatchEngine)
    try:
        call(instance, "a", "create", {"seed": 2}, "create")
        policy = call(instance, "a", "publish", {"revision": 0}, "publish")["policy"]
        with instance.condition:
            for index in range(8):
                instance.submit(f"rollout-{index}", "a", "sample", {
                    "policy": policy, "prompt_token_ids": [1], "max_tokens": 1,
                })
        for index in range(8):
            op = instance.wait(f"rollout-{index}")
            assert op["status"] == "succeeded"
            assert op["result"]["sampled_tokens"] == [2]
        batches = [e[1] for e in instance.engine.events if e[0] == "batch"]
        assert batches == ([] if limit == 1 else [limit] * (8 // limit))
        metrics = instance.inference_metrics.readout()
        assert metrics["generated_tokens_total"] == 8
        assert metrics["completed_requests_total"] == 8
        assert metrics["active_requests"] == metrics["queued_requests"] == 0
    finally:
        instance.close()


@pytest.mark.parametrize("fail", [False, True])
def test_live_batch_metrics_track_finished_members_and_partial_failures(tmp_path, fail):
    progressed, release = threading.Event(), threading.Event()

    class ProgressEngine(Engine):
        def sample_batch(self, bodies):
            yield {"phase": "batched_decode", "generated_tokens": [1, 1],
                   "finished_requests": [True, False]}
            # The owner has consumed and observed the previous yield.
            progressed.set()
            assert release.wait(5)
            if fail:
                raise RuntimeError("injected generation failure")
            yield {"phase": "batched_decode", "generated_tokens": [1, 3],
                   "finished_requests": [True, True]}
            return [{"sampled_tokens": [1]}, {"sampled_tokens": [1, 2, 3]}]

    instance = Service(tmp_path, ProgressEngine)
    try:
        call(instance, "a", "create", {}, "create")
        policy = call(instance, "a", "publish", {"revision": 0}, "publish")["policy"]
        body = {"policy": policy, "prompt_token_ids": [1], "max_tokens": 3}
        with instance.condition:
            instance.submit("one", "a", "sample", body)
            instance.submit("two", "a", "sample", body)
        assert progressed.wait(5)
        metrics = instance.inference_metrics.readout()
        assert metrics["generated_tokens_total"] == 2 and metrics["active_requests"] == 1
        assert metrics["queued_requests"] == 0
        # An HTTP retry of the same durable operation is not more inference.
        instance.submit("one", "a", "sample", body)
        assert instance.inference_metrics.readout()["queued_requests"] == 0
        release.set()
        for request in ("one", "two"):
            assert instance.wait(request)["status"] == ("failed" if fail else "succeeded")
        metrics = instance.inference_metrics.readout()
        assert metrics["generated_tokens_total"] == (2 if fail else 4)
        assert metrics["failed_requests_total"] == (2 if fail else 0)
        assert metrics["active_requests"] == metrics["queued_requests"] == 0
    finally:
        release.set()
        instance.close()
    rows = [json.loads(line) for line in (tmp_path / "inference-telemetry.jsonl").read_text().splitlines()]
    assert sum(row["generated_tokens"] for row in rows) == (2 if fail else 4)
    assert rows[-1]["active_requests"] == 0


def test_scoring_and_durable_retries_do_not_inflate_generation(service):
    call(service, "a", "create", {}, "create")
    policy = call(service, "a", "publish", {"revision": 0}, "publish")["policy"]
    body = {"policy": policy, "prompt_token_ids": [1], "max_tokens": 1}
    call(service, "a", "sample", body, "draw")
    call(service, "a", "sample", body, "draw")
    # The fixture has no scoring implementation: a failed scoring operation
    # still must not enter the sampling population.
    service.submit("score", "a", "logprobs", {"policy": policy, "prompt_token_ids": [1], "target_token_ids": [2]})
    assert service.wait("score")["status"] == "failed"
    metrics = service.inference_metrics.readout()
    assert metrics["generated_tokens_total"] == metrics["completed_requests_total"] == 1
    assert metrics["failed_requests_total"] == 0


def test_invalid_policy_leaves_no_active_sampling_gauge(service):
    call(service, "a", "create", {}, "create")
    service.submit("bad", "a", "sample", {"policy": "unpublished", "prompt_token_ids": [1], "max_tokens": 1})
    assert service.wait("bad")["status"] == "failed"
    metrics = service.inference_metrics.readout()
    assert metrics["active_requests"] == metrics["queued_requests"] == 0
    assert metrics["failed_requests_total"] == 1
    assert metrics["admitted_batches_total"] == 0


def test_materialized_token_before_next_gpu_step_failure_is_counted(tmp_path):
    class ObserverEngine(Engine):
        def set_inference_observer(self, observer):
            assert threading.get_ident() == self.thread
            self.observer = observer

        def work(self, client, kind, body):
            if kind == "sample":
                self.observer(body["operation_id"], 1, finished=False)
                raise RuntimeError("Next GPU evaluation failed before progress yield")
            return (yield from super().work(client, kind, body))

    instance = Service(tmp_path, ObserverEngine)
    try:
        call(instance, "a", "create", {}, "create")
        policy = call(instance, "a", "publish", {"revision": 0}, "publish")["policy"]
        instance.submit("draw", "a", "sample", {"policy": policy, "prompt_token_ids": [1], "max_tokens": 2})
        assert instance.wait("draw")["status"] == "failed"
        metrics = instance.inference_metrics.readout()
        assert metrics["generated_tokens_total"] == metrics["failed_requests_total"] == 1
        assert metrics["active_requests"] == 0
    finally:
        instance.close()


def test_exact_admission_and_overflow(service):
    call(service, "a", "create", {}, "c")
    policy = call(service, "a", "publish", {"revision": 0}, "p")["policy"]
    call(
        service,
        "a",
        "sample",
        {"policy": policy, "prompt_token_ids": [1] * 31, "max_tokens": 1},
        "exact",
    )
    with pytest.raises(ValueError, match="overflow"):
        service.submit(
            "overflow",
            "a",
            "sample",
            {"policy": policy, "prompt_token_ids": [1] * 32, "max_tokens": 1},
        )
    with pytest.raises(KeyError):
        service.operation("overflow")
    # Internal benchmark forcing must never become a public sampling escape
    # hatch around the reserved completion count or vocabulary validation.
    with pytest.raises(ValueError, match="Use logprobs"):
        service.submit(
            "forced-overflow",
            "a",
            "sample",
            {
                "policy": policy,
                "prompt_token_ids": [1] * 31,
                "max_tokens": 1,
                "target_token_ids": [2, 3],
            },
        )
    with pytest.raises(KeyError):
        service.operation("forced-overflow")


def test_checkpoint_failure_rolls_back_and_publication_rejects(service):
    call(service, "a", "create", {}, "c")
    call(service, "a", "forward_backward", batch(), "b")
    service.engine.write_failure = True
    service.submit(
        "o", "a", "optim_step", {"learning_rate": 1.0, "backward_operation": "b"}
    )
    assert service.wait("o")["status"] == "failed"
    assert service.engine.clients["a"]["step"] == 0
    service.submit("p", "a", "publish", {"revision": 1})
    assert service.wait("p")["status"] == "failed"


def test_restart_recovers_receipts_and_interrupted_work(tmp_path):
    service = Service(tmp_path, Engine)
    created = call(service, "a", "create", {"seed": 2}, "c")
    service.submit("b", "a", "forward_backward", batch())
    while service.operation("b")["status"] == "queued":
        time.sleep(0.001)
    service.close()
    fresh = Service(tmp_path, Engine)
    try:
        assert fresh.operation("b")["status"] == "interrupted"
        assert fresh.operation("c")["result"] == created
        assert not fresh.engine.clients  # recovered clients start cold
        assert call(fresh, "a", "inspect", {}, "probe")["weight"] == 2
    finally:
        fresh.close()


def test_checkpoint_commit_gap_is_recovered_without_reexecution(tmp_path):
    store = Store(tmp_path)
    store.submit("create", "a", "create", {})
    store.running("create")

    def writer(folder):
        (folder / "state.json").write_text('{"step":0,"weight":9}')

    state, result = store.checkpoint(
        "create",
        writer,
        identity=Engine.identity,
        state={"revision": 0, "seed": 9, "closed": False},
        result={"revision": 0},
    )
    store.close()
    service = Service(tmp_path, Engine)
    try:
        assert service.operation("create")["result"] == result
        assert not service.engine.clients
        assert call(service, "a", "inspect", {}, "probe")["weight"] == 9
    finally:
        service.close()


def test_failure_after_optimizer_checkpoint_rename_commits_original_receipt(
    service, monkeypatch
):
    call(service, "a", "create", {}, "create")
    call(service, "a", "forward_backward", batch(), "back")
    original = service.store.finish
    failed = []

    def finish(request_id, *args, **kwargs):
        if request_id == "optimizer" and not failed:
            failed.append(request_id)
            raise OSError("failure after durable checkpoint rename")
        return original(request_id, *args, **kwargs)

    monkeypatch.setattr(service.store, "finish", finish)
    result = call(
        service,
        "a",
        "optim_step",
        {"learning_rate": 1.0, "backward_operation": "back"},
        "optimizer",
    )
    assert failed == ["optimizer"] and result["optim_metrics"]["optimizer_steps"] == 1
    assert service.engine.clients["a"]["step"] == 1


def test_completed_backward_without_optimizer_is_interrupted_after_restart(tmp_path):
    service = Service(tmp_path, Engine)
    call(service, "a", "create", {}, "create")
    call(service, "a", "forward_backward", batch(), "back")
    service.close()
    fresh = Service(tmp_path, Engine)
    try:
        assert fresh.operation("back")["status"] == "interrupted"
        fresh.submit(
            "optim",
            "a",
            "optim_step",
            {"learning_rate": 1.0, "backward_operation": "back"},
        )
        assert fresh.wait("optim")["status"] == "failed"
        assert fresh.engine.clients["a"]["step"] == 0
    finally:
        fresh.close()


def test_corruption_fails_closed(tmp_path):
    service = Service(tmp_path, Engine)
    checkpoint = call(service, "a", "create", {}, "c")["checkpoint"]
    folder, _ = service.store.verify(checkpoint)
    service.close()
    (folder / "state.json").write_text("corrupted")
    with pytest.raises(ValueError, match="corruption"):
        Service(tmp_path, Engine)


def test_http_client_without_accelerator_imports(service):
    http = server(service, ("127.0.0.1", 0))
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        client = Client(f"http://127.0.0.1:{http.server_port}", "remote")
        assert client.capabilities()["context_window"] == 32
        assert client.metrics()["generated_tokens_total"] == 0
        scoped = client.metrics(client_id="remote")
        assert scoped["scope"] == "service"
        assert scoped["client_metrics"]["client_id"] == "remote"
        assert scoped["client_metrics"]["scope"] == "client"
        assert scoped["max_inference_batch"] == 1
        client.call("create", {}, request_id="create-remote")
        phases = []
        result = asyncio.run(
            client.update(batch()["sequences"], learning_rate=1.0, request_id="update",
                          on_phase=phases.append)
        )
        assert result["optim_metrics"]["optimizer_steps"] == 1
        assert phases == ["forward_backward", "optimizer"]
        script = "import sys; import studio_compute.client; assert not any(x.startswith(('mlx', 'shellsmith')) for x in sys.modules)"
        subprocess.run([sys.executable, "-c", script], check=True)
    finally:
        http.shutdown()
        http.server_close()


def test_live_training_progress_never_exposes_request_body(service):
    with service.condition:
        service.store.submit("telemetry-progress", "example", "forward_backward", {"secret": "not-visible"})
        service.store.running("telemetry-progress", {"phase": "backward", "completed_tokens": 4})
        service.active["example"] = ("telemetry-progress", "example", "forward_backward", iter(()))
        row = service.telemetry()
        del service.active["example"]
    assert row["training_operations"]["telemetry-progress"]["progress"]["completed_tokens"] == 4
    assert "not-visible" not in str(row)
    assert row["memory_peak_scope"] == "process_lifetime"



@pytest.mark.parametrize("invalid", ["missing", "stale", "closed", "all_missing"])
def test_batch_occupancy_excludes_rejected_members(tmp_path, invalid):
    observed = []

    class BatchEngine(Engine):
        def sample_batch(self, bodies):
            observed.append(len(bodies))
            yield {"phase": "batched_decode", "generated_tokens": [1] * len(bodies)}
            return [{"sampled_tokens": [1]} for _ in bodies]

    instance = Service(tmp_path, BatchEngine)
    try:
        call(instance, "a", "create", {}, "create")
        policy = call(instance, "a", "publish", {"revision": 0}, "publish")["policy"]
        if invalid == "closed":
            call(instance, "b", "create", {}, "create-b")
            call(instance, "b", "close", {}, "close-b")
        body = {"policy": policy, "prompt_token_ids": [1], "max_tokens": 1}
        bad_client = "b" if invalid == "closed" else "missing"
        bad_body = body
        if invalid == "stale":
            bad_client, bad_body = "a", body | {"expected_owner": "stale"}
        with instance.condition:
            instance.submit("bad", bad_client, "sample", bad_body)
            instance.submit("other", "missing" if invalid == "all_missing" else "a",
                            "sample", body)
        assert instance.wait("bad")["status"] == "failed"
        assert instance.wait("other")["status"] == (
            "failed" if invalid == "all_missing" else "succeeded")
        count = 0 if invalid == "all_missing" else 1
        assert observed == ([1] if count else [])
        metrics = instance.inference_metrics.readout()
        assert metrics["admitted_batches_total"] == count
        assert metrics["last_admitted_batch_size"] == count
        assert metrics["mean_admitted_batch_size"] == count
    finally:
        instance.close()


def test_metrics_iteration_survives_concurrent_operation_completion(
    tmp_path, monkeypatch,
) -> None:
    """
    Purpose: Keep metrics reads valid while the owner retires a completed lane.
    Input: Two active lanes and a metrics read at the first lane's completion.
    Output: A complete response without a dictionary-size error.
    """
    instance = Service(tmp_path, Engine)
    completed, release = threading.Event(), threading.Event()
    execute = instance._execute
    operation = instance.store.operation

    def paused_completion(request_id, client, kind, body):
        result = yield from execute(request_id, client, kind, body)
        if request_id == "back-a":
            completed.set()
            assert release.wait(5)
        return result

    def read_during_completion(request_id):
        if request_id == "back-a" and threading.current_thread() is not instance.owner:
            release.set()
            deadline = time.monotonic() + .1
            while "training:a" in instance.active and time.monotonic() < deadline:
                time.sleep(.001)
        return operation(request_id)

    try:
        call(instance, "a", "create", {}, "create-a")
        call(instance, "b", "create", {}, "create-b")
        monkeypatch.setattr(instance, "_execute", paused_completion)
        with instance.condition:
            instance.submit("back-a", "a", "forward_backward", batch())
            instance.submit("back-b", "b", "forward_backward", batch())
        assert completed.wait(5)
        monkeypatch.setattr(instance.store, "operation", read_during_completion)
        row = instance.telemetry()
        assert set(row["training_operations"]) == {"back-a", "back-b"}
        assert instance.wait("back-a")["status"] == "succeeded"
        assert instance.wait("back-b")["status"] == "succeeded"
    finally:
        release.set()
        instance.close()
