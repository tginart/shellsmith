"""Persistent scheduler regressions with deterministic, CPU-only request state."""

import hashlib
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from types import SimpleNamespace
from typing import ClassVar

import pytest
from test_studio_service import Engine, call

from studio_compute.cli import parse_args
from studio_compute.mlx_racing import ContinuousBatcher
from studio_compute.service import Service


def make_scheduler(concurrency=3, *, memory_slots=None):
    events, released, completed, failed = [], [], {}, {}
    live = set()

    def create(request_id, body):
        if body.get("invalid"):
            raise ValueError("invalid request")
        if memory_slots is not None and len(live) >= memory_slots:
            return None
        live.add(request_id)
        events.append(("admit", request_id))
        return SimpleNamespace(
            request_id=request_id,
            body=body,
            tokens=[],
            offset=0,
            logits=None,
            needs_decode=False,
            reason=None,
        )

    def prefill(request):
        request.offset += min(2, request.body["prompt_length"] - request.offset)
        events.append(("prefill", request.request_id, request.offset))
        if request.offset == request.body["prompt_length"]:
            request.logits = True

    def advance(requests):
        events.append(("advance", tuple(r.request_id for r in requests)))
        for request in requests:
            request.needs_decode = False

    def sample(requests):
        events.append(("sample", tuple(r.request_id for r in requests)))
        for request in requests:
            if request.body.get("fail"):
                raise RuntimeError("kernel failed")
            request.tokens.append(len(request.tokens))
            request.reason = (
                "length" if len(request.tokens) == request.body["maximum"] else None
            )
            request.needs_decode = request.reason is None

    def release(request):
        live.remove(request.request_id)
        released.append(request.request_id)

    scheduler = ContinuousBatcher(
        concurrency,
        create=create,
        prefill=prefill,
        advance=advance,
        sample=sample,
        finish=lambda r: completed.update({r.request_id: list(r.tokens)}),
        fail=lambda name, error: failed.update({name: str(error)}),
        release=release,
    )
    return scheduler, events, released, completed, failed


def body(length=2, maximum=8, policy="p", **extra):
    return {"policy": policy, "prompt_length": length, "maximum": maximum, **extra}


@pytest.mark.parametrize("concurrency", [1, 2, 3, 5, 8])
def test_mixed_lengths_refill_and_bounded_prefill(concurrency):
    scheduler, events, released, completed, failed = make_scheduler(concurrency)
    for i in range(20):
        scheduler.enqueue(str(i), body(length=1 + i % 7, maximum=1 + i % 9))
    steps = []
    while scheduler.pending or scheduler.active:
        steps.append(scheduler.step())
        assert len(scheduler.active) <= concurrency
        assert len(steps) < 200
    assert len(completed) == len(set(released)) == 20 and not failed
    assert all(len(tokens) == 1 + int(name) % 9 for name, tokens in completed.items())
    if concurrency > 1:
        assert any(max(step["batch_sizes"], default=0) > 1 for step in steps)
        admissions = [i for i, event in enumerate(events) if event[0] == "admit"]
        assert any(event[0] == "advance" for event in events[admissions[concurrency] :])
    # Each turn contains at most one globally bounded prompt chunk.
    assert sum(event[0] == "prefill" for event in events) <= len(steps)


def test_live_arrival_refills_before_survivor_finishes():
    scheduler, events, _, completed, _ = make_scheduler(2)
    scheduler.enqueue("long", body(maximum=10))
    scheduler.enqueue("short", body(maximum=1))
    scheduler.step()
    scheduler.step()
    assert "short" in completed and "long" not in completed
    scheduler.enqueue("new", body(maximum=2))
    scheduler.step()
    assert ("sample", ("long", "new")) in events
    scheduler.close()


def test_memory_deferral_cancel_and_failure_release():
    scheduler, _, released, completed, failed = make_scheduler(7, memory_slots=2)
    scheduler.enqueue("bad", body(invalid=True))
    scheduler.enqueue("a", body(maximum=2))
    scheduler.enqueue("b", body(length=5))
    scheduler.enqueue("c", body(policy="other", fail=True))
    scheduler.enqueue("queued-cancel", body())
    scheduler.step()
    assert "bad" in failed and len(scheduler.active) == 2
    scheduler.cancel("queued-cancel")
    scheduler.cancel("b")
    while scheduler.pending or scheduler.active:
        scheduler.step()
    assert "c" in failed and "a" in completed
    assert set(released) == {"a", "b", "c"}
    assert not scheduler.active and not scheduler.pending


def test_policies_never_share_a_decode_group():
    scheduler, events, *_ = make_scheduler(6)
    for name, policy in (("a", "p"), ("b", "other"), ("c", "p")):
        scheduler.enqueue(name, body(policy=policy))
    for _ in range(5):
        scheduler.step()
    for event in events:
        if event[0] in ("advance", "sample"):
            assert not ("b" in event[1] and len(event[1]) > 1)
    scheduler.close()


@pytest.mark.parametrize("value", [0, -1])
def test_cli_rejects_nonpositive_concurrency(value):
    with pytest.raises(SystemExit):
        parse_args(["serve", "--model", "tiny", "--inference-concurrency", str(value)])


def test_cli_and_toml_concurrency_independent_of_episode_workers(tmp_path):
    path = tmp_path / "service.toml"
    path.write_text('[service]\nmodel="tiny"\ninference_concurrency=12\n')
    assert parse_args(["serve", "--config", str(path)]).inference_concurrency == 12
    assert (
        parse_args(
            ["serve", "--config", str(path), "--inference-concurrency", "7"]
        ).inference_concurrency
        == 7
    )


def test_service_refill_receipts_rng_cancellation_and_restart(tmp_path, monkeypatch):
    """
    Purpose: Exercise the production queue bridge and durable receipts without MLX.
    Input: Mixed requests, late arrivals, one invalid policy, and one cancellation.
    Output: Stable receipts, one RNG assignment per request, and complete cleanup.
    """
    from studio_compute import inference

    started, release = threading.Event(), threading.Event()
    live, seen_batches = set(), []

    class ContinuousEngine(Engine):
        capabilities: ClassVar = {
            **Engine.capabilities,
            "continuous_batching": True,
            "max_inference_batch": 5,
        }

        def inference_scheduler(self, *, create, finish, fail):
            def prefill(request):
                request.logits = True

            def advance(requests):
                seen_batches.append(tuple(r.request_id for r in requests))
                time.sleep(0.001)

            def sample(requests):
                started.set()
                assert release.wait(5)
                for request in requests:
                    request.tokens.append(2)
                    request.needs_decode = True
                    if len(request.tokens) >= request.body["max_tokens"]:
                        request.reason = "length"

            def cleanup(request):
                live.remove(request.request_id)

            return ContinuousBatcher(
                5,
                create=create,
                prefill=prefill,
                advance=advance,
                sample=sample,
                finish=finish,
                fail=fail,
                release=cleanup,
            )

    def create(engine, name, prepared):
        live.add(name)
        return SimpleNamespace(
            request_id=name,
            body=prepared,
            logits=None,
            tokens=[],
            offset=0,
            needs_decode=False,
            reason=None,
        )

    monkeypatch.setattr(inference, "create_request", create)
    monkeypatch.setattr(
        inference,
        "request_result",
        lambda engine, r: {
            "sampled_tokens": list(r.tokens),
            "policy": r.body["policy"],
            "policy_revision": r.body["revision"],
            "completion_tokens": len(r.tokens),
        },
    )
    service = Service(tmp_path, ContinuousEngine)
    try:
        call(service, "a", "create", {"seed": 17}, "create")
        policy = call(service, "a", "publish", {"revision": 0}, "publish")["policy"]
        request = {"policy": policy, "prompt_token_ids": [1], "max_tokens": 25}
        with service.condition:
            for i in range(9):
                service.submit(str(i), "a", "sample", {**request, "max_tokens": 1 + i})
        assert started.wait(5)
        service.cancel("8")
        release.set()
        for i in range(9):
            op = service.wait(str(i), 10)
            assert op["status"] == ("interrupted" if i == 8 else "succeeded"), op
        assert any(len(rows) > 1 for rows in seen_batches)
        assert not live
        assert service.clients["a"]["rng_counter"] == 8
        saved = service.operation("3")
        assert service.submit("3", "a", "sample", {**request, "max_tokens": 4}) == saved
    finally:
        release.set()
        service.close()
    recovered = Service(tmp_path, ContinuousEngine)
    try:
        assert recovered.operation("3") == saved
        assert recovered.clients["a"]["rng_counter"] == 8
    finally:
        recovered.close()


def test_creation_defers_temporary_pressure_and_fails_oversized_work():
    from studio_compute.inference import create_request

    warm_prefixes = ["reusable-prefix"]

    def admit(needed):
        warm_prefixes.clear()
        raise MemoryError("live memory pressure")

    engine = SimpleNamespace(
        resident=SimpleNamespace(estimate_inference=lambda length, **options: 10),
        prefixes=SimpleNamespace(limit=0, bytes=0),
        policies={"p": {}},
        reserved={"replay": 40},
        persistent_bytes=lambda: 30,
        admission_budget_bytes=lambda: 60,
        admit=admit,
    )
    request = {"policy": "p", "prompt_token_ids": [1], "max_tokens": 4}
    assert create_request(engine, "later", request) is None
    assert warm_prefixes == ["reusable-prefix"]
    engine.reserved.clear()
    with pytest.raises(MemoryError, match="live memory pressure"):
        create_request(engine, "now", request)
    engine.reserved["replay"] = 40
    engine.persistent_bytes = lambda: 55
    with pytest.raises(MemoryError, match="live memory pressure"):
        create_request(engine, "oversized", request)


def test_cache_failure_after_policy_pin_releases_every_owner():
    from studio_compute.inference import create_request

    pinned = []

    @contextmanager
    def borrow(reference):
        pinned.append(reference)
        try:
            yield {}
        finally:
            pinned.remove(reference)

    def broken_cache(*args, **kwargs):
        raise ValueError("corrupt cache")

    engine = SimpleNamespace(
        resident=SimpleNamespace(estimate_inference=lambda length, **options: 10),
        prefixes=SimpleNamespace(limit=0, get=broken_cache),
        policies={"p": {}},
        prefix_identity="identity",
        reserved={},
        admit=lambda needed: None,
        _borrow_policy=borrow,
    )
    with pytest.raises(ValueError, match="corrupt cache"):
        create_request(
            engine, "failed", {"policy": "p", "prompt_token_ids": [1], "max_tokens": 4}
        )
    assert not pinned and not engine.reserved


def test_evicted_requested_policy_defers_reload_until_live_owner_releases(tmp_path):
    """
    Purpose: Keep an evicted policy request queued while another owner holds memory.
    Input: Two resident policies and one live replay reservation.
    Output: A clean deferral, then a successful retry after the live owner releases memory.
    """
    from studio_compute.engine import ResidentEngine
    from studio_compute.inference import create_request, release_request

    engine = ResidentEngine.__new__(ResidentEngine)
    factors = {"weight": SimpleNamespace(nbytes=40)}
    engine.policies = OrderedDict(
        (reference, {"factors": dict(factors), "revision": 0})
        for reference in ("requested", "live")
    )
    engine.policy_users = {"live": 1}
    engine.policy_cache_limit = 4
    engine.idle_factors = factors
    engine.reserved = {"replay": 30}
    engine.execution_profile = "legacy"
    engine.persistent_bytes = lambda: 20 + 40 * len(engine.policies)

    def admit(additional, reserved, *, persistent):
        if persistent + reserved + additional > 100:
            raise MemoryError("policy reload exceeds budget")

    engine.resident = SimpleNamespace(
        budget=100,
        selected=None,
        estimate_inference=lambda length, **options: 10,
        admit=admit,
        adapter=SimpleNamespace(validate=lambda values: None),
    )
    engine.mx = SimpleNamespace(
        load=lambda path: dict(factors),
        eval=lambda values: None,
        random=SimpleNamespace(key=lambda seed: seed),
    )
    asset = tmp_path / "adapters.safetensors"
    asset.write_bytes(b"immutable policy")
    engine.policy_sources = {
        "requested": (tmp_path, 0, hashlib.sha256(asset.read_bytes()).hexdigest())
    }
    engine.prefix_identity = "identity"
    engine.prefixes = SimpleNamespace(
        limit=0, bytes=0, get=lambda *args, **kwargs: ([], object(), 1)
    )
    request_body = {
        "policy": "requested",
        "prompt_token_ids": [1],
        "max_tokens": 4,
        "seed": 17,
    }
    assert create_request(engine, "retry", request_body) is None
    assert "requested" not in engine.policies
    assert engine.reserved == {"replay": 30}
    assert engine.policy_users == {"live": 1}

    engine.reserved.clear()
    engine.policy_users.clear()
    request = create_request(engine, "retry", request_body)
    assert request is not None
    assert engine.policy_users == {"requested": 1}
    release_request(engine, request)
    assert not engine.reserved and not engine.policy_users


def test_request_timing_includes_admission_policy_and_cache_work(monkeypatch):
    from studio_compute import inference

    clock = [100.0]
    monkeypatch.setattr(inference.time, "monotonic", lambda: clock[0])

    @contextmanager
    def borrow(reference):
        clock[0] += 3
        yield {}

    def admit(needed):
        clock[0] += 2

    def cached_prefix(*args, **kwargs):
        clock[0] += 5
        return [], object(), 1

    engine = SimpleNamespace(
        resident=SimpleNamespace(estimate_inference=lambda length, **options: 10),
        prefixes=SimpleNamespace(limit=0, get=cached_prefix),
        policies={"p": {}},
        prefix_identity="identity",
        reserved={},
        admit=admit,
        _borrow_policy=borrow,
        mx=SimpleNamespace(random=SimpleNamespace(key=lambda seed: seed)),
        metrics=lambda started: {"seconds": clock[0] - started},
    )
    request = inference.create_request(
        engine,
        "timed",
        {
            "policy": "p",
            "revision": 0,
            "prompt_token_ids": [1],
            "max_tokens": 4,
            "seed": 17,
        },
    )
    clock[0] += 2
    result = inference.request_result(engine, request)
    assert result["execution_metrics"]["seconds"] == 12
    assert result["prefill_seconds"] == 10
    assert result["decode_seconds"] == 2
    inference.release_request(engine, request)


def test_memory_retries_prepare_rng_once_and_recheck_acquisition(monkeypatch):
    from studio_compute import inference
    from studio_compute.service import ClientOwnershipConflict

    service = Service.__new__(Service)
    service.engine = object()
    service.clients = {"a": {"owner_operation": "create"}}
    service.live_clients = {"a"}
    prepared = []
    service._inference_body = lambda *args: prepared.append(args) or args[-1]
    monkeypatch.setattr(inference, "create_request", lambda *args: None)
    item = {"client": "a", "kind": "sample", "body": {"expected_owner": "create"}}
    for _ in range(3):
        assert service._create_continuous("deferred", item) is None
    assert len(prepared) == 1
    service.clients["a"]["owner_operation"] = "replacement"
    with pytest.raises(ClientOwnershipConflict, match="no longer live"):
        service._create_continuous("deferred", item)


def test_queue_failure_cannot_overwrite_a_completed_leader(tmp_path, monkeypatch):
    """
    Purpose: Protect an early successful receipt when a later queue boundary fails.
    Input: Two queued samples and an injected scheduler failure.
    Output: The leader stays successful. The survivor becomes interrupted.
    """
    from studio_compute import inference

    class BrokenQueue:
        def __init__(self, finish):
            self.finish, self.pending, self.active = finish, [], {}
            self.first = True

        def enqueue(self, name, body):
            self.pending.append(name)

        def step(self):
            if not self.first:
                raise RuntimeError("queue boundary failed")
            self.first = False
            name = self.pending.pop(0)
            self.finish(SimpleNamespace(request_id=name, tokens=[2]))
            return {
                "batch_sizes": [1],
                "request_ids": [name],
                "generated_tokens": [1],
                "finished_requests": [True],
                "prefill": None,
            }

        def close(self):
            self.pending.clear()

    class BrokenEngine(Engine):
        def __init__(self):
            super().__init__()
            self.capabilities = {**self.capabilities, "continuous_batching": True}

        def inference_scheduler(self, *, create, finish, fail):
            return BrokenQueue(finish)

    monkeypatch.setattr(
        inference, "request_result", lambda *args: {"sampled_tokens": [2]}
    )
    service = Service(tmp_path, BrokenEngine)
    try:
        call(service, "a", "create", {}, "create")
        policy = call(service, "a", "publish", {"revision": 0}, "publish")["policy"]
        with service.condition:
            for name in ("leader", "survivor"):
                service.submit(
                    name,
                    "a",
                    "sample",
                    {
                        "policy": policy,
                        "prompt_token_ids": [1],
                        "max_tokens": 2,
                    },
                )
        service.owner.join(timeout=5)
        assert not service.owner.is_alive()
        assert service.operation("leader")["status"] == "succeeded"
        assert service.operation("survivor")["status"] == "interrupted"
        assert "queue boundary failed" in service.fatal_error
    finally:
        service.close()


@pytest.mark.parametrize(
    "is_resident,has_reservation,should_defer",
    [
        (False, True, True),
        (True, True, False),
        (False, False, False),
    ],
)
def test_cold_rng_owner_waits_for_memory_before_key_assignment(
    is_resident,
    has_reservation,
    should_defer,
):
    service = Service.__new__(Service)
    service.engine = SimpleNamespace(
        next_rng=lambda client: None, reserved={"replay": 1} if has_reservation else {}
    )
    service.clients = {"a": {}}
    service.live_clients = set()
    service.resident_clients = {"a"} if is_resident else set()

    def prepare(*args):
        raise MemoryError("cold RNG owner cannot fit")

    service._inference_body = prepare
    item = {"client": "a", "kind": "sample", "body": {}}
    if should_defer:
        assert service._create_continuous("deferred", item) is None
        assert "prepared" not in item
    else:
        with pytest.raises(MemoryError, match="cold RNG owner"):
            service._create_continuous("failed", item)
