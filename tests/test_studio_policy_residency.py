"""CPU lifecycle probes: durable references do not own accelerator factors."""

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import weakref
import hashlib

import pytest

from studio_compute.engine import ResidentEngine


class Array:
    nbytes = 10


def engine(tmp_path):
    instance = ResidentEngine.__new__(ResidentEngine)
    instance.execution_profile = "legacy"
    instance.root = tmp_path
    instance.clients, instance.policies = {}, OrderedDict()
    instance.policy_sources, instance.policy_users = {}, {}
    instance.policy_cache_limit = 4
    instance.idle_factors = {"idle": Array()}
    instance.reserved = {}
    instance.prefixes = SimpleNamespace(bytes=0)
    loads, selected = [], []
    resident = SimpleNamespace(frozen=[], selected=None, budget=1000)

    def select(factors):
        resident.selected = factors
        selected[:] = list(factors.values())  # decoder module references

    def admit(additional, reserved, *, persistent):
        if persistent + reserved + additional > resident.budget:
            raise MemoryError("fixture budget")

    def load(path):
        loads.append(path)
        return {"weight": Array()}

    resident.select, resident.admit = select, admit
    resident.adapter = SimpleNamespace(validate=lambda factors: None)
    instance.resident = resident
    instance.mx = SimpleNamespace(load=load, eval=lambda factors: None)
    instance.loads = loads
    return instance


def register(instance, count):
    for i in range(count):
        folder = instance.root / f"checkpoint-{i}"
        folder.mkdir()
        data = f"finite-factors-{i}".encode()
        (folder / "adapters.safetensors").write_bytes(data)
        instance.policy(f"p{i}", folder, {"revision": i}, adapter_digest=hashlib.sha256(data).hexdigest())


def test_historical_registration_is_lazy_and_factor_cache_is_bounded(tmp_path):
    instance = engine(tmp_path)
    register(instance, 100)
    assert not instance.loads and not instance.policies
    for i in range(100):
        with instance._borrow_policy(f"p{i}") as factors:
            assert instance.policies[f"p{i}"]["factors"] is factors
        assert len(instance.policies) <= 4
    assert len(instance.policy_sources) == 100
    assert len(instance.loads) == 100
    with instance._borrow_policy("p99"):
        pass
    assert len(instance.loads) == 100  # warm reuse
    with instance._borrow_policy("p0"):
        pass
    assert len(instance.loads) == 101  # durable old revision reloads
    # Another full-state save at the same revision may supply a new folder.
    instance.policy("p0", Path("another-equivalent-checkpoint"), {"revision": 0}, adapter_digest=instance.policy_sources["p0"][2])
    with pytest.raises(ValueError, match="immutable"):
        instance.policy("p0", Path("bad"), {"revision": 1}, adapter_digest=instance.policy_sources["p0"][2])


def test_pinned_revisions_survive_interleaving_close_and_memory_pressure(tmp_path):
    instance = engine(tmp_path)
    register(instance, 20)
    with instance._borrow_policy("p0") as original:
        for i in range(1, 20):
            with instance._borrow_policy(f"p{i}"):
                pass
        assert instance.policies["p0"]["factors"] is original
        instance.close_client("closed-publisher")
        assert list(instance.policies) == ["p0"]
        instance.resident.budget = 20  # baseline + pinned revision
        with pytest.raises(MemoryError):
            instance.admit(1)
        assert instance.policies["p0"]["factors"] is original
    instance.admit(1)
    assert not instance.policies and not instance.policy_users


def test_eviction_releases_decoder_selected_factors(tmp_path):
    instance = engine(tmp_path)
    register(instance, 1)
    with instance._borrow_policy("p0") as factors:
        instance.resident.select(factors)
        ref = weakref.ref(factors["weight"])
    del factors
    instance.resident.budget = 11
    instance.admit(1)
    assert ref() is None
    assert instance.resident.selected is instance.idle_factors
    assert instance.persistent_bytes() == 10


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("failure", ["close", "raise", "finish"])
def test_sample_generators_pin_and_release_on_every_exit(tmp_path, batched, failure):
    instance = engine(tmp_path)
    register(instance, 1)
    def sample(*args):
        yield {"phase": "decode"}
        if failure == "raise":
            raise RuntimeError("injected decode failure")
        return {"tokens": [1]}
    instance._sample_policy_units = instance._sample_batch_policy_units = sample
    body = {"policy": "p0"}
    work = instance._sample_batch_units([body]) if batched else instance._sample_units(body, 0)
    next(work)
    instance.close_client("publisher")
    assert instance.policy_users == {"p0": 1} and "p0" in instance.policies
    if failure == "close":
        work.close()
    elif failure == "raise":
        with pytest.raises(RuntimeError, match="injected"):
            next(work)
    else:
        with pytest.raises(StopIteration):
            next(work)
    assert not instance.policy_users
    instance.close_client("publisher")
    assert not instance.policies


def test_all_active_revisions_can_exceed_idle_limit_then_shrink(tmp_path):
    from contextlib import ExitStack
    instance = engine(tmp_path)
    register(instance, 8)
    with ExitStack() as stack:
        factors = [stack.enter_context(instance._borrow_policy(f"p{i}")) for i in range(8)]
        assert len(instance.policies) == 8
        assert instance.persistent_bytes() == 90
        assert len({id(f) for f in factors}) == 8
    assert len(instance.policies) == 4 and not instance.policy_users


def test_service_restart_registers_history_without_loading_factors(tmp_path):
    from studio_compute.service import Service

    instances = []
    def factory():
        instance = engine(tmp_path)
        instance.identity = {"model": "fixture", "numerics": "cpu-lifecycle"}
        instance.capabilities = {"context_window": 32, "vocabulary_size": 16, "cache_precisions": ["bf16"]}
        instance.create = lambda client, body: {}
        instance.restore = lambda client, folder, state: None
        instance.write_state = lambda client, folder: (folder / "adapters.safetensors").write_bytes(b"fixture")
        instance.close = lambda: None
        def work(client, kind, body):
            assert kind == "sample"
            with instance._borrow_policy(body["policy"]):
                yield {"phase": "decode"}
                return {"sampled_tokens": [3], "policy_revision": body["revision"]}
        instance.work = work
        instances.append(instance)
        return instance

    def call(service, client, kind, body, request_id):
        service.submit(request_id, client, kind, body)
        op = service.wait(request_id)
        assert op["status"] == "succeeded", op
        return op["result"]

    service = Service(tmp_path, factory)
    try:
        policies = []
        for i in range(12):
            call(service, f"a{i}", "create", {}, f"create-{i}")
            policies.append(call(service, f"a{i}", "publish", {"revision": 0}, f"publish-{i}")["policy"])
            call(service, f"a{i}", "close", {}, f"close-{i}")
        assert not instances[0].loads and not instances[0].policies
    finally:
        service.close()
    service = Service(tmp_path, factory)
    try:
        assert len(instances[1].policy_sources) == 12
        assert not instances[1].loads and not instances[1].policies
        call(service, "consumer", "create", {}, "create-consumer")
        for i, policy in enumerate(policies):
            sampled = call(service, "consumer", "sample", {
                "policy": policy, "prompt_token_ids": [1], "max_tokens": 1, "seed": 0,
            }, f"sample-{i}")
            assert sampled == {"sampled_tokens": [3], "policy_revision": 0}
        assert len(instances[1].policies) == 4
    finally:
        service.close()


@pytest.mark.parametrize("warm", [False, True])
def test_lazy_policy_reload_rejects_changed_finite_factors(tmp_path, warm):
    instance = engine(tmp_path)
    register(instance, 1)
    if warm:
        with instance._borrow_policy("p0") as factors:
            original = factors
        # The resident immutable factors remain correct until evicted.
    folder, _, expected = instance.policy_sources["p0"]
    (folder / "adapters.safetensors").write_bytes(b"different-but-finite-factors")
    if warm:
        with instance._borrow_policy("p0") as factors:
            assert factors is original
        instance.close_client("publisher")
    loads = list(instance.loads)
    with pytest.raises(ValueError, match="Published policy adapter corruption"):
        with instance._borrow_policy("p0"):
            pytest.fail("Corrupt factors must not be admitted")
    assert instance.loads == loads
    assert not instance.policies and not instance.policy_users
    with pytest.raises(ValueError, match="immutable"):
        instance.policy("p0", folder, {"revision": 0}, adapter_digest="0" * 64)
    assert instance.policy_sources["p0"][2] == expected
