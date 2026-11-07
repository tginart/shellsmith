"""Exact transport admission, local client validation, exclusive durable ownership."""

import asyncio
from dataclasses import replace
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from shellsmith.train.backend import TokenSequence, TokenBatch
from shellsmith.train.config import TrainConfig, RunSection, ComputeSection, validate_train_config
from shellsmith.train.studio import StudioBackend
from shellsmith.train.trainer import submit_update
from studio_compute import protocol
from studio_compute.client import Client
from studio_compute.service import Service
from studio_compute.storage import Store
from test_studio_service import Engine


def backend():
    value = StudioBackend.__new__(StudioBackend)
    value.client = Client(client_id="transport-client")
    value.update_request_id = "run:segment:1:update:1"
    value.revision = 0
    return value


def sequence():
    return TokenSequence((1, 2, 3), (4, 5), (-1., -1.), (.1, .1), (1., 1.), "sample")


def test_exact_canonical_request_boundary_and_no_network_on_overflow(monkeypatch):
    from studio_compute import client
    rpc = Client()
    body = {"note": "Unicode: \u03b1", "tokens": [1, 2, 3]}
    envelope = {"id": "request", "client": "default", "kind": "fixture", "body": body}
    size = len(protocol.canonical(envelope))
    monkeypatch.setattr(protocol, "MAX_REQUEST_BYTES", size)
    assert rpc.validate_submission("fixture", body, request_id="request") == size
    seen = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{"ok":true}'
    def open_request(request, **kwargs):
        seen.append(request.data)
        return Response()
    monkeypatch.setattr(client, "urlopen", open_request)
    assert rpc.submit("fixture", body, request_id="request") == {"ok": True}
    assert seen == [protocol.canonical(envelope)]
    monkeypatch.setattr(protocol, "MAX_REQUEST_BYTES", size - 1)
    with pytest.raises(ValueError, match="No RPC submitted"):
        rpc.submit("fixture", body, request_id="request")
    assert len(seen) == 1


def test_oversized_update_rejected_before_receipt_lookup_or_rpc(monkeypatch):
    rpc = Client()
    monkeypatch.setattr(protocol, "MAX_REQUEST_BYTES", 1)
    monkeypatch.setattr(rpc, "_request", lambda *a, **k: pytest.fail("No RPC, including receipt lookup"))
    with pytest.raises(ValueError, match="HTTP limit"):
        asyncio.run(rpc.update([{"tokens": list(range(20))}], learning_rate=.01, request_id="update"))


def test_size_guard_precedes_durable_training_intent_with_unlimited_token_caps(monkeypatch):
    value = backend()
    value.update = AsyncMock()
    config = TrainConfig(run=RunSection(run_id="transport"), compute=ComputeSection(backend="studio"))
    assert config.train.max_update_prompt_tokens == config.train.max_update_completion_tokens == 0
    markers = []
    monkeypatch.setattr(protocol, "MAX_REQUEST_BYTES", 1)
    with pytest.raises(ValueError, match="No RPC submitted"):
        asyncio.run(submit_update(deps=None, training_client=value, reference_client=None,
            datums=[SimpleNamespace(datum=sequence())], config=config,
            before_submit=lambda: markers.append("pending")))
    assert not markers
    value.update.assert_not_called()


def test_shellsmith_size_preflight_uses_actual_arrays_and_exact_update_id(monkeypatch):
    value = backend()
    batch = TokenBatch((sequence(),))
    body = protocol.update_body(value._sequences(batch), 0., 4.)
    envelope = {"id": value.update_request_id + ":backward", "client": value.client.client_id,
                "kind": "forward_backward", "body": body}
    size = len(protocol.canonical(envelope))
    monkeypatch.setattr(protocol, "MAX_REQUEST_BYTES", size)
    value.validate_update(batch, learning_rate=.001, clip_low=0., clip_high=4.)
    monkeypatch.setattr(protocol, "MAX_REQUEST_BYTES", size - 1)
    with pytest.raises(ValueError, match="HTTP limit"):
        value.validate_update(batch, learning_rate=.001, clip_low=0., clip_high=4.)


@pytest.mark.parametrize("explicit", [False, True])
def test_selected_training_client_limit_checked_in_configuration(explicit):
    config = TrainConfig(run=RunSection(run_id="r" * (1 if explicit else 200)),
                         compute=ComputeSection(backend="studio"))
    if explicit:
        config = replace(config, studio=replace(config.studio, training_client="c" * 200))
    validate_train_config(config)
    if explicit:
        invalid = replace(config, studio=replace(config.studio, training_client="c" * 201))
    else:
        invalid = replace(config, run=replace(config.run, run_id="r" * 201))
    with pytest.raises(ValueError, match="Selected Studio training client ID"):
        validate_train_config(invalid)


def test_unused_training_client_does_not_constrain_inference():
    config = TrainConfig(run=RunSection(run_id="inference"), compute=ComputeSection(backend="studio"))
    config = replace(config, studio=replace(config.studio, training_client="unused" * 200))
    validate_train_config(config, campaign_mode="inference")


def test_backend_rejects_invalid_client_before_discovery(monkeypatch):
    from shellsmith.train import studio
    config = TrainConfig(run=RunSection(run_id="valid"), compute=ComputeSection(backend="studio"))
    config = replace(config, studio=replace(config.studio, training_client="c" * 201))
    monkeypatch.setattr(studio, "Client", lambda *a, **k: pytest.fail("No discovery or tokenizer work"))
    with pytest.raises(ValueError, match="1–200"):
        StudioBackend(config)


def test_service_state_root_lock_precedes_engine_even_through_alias(tmp_path):
    root = tmp_path / "state"
    owner = Service(root, Engine)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    try:
        for path in (root, alias):
            with pytest.raises(RuntimeError, match="owns Studio state root"):
                Service(path, lambda: pytest.fail("Second engine must never load"))
        separate = Service(tmp_path / "separate", Engine)
        separate.close()
    finally:
        owner.close()
    recovered = Service(alias, Engine)
    recovered.close()
    assert (root / "owner.lock").exists()  # Never unlink a shared lock inode.


def test_state_lock_released_after_database_initialization_failure(tmp_path, monkeypatch):
    original = Store._open_database
    def fail(value):
        original(value)
        raise OSError("injected startup failure")
    monkeypatch.setattr(Store, "_open_database", fail)
    with pytest.raises(OSError, match="injected"):
        Store(tmp_path)
    monkeypatch.setattr(Store, "_open_database", original)
    store = Store(tmp_path)
    store.close()


def test_state_lock_excludes_fresh_process_and_recovers_after_owner_death(tmp_path):
    code = "from studio_compute.storage import Store;import sys; s=Store(sys.argv[1]); print('owned',flush=True);sys.stdin.read()"
    owner = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert owner.stdout.readline().strip() == "owned"
        with pytest.raises(RuntimeError, match="owns Studio state root"):
            Store(tmp_path)
        second = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                                input="", capture_output=True, text=True, timeout=10)
        assert second.returncode != 0 and "owns Studio state root" in second.stderr
    finally:
        owner.kill()
        owner.communicate(timeout=10)
    recovered = Store(tmp_path)
    recovered.close()


def test_transport_limit_is_advertised(tmp_path):
    service = Service(tmp_path, Engine)
    try:
        assert service.capabilities["max_request_bytes"] == 128 * 1024 * 1024
    finally:
        service.close()
