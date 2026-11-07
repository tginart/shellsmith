"""Trainer setup/restore uses the external SDK's async interface.

The explicit external fixture adapts inherited synchronous fault hooks in
worker threads. Guard both sides: production awaits async methods on the loop,
while only that fixture executes blocking hooks off-loop. No production fallback.
"""
import asyncio
from dataclasses import replace
import threading

from shellsmith.policy.deps import load_tinker_deps
from shellsmith.train import trainer
from test_trainer_resume import _runtime, external_tinker_sdk, training_sdk


def test_async_sdk_setup_and_restore_with_off_loop_external_fixture_hooks(tmp_path, monkeypatch, training_sdk):
    config, _audit = _runtime(tmp_path, monkeypatch)
    deps = load_tinker_deps()
    client_type = deps.tinker.ServiceClient
    observed = []
    async_observed = []
    owner_thread = threading.get_ident()
    for name in ("get_server_capabilities_async", "create_lora_training_client_async",
                 "create_training_client_from_state_with_optimizer_async", "create_sampling_client_async"):
        original = getattr(client_type, name)

        async def async_only(self, *args, _name=name, _original=original, **kwargs):
            assert asyncio.get_running_loop() is not None and threading.get_ident() == owner_thread
            async_observed.append(_name)
            return await _original(self, *args, **kwargs)

        monkeypatch.setattr(client_type, name, async_only)
    for name in ("get_server_capabilities", "create_lora_training_client", "create_training_client_from_state_with_optimizer"):
        original = getattr(client_type, name)

        def synchronous_only(self, *args, _name=name, _original=original, **kwargs):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"Sync-only SDK {_name} called inside running event loop")
            observed.append((_name, threading.get_ident()))
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(client_type, name, synchronous_only)
    state = asyncio.run(trainer.run_train(config, deps=deps))
    assert state["step"] == 1 and state["status"] == "complete"
    later = replace(config, run=replace(config.run, run_id="later-auto-id"))
    resumed = asyncio.run(trainer.run_train(later, deps=deps))
    assert resumed["step"] == 1 and resumed["resumes"] == 1
    assert [name for name, _ in observed] == [
        "get_server_capabilities", "create_lora_training_client",
        "get_server_capabilities", "create_training_client_from_state_with_optimizer",
    ]
    assert all(thread != owner_thread for _, thread in observed)
    assert async_observed == [
        "get_server_capabilities_async", "create_lora_training_client_async", "create_sampling_client_async",
        "create_sampling_client_async", "get_server_capabilities_async",
        "create_training_client_from_state_with_optimizer_async", "create_sampling_client_async",
    ]
