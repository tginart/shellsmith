"""Recovery/publication failures must be rejected before touching the GPU."""

import asyncio
from types import SimpleNamespace
import threading

import pytest

import shellsmith.mlx.backend as backend
from shellsmith.train.config import RetrySection


@pytest.mark.parametrize("field,wrong", [("next_world_offset", 9), ("metrics_head", {"bad": True}),
                                       ("committed_group_ids", ["later-uncommitted-world"])])
def test_recovery_rejects_mismatched_durable_trainer_position_before_model_load(monkeypatch, tmp_path, field, wrong):
    identity = {"identity_sha256": "frozen"}
    position = {"step": 2, "next_world_offset": 4, "metrics_head": {"sha256": "committed"},
                "committed_group_ids": ["step1:world0", "step2:world1"], "sampler_path": "/sampler"}
    monkeypatch.setattr(backend, "require_ready", lambda settings: None)
    monkeypatch.setattr(backend, "resolve_identity", lambda settings: identity)
    monkeypatch.setattr(backend, "verify_checkpoint", lambda *args, **kwargs:
                        {"policy_identity": identity, "sampler_path": "/sampler", "trainer_position": position})
    checkpoint = {"path": "/state", **position, field: wrong, "policy_identity": identity}
    with pytest.raises(ValueError, match=field):
        backend.MLXSession(SimpleNamespace(mlx=object(), retry=RetrySection()), checkpoint_root=tmp_path / "uncreated", checkpoint=checkpoint)
    assert not (tmp_path / "uncreated").exists()


def test_publication_requires_durable_optimizer_boundary():
    session = backend.MLXSession.__new__(backend.MLXSession)
    session.lock = threading.RLock()
    session.optimizer = SimpleNamespace(steps=1)
    session._pending_ref, session._pending, session._durable_version = "/pending-sampler", True, None
    session.mx = SimpleNamespace(clear_cache=lambda: None)
    with pytest.raises(RuntimeError, match="committed"):
        asyncio.run(session.publish_policy(1))
    session._durable_version = 1
    assert asyncio.run(session.publish_policy(1)) == "/pending-sampler"
    assert session.policy_version == 1 and session._pending is False


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_constructor_failure_cleans_partial_allocation_and_preserves_exception(monkeypatch, tmp_path, cleanup_fails):
    monkeypatch.setattr(backend, "require_ready", lambda settings: None)
    monkeypatch.setattr(backend, "resolve_identity", lambda settings: {"identity_sha256": "frozen"})
    allocated, cleanup = [], []
    failure = ValueError("adapter validation failed after model allocation")

    def clear_cache():
        cleanup.append("cache")
        if cleanup_fails:
            raise RuntimeError("cleanup failed")

    def fail_initialization(session, *args):
        allocated.append(session)
        session.mx = SimpleNamespace(clear_cache=clear_cache)
        session.model = session.lm = session.adapter = session.optimizer = session.sampler = session.rng = object()
        raise failure

    monkeypatch.setattr(backend.MLXSession, "_initialize", fail_initialization)
    with pytest.raises(ValueError) as caught:
        backend.MLXSession(SimpleNamespace(mlx=object(), retry=RetrySection()), checkpoint_root=tmp_path)
    assert caught.value is failure
    session = allocated[0]
    assert session.closed and cleanup == ["cache"]
    assert all(getattr(session, name) is None for name in ("model", "lm", "adapter", "optimizer", "sampler", "rng"))


@pytest.mark.parametrize("failure_phase", [None, "construction", "cleanup"])
def test_cancelled_construction_settles_and_closes_before_returning(monkeypatch, tmp_path, failure_phase):
    async def exercise():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        events = []

        class DelayedSession:
            def __init__(self, *args, **kwargs):
                loop.call_soon_threadsafe(started.set)
                assert release.wait(timeout=5)
                events.append("constructed")
                if failure_phase == "construction":
                    raise ValueError("constructor failure after cancellation")

            def close(self):
                events.append("closed")
                if failure_phase == "cleanup":
                    raise ValueError("cleanup failure after cancellation")

        monkeypatch.setattr(backend, "MLXSession", DelayedSession)
        task = asyncio.create_task(backend.create_session(object(), checkpoint_root=tmp_path))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()  # The caller must continue holding its GPU lease.
            task.cancel()  # Repeated cancellation must not abandon the worker either.
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
            assert events == (["constructed"] if failure_phase == "construction" else ["constructed", "closed"])
        finally:
            release.set()

    asyncio.run(exercise())
