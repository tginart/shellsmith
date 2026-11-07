"""Run-directory ownership wraps the actual trainer, including state and SDK I/O."""
import asyncio
from types import SimpleNamespace

import pytest

from shellsmith.policy.deps import load_tinker_deps
from shellsmith.train import trainer
from shellsmith.train.config import load_train_config
from shellsmith.train.run_lock import TrainRunLockedError, lock_train_run
from test_trainer_resume import _runtime, external_tinker_sdk, training_sdk


def test_trainer_contender_cannot_even_inspect_or_rewrite_saved_state(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    state = root / trainer.STATE_FILE
    state.write_bytes(b"deliberately-invalid-owner-state")
    config = load_train_config(None, log_dir=root)
    with lock_train_run(root):
        with pytest.raises(TrainRunLockedError, match="already locked"):
            asyncio.run(trainer.run_train(config, deps=SimpleNamespace()))
    assert state.read_bytes() == b"deliberately-invalid-owner-state"
    assert sorted(path.name for path in root.iterdir()) == [trainer.STATE_FILE]


def test_actual_training_holds_lock_through_update_checkpoint_and_resume(tmp_path, monkeypatch, training_sdk):
    config, _audit = _runtime(tmp_path, monkeypatch)
    observed = []

    def require_ownership():
        with pytest.raises(TrainRunLockedError):
            with lock_train_run(config.resolved_log_dir()):
                pytest.fail("Trainer released run ownership during model/update I/O")
        observed.append(True)

    training_sdk.observe = require_ownership
    deps = load_tinker_deps()
    state = asyncio.run(trainer.run_train(config, deps=deps))
    assert state["step"] == 1 and state["status"] == "complete"
    assert observed
    with lock_train_run(config.resolved_log_dir()):
        pass  # released after final state publication
    before = len(observed)
    resumed = asyncio.run(trainer.run_train(config, deps=deps))
    assert resumed["resumes"] == 1 and len(observed) > before
    with lock_train_run(config.resolved_log_dir()):
        pass
