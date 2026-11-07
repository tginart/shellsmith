"""Require numerical progress rather than publication traffic in qualification."""

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts/observe_studio_metrics.py"
SPEC = importlib.util.spec_from_file_location("observe_studio_metrics", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_publication_is_not_backward_progress():
    assert not MODULE.has_backward_progress({
        "training_operations": {"publish-0": {"kind": "publish", "progress": None}},
    })


def test_backward_requires_a_progress_receipt():
    assert not MODULE.has_backward_progress({
        "training_operations": {"update-1": {"kind": "forward_backward", "progress": None}},
    })
    assert MODULE.has_backward_progress({
        "training_operations": {"update-1": {"kind": "forward_backward", "progress": {"phase": "backward"}}},
    })


@pytest.mark.parametrize("phase", ["replay_shared_forward", "replay_branch_forward", "head"])
def test_forward_only_training_progress_is_not_backward(phase: str) -> None:
    assert not MODULE.has_backward_progress({"training_operations": {
        "update": {"kind": "forward_backward", "progress": {
            "phase": phase, "phase_seconds": {phase: .5, "replay_branch_backward": 0.}}},
    }})


@pytest.mark.parametrize("phase", ["backward", "head_backward", "replay_branch_backward", "replay_shared_backward"])
def test_numerical_backward_phases_and_completed_counters(phase: str) -> None:
    for progress in [{"phase": phase},
                     {"phase": "segment_complete", "phase_seconds": {phase: .5}}]:
        assert MODULE.has_backward_progress({"training_operations": {
            "update": {"kind": "forward_backward", "progress": progress},
        }})
    assert not MODULE.has_backward_progress({"training_operations": {
        "update": {"kind": "forward", "progress": {"phase": phase}},
    }})
