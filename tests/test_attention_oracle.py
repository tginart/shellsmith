"""Independent derivative and causal-offset checks for benchmark qualification."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")


def test_gqa_oracle_derivatives_match_finite_differences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Purpose: Verify shared KV-head accumulation and all three derivatives."""
    module = _load(monkeypatch)
    rng = np.random.default_rng(19)
    inputs = [
        rng.normal(size=shape) for shape in ((1, 2, 2, 3), (1, 1, 3, 3), (1, 1, 3, 3))
    ]
    dy_BHQD = rng.normal(size=(1, 2, 2, 3))
    gradients = module.numpy_attention(*inputs, dy_BHQD, scale=0.5)[1:]
    epsilon = 1e-5
    for index, value in enumerate(inputs):
        for coordinate in np.ndindex(value.shape):
            plus = [x.copy() for x in inputs]
            minus = [x.copy() for x in inputs]
            plus[index][coordinate] += epsilon
            minus[index][coordinate] -= epsilon
            high = np.sum(
                module.numpy_attention(*plus, dy_BHQD, scale=0.5)[0] * dy_BHQD
            )
            low = np.sum(
                module.numpy_attention(*minus, dy_BHQD, scale=0.5)[0] * dy_BHQD
            )
            expected = (high - low) / (2 * epsilon)
            assert gradients[index][coordinate] == pytest.approx(
                expected, rel=1e-7, abs=1e-7
            )


def test_suffix_queries_observe_their_actual_causal_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Purpose: Detect top-left masks and accidental truncation of old keys."""
    module = _load(monkeypatch)
    q_BHQD = np.zeros((1, 2, 2, 1))
    k_BJKD = np.zeros((1, 1, 3, 1))
    v_BJKD = np.array([[[[1.0], [3.0], [8.0]]]])
    actual_BHQD = module.numpy_attention(
        q_BHQD, k_BJKD, v_BJKD, np.ones_like(q_BHQD), scale=1.0
    )[0]
    np.testing.assert_allclose(actual_BHQD[0, :, 0, 0], [2.0, 2.0])
    np.testing.assert_allclose(actual_BHQD[0, :, 1, 0], [4.0, 4.0])


def _load(monkeypatch: pytest.MonkeyPatch) -> Any:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "attention_qualify", scripts / "mlx_attention_qualify.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
