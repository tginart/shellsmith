"""
Purpose: Verify the BF16 profile and counter-based stochastic rounding.
Input: A small decoder fixture. Output: Dtype and checkpoint assertions.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
from studio_compute.numerics.bf16_adam import update
from studio_compute.numerics.optimizer import MasterAdam
from studio_compute.numerics.replay import Replay
from test_studio_numerics import branch, drain, tiny


def test_bf16_replay_gradients_and_exact_optimizer_restore() -> None:
    lm, adapter = tiny()
    sequence = branch(lm, tuple(range(1, 14)), (17, 23, 11, 8, 29))
    sequence["objective"] = lambda current: -mx.sum(current)
    currents, loss, gradients, stats = drain(
        Replay(
            SimpleNamespace(language_model=lm),
            segment_tokens=7,
            checkpoint_layers=1,
            execution_profile="bf16-fast",
        ).run_group([sequence], capture_prefix_gradient=True)
    )
    assert currents[0].dtype == loss.dtype == mx.bfloat16
    assert stats["prefix_gradient"].dtype == mx.bfloat16
    assert all(value.dtype == mx.bfloat16 for value in gradients.values())
    optimizer = MasterAdam(adapter, 3e-4, execution_profile="bf16-fast", seed=23)
    optimizer.accumulate(gradients)
    assert optimizer.step() > 0
    tensors, metadata = optimizer.state()
    assert all(value.dtype == mx.bfloat16 for value in tensors.values())
    restored = MasterAdam(adapter, 3e-4, execution_profile="bf16-fast", seed=91)
    restored.load(tensors, metadata)
    for instance in (optimizer, restored):
        instance.accumulate(gradients)
        instance.step()
    assert all(
        bool(mx.array_equal(value, restored.master[name]).item())
        for name, value in optimizer.master.items()
    )
    with pytest.raises(ValueError, match="profile"):
        MasterAdam(adapter, 3e-4).load(tensors, metadata)


def test_stochastic_rounding_small_updates_and_zero_rate() -> None:
    parameter_N = mx.ones((32768,), mx.bfloat16)
    zero_N = mx.zeros_like(parameter_N)
    gradient_N = mx.ones_like(parameter_N)
    constants_C = mx.array([0, 0, 1e-4, 1, 1e-8], mx.bfloat16)
    counter_R = mx.array([23, 1, 19], mx.uint32)
    result_N, _, _ = update(
        parameter_N, zero_N, zero_N, gradient_N, constants_C, counter_R
    )
    same_N, _, _ = update(
        parameter_N, zero_N, zero_N, gradient_N, constants_C, counter_R
    )
    assert bool(mx.array_equal(result_N, same_N).item())
    assert 300 < int(mx.sum(result_N != parameter_N).item()) < 1600
    mean_delta = float(
        mx.mean(parameter_N.astype(mx.float32) - result_N.astype(mx.float32)).item()
    )
    assert mean_delta == pytest.approx(1e-4, rel=0.15)
    constants_C = mx.array([0.9, 0.95, 0, 1, 1e-8], mx.bfloat16)
    unchanged_N, _, _ = update(
        parameter_N, zero_N, zero_N, gradient_N, constants_C, counter_R
    )
    assert bool(mx.array_equal(unchanged_N, parameter_N).item())


@pytest.mark.parametrize("clip_low", [0.0, 0.5])
def test_bf16_cispo_preserves_detached_ratios_masks_and_sum(clip_low: float) -> None:
    """
    Purpose: Verify the BF16 production objective against independent scalar equations.
    Input: Positive and negative credit, a masked token, and both clip bounds.
    Output: BF16 loss and gradients with the original detached-ratio semantics.
    """
    from studio_compute.numerics.cispo import (
        cispo_loss,
        reference_loss_and_gradient,
    )

    current_N = mx.array([-2, -1, -4, -2, -6], mx.bfloat16)
    old_N = [-3, -4, -1, -2, -2]
    advantages_N = [0.125, -0.0625, 0.5, 0.125, 0.125]
    mask_N = [1, 1, 0, 1, 1]
    loss, gradient_N = mx.value_and_grad(
        lambda values_N: cispo_loss(
            values_N,
            old_N,
            advantages_N,
            mask_N,
            clip_low=clip_low,
            clip_high=4.0,
            dtype=mx.bfloat16,
        )
    )(current_N)
    expected_loss, expected_gradient = reference_loss_and_gradient(
        current_N.tolist(),
        old_N,
        advantages_N,
        mask_N,
        clip_low=clip_low,
        clip_high=4.0,
    )
    assert loss.dtype == gradient_N.dtype == mx.bfloat16
    assert float(loss.item()) == pytest.approx(expected_loss, rel=0.02)
    assert gradient_N.tolist() == pytest.approx(expected_gradient, rel=0.02, abs=1e-4)


def test_engine_defaults_to_the_qualified_bf16_profile():
    """Purpose: Keep the engine default aligned with the qualified CLI profile."""
    from studio_compute.runtime import RuntimeEngine
    from test_studio_numerics import TinyResident

    engine = RuntimeEngine("tiny", model_factory=TinyResident)
    try:
        assert engine.execution_profile == "bf16-fast"
        assert engine.compute_dtype == mx.bfloat16
        assert engine.capabilities["execution"]["saved_activation_backward"]
        assert engine.identity["execution"]["checkpoint_layers"] == "auto"
    finally:
        engine.close()


@pytest.mark.parametrize("betas", [(0.9, 0.95), (0.8, 0.98)])
def test_bf16_adam_bias_correction_uses_effective_moment_betas(betas):
    """
    Purpose: Compare Adam bias correction with independent BF16 scalar equations.
    Input: Seven updates with configured betas that change under BF16 rounding.
    Output: Moments and zero-origin parameter deltas match the scalar reference.
    """
    import math

    class Adapter:
        def __init__(self):
            self.factors = {"factor": mx.zeros((8,), mx.bfloat16)}

        def state(self):
            return self.factors

        def load(self, factors):
            self.factors = factors

    def bf16(value):
        return float(mx.array(value, mx.bfloat16).item())

    adapter = Adapter()
    optimizer = MasterAdam(
        adapter, 3e-4, beta1=betas[0], beta2=betas[1],
        execution_profile="bf16-fast", seed=23,
    )
    beta1, beta2 = map(bf16, betas)
    first = second = 0.0
    for count in range(1, 8):
        gradient = 1.0 if count % 2 else 0.25
        first = bf16(beta1 * first + (1 - beta1) * gradient)
        second = bf16(beta2 * second + (1 - beta2) * gradient * gradient)
        size = bf16(3e-4 / (1 - beta1 ** count))
        correction = bf16(math.sqrt(1 - beta2 ** count))
        delta = bf16(size * first / (math.sqrt(second) / correction + bf16(1e-8)))
        optimizer.master["factor"] = mx.zeros((8,), mx.bfloat16)
        optimizer.accumulate({"factor": mx.full((8,), gradient, mx.bfloat16)})
        optimizer.step()
        assert optimizer.first["factor"].tolist() == [first] * 8
        assert optimizer.second["factor"].tolist() == [second] * 8
        assert optimizer.master["factor"].tolist() == [-delta] * 8
    _, metadata = optimizer.state()
    assert (metadata["beta1"], metadata["beta2"]) == betas


def test_bf16_adam_rejects_betas_that_round_to_one():
    """Reject decay constants with a zero BF16 bias-correction denominator."""
    _, adapter = tiny()
    with pytest.raises(ValueError, match="BF16.*below one"):
        MasterAdam(adapter, 3e-4, beta1=0.999, execution_profile="bf16-fast")
