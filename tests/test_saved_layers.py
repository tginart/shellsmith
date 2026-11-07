"""
Purpose: Compare explicit saved-layer derivatives with full decoder-layer autodiff.
Input: Quantized tiny layers, nonzero adapters, and hidden and carry cotangents.
Output: Gradient, ownership, projection-count, and policy-binding assertions.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
from mlx.utils import tree_flatten, tree_unflatten
from mlx_vlm.models.qwen3_5.config import TextConfig
from mlx_vlm.models.qwen3_5.language import LanguageModel
from studio_compute.numerics.adapter import Adapter
from studio_compute.numerics.ops import segment_layer, initial_states
from studio_compute.numerics.saved import forward_saved, backward_saved, _projections


def tiny_saved_model():
    mx.random.seed(17)
    config = TextConfig(
        model_type="qwen3_5_text",
        hidden_size=64,
        intermediate_size=128,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=24,
        rms_norm_eps=1e-6,
        vocab_size=96,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        head_dim=256,
        rope_parameters={
            "type": "default",
            "partial_rotary_factor": 0.5,
            "rope_theta": 10000,
            "mrope_section": [16, 24, 24],
            "mrope_interleaved": True,
        },
    )
    model = LanguageModel(config)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=8)
    adapter = Adapter(model, ["linear_attention"] * 3 + ["full_attention"])
    adapter.load(
        {
            name: value + 0.001 if name.endswith(".b") else value
            for name, value in adapter.initial(23).items()
        }
    )
    model.train()
    mx.eval(model.parameters())
    return model, adapter


@pytest.mark.parametrize("is_linear", [True, False])
@pytest.mark.parametrize("retain_all", [True, False])
@pytest.mark.parametrize("hidden_edge", [True, False])
@pytest.mark.parametrize("retain_middle", [True, False])
def test_saved_layer_all_input_adapter_and_carry_derivatives(
    is_linear, retain_all, hidden_edge, retain_middle
):
    lm, adapter = tiny_saved_model()
    layer = lm.layers[0 if is_linear else 3]
    x = mx.random.normal((1, 7, 64)).astype(mx.bfloat16) * 0.1
    pos = mx.broadcast_to(mx.arange(5, 12, dtype=mx.int32)[None, None], (3, 1, 7))
    if is_linear:
        carries = [
            value + 0.001
            for _, value in initial_states(
                [layer], mx.bfloat16, recurrent_dtype=mx.bfloat16
            )[0]
        ]
    else:
        carries = [
            mx.random.normal((1, 4, 5, 256)).astype(mx.bfloat16) * 0.1 for _ in range(2)
        ]
    retained = (
        frozenset(name for name, _ in _projections(layer))
        if retain_all
        else frozenset()
    )
    out, first, second, tape = forward_saved(
        layer, x, carries, pos, retain=retained, retain_middle=retain_middle
    )
    cots = [
        mx.random.normal(value.shape).astype(mx.bfloat16) * 0.1
        for value in (out, first, second)
    ]
    if not hidden_edge:
        cots[0] = None
    names, back = backward_saved(tape, cots)
    mx.eval(back)
    original = tree_flatten(layer.trainable_parameters())
    values = [value for _, value in original]
    active = [i for i in (1, 2, 0) if cots[i] is not None]

    def reference(*arguments):
        layer.update(tree_unflatten(list(zip(names, arguments[: len(names)]))))
        outputs = segment_layer(
            layer,
            arguments[len(names)],
            list(arguments[len(names) + 1 :]),
            pos,
            execution_profile="bf16-fast",
        )
        return [outputs[i] for i in active]

    try:
        primal, expected = mx.vjp(
            reference, [*values, x, *carries], [cots[i] for i in active]
        )
        mx.eval(primal, expected)
    finally:
        layer.update(tree_unflatten(original))
    for name, actual, wanted in zip(
        [*names, "hidden", *["carry" + str(i) for i in range(len(carries))]],
        back,
        expected,
        strict=True,
    ):
        error = mx.linalg.norm((actual - wanted).astype(mx.float32))
        norm = mx.linalg.norm(wanted.astype(mx.float32))
        assert float(error.item()) <= 1e-5 + float(norm.item()) * 0.09, name
    if retain_all:
        assert not any(
            tape.counters.get("reconstructed." + name, 0)
            for name, _ in _projections(layer)
        )
    assert all(
        count == 1 for key, count in tape.counters.items() if key.startswith("forward.")
    )
    assert all(value.dtype == mx.bfloat16 for value in back)


def test_saved_layer_preserves_policy_bindings_after_interleaving():
    lm, adapter = tiny_saved_model()
    layer = lm.layers[0]
    x = mx.ones((1, 3, 64), mx.bfloat16) * 0.1
    pos = mx.broadcast_to(mx.arange(3, dtype=mx.int32)[None, None], (3, 1, 3))
    carries = [
        value
        for _, value in initial_states(
            [layer], mx.bfloat16, recurrent_dtype=mx.bfloat16
        )[0]
    ]
    out, _, _, tape = forward_saved(layer, x, carries, pos)
    cot = [mx.ones_like(out), None, None]
    _, expected = backward_saved(tape, cot)
    mx.eval(expected)
    adapter.load({name: value + 0.002 for name, value in adapter.state().items()})
    current = tree_flatten(layer.trainable_parameters())
    _, actual = backward_saved(tape, cot)
    mx.eval(actual)
    for got, want in zip(actual, expected, strict=True):
        assert bool(mx.array_equal(got, want).item())
    for (_, got), (_, want) in zip(
        tree_flatten(layer.trainable_parameters()), current, strict=True
    ):
        assert got is want


@pytest.mark.parametrize("checkpoint_layers", [1, 2, 4])
def test_saved_replay_preserves_shared_context_and_checkpoint_groups(checkpoint_layers):
    from types import SimpleNamespace
    from studio_compute.numerics.replay import Replay
    from test_studio_numerics import branch, drain, flat

    lm, adapter = tiny_saved_model()
    branches = [
        branch(lm, tuple(range(1, 18)), targets)
        for targets in ((17, 23, 11, 8, 29), (21, 15, 9))
    ]
    expected = drain(
        Replay(
            SimpleNamespace(language_model=lm),
            segment_tokens=7,
            checkpoint_layers=1,
            execution_profile="bf16-fast",
        ).run_group(branches, capture_prefix_gradient=True)
    )
    retained = lambda segment, index: frozenset(
        name for name, _ in _projections(lm.layers[index])
    )
    actual = drain(
        Replay(
            SimpleNamespace(language_model=lm),
            segment_tokens=7,
            checkpoint_layers=checkpoint_layers,
            execution_profile="bf16-fast",
            saved_activations=True,
            retained_projections=retained,
        ).run_group(branches, capture_prefix_gradient=True)
    )
    error = float(mx.linalg.norm(flat(actual[2]) - flat(expected[2])).item())
    norm = float(mx.linalg.norm(flat(expected[2])).item())
    assert error / max(norm, 1e-8) < 0.12
    prefix = actual[3]["prefix_gradient"].astype(mx.float32)
    wanted = expected[3]["prefix_gradient"].astype(mx.float32)
    assert (
        float(mx.linalg.norm(prefix - wanted).item())
        / float(mx.linalg.norm(wanted).item())
        < 0.12
    )
    assert actual[3]["shared_segments"] == 2
    counts = actual[3]["projection_counts"]
    assert not any(
        counts.get("reconstructed." + name, 0)
        for layer in lm.layers
        for name, _ in _projections(layer)
    )


@pytest.mark.parametrize("checkpoint_layers", [1, 4])
def test_saved_replay_zero_optional_budget_matches_retained_gradients(
    checkpoint_layers,
):
    from types import SimpleNamespace
    from studio_compute.numerics.replay import Replay
    from test_studio_numerics import branch, drain, flat

    lm, adapter = tiny_saved_model()
    sequence = branch(lm, tuple(range(1, 18)), (17, 23, 11, 8, 29))
    holder = SimpleNamespace(language_model=lm)
    reference = drain(
        Replay(
            holder,
            segment_tokens=7,
            checkpoint_layers=1,
            execution_profile="bf16-fast",
            saved_activations=True,
            retained_projections=lambda segment, index: frozenset(
                name for name, _ in _projections(lm.layers[index])
            ),
        ).run_group([sequence], capture_prefix_gradient=True)
    )
    actual = drain(
        Replay(
            holder,
            segment_tokens=7,
            checkpoint_layers=checkpoint_layers,
            execution_profile="bf16-fast",
            saved_activations=True,
            activation_budget_bytes=0,
        ).run_group([sequence], capture_prefix_gradient=True)
    )
    difference = float(mx.linalg.norm(flat(actual[2]) - flat(reference[2])).item())
    norm = float(mx.linalg.norm(flat(reference[2])).item())
    assert difference / max(norm, 1e-8) < 0.02
    assert actual[3]["activation_retention"]["retained_bytes"] == 0
    assert actual[3]["projection_counts"]["reconstructed.mlp.gate_proj"] > 0
