"""Real Metal math tests; skipped on portable hosts without optional MLX deps."""

import math
import os

import pytest

# Native FP32 oracle qualification disables MLX's reduced-precision FP32
# matrix operations before loading the numerical runtime.
os.environ["MLX_ENABLE_TF32"] = "0"
mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_vlm")

from mlx_vlm.models.qwen3_5.config import TextConfig
from mlx_vlm.models.qwen3_5.language import LanguageModel
from mlx_vlm.models.qwen3_5.gated_delta import gated_delta_ops, gated_delta_chunked as native_chunked

from shellsmith.mlx.adapter import Adapter
from shellsmith.mlx.cispo import cispo_loss, frozen_head_logprobs, reference_loss_and_gradient
from shellsmith.mlx.ops import gated_delta_chunked, initial_states, segment_layer
from shellsmith.mlx.optimizer import MasterAdam
from shellsmith.mlx.replay import Replay


def tiny_model(*, quantized=False, layers=4):
    mx.random.seed(17)
    config = TextConfig(model_type="qwen3_5_text", hidden_size=64, intermediate_size=128,
                        linear_num_value_heads=2, linear_num_key_heads=1, linear_key_head_dim=32,
                        linear_value_head_dim=32, linear_conv_kernel_dim=4, num_hidden_layers=layers,
                        num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=96,
                        num_key_value_heads=2, max_position_embeddings=4096, head_dim=16,
                        rope_parameters={"type": "default", "partial_rotary_factor": 0.5, "rope_theta": 10000,
                                         "mrope_section": [2, 1, 1], "mrope_interleaved": True})
    model = LanguageModel(config)
    model.set_dtype(mx.bfloat16)
    if quantized:
        nn.quantize(model, group_size=64, bits=8)
    adapter = Adapter(model, ["full_attention" if (index + 1) % 4 == 0 else "linear_attention"
                              for index in range(layers)])
    adapter.load(adapter.initial(23))
    # Nonzero B exercises prompt gradients through every adapter factor.
    adapter.load({name: value + 0.001 if name.endswith(".b") else value for name, value in adapter.state().items()})
    model.train()
    mx.eval(model.parameters())
    return model, adapter


def assert_relative(actual, expected, *, tolerance=0.07):
    mx.eval(actual, expected)
    error = float(mx.sqrt(mx.sum((actual.astype(mx.float32) - expected.astype(mx.float32)) ** 2)).item())
    scale = max(1e-8, float(mx.sqrt(mx.sum(expected.astype(mx.float32) ** 2)).item()))
    assert error / scale < tolerance


def test_cispo_gradients_match_detached_scalar_oracle():
    current, old, advantages, mask = [-2., -1., -4., -2.], [-3., -4., -1., -2.], [0.1, -0.25, 0.5, 1.], [1, 1, 1, 0]
    expected_loss, expected_gradients = reference_loss_and_gradient(current, old, advantages, mask)
    loss, gradient = mx.value_and_grad(lambda x: cispo_loss(x, old, advantages, mask))(mx.array(current))
    assert float(loss.item()) == pytest.approx(expected_loss, abs=1e-6)
    assert gradient.tolist() == pytest.approx(expected_gradients, abs=1e-6)


@pytest.mark.parametrize("length", [1, 17, 70])
def test_native_gdn_recurrence_and_carry_match_reference(length):
    mx.random.seed(31)
    q, k = (mx.random.normal((1, length, 1, 8)) * 0.12 for _ in range(2))
    v = mx.random.normal((1, length, 2, 8)) * 0.15
    g = mx.full((1, length, 2), 0.97)
    beta = mx.full((1, length, 2), 0.4)
    state = mx.random.normal((1, 2, 8, 8)) * 0.05
    actual, carried = gated_delta_chunked(q, k, v, g, beta, state)
    expected, expected_carry = gated_delta_ops(q, k, v, g, beta, state)
    # The parallel FP32 matmul path rounds differently from per-token elementwise
    # recurrence on Metal. Check both the native parallel implementation and
    # the independent sequential reference, below the BF16 precision budget.
    parallel, parallel_carry = native_chunked(q, k, v, g, beta, state)
    assert_relative(actual, parallel, tolerance=2e-5)
    assert_relative(carried, parallel_carry, tolerance=2e-5)
    assert_relative(actual, expected, tolerance=0.002)
    assert_relative(carried, expected_carry, tolerance=0.002)


def test_fullprompt_segmented_gradients_match_unsplit_autodiff():
    model, adapter = tiny_model()
    prompt, tokens = tuple(range(1, 14)), (17, 23, 11, 8, 29)
    replay = Replay(model, segment_tokens=7, memory_gib=None)
    objective = lambda x: cispo_loss(x, [-5.] * len(tokens), [0.2, -0.1, 0., 0.3, 0.1])
    current, loss, gradients, stats = replay.run(prompt, tokens, objective, capture_prefix_gradient=True)
    ids = mx.array([prompt + tokens[:-1]], dtype=mx.int32)
    embeddings = model.model.embed_tokens(ids).astype(mx.bfloat16)
    positions = mx.broadcast_to(mx.arange(ids.shape[1], dtype=mx.int32)[None, None], (3, 1, ids.shape[1]))
    original = adapter.state()

    def unsplit(factors):
        for name, module in adapter.modules.items():
            module.a, module.b = factors[name + ".a"], factors[name + ".b"]
        hidden = embeddings
        states = initial_states(model.layers, hidden.dtype)
        for layer, incoming in zip(model.layers, states):
            hidden, *_ = segment_layer(layer, hidden, [value for _, value in incoming], positions)
        scored = frozen_head_logprobs(model, hidden[0, len(prompt) - 1:], mx.array(tokens))
        return objective(scored)

    try:
        expected_loss, expected_gradients = mx.value_and_grad(unsplit)(
            {name: value.astype(mx.float32) for name, value in original.items()})
        mx.eval(expected_loss, expected_gradients)
    finally:
        adapter.load(original)
    assert float(loss.item()) == pytest.approx(float(expected_loss.item()), rel=0.025, abs=0.003)
    assert set(gradients) == set(original)
    # BF16 segment rounding differs; compare whole-adapter relative norm.
    a = mx.concatenate([gradients[name].reshape(-1) for name in sorted(gradients)])
    b = mx.concatenate([expected_gradients[name].reshape(-1) for name in sorted(gradients)])
    assert_relative(a, b, tolerance=0.08)
    assert float(mx.sum(stats["prefix_gradient"] ** 2).item()) > 0
    assert stats["segments"] == 3
    assert stats["hidden_checkpoint_storage"] == "bf16"


@pytest.mark.parametrize("segment_tokens,completion_length", [(7, 5), (64, 257)])
@pytest.mark.parametrize("checkpoint_layers", [1, 2, 4])
def test_fp32_replay_likelihood_and_gradients_match_independent_native_decoder(segment_tokens, completion_length, checkpoint_layers):
    model, adapter = tiny_model()
    model.set_dtype(mx.float32)
    model.train()
    prompt = tuple(range(1, 14))
    tokens = tuple((17, 23, 11, 8, 29)[index % 5] for index in range(completion_length))
    advantages = tuple((0.2, -0.1, 0., 0.3, 0.1)[index % 5] for index in range(completion_length))
    replay = Replay(model, segment_tokens=segment_tokens, memory_gib=None, qualification_dtype=mx.float32,
                    checkpoint_layers=checkpoint_layers)
    objective = lambda x: cispo_loss(x, [-5.] * len(tokens), advantages)
    current, loss, gradients, _ = replay.run(prompt, tokens, objective)
    ids = mx.array([prompt + tokens[:-1]], dtype=mx.int32)
    positions = mx.broadcast_to(mx.arange(ids.shape[1], dtype=mx.int32)[None, None], (3, 1, ids.shape[1]))
    original = adapter.state()

    def native(factors):
        for name, module in adapter.modules.items():
            module.a, module.b = factors[name + ".a"], factors[name + ".b"]
        # This uses the unmodified pinned dependency's model forward, native
        # attention, GDN implementation and cache. It calls no replay unit.
        logits = model(ids, cache=model.make_cache(), position_ids=positions).logits[0, len(prompt) - 1:]
        logp = mx.take_along_axis(logits, mx.array(tokens)[:, None], axis=-1)[:, 0] - mx.logsumexp(logits, axis=-1)
        return objective(logp)

    try:
        full_logits = model(ids, cache=model.make_cache(), position_ids=positions).logits[0, len(prompt) - 1:]
        expected_logp = mx.take_along_axis(full_logits, mx.array(tokens)[:, None], axis=-1)[:, 0] - mx.logsumexp(full_logits, axis=-1)
        expected_loss, expected_gradients = mx.value_and_grad(native)(
            {name: value.astype(mx.float32) for name, value in original.items()})
        mx.eval(expected_logp, expected_loss, expected_gradients)
    finally:
        # Preserve FP32 factor dtype for the qualification fixture.
        for name, module in adapter.modules.items():
            module.a, module.b = original[name + ".a"], original[name + ".b"]
    assert bool(mx.all(mx.abs(current - expected_logp) <= 2e-6 + 1e-5 * mx.abs(expected_logp)).item())
    assert float(loss.item()) == pytest.approx(float(expected_loss.item()), rel=1e-5, abs=2e-6)
    assert set(gradients) == set(expected_gradients)
    for name, actual in gradients.items():
        wanted = expected_gradients[name]
        assert bool(mx.all(mx.abs(actual - wanted) <= 2e-6 + 3e-4 * mx.abs(wanted)).item()), name


@pytest.mark.parametrize("checkpoint_layers", [2, 4, 8])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32], ids=["bf16", "fp32"])
def test_recomputed_boundaries_preserve_every_factor_prompt_and_carry_adjoint(checkpoint_layers, dtype):
    model, adapter = tiny_model(quantized=True, layers=12 if checkpoint_layers == 8 else 4)
    model.set_dtype(dtype)
    prompt, tokens = tuple(range(1, 14)), (17, 23, 11, 8, 29)
    objective = lambda x: cispo_loss(x, [-5.] * len(tokens), [0.2, -0.1, 0., 0.3, 0.1])
    results = [Replay(model, segment_tokens=7, memory_gib=None, checkpoint_layers=interval, qualification_dtype=dtype).run(
        prompt, tokens, objective, capture_prefix_gradient=True) for interval in (1, checkpoint_layers)]
    expected, actual = results
    for got, wanted in zip(actual[:2], expected[:2]):
        assert bool(mx.array_equal(got, wanted).item())
    assert set(actual[2]) == set(adapter.state())
    for name, wanted in expected[2].items():
        assert bool(mx.array_equal(actual[2][name], wanted).item()), name
    assert bool(mx.array_equal(actual[3]["prefix_gradient"], expected[3]["prefix_gradient"]).item())
    assert set(actual[3]["carry_gradients"]) == set(expected[3]["carry_gradients"])
    assert actual[3]["carry_gradients"]
    for key, wanted in expected[3]["carry_gradients"].items():
        assert bool(mx.array_equal(actual[3]["carry_gradients"][key], wanted).item()), key
    assert actual[3]["boundary_stored_bytes"] < expected[3]["boundary_stored_bytes"]


def test_replay_memory_estimate_counts_only_saved_layers_and_recompute_block():
    model, _ = tiny_model()
    estimate = Replay(model, segment_tokens=7, checkpoint_layers=4).memory_estimate(17, 64, 2, True)
    assert estimate["boundary_bytes"] == 17 * 64 * 2
    assert estimate["recompute_working_bytes"] == 7 * 64 * 4 * 2
    assert estimate["carry_adjoint_bytes"] > 0
    inference = Replay(model, segment_tokens=7, checkpoint_layers=4).memory_estimate(17, 64, 2, False)
    assert inference["carry_adjoint_bytes"] == inference["boundary_bytes"] == 0


def test_master_adam_matches_scalar_math_and_accepts_zero_gradient():
    class Factors:
        def __init__(self): self.values = {"w": mx.array([1., -2.], mx.bfloat16)}
        def state(self): return self.values
        def load(self, state): self.values = state

    adapter = Factors()
    optimizer = MasterAdam(adapter, 0.01)
    optimizer.accumulate({"w": mx.array([0.3, -0.4])})
    optimizer.accumulate({"w": mx.array([0.2, -0.1])})
    norm = optimizer.step()
    assert norm == pytest.approx(math.sqrt(0.5), rel=1e-6)
    assert optimizer.master["w"].tolist() == pytest.approx([0.99, -1.99], abs=1e-6)
    tensors, metadata = optimizer.state()
    restored = MasterAdam(Factors(), 0.01)
    restored.load(tensors, metadata)
    for instance in (optimizer, restored):
        instance.accumulate({"w": mx.zeros((2,))})
        assert instance.step() == 0.0
    assert bool(mx.array_equal(optimizer.master["w"], restored.master["w"]).item())
    assert optimizer.steps == 2


def test_quantized_canonical_base_remains_frozen():
    model, adapter = tiny_model(quantized=True)
    from mlx.utils import tree_flatten
    trainable = dict(tree_flatten(model.trainable_parameters()))
    assert len(trainable) == len(adapter.state())
    assert all(name.endswith((".a", ".b")) for name in trainable)
    assert all(module.base.bits == 8 for module in adapter.modules.values())
