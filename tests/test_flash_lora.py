"""
Purpose: Check Flash LoRA, explicit carried state, and shared-prefix derivatives.
Input: A tiny decoder with real Flash layer classes and mixed quantized projections.
Output: Comparisons against an independent whole-sequence autograd objective.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_vlm.models.qwen4_exp")
from mlx.utils import tree_flatten
from mlx_vlm.models.qwen4_exp.config import TextConfig
from mlx_vlm.models.qwen4_exp.language import LanguageModel

from studio_compute.numerics.flash_adapter import FlashAdapter
from studio_compute.numerics.flash_ops import FlashOps, decode_hidden, decode_rows
from studio_compute.mlx_request_cache import fork_cache
from studio_compute.numerics.replay import Replay


def tiny_flash(*, indexer_budget: int = 2048) -> tuple:
    """Create all Flash layer types with affine expert and dense quantization."""
    mx.random.seed(17)
    config = TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=256,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=128,
        moe_intermediate_size=128,
        rms_norm_eps=1e-6,
        vocab_size=96,
        max_position_embeddings=4096,
        hc_count=4,
        hc_lowrank=64,
        ple_layer_ids=[2],
        ple_embed_dim=64,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        split_ngram_parts=4,
        eos_token_id=95,
        indexer_budget=indexer_budget,
        rope_parameters={
            "type": "default",
            "partial_rotary_factor": 0.5,
            "mrope_section": [16, 24, 24],
            "rope_theta": 10000,
        },
    )
    lm = LanguageModel(config)
    lm.set_dtype(mx.bfloat16)

    def quantization(path: str, module: object) -> object:
        if ".switch_mlp." in path and hasattr(module, "to_quantized"):
            return {"bits": 4, "group_size": 64}
        return isinstance(module, nn.Linear)

    nn.quantize(lm, group_size=64, bits=8, class_predicate=quantization)
    adapter = FlashAdapter(lm, config.layer_types)
    state = {
        name: value + 0.001 if name.endswith(".b") else value
        for name, value in adapter.initial(23).items()
    }
    adapter.load(state)
    lm.train()
    mx.eval(lm.parameters())
    return lm, adapter


def consume(work: object) -> object:
    while True:
        try:
            next(work)
        except StopIteration as result:
            return result.value


def branch(lm: object, prefix: list, tokens: list) -> dict:
    ids_BL = mx.array([prefix + tokens[:-1]], mx.int32)
    hidden_BLD = mx.tile(lm.model.embed_tokens(ids_BL), (1, 1, lm.args.hc_count))
    positions = mx.broadcast_to(
        mx.arange(ids_BL.shape[1], dtype=mx.int32)[None, None], (3, 1, ids_BL.shape[1])
    )
    return {
        "input_ids": ids_BL,
        "embeddings": hidden_BLD,
        "positions": positions,
        "prefix_length": len(prefix),
        "tokens": tokens,
        "objective": lambda lp: -mx.sum(lp),
    }


def whole_sequence(lm: object, item: dict) -> object:
    """Evaluate one complete sequence without replay segments or shared states."""
    ops = FlashOps()
    hidden_BLD = item["embeddings"]
    states = ops.initial_states(lm.layers, hidden_BLD.dtype)
    context = ops.context(item, 0, hidden_BLD.shape[1])
    for layer, values in zip(lm.layers, states):
        hidden_BLD, *_ = ops.layer(
            layer,
            hidden_BLD,
            [value for _, value in values],
            item["positions"],
            context,
        )
    start = item["prefix_length"] - 1
    return ops.head(
        lm, hidden_BLD[0, start:], mx.array(item["tokens"], mx.int32), mx.bfloat16
    )


def test_flash_targets_cover_every_layer_shared_and_routed_experts() -> None:
    lm, adapter = tiny_flash()
    assert len(adapter.modules) == 3 * (5 + 6) + (4 + 6)
    assert all(
        module.a.dtype == module.b.dtype == mx.bfloat16
        for module in adapter.modules.values()
    )
    for index in range(len(lm.layers)):
        routed = adapter.modules[f"layers.{index}.mlp.switch_mlp.up_proj"]
        assert routed.a.shape == (4, 16, 64)
        assert routed.b.shape == (4, 128, 16)
        assert routed.base.bits == 4
    names = [name for name, _ in tree_flatten(lm.trainable_parameters())]
    assert names and all(name.endswith((".a", ".b")) for name in names)


@pytest.mark.parametrize("checkpoint_layers", [1, 3])
@pytest.mark.parametrize("segment_tokens", [4, 64])
@pytest.mark.parametrize("indexer_budget", [8, 2048])
def test_flash_replay_matches_whole_sequence_loss_and_all_adapter_derivatives(
    segment_tokens: int, indexer_budget: int, checkpoint_layers: int
) -> None:
    lm, adapter = tiny_flash(indexer_budget=indexer_budget)
    items = [
        branch(lm, list(range(1, 11)), [11, 12, 13]),
        branch(lm, list(range(1, 11)), [14, 15, 16, 17]),
    ]
    original = adapter.state()

    def objective(factors: dict) -> object:
        for name, module in adapter.modules.items():
            module.a, module.b = factors[name + ".a"], factors[name + ".b"]
        return sum(-mx.sum(whole_sequence(lm, item)) for item in items)

    oracle_loss, oracle_gradients = mx.value_and_grad(objective)(original)
    mx.eval(oracle_loss, oracle_gradients)
    for name, module in adapter.modules.items():
        module.a, module.b = original[name + ".a"], original[name + ".b"]
    replay = Replay(
        SimpleNamespace(language_model=lm),
        segment_tokens=segment_tokens,
        checkpoint_layers=checkpoint_layers,
        execution_profile="bf16-fast",
        model_ops=FlashOps(),
    )
    probabilities, loss, gradients, stats = consume(replay.run_group(items))
    mx.eval(loss, gradients)
    assert set(gradients) == set(original)
    assert bool(mx.allclose(loss, oracle_loss, rtol=0.04, atol=0.1).item())
    for name, expected in oracle_gradients.items():
        actual = gradients[name]
        assert bool(mx.all(mx.isfinite(actual)).item()), name
        error = float(
            mx.linalg.norm(
                actual.astype(mx.float32) - expected.astype(mx.float32)
            ).item()
        )
        magnitude = float(mx.linalg.norm(expected.astype(mx.float32)).item())
        assert error <= max(0.03, magnitude * 0.15), (name, error, magnitude)
    assert stats["shared_segments"] == (9 // segment_tokens)


def test_flash_interleaved_inference_does_not_change_replay_gradients() -> None:
    """
    Purpose: Check factor isolation across cooperative inference and backward turns.
    Input: Different published factors and one fixed replay objective.
    Output: Equal loss and derivatives within the BF16 reduction tolerance.
    Note: Expert scatter reductions can change their floating-point addition order.
    """
    lm, adapter = tiny_flash(indexer_budget=8)
    item = branch(lm, list(range(1, 11)), [11, 12, 13])
    factors = adapter.state()
    replay = Replay(
        SimpleNamespace(language_model=lm),
        segment_tokens=4,
        checkpoint_layers=1,
        execution_profile="bf16-fast",
        model_ops=FlashOps(),
    )
    expected = consume(replay.run_group([item]))
    mx.eval(expected[1], expected[2])
    work = replay.run_group([item])
    lm.train()
    turns = 0
    while True:
        try:
            next(work)
        except StopIteration as result:
            actual = result.value
            break
        if turns % 7 == 0:
            policy = {name: value + 0.01 for name, value in factors.items()}
            adapter.load(policy)
            lm.eval()
            ids = mx.array([[1, 2, 3]], mx.int32)
            positions = mx.broadcast_to(mx.arange(3)[None, None], (3, 1, 3))
            hidden = decode_hidden(lm, ids, positions, lm.make_cache())
            mx.eval(hidden)
            adapter.load(factors)
            lm.train()
        turns += 1
    mx.eval(actual[1], actual[2])
    assert turns > 7
    assert bool(mx.array_equal(actual[1], expected[1]).item())
    for name in factors:
        reference = expected[2][name].astype(mx.float32)
        error = float(
            mx.linalg.norm(actual[2][name].astype(mx.float32) - reference).item()
        )
        magnitude = float(mx.linalg.norm(reference).item())
        assert error <= max(1e-5, 0.02 * magnitude), (name, error, magnitude)


def test_flash_zero_factors_preserve_native_model_outputs() -> None:
    lm, adapter = tiny_flash()
    zero = {name: mx.zeros_like(value) for name, value in adapter.state().items()}
    adapter.load(zero)
    item = branch(lm, [1, 2, 3, 4], [5, 6, 7])
    lm.eval()
    output = lm(item["input_ids"], position_ids=item["positions"])
    logits = output.logits[0, 3:]
    tokens = mx.array(item["tokens"], mx.int32)
    expected = mx.take_along_axis(logits, tokens[:, None], axis=-1)[
        :, 0
    ] - mx.logsumexp(logits, axis=-1)
    lm.train()
    actual = whole_sequence(lm, item)
    mx.eval(actual, expected)
    assert bool(mx.allclose(actual, expected, rtol=0.03, atol=0.08).item())


@pytest.mark.parametrize("cache_precision", ["bf16", "8bit"])
@pytest.mark.parametrize("indexer_budget", [8, 2048])
def test_flash_real_caches_match_replay_and_keep_forked_prefixes_immutable(
    cache_precision: str, indexer_budget: int
) -> None:
    lm, adapter = tiny_flash(indexer_budget=indexer_budget)
    item = branch(lm, list(range(1, 9)), [11, 12, 13])
    expected = whole_sequence(lm, item)
    lm.eval()
    caches = _request_caches(lm, cache_precision)
    hidden = decode_hidden(
        lm, item["input_ids"][:, :8], item["positions"][:, :, :8], caches
    )
    snapshots = [fork_cache(cache) for cache in caches]
    saved = [_copy_state(cache.state) for cache in snapshots]
    probabilities = []
    for index, token in enumerate(item["tokens"]):
        logits = lm.lm_head(hidden[:, -1])[0]
        probabilities.append(logits[token] - mx.logsumexp(logits))
        if index + 1 < len(item["tokens"]):
            offset = 8 + index
            hidden = decode_hidden(
                lm,
                mx.array([[token]], mx.int32),
                item["positions"][:, :, offset : offset + 1],
                caches,
            )
    actual = mx.stack(probabilities)
    mx.eval(actual, expected)
    assert bool(mx.allclose(actual, expected, rtol=0.04, atol=0.1).item())
    for cache, original in zip(snapshots, saved):
        _assert_state_equal(cache.state, original)
    forked = [fork_cache(cache) for cache in snapshots]
    decode_hidden(lm, mx.array([[11]], mx.int32), item["positions"][:, :, 8:9], forked)
    for cache, original in zip(snapshots, saved):
        _assert_state_equal(cache.state, original)


@pytest.mark.parametrize("cache_precision", ["bf16", "8bit"])
@pytest.mark.parametrize("indexer_budget", [8, 2048])
def test_flash_batched_ragged_decode_matches_independent_requests(
    cache_precision: str, indexer_budget: int
) -> None:
    lm, adapter = tiny_flash(indexer_budget=indexer_budget)
    lm.eval()
    requests, expected = [], []
    for length in (1, 4, 6, 8, 9, 12, 17, 20):
        prefix = list(range(1, length + 1))
        ids = mx.array([prefix], mx.int32)
        positions = mx.broadcast_to(
            mx.arange(length, dtype=mx.int32)[None, None], (3, 1, length)
        )
        caches = _request_caches(lm, cache_precision)
        decode_hidden(lm, ids, positions, caches)
        requests.append(
            SimpleNamespace(
                cache=caches,
                tokens=[11],
                body={"prompt_token_ids": prefix},
                needs_decode=True,
            )
        )
        independent = [fork_cache(cache) for cache in caches]
        hidden = decode_hidden(
            lm,
            mx.array([[11]], mx.int32),
            mx.full((3, 1, 1), length, mx.int32),
            independent,
        )
        expected.append(lm.lm_head(hidden)[:, -1])
    engine = SimpleNamespace(lm=lm, compute_dtype=mx.bfloat16)
    decode_rows(engine, requests)
    for request, logits in zip(requests, expected):
        mx.eval(request.logits, logits)
        assert not request.needs_decode
        assert bool(mx.allclose(request.logits, logits, rtol=0.04, atol=0.1).item())


def test_flash_quantized_cache_payload_and_bf16_backward_are_separate() -> None:
    """
    Purpose: Check packed KV payloads and BF16 replay after quantized inference.
    Input: A tiny Flash model with sparse attention and nonzero LoRA factors.
    Output: Exact payload sizes, finite replay gradients, and unchanged BF16 factors.
    """
    from mlx_vlm.models.qwen4_exp.language import QSAQuantizedKVCache

    lm, adapter = tiny_flash(indexer_budget=8)
    lm.eval()
    ids_BL = mx.array([[1 + index % 80 for index in range(256)]], mx.int32)
    positions = mx.broadcast_to(mx.arange(256)[None, None], (3, 1, 256))
    caches = _request_caches(lm, "8bit")
    hidden_BLD = decode_hidden(lm, ids_BL, positions, caches)
    mx.eval(hidden_BLD)
    attention = [cache for cache in caches if isinstance(cache, QSAQuantizedKVCache)]
    assert attention
    for cache in attention:
        assert cache.bits == 8 and cache.group_size == 64 and cache.offset == 256
        for packed, scales, biases in (cache.keys, cache.values):
            assert packed.dtype == mx.uint32
            assert scales.dtype == biases.dtype == mx.bfloat16
            packed_bytes = packed.nbytes + scales.nbytes + biases.nbytes
            bf16_bytes = 1 * 1 * 256 * 256 * 2
            assert packed_bytes == bf16_bytes * 17 // 32
        assert cache.index_keys.dtype == mx.bfloat16
    lm.train()
    replay = Replay(
        SimpleNamespace(language_model=lm),
        segment_tokens=4,
        checkpoint_layers=1,
        execution_profile="bf16-fast",
        model_ops=FlashOps(),
    )
    _, loss, gradients, _ = consume(
        replay.run_group([branch(lm, list(range(1, 11)), [11, 12, 13])])
    )
    mx.eval(loss, gradients)
    assert bool(mx.isfinite(loss).item())
    assert set(gradients) == set(adapter.state())
    assert all(value.dtype == mx.bfloat16 for value in adapter.state().values())
    assert all(bool(mx.all(mx.isfinite(value)).item()) for value in gradients.values())


def _request_caches(lm: object, precision: str) -> list:
    return [
        cache.to_quantized(bits=8, group_size=64)
        if precision == "8bit" and hasattr(cache, "to_quantized")
        else cache
        for cache in lm.make_cache()
    ]


def _copy_state(value: object) -> object:
    if isinstance(value, mx.array):
        copied = value + mx.zeros_like(value)
        mx.eval(copied)
        return copied
    if isinstance(value, (list, tuple)):
        return type(value)(_copy_state(item) for item in value)
    return value


def _assert_state_equal(actual: object, expected: object) -> None:
    if isinstance(actual, mx.array):
        assert bool(mx.array_equal(actual, expected).item())
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for value, prior in zip(actual, expected, strict=True):
            _assert_state_equal(value, prior)
    else:
        assert actual == expected


@pytest.mark.parametrize("indexer_budget", [8, 2048])
def test_quantized_attention_shares_projections_and_keeps_ragged_caches_private(
    indexer_budget: int, monkeypatch
) -> None:
    from studio_compute.numerics.flash_ops import _quantized_attention_rows

    lm, _ = tiny_flash(indexer_budget=indexer_budget)
    lm.eval()
    layer_index = next(i for i, layer in enumerate(lm.layers) if not layer.is_linear)
    attention = lm.layers[layer_index].self_attn
    lengths = (1, 4, 6, 8, 9, 12, 17, 20)
    caches, references, snapshots = [], [], []
    for length in lengths:
        all_caches = _request_caches(lm, "8bit")
        ids_BL = mx.array([list(range(1, length + 1))], mx.int32)
        positions = mx.broadcast_to(mx.arange(length)[None, None], (3, 1, length))
        hidden_BLD = decode_hidden(lm, ids_BL, positions, all_caches)
        mx.eval(hidden_BLD, [cache.state for cache in all_caches])
        cache = all_caches[layer_index]
        caches.append(cache)
        references.append(fork_cache(cache))
        snapshots.append((fork_cache(cache), _copy_state(cache.state)))
    hidden_BLD = mx.random.normal((8, 1, lm.args.hidden_size)).astype(mx.bfloat16)
    positions = mx.broadcast_to(mx.array(lengths)[None, :, None], (3, 8, 1))
    expected_BLD = mx.concatenate([
        attention(hidden_BLD[row:row + 1], mask="causal", cache=cache,
                  position_ids=positions[:, row:row + 1])
        for row, cache in enumerate(references)
    ])
    mx.eval(expected_BLD)
    calls = {}

    from studio_compute.numerics import flash_ops

    names = {id(getattr(attention, name)): name for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
    names[id(attention.indexer.index_qk_proj)] = "index"
    original_project = flash_ops._project_rows

    def project(linear, value):
        calls.setdefault(names[id(linear)], []).append(value.shape)
        return original_project(linear, value)

    monkeypatch.setattr(flash_ops, "_project_rows", project)
    actual_BLD = _quantized_attention_rows(attention, hidden_BLD, caches, positions)
    mx.eval(actual_BLD)
    assert bool(mx.allclose(actual_BLD, expected_BLD, rtol=0.02, atol=0.03125).item())
    assert set(calls) == {"q_proj", "k_proj", "v_proj", "o_proj", "index"}
    assert all(len(shapes) == 1 and shapes[0][:2] == (8, 1) for shapes in calls.values())
    assert len({id(cache) for cache in caches}) == 8
    for length, cache, (snapshot, original) in zip(lengths, caches, snapshots, strict=True):
        assert cache.offset == length + 1
        assert cache.bits == 8 and cache.group_size == 64
        assert cache.index_keys.dtype == mx.bfloat16
        _assert_state_equal(snapshot.state, original)


@pytest.mark.parametrize("rows", [1, 8, 64, 256])
def test_inference_expert_order_preserves_original_row_reductions(rows: int) -> None:
    from studio_compute.numerics.flash_ops import _moe

    lm, _ = tiny_flash()
    lm.eval()
    hidden_BLD = mx.random.normal((1, rows, lm.args.hidden_size)).astype(mx.bfloat16)
    expected_BLD = _moe(lm.layers[0].mlp, hidden_BLD)
    actual_BLD = _moe(lm.layers[0].mlp, hidden_BLD, sort_rows=True)
    mx.eval(expected_BLD, actual_BLD)
    assert bool(mx.array_equal(expected_BLD, actual_BLD).item())


def test_segment_trie_shares_only_complete_segments_with_later_segments() -> None:
    from studio_compute.numerics.replay import _count_nodes, _segment_trie

    shared = list(range(8))
    tree = _segment_trie(
        [shared + [8, 9], shared + [8, 9], shared + [10], shared[:4], shared], 4
    )
    assert tree == [
        (0, 1, [0, 1, 2, 4], [
            (1, 2, [0, 1, 2], [(2, 3, [0], []), (2, 3, [1], []), (2, 3, [2], [])]),
            (1, 2, [4], []),
        ]),
        (0, 1, [3], []),
    ]
    assert _count_nodes(tree) == 7


@pytest.mark.parametrize("segment_tokens", [4, 64])
def test_flash_tree_replay_matches_independent_replay(segment_tokens: int) -> None:
    lm, _ = tiny_flash(indexer_budget=8)
    prefix = list(range(1, 11))

    def items() -> list:
        values = [
            branch(lm, prefix, [11, 12, 13]),
            branch(lm, prefix + [11, 12], [20, 21, 22, 23]),
            branch(lm, list(range(30, 39)), [40, 41]),
        ]
        for value in values:
            value["token_ids"] = value["input_ids"][0].tolist()
        return values

    replay = Replay(
        SimpleNamespace(language_model=lm),
        segment_tokens=segment_tokens,
        checkpoint_layers=1,
        execution_profile="bf16-fast",
        model_ops=FlashOps(),
    )
    expected = [consume(replay.run_group([item])) for item in items()]
    probabilities, loss, gradients, stats = consume(replay.run_tree(items()))
    mx.eval(loss, gradients)
    for actual, (reference, *_rest) in zip(probabilities, expected):
        assert bool(mx.array_equal(actual, reference[0]).item())
    expected_loss = sum(float(value[1].item()) for value in expected)
    # Both losses are BF16 sums of per-branch BF16 losses; allow one BF16 ulp.
    assert abs(float(loss.item()) - expected_loss) <= 2**-7 * max(1.0, abs(expected_loss))
    for name, actual in gradients.items():
        reference = sum(value[2][name].astype(mx.float32) for value in expected)
        error = float(mx.linalg.norm(actual.astype(mx.float32) - reference).item())
        magnitude = float(mx.linalg.norm(reference).item())
        assert error <= max(1e-3, 0.02 * magnitude), (name, error, magnitude)
    assert stats["shared_segments"] == (2 if segment_tokens == 4 else 0)
    assert stats["forward_segments"] < stats["independent_segments"] or segment_tokens == 64


@pytest.mark.parametrize("tokens", [8, 256])
def test_flash_prefill_expert_tiles_match_sorted_rows(tokens: int) -> None:
    """
    Purpose: Check grouped expert tiles for prompt prefill against sorted rows.
    Input: A tiny Flash model, one prompt, and 8-bit request caches.
    Output: Close hidden rows and close next-token hidden rows after the prefill.
    Note: Prompts shorter than PREFILL_TILE_MIN_TOKENS keep sorted rows bitwise.
    """
    from studio_compute.numerics.flash_ops import PREFILL_TILE_MIN_TOKENS

    lm, _ = tiny_flash(indexer_budget=8)
    lm.eval()
    ids_BL = mx.array([[1 + index % 80 for index in range(tokens)]], mx.int32)
    positions = mx.broadcast_to(mx.arange(tokens)[None, None], (3, 1, tokens))
    next_BL = mx.array([[7]], mx.int32)
    next_positions = mx.full((3, 1, 1), tokens, mx.int32)
    results = []
    for moe_chunk in (None, 4):
        caches = _request_caches(lm, "8bit")
        hidden_BLD = decode_hidden(lm, ids_BL, positions, caches, moe_chunk=moe_chunk)
        next_BLD = decode_hidden(lm, next_BL, next_positions, caches)
        mx.eval(hidden_BLD, next_BLD)
        results.append((hidden_BLD.astype(mx.float32), next_BLD.astype(mx.float32)))
    (sorted_BLD, sorted_next), (tiled_BLD, tiled_next) = results
    if tokens < PREFILL_TILE_MIN_TOKENS:
        assert mx.array_equal(sorted_BLD, tiled_BLD).item()
        assert mx.array_equal(sorted_next, tiled_next).item()
        return
    for expected, actual in ((sorted_BLD, tiled_BLD), (sorted_next, tiled_next)):
        error = mx.linalg.norm(actual - expected) / mx.linalg.norm(expected)
        assert error.item() < 2**-5


def test_flash_sorted_training_experts_match_per_row_experts() -> None:
    """
    Purpose: Check exact sorted expert segments in training replay against per-row experts.
    Input: A tiny Flash model and two branches that share a prompt.
    Output: Equal logprobs within BF16 noise and close adapter gradients.
    """
    lm, _ = tiny_flash(indexer_budget=8)
    results = []
    for chunk in (None, 32):
        items = [
            branch(lm, list(range(1, 11)), [11, 12, 13]),
            branch(lm, list(range(1, 11)), [14, 15, 16, 17]),
        ]
        replay = Replay(
            SimpleNamespace(language_model=lm), segment_tokens=4,
            checkpoint_layers=2, execution_profile="bf16-fast",
            model_ops=FlashOps(training_moe_chunk=chunk),
        )
        results.append(consume(replay.run_group(items)))
    (rows_logp, rows_loss, rows_grad, _), (sorted_logp, sorted_loss, sorted_grad, _) = results
    for expected, actual in zip(rows_logp, sorted_logp):
        assert float(mx.abs(expected - actual).max().item()) <= 0.1
    assert abs(float(rows_loss.item()) - float(sorted_loss.item())) <= 0.05 * max(
        1.0, abs(float(rows_loss.item())))
    assert set(rows_grad) == set(sorted_grad)
    for name, expected in rows_grad.items():
        expected = expected.astype(mx.float32)
        error = float(mx.linalg.norm(sorted_grad[name].astype(mx.float32) - expected).item())
        magnitude = float(mx.linalg.norm(expected).item())
        assert error <= max(1e-3, 0.05 * magnitude), (name, error, magnitude)
