"""Native Metal parity for mixed prompt lengths, warm forks, and slot refill."""

import time

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm")

from test_studio_numerics import TinyResident, drain

from studio_compute import inference
from studio_compute.mlx_request_cache import fork_cache
from studio_compute.runtime import RuntimeEngine


@pytest.mark.parametrize("profile", ["legacy", "bf16-fast"])
@pytest.mark.parametrize("concurrency", [1, 3, 6])
def test_variable_age_rows_match_singleton(profile, concurrency):
    engine = RuntimeEngine(
        "tiny",
        model_factory=TinyResident,
        execution_profile=profile,
        checkpoint_layers=2,
        prefill_tokens=7,
        inference_concurrency=concurrency,
    )
    try:
        engine.create("a", {"seed": 23})
        factors = engine.clients["a"]["adapter"].state()
        engine.policies["p"] = {"factors": factors, "revision": 0}
        bodies = [
            {
                "policy": "p",
                "revision": 0,
                "prompt_token_ids": list(range(1, length + 1)),
                "target_token_ids": [17, 23, 11, 8, 29, 30, 31][:maximum],
                "max_tokens": maximum,
                "seed": 17 + i,
            }
            for i, (length, maximum) in enumerate(
                [(3, 1), (15, 7), (8, 3), (21, 6), (11, 2), (6, 5), (9, 4)]
            )
        ]
        expected = [
            drain(engine._sample({**body, "use_prefix_cache": False}, time.monotonic()))
            for body in bodies
        ]
        # Reuse a smaller cached prefix across different prompt ages.
        drain(
            engine._sample(
                {**bodies[0], "prompt_token_ids": [1, 2, 3]}, time.monotonic()
            )
        )
        results, errors, progress = {}, {}, []
        scheduler = engine.inference_scheduler(
            create=lambda name, body: inference.create_request(engine, name, body),
            finish=lambda r: results.update(
                {r.request_id: inference.request_result(engine, r)}
            ),
            fail=lambda name, error: errors.update({name: str(error)}),
        )
        for i, body in enumerate(bodies):
            scheduler.enqueue(str(i), body)
        while scheduler.pending or scheduler.active:
            progress.append(scheduler.step())
            assert len(progress) < 100
        assert not errors
        assert not engine.reserved and not engine.policy_users
        for i, reference in enumerate(expected):
            actual = results[str(i)]
            assert actual["sampled_tokens"] == reference["sampled_tokens"]
            assert actual["prefix_cache_tokens"] >= 3
            error = max(
                abs(a - b)
                for a, b in zip(
                    actual["behavior_logprobs"], reference["behavior_logprobs"]
                )
            )
            # BF16 matrix kernels may choose a different reduction for several rows.
            assert error <= (0.125 if profile == "bf16-fast" else 0.03), error
        if concurrency > 1:
            assert any(
                max(row["decode_batch_sizes"], default=0) > 1 for row in progress
            )
    finally:
        engine.close()


def test_fork_writers_do_not_change_snapshot_or_each_other():
    from mlx_vlm.models.cache import ArraysCache, KVCache

    native = KVCache()
    keys_BHLD = mx.ones((1, 2, 5, 8), mx.bfloat16)
    native.update_and_fetch(keys_BHLD, keys_BHLD)
    left, right = fork_cache(native), fork_cache(native)
    left.update_and_fetch(keys_BHLD[:, :, :1] * 2, keys_BHLD[:, :, :1] * 3)
    right.update_and_fetch(keys_BHLD[:, :, :2] * 4, keys_BHLD[:, :, :2] * 5)
    mx.eval(left.state, right.state, native.state)
    assert (native.offset, left.offset, right.offset) == (5, 6, 7)
    assert bool(mx.all(native.state[0] == 1).item())
    assert bool(mx.all(left.state[0][:, :, -1:] == 2).item())
    assert bool(mx.all(right.state[0][:, :, -1:] == 4).item())
    recurrent = ArraysCache(size=2)
    recurrent.cache = [mx.ones((1, 3, 8)), mx.ones((1, 2, 8, 8))]
    fork = fork_cache(recurrent)
    fork.cache[0][0, 0, 0] = 9
    mx.eval(fork.state, recurrent.state)
    assert recurrent.cache[0][0, 0, 0].item() == 1


def test_request_cancellation_releases_policy_cache_and_reservation():
    engine = RuntimeEngine(
        "tiny",
        model_factory=TinyResident,
        execution_profile="legacy",
        checkpoint_layers=2,
    )
    try:
        engine.policies["p"] = {"factors": engine.idle_factors, "revision": 0}
        body = {
            "policy": "p",
            "revision": 0,
            "prompt_token_ids": [1, 2, 3],
            "max_tokens": 20,
            "seed": 17,
        }
        scheduler = engine.inference_scheduler(
            create=lambda name, body: inference.create_request(engine, name, body),
            finish=lambda request: None,
            fail=lambda *args: None,
        )
        scheduler.enqueue("cancel", body)
        scheduler.step()
        request = scheduler.active["cancel"]
        assert engine.reserved and engine.policy_users == {"p": 1}
        scheduler.cancel("cancel")
        assert not engine.reserved and not engine.policy_users
        assert request.cache == [] and request.logits is None
    finally:
        engine.close()


def test_rng_stream_does_not_depend_on_batch_neighbors():
    """
    Purpose: Check categorical keys and draw shapes independently of projection roundoff.
    Input: Fixed logits and distinct per-request seeds. Output: Exact singleton token streams.
    """
    from types import SimpleNamespace

    from studio_compute.mlx_racing import DecodeRequest

    def observe(*args):
        pass

    engine = SimpleNamespace(
        mx=mx,
        _draw_token=lambda lp, key: mx.random.categorical(lp, key=key),
        _observe_generated=observe,
    )
    requests, expected = [], []
    for i in range(6):
        logits_BV = mx.arange(96, dtype=mx.float32)[None] / 96
        request = DecodeRequest(
            str(i),
            {"max_tokens": 9 + i},
            [],
            None,
            None,
            mx.random.key(17 + i),
            time.monotonic(),
            logits=logits_BV,
        )
        requests.append(request)
        key, tokens = mx.random.key(17 + i), []
        lp_BV = logits_BV - mx.logsumexp(logits_BV, -1, keepdims=True)
        for _ in range(9 + i):
            key, draw_key = mx.random.split(key)
            tokens.append(int(mx.random.categorical(lp_BV, key=draw_key).item()))
        expected.append(tokens)
    for step in range(14):
        # Neighbors retire at different boundaries without changing another key.
        inference.sample_requests(engine, [r for r in requests if r.reason is None])
    assert [r.tokens for r in requests] == expected


def test_packed_cache_rows_keep_independent_positions_and_storage():
    """
    Purpose: Compare mixed-age packed KV rows against native singleton attention.
    Input: Different prefix lengths and two fixed decode tokens.
    Output: BF16 logits within one rounding interval and intact packed cache storage.
    """
    from types import SimpleNamespace

    from test_saved_layers import tiny_saved_model

    from studio_compute.mlx_decode import decode_rows
    from studio_compute.numerics.bf16 import decode_hidden

    lm, _ = tiny_saved_model()
    lm.eval()
    engine = SimpleNamespace(
        lm=lm, execution_profile="bf16-fast", compute_dtype=mx.bfloat16
    )
    requests, references = [], []
    for length in (3, 11, 7):
        caches = [
            cache.to_quantized(bits=8, group_size=64)
            if hasattr(cache, "to_quantized")
            else cache
            for cache in lm.make_cache()
        ]
        positions = mx.broadcast_to(mx.arange(length)[None, None], (3, 1, length))
        hidden_BLD = decode_hidden(
            lm, mx.array([list(range(1, length + 1))]), positions, caches
        )
        mx.eval(hidden_BLD, [cache.state for cache in caches])
        references.append([fork_cache(cache) for cache in caches])
        requests.append(
            SimpleNamespace(
                body={"prompt_token_ids": [1] * length},
                cache=[fork_cache(cache) for cache in caches],
                tokens=[],
                logits=None,
            )
        )
    for index, token in enumerate((17, 23)):
        for request in requests:
            request.tokens.append(token)
        decode_rows(engine, requests)
        for request, cache in zip(requests, references, strict=True):
            length = len(request.body["prompt_token_ids"])
            hidden_BLD = decode_hidden(
                lm,
                mx.array([[token]]),
                mx.full((3, 1, 1), length + index, mx.int32),
                cache,
            )
            expected_BV = lm.lm_head(hidden_BLD)[:, -1].astype(mx.bfloat16)
            mx.eval(expected_BV)
            assert float(mx.max(mx.abs(request.logits - expected_BV)).item()) <= 0.125
            assert request.cache[3].offset == length + index + 1
            assert request.cache[3].state[0][0].dtype == mx.uint32


def test_policy_pins_survive_selection_changes_between_boundaries():
    engine = RuntimeEngine(
        "tiny",
        model_factory=TinyResident,
        execution_profile="legacy",
        checkpoint_layers=2,
        prefill_tokens=7,
        inference_concurrency=3,
    )
    try:
        first = engine.idle_factors
        second = {
            name: value + 0.01 if name.endswith(".b") else value
            for name, value in first.items()
        }
        mx.eval(second)
        engine.policies["p"] = {"factors": first, "revision": 0}
        engine.policies["q"] = {"factors": second, "revision": 1}
        bodies = [
            {
                "policy": policy,
                "revision": revision,
                "prompt_token_ids": list(range(1, length)),
                "target_token_ids": [17, 23, 11, 8],
                "max_tokens": 4,
                "use_prefix_cache": False,
                "seed": 17,
            }
            for policy, revision, length in (("p", 0, 5), ("q", 1, 14), ("p", 0, 9))
        ]
        expected = [drain(engine._sample(body, time.monotonic())) for body in bodies]
        actual, errors = {}, []
        scheduler = engine.inference_scheduler(
            create=lambda name, body: inference.create_request(engine, name, body),
            finish=lambda r: actual.update(
                {r.request_id: inference.request_result(engine, r)}
            ),
            fail=lambda name, error: errors.append(str(error)),
        )
        for i, body in enumerate(bodies):
            scheduler.enqueue(str(i), body)
        while scheduler.pending or scheduler.active:
            # A training turn can select another adapter between inference turns.
            engine.resident.select(second)
            scheduler.step()
            if len(scheduler.active) == 3:
                assert not engine._evict_policy()
        assert not errors and not engine.policy_users and not engine.reserved
        for i, reference in enumerate(expected):
            assert actual[str(i)]["policy_revision"] == reference["policy_revision"]
            assert (
                max(
                    abs(a - b)
                    for a, b in zip(
                        actual[str(i)]["behavior_logprobs"],
                        reference["behavior_logprobs"],
                    )
                )
                <= 0.03
            )
    finally:
        engine.close()


def test_eos_retires_before_refill_without_a_decode_pad():
    engine = RuntimeEngine(
        "tiny",
        model_factory=TinyResident,
        execution_profile="legacy",
        checkpoint_layers=2,
        prefill_tokens=7,
        inference_concurrency=2,
    )
    try:
        engine.policies["p"] = {"factors": engine.idle_factors, "revision": 0}
        body = {
            "policy": "p",
            "revision": 0,
            "prompt_token_ids": [1, 2, 3],
            "max_tokens": 1,
            "seed": 17,
            "use_prefix_cache": False,
        }
        first = drain(engine._sample(body, time.monotonic()))["sampled_tokens"][0]
        results, errors, progress = {}, [], []
        scheduler = engine.inference_scheduler(
            create=lambda name, value: inference.create_request(engine, name, value),
            finish=lambda r: results.update(
                {r.request_id: inference.request_result(engine, r)}
            ),
            fail=lambda name, error: errors.append(str(error)),
        )
        scheduler.enqueue("eos", {**body, "max_tokens": 20, "stop_token_ids": [first]})
        scheduler.enqueue(
            "survivor",
            {
                **body,
                "prompt_token_ids": list(range(1, 15)),
                "seed": 23,
                "max_tokens": 5,
            },
        )
        scheduler.enqueue(
            "refill",
            {**body, "prompt_token_ids": [1, 2, 3, 4], "seed": 29, "max_tokens": 2},
        )
        progress.append(scheduler.step())
        assert results["eos"]["finish_reason"] == "stop"
        assert results["eos"]["sampled_tokens"] == [first]
        assert "eos" not in scheduler.active
        progress.append(scheduler.step())
        assert "refill" in scheduler.active and "survivor" in scheduler.active
        while scheduler.pending or scheduler.active:
            progress.append(scheduler.step())
        assert not errors
        assert results["survivor"]["completion_tokens"] == 5
        assert results["refill"]["completion_tokens"] == 2
        assert sum(sum(row["decode_batch_sizes"]) for row in progress) == 5
        assert not engine.reserved and not engine.policy_users
    finally:
        engine.close()
