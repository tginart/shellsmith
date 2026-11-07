"""
Purpose: Own inference resources across cooperative scheduler turns.
Input: Validated requests and immutable policy references. Output: Existing sample receipts.
Note: Training and checkpoint code remain in ResidentEngine.
"""

import math
import time
from typing import Any

from .mlx_racing import DecodeRequest


def create_request(engine: Any, request_id: str, body: dict) -> DecodeRequest | None:
    """
    Purpose: Reserve a complete request peak before cache or policy allocation.
    Input: A prepared request with its durable RNG key. Output: A request or a memory deferral.
    Note: Oversized work fails when no other reservation can release memory.
    """
    started = time.monotonic()
    maximum = len(body.get("target_token_ids", [])) or body["max_tokens"]
    precision = body.get("cache_precision", "bf16")
    peak = engine.resident.estimate_inference(
        len(body["prompt_token_ids"]) + maximum,
        cache_precision=precision,
    )
    if body.get("use_prefix_cache", True):
        peak += min(
            engine.prefixes.limit,
            engine.resident.estimate_inference(
                len(body["prompt_token_ids"]), cache_precision=precision
            ),
        )
    policy_bytes = (
        0
        if body["policy"] in engine.policies
        else sum(value.nbytes for value in engine.idle_factors.values())
    )
    if engine.reserved:
        noncache = engine.persistent_bytes() - engine.prefixes.bytes
        needed = noncache + peak + policy_bytes
        budget = engine.admission_budget_bytes()
        if needed <= budget < needed + sum(engine.reserved.values()):
            # Snapshot eviction cannot free a live peak reservation. Keep warm
            # prefixes until another request or replay releases that reservation.
            return None
    try:
        engine.admit(peak + policy_bytes)
    except MemoryError:
        # Other inference and replay owners release these reservations at boundaries.
        # Reject a request that cannot fit even after those owners finish.
        noncache = engine.persistent_bytes() - engine.prefixes.bytes
        if engine.reserved and noncache + peak <= engine.admission_budget_bytes():
            return None
        raise
    reservation = ("continuous_inference", request_id)
    engine.reserved[reservation] = peak
    lease = engine._borrow_policy(body["policy"])
    entered = False
    try:
        try:
            factors = lease.__enter__()
        except MemoryError:
            # Admission can evict the requested policy. Its reload must wait
            # for other live owners when the newly cold policy no longer fits.
            engine.reserved.pop(reservation, None)
            if engine.reserved:
                return None
            raise
        entered = True
        precision = body.get("cache_precision", "bf16")
        hit = (
            engine.prefixes.get(
                engine.prefix_identity,
                body["policy"],
                precision,
                body["prompt_token_ids"],
                fork=True,
            )
            if body.get("use_prefix_cache", True)
            else None
        )
        offset, logits = 0, None
        if hit is not None:
            cache, logits, offset = hit
        else:
            cache = engine.lm.make_cache()
            if precision == "8bit":
                cache = [
                    item.to_quantized(bits=8, group_size=64)
                    if hasattr(item, "to_quantized")
                    else item
                    for item in cache
                ]
        mx = engine.mx
        rng = (
            mx.array(body["rng_key"], mx.uint32)
            if "rng_key" in body
            else mx.random.key(body["seed"])
        )
        request = DecodeRequest(
            request_id,
            body,
            cache,
            factors,
            lease,
            rng,
            started,
            offset=offset,
            cached_tokens=offset,
        )
        if offset == len(body["prompt_token_ids"]):
            request.logits = logits
            request.decode_started = time.monotonic()
            request.prefill_seconds = request.decode_started - started
        return request
    except BaseException:
        if entered:
            lease.__exit__(None, None, None)
        engine.reserved.pop(reservation, None)
        raise


def prefill_request(engine: Any, request: DecodeRequest) -> None:
    """
    Purpose: Advance at most one configured prompt chunk.
    Input: A request with an incomplete prompt. Output: Updated prompt position and final logits.
    """
    mx, body = engine.mx, request.body
    prompt = body["prompt_token_ids"]
    start = request.offset
    stop = min(len(prompt), start + engine.prefill_tokens)
    engine.resident.select(request.factors)
    engine.resident.mode(False)
    positions = mx.broadcast_to(
        mx.arange(start, stop, dtype=mx.int32)[None, None], (3, 1, stop - start)
    )
    hidden_BLD = engine._decode_hidden(
        mx.array([prompt[start:stop]], mx.int32),
        position_ids=positions,
        cache=request.cache,
    )
    mx.eval(hidden_BLD, [item.state for item in request.cache])
    request.offset = stop
    if stop < len(prompt):
        return
    lm = engine.lm
    head = (
        lm.model.embed_tokens.as_linear if lm.args.tie_word_embeddings else lm.lm_head
    )
    request.logits = head(hidden_BLD[:, -1:])[:, -1].astype(engine.compute_dtype)
    mx.eval(request.logits)
    if body.get("use_prefix_cache", True):
        engine.prefixes.put(
            engine.prefix_identity,
            body["policy"],
            body.get("cache_precision", "bf16"),
            prompt,
            request.cache,
            request.logits,
        )
    request.prefill_seconds = time.monotonic() - request.started
    request.decode_started = time.monotonic()


def advance_requests(engine: Any, requests: list[DecodeRequest]) -> None:
    engine.resident.select(requests[0].factors)
    engine.resident.mode(False)
    from .mlx_decode import decode_rows

    decode_rows(engine, requests)
    for request in requests:
        request.batch_size = max(request.batch_size, len(requests))


def sample_requests(engine: Any, requests: list[DecodeRequest]) -> None:
    """
    Purpose: Preserve per-request categorical draws and exact behavior records.
    Input: Independent logits and RNG keys. Output: One materialized token per request.
    Note: Forced targets serve internal parity probes. RPC sampling never accepts them.
    """
    mx, pending = engine.mx, []
    for request in requests:
        logits_BV = request.logits
        lp_BV = logits_BV - mx.logsumexp(logits_BV, axis=-1, keepdims=True)
        forced = request.body.get("target_token_ids")
        if forced:
            draw_S = mx.array(forced[len(request.tokens)], mx.int32)
        else:
            request.rng, key = mx.random.split(request.rng)
            draw_S = engine._draw_token(lp_BV, key)
        probability_S = mx.take(lp_BV[0], draw_S, axis=-1)
        pending.append((request, draw_S, probability_S))
    mx.eval(
        [draw for _, draw, _ in pending],
        [lp for _, _, lp in pending],
        [r.rng for r in requests],
    )
    for request, draw_S, probability_S in pending:
        token, probability = int(draw_S.item()), float(probability_S.item())
        if not math.isfinite(probability):
            raise FloatingPointError("Nonfinite behavior logprob.")
        request.tokens.append(token)
        request.logprobs.append(probability)
        body = request.body
        forced = body.get("target_token_ids")
        maximum = len(forced) if forced else body["max_tokens"]
        if not forced and token in body.get("stop_token_ids", []):
            request.reason = "stop"
        elif len(request.tokens) == maximum:
            request.reason = "length"
        request.needs_decode = request.reason is None
        if not forced:
            engine._observe_generated(
                body, len(request.tokens), request.reason is not None
            )


def request_result(engine: Any, request: DecodeRequest) -> dict:
    body = request.body
    return {
        "sampled_tokens": request.tokens,
        "behavior_logprobs": request.logprobs,
        "logprobs": request.logprobs,
        "prompt_tokens": len(body["prompt_token_ids"]),
        "completion_tokens": len(request.tokens),
        "policy": body["policy"],
        "policy_revision": body["revision"],
        "finish_reason": request.reason,
        "cache_precision": body.get("cache_precision", "bf16"),
        "prefix_cache_tokens": request.cached_tokens,
        "batch_size": request.batch_size,
        "prefill_seconds": request.prefill_seconds,
        "decode_seconds": time.monotonic() - request.decode_started,
        "execution_metrics": engine.metrics(request.started),
    }


def release_request(engine: Any, request: DecodeRequest) -> None:
    """Purpose: Release all request resources after success, failure, or cancellation."""
    try:
        request.cache.clear()
        request.logits = request.rng = request.factors = None
        request.lease.__exit__(None, None, None)
    finally:
        engine.reserved.pop(("continuous_inference", request.request_id), None)
