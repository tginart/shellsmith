"""Portable compute boundary. JSON values only; no application or GPU imports."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


MAX_REQUEST_BYTES = 128 * 1024 * 1024


def update_body(sequences, clip_low, clip_high):
    return {
        "sequences": sequences, "loss": "cispo", "reduction": "sum",
        "clip_low": clip_low, "clip_high": clip_high,
    }


def encoded_request(payload):
    data = canonical(payload)
    if len(data) > MAX_REQUEST_BYTES:
        raise ValueError(
            f"Studio request is {len(data)} bytes; HTTP limit is {MAX_REQUEST_BYTES} bytes. "
            "No RPC submitted."
        )
    return data


class Conflict(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 200:
        raise ValueError("A durable request ID of 1–200 characters is required.")
    return value


def bounded_identifier(value: str, *, limit: int = 200) -> str:
    """Keep generated durable names stable while bounding repeated run labels."""
    if type(limit) is not int or not 74 <= limit <= 200:
        raise ValueError("Generated ID limit must fit a digest and be at most 200.")
    if not isinstance(value, str) or not value:
        raise ValueError("A nonempty operation label is required.")
    return value if len(value) <= limit else "studio-op:" + hashlib.sha256(value.encode()).hexdigest()


def tokens(value: Any, vocabulary: int) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(x) is not int or not 0 <= x < vocabulary for x in value)
    ):
        raise ValueError(
            "Token IDs must be a nonempty array within the served vocabulary."
        )
    return value


def numbers(value: Any, count: int) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(type(x) not in (float, int) or not math.isfinite(x) for x in value)
    ):
        raise ValueError("Numerical arrays must be finite and aligned with targets.")
    return value


def validate_request(kind: str, body: dict, capabilities: dict) -> None:
    """Reject oversized or malformed work before it touches RNG or the GPU."""
    context, vocabulary = (
        capabilities["context_window"],
        capabilities["vocabulary_size"],
    )
    if kind in ("sample", "logprobs"):
        prompt = tokens(body.get("prompt_token_ids"), vocabulary)
        if not isinstance(body.get("policy"), str):
            raise ValueError("Sampling/scoring requires an immutable policy reference.")
        if body.get("cache_precision", "bf16") not in capabilities["cache_precisions"]:
            raise ValueError("Unsupported inference cache precision.")
        if kind == "sample":
            if "target_token_ids" in body:
                raise ValueError(
                    "Use logprobs to score target IDs; sample draws its own tokens."
                )
            maximum = body.get("max_tokens")
            if type(maximum) is not int or maximum < 1:
                raise ValueError("max_tokens must be a positive integer.")
            if body.get("temperature", 1.0) != 1.0 or body.get("top_p", 1.0) != 1.0:
                raise ValueError(
                    "V1 sampling requires temperature=1 and top_p=1 for exact behavior logprobs."
                )
            if "seed" in body and (
                type(body["seed"]) is not int or not 0 <= body["seed"] < 2**32
            ):
                raise ValueError("Sample seed must be a uint32.")
            stops = body.get("stop_token_ids", [])
            if not isinstance(stops, list) or any(
                type(x) is not int or not 0 <= x < vocabulary for x in stops
            ):
                raise ValueError("Invalid stop token IDs.")
        else:
            maximum = len(tokens(body.get("target_token_ids"), vocabulary))
        if len(prompt) + maximum > context:
            raise ValueError(
                f"Context overflow: prompt plus completion exceeds {context} tokens."
            )
    elif kind == "forward_backward":
        if body.get("reduction", "sum") != "sum":
            raise ValueError(
                "Losses use sum reduction; normalization belongs to the caller."
            )
        loss = body.get("loss", "cispo")
        if loss not in ("cispo", "weighted_cross_entropy"):
            raise ValueError("Unsupported loss.")
        sequences = body.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise ValueError("A nonempty sequence batch is required.")
        for sequence in sequences:
            if not isinstance(sequence, dict):
                raise ValueError("Each training sequence must be an object.")
            prompt = tokens(sequence.get("prompt_token_ids"), vocabulary)
            targets = tokens(sequence.get("sampled_tokens"), vocabulary)
            if len(prompt) + len(targets) > context:
                raise ValueError(
                    f"Context overflow: prompt plus targets exceeds {context} tokens."
                )
            mask = numbers(
                sequence.get("loss_mask", [1.0] * len(targets)), len(targets)
            )
            if any(x not in (0, 1) for x in mask):
                raise ValueError("Loss masks must be binary.")
            spans = sequence.get("source_spans", [])
            if not isinstance(spans, (list, tuple)):
                raise ValueError("Source spans must be an array.")
            covered, previous, names = set(), len(prompt), set()
            for span in spans:
                if not isinstance(span, (list, tuple)) or len(span) != 3:
                    raise ValueError("Invalid sampled source span.")
                name, start, stop = span
                if (not isinstance(name, str) or not name or name in names
                        or type(start) is not int or type(stop) is not int
                        or not previous <= start < stop <= len(prompt) + len(targets)):
                    raise ValueError("Source spans must be unique, ordered, and target-aligned.")
                names.add(name); previous = stop
                covered.update(range(start-len(prompt), stop-len(prompt)))
            if spans and "loss_mask" not in sequence:
                raise ValueError("Packed sequences require an explicit loss mask.")
            if spans and any(m and i not in covered for i,m in enumerate(mask)):
                raise ValueError("Context outside sampled source spans cannot receive supervision.")
            if loss == "cispo":
                numbers(sequence.get("behavior_logprobs"), len(targets))
                advantage = numbers(sequence.get("advantages"), len(targets))
                if any(x != 0 for x, enabled in zip(advantage, mask) if not enabled):
                    raise ValueError("Masked advantages must be zero.")
            else:
                numbers(sequence.get("weights"), len(targets))
        low, high = body.get("clip_low", 0.0), body.get("clip_high", 4.0)
        if (
            not all(type(x) in (int, float) and math.isfinite(x) for x in (low, high))
            or not 0 <= low <= high
            or high <= 0
        ):
            raise ValueError("Invalid CISPO thresholds.")
    elif kind == "create":
        optimizer = body.get("optimizer", {})
        if not isinstance(optimizer, dict) or set(optimizer) - {"beta1", "beta2", "epsilon"}:
            raise ValueError("Invalid optimizer configuration.")
        for name, default in (("beta1", 0.9), ("beta2", 0.95), ("epsilon", 1e-8)):
            value = optimizer.get(name, default)
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or (not 0 <= value < 1 if name != "epsilon" else value <= 0)):
                raise ValueError(f"Invalid optimizer {name}.")
    elif kind == "save":
        if body.get("kind", "state") not in ("state", "sampler"):
            raise ValueError("Checkpoint kind must be state or sampler.")
    elif kind == "optim_step":
        rate = body.get("learning_rate")
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("Learning rate must be finite and positive.")
    elif kind not in ("create", "close", "suspend", "save", "restore", "publish", "inspect"):
        raise ValueError(f"Unknown operation: {kind}")
