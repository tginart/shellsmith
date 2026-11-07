"""Sum-reduced CISPO using Shellsmith's already-normalized token advantages."""

from __future__ import annotations

import math
from typing import Any, Iterable


def validate_clip(low: float, high: float) -> None:
    if not all(math.isfinite(x) for x in (low, high)) or not 0 <= low <= high or high <= 0:
        raise ValueError("CISPO clip thresholds require finite 0 <= low <= high and high > 0.")


def cispo_loss(current: Any, old: Any, advantages: Any, mask: Any = None,
               *, clip_low: float = 0.0, clip_high: float = 4.0) -> Any:
    """Negative sum of detached clipped importance ratios * advantage * logprob.

    The credit bridge supplies rollout-wide normalized advantages; dividing by
    sequence length, datum count, or world count here changes the objective.
    """
    import mlx.core as mx

    validate_clip(clip_low, clip_high)
    old = mx.stop_gradient(mx.array(old, dtype=mx.float32))
    advantages = mx.stop_gradient(mx.array(advantages, dtype=mx.float32))
    if current.ndim != 1 or current.shape != old.shape or current.shape != advantages.shape:
        raise ValueError("CISPO requires aligned token probabilities and advantages.")
    current = current.astype(mx.float32)
    log_ratio = mx.minimum(mx.stop_gradient(current) - old, math.log(clip_high))
    if clip_low > 0:
        log_ratio = mx.maximum(log_ratio, math.log(clip_low))
    weighted = mx.exp(log_ratio) * advantages * current
    if mask is not None:
        mask = mx.array(mask, dtype=mx.bool_)
        if mask.shape != current.shape:
            raise ValueError("CISPO requires an aligned token loss mask.")
        weighted = mx.where(mask, weighted, 0.0)
    return -mx.sum(weighted)


def reference_loss_and_gradient(current: Iterable[float], old: Iterable[float], advantages: Iterable[float],
                                mask: Iterable[float] | None = None, *, clip_low: float = 0.0,
                                clip_high: float = 4.0) -> tuple[float, tuple[float, ...]]:
    """Portable scalar oracle for detached-ratio CISPO, including zero gradients."""
    validate_clip(clip_low, clip_high)
    current, old, advantages = tuple(current), tuple(old), tuple(advantages)
    mask = tuple(mask) if mask is not None else (1.0,) * len(current)
    if not current or not (len(current) == len(old) == len(advantages) == len(mask)):
        raise ValueError("CISPO oracle requires nonempty aligned arrays.")
    losses, gradients = [], []
    for new, prior, advantage, enabled in zip(current, old, advantages, mask):
        if not all(math.isfinite(x) for x in (new, prior, advantage, enabled)) or enabled not in (0, 1):
            raise ValueError("CISPO oracle requires finite probabilities/advantages and binary masks.")
        ratio = math.exp(min(new - prior, math.log(clip_high)))
        ratio = max(clip_low, ratio)
        gradient = -ratio * advantage if enabled else 0.0
        gradients.append(gradient)
        losses.append(gradient * new)
    return math.fsum(losses), tuple(gradients)


def frozen_head_logprobs(language_model: Any, hidden: Any, tokens: Any) -> Any:
    """Bounded head tile: normalize hidden rows and select FP32 token logprobs."""
    import mlx.core as mx

    decoder = language_model.model
    normalized = decoder.norm(hidden)
    head = decoder.embed_tokens.as_linear if language_model.args.tie_word_embeddings else language_model.lm_head
    logits = head(normalized).astype(mx.float32)
    return mx.take_along_axis(logits, tokens[:, None], axis=-1)[:, 0] - mx.logsumexp(logits, axis=-1)
