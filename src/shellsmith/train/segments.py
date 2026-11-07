"""Causally exact rollout packing. Loss arrays align to *target* token IDs.

A logical segment ends at compaction. Renderers can also rewrite history (for
example stripping reasoning or reserializing tool calls). Such a discontinuity
ends a physical path, is reported, and preserves the preceding supervision.
Never transfer a loss to a different conditioning prefix, even if text matches.
"""

from dataclasses import dataclass

from .backend import TokenSequence


@dataclass(frozen=True)
class PackedRollout:
    sequences: tuple[TokenSequence, ...]
    rewrites: tuple[dict, ...]


def pack_rollout_segments(turns, *, on_incompatible="split") -> PackedRollout:
    """Pack ordered (logical segment index, TokenSequence) pairs without renormalizing.

    ``error`` is useful for harnesses promising append-only histories. ``split``
    explicitly retains both actual paths when a template rewrites the history.
    Source spans are half-open offsets in prompt+tail, including masked injected
    tokens. Every source request occurs once, including zero-credit actions.
    """
    if on_incompatible not in ("split", "error"):
        raise ValueError("Unknown incompatible-history policy")
    packed, rewrites, seen = [], [], set()
    active, logical, full, spans = None, None, (), []

    def finish():
        if active is not None:
            packed.append(
                TokenSequence(
                    prompt_token_ids=active.prompt_token_ids,
                    sampled_tokens=tuple(tail),
                    behavior_logprobs=tuple(old),
                    advantages=tuple(advantages),
                    loss_mask=tuple(mask),
                    request_id=active.request_id,
                    weights=tuple(weights) if weighted else None,
                    source_spans=tuple(spans),
                )
            )

    for index, sequence in turns:
        if sequence.request_id in seen:
            raise ValueError("Duplicate source request in rollout")
        seen.add(sequence.request_id)
        if (
            type(index) is not int
            or index < 0
            or (logical is not None and index < logical)
        ):
            raise ValueError("Rollout segment indices must be ordered and nonnegative")
        prompt = sequence.prompt_token_ids
        compatible = (
            active is not None and index == logical and prompt[: len(full)] == full
        )
        if active is not None and index == logical and not compatible:
            common = next(
                (i for i, (a, b) in enumerate(zip(full, prompt)) if a != b),
                min(len(full), len(prompt)),
            )
            detail = {
                "previous_request": spans[-1][0],
                "request_id": sequence.request_id,
                "logical_segment": index,
                "common_prefix_tokens": common,
                "previous_path_tokens": len(full),
                "next_prompt_tokens": len(prompt),
            }
            if on_incompatible == "error":
                raise ValueError(f"Incompatible causal history: {detail}")
            rewrites.append(detail)
        if not compatible:
            finish()
            active, logical = sequence, index
            full, spans = prompt, []
            tail, old, advantages, mask, weights = [], [], [], [], []
            weighted = sequence.weights is not None
        if weighted != (sequence.weights is not None):
            raise ValueError("Cannot mix weighted cross entropy and CISPO in a segment")
        gap = prompt[len(full) :]
        tail.extend(gap)
        old.extend([0.0] * len(gap))
        advantages.extend([0.0] * len(gap))
        mask.extend([0.0] * len(gap))
        weights.extend([0.0] * len(gap))
        start = len(prompt)
        tail.extend(sequence.sampled_tokens)
        old.extend(sequence.behavior_logprobs)
        advantages.extend(sequence.advantages)
        mask.extend(sequence.loss_mask)
        weights.extend(sequence.weights or [0.0] * len(sequence.sampled_tokens))
        full = prompt + sequence.sampled_tokens
        spans.append((sequence.request_id, start, len(full)))
    finish()
    return PackedRollout(tuple(packed), tuple(rewrites))
