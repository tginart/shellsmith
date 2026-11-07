#!/usr/bin/env python3
"""Audit a preserved per-response batch against exact rollout-segment packing.

Input JSON (or .gz): body.sequences is the original response batch;
captures[].turns contains request_id, kind and sequence_index from the harness
compaction adapter. No model execution, checkpoint writes, or data truncation.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from shellsmith.train.backend import TokenSequence
from shellsmith.train.segments import pack_rollout_segments


def analyze(path):
    raw = Path(path).read_bytes()
    payload = json.loads(gzip.decompress(raw) if str(path).endswith(".gz") else raw)
    sequences = payload["body"]["sequences"]
    by_id = {
        s["request_id"]: TokenSequence(
            **{k: tuple(v) if isinstance(v, list) else v for k, v in s.items()}
        )
        for s in sequences
    }
    if len(by_id) != len(sequences):
        raise ValueError("Duplicate requests in original batch")
    packed, rewrites, seen = [], [], set()
    for capture in payload["captures"]:
        turns = []
        for turn in capture["turns"]:
            name = turn["request_id"]
            if name not in by_id:
                raise ValueError(f"Missing original response: {name}")
            if name in seen:
                raise ValueError(f"Response appears in multiple rollouts: {name}")
            seen.add(name)
            turns.append((turn["sequence_index"], by_id[name]))
        result = pack_rollout_segments(turns)
        rewrites.extend(
            {**row, "rollout_id": capture["rollout_id"]} for row in result.rewrites
        )
        packed.extend(result.sequences)
    if seen != set(by_id):
        raise ValueError("Some original responses lack rollout provenance")
    covered = set()
    for segment in packed:
        full = segment.prompt_token_ids + segment.sampled_tokens
        for name, start, stop in segment.source_spans:
            original = by_id[name]
            if name in covered:
                raise ValueError("Supervision assigned twice")
            covered.add(name)
            if (
                full[:start] != original.prompt_token_ids
                or full[start:stop] != original.sampled_tokens
            ):
                raise ValueError(
                    "Packing changed a response's conditioning history or target IDs"
                )
            offset = start - len(segment.prompt_token_ids)
            for field in ("behavior_logprobs", "advantages", "loss_mask", "weights"):
                before = getattr(original, field)
                after = getattr(segment, field)
                if (before is None) != (after is None) or (
                    before is not None
                    and after[offset : offset + len(before)] != before
                ):
                    raise ValueError(f"Packing changed {field} for {name}")
    if covered != seen:
        raise ValueError("Incomplete supervision coverage")
    lengths = [len(s.prompt_token_ids) + len(s.sampled_tokens) - 1 for s in packed]
    old = sum(
        len(s.prompt_token_ids) + len(s.sampled_tokens) - 1 for s in by_id.values()
    )
    return {
        "schema": "shellsmith.segment_reconstruction.v1",
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "provenance": payload.get("provenance"),
        "historical_source_sha256": payload.get("source_sha256"),
        "rollouts": len(payload["captures"]),
        "responses": len(sequences),
        "prompt_tokens": sum(len(s.prompt_token_ids) for s in by_id.values()),
        "target_tokens": sum(len(s.sampled_tokens) for s in by_id.values()),
        "packed_segments": len(packed),
        "packed_shifted_lengths": lengths,
        "old_replay_tokens": old,
        "packed_replay_tokens": sum(lengths),
        "replay_reduction_fraction": 1 - sum(lengths) / old,
        "supervised_tokens_before": sum(sum(s.loss_mask) for s in by_id.values()),
        "supervised_tokens_after": sum(sum(s.loss_mask) for s in packed),
        "source_requests_verified": len(covered),
        "history_rewrites": rewrites,
        "causal_prefixes_targets_credit_and_logprobs_exact": True,
        "qualification": "Token replay reduction; not a measured full-batch speedup.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("batch")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    report = analyze(a.batch)
    Path(a.output).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "history_rewrites"}, indent=2
        )
    )
