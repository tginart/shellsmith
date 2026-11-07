"""
Purpose: Rebuild the exact Studio forward_backward sequences of one committed update.
Input: A run log directory, the train config, the update step, and the service journal.
Output: A JSON file with the RPC sequences and a check against the journal receipt.
Note: The script only reads the run. It uses the trainer's own credit and packing code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from shellsmith.credit.to_tinker_bridge import build_tinker_group
from shellsmith.credit.turn_credit import TurnAdvantageConfig
from shellsmith.train.backend import TokenBatch
from shellsmith.train.config import load_train_config
from shellsmith.train.durable_capture import load_captures_from_index
from shellsmith.train.studio import StudioBackend


def main() -> None:
    args = _parse_args()
    config = load_train_config(args.config)
    metrics = _update_metrics(args.log_dir / "metrics.jsonl", step=args.step)
    captures = load_captures_from_index(args.log_dir / "segments" / f"{args.step:03d}")
    segments = []
    for world in metrics["worlds"]:
        if world["decision"] != "group_ready":
            continue
        rollouts = tuple(
            capture
            for episode_id, capture in sorted(captures.items())
            if f"-{world['world_id']}-" in episode_id
        )
        bridge = build_tinker_group(
            deps=None,
            group_id=f"step{args.step}:{world['world_id']}",
            step=args.step,
            rollouts=rollouts,
            advantage_std_floor=config.credit.advantage_std_floor,
            advantage_cap=config.credit.advantage_cap,
            turn_config=TurnAdvantageConfig(
                **{
                    name: getattr(config.credit, name)
                    for name in TurnAdvantageConfig.__dataclass_fields__
                }
            ),
        )
        assert len(bridge.segments) == world["training_segments"], world["world_id"]
        segments.extend(bridge.segments)
    sequences = StudioBackend._sequences(TokenBatch(tuple(segments)))
    receipt = _journal_receipt(args.journal)
    check = _check_receipt(sequences, receipt)
    payload = {
        "step": args.step,
        "sequences": sequences,
        "replay_logprobs": receipt["logprobs"],
        "logprob_target_indices": receipt["logprob_target_indices"],
        "check": check,
    }
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    args.output.write_bytes(encoded)
    print(json.dumps({**check, "sha256": hashlib.sha256(encoded).hexdigest()}))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True, help="Run train/ dir.")
    parser.add_argument("--config", type=Path, required=True, help="Run train.toml.")
    parser.add_argument("--journal", type=Path, required=True, help="Journal copy.")
    parser.add_argument("--step", type=int, default=0, help="Update step index.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON.")
    return parser.parse_args()


def _update_metrics(path: Path, *, step: int) -> dict:
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["step"] == step:
            return row
    raise ValueError(f"No metrics row for step {step}.")


def _journal_receipt(path: Path) -> dict:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = db.execute(
        "SELECT result FROM operations WHERE kind='forward_backward' "
        "AND status='succeeded'"
    ).fetchall()
    assert len(rows) == 1, "Expected exactly one committed forward_backward."
    return json.loads(rows[0][0])


def _check_receipt(sequences: list, receipt: dict) -> dict:
    """Require the same sequence count, token totals, and target index sets."""
    metrics = receipt["forward_metrics"]
    expected = [
        [i for i, mask in enumerate(s["loss_mask"]) if mask]
        if s.get("source_spans")
        else None
        for s in sequences
    ]
    assert len(sequences) == metrics["sequences"]
    assert sum(len(s["prompt_token_ids"]) for s in sequences) == metrics["prompt_tokens"]
    assert sum(len(s["sampled_tokens"]) for s in sequences) == metrics[
        "completion_tokens"
    ]
    assert expected == receipt["logprob_target_indices"]
    return {
        "sequences": len(sequences),
        "prompt_tokens": metrics["prompt_tokens"],
        "completion_tokens": metrics["completion_tokens"],
        "target_indices_match": True,
    }


if __name__ == "__main__":
    main()
