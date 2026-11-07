"""
Purpose: Decompose the Flash sampling-versus-replay logprob error into causes.
Input: Reconstructed update sequences, player records, and the Flash model.
Output: Per-variant logprobs for selected sampled spans and pairwise error statistics.
Note: The script owns the GPU lease. It never opens a service state root.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

LEASE = Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"
VARIANTS = (
    "inference_8bit",
    "inference_bf16",
    "inference_8bit_unsorted",
    "replay_unsorted",
    "replay_sorted",
)


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.sequences.read_text())
    records = _load_records(args.records)
    spans = _select_spans(payload["sequences"], records, args.spans_per_sequence,
                          args.sequence_indices, args.max_context)
    lease = LEASE.open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    _write_identity(args.output / "process.json")
    engine = _load_engine(args.model)
    results = []
    for span in spans:
        row = {key: span[key] for key in ("sequence", "request_id", "start", "stop")}
        row["behavior"] = span["behavior"]
        for variant in args.variants:
            started = time.perf_counter()
            row[variant] = _run_variant(engine, span, variant)
            row[variant + "_seconds"] = time.perf_counter() - started
        results.append(row)
        _write_json(args.output / "spans.json", results)
        print(json.dumps({"done": row["request_id"], "tokens": len(row["behavior"])}),
              flush=True)
    _write_json(args.output / "summary.json", _summarize(results, args.variants))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, required=True,
                        help="Reconstructed update JSON from reconstruct_studio_update.")
    parser.add_argument("--records", type=Path, required=True,
                        help="The segment player_records directory.")
    parser.add_argument("--model", type=Path, required=True, help="Flash model path.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--sequence-indices", type=int, nargs="+", default=[11, 0],
                        help="Update sequence indices to probe.")
    parser.add_argument("--spans-per-sequence", type=int, default=4,
                        help="Sampled spans per sequence, from the start.")
    parser.add_argument("--max-context", type=int, default=8192,
                        help="Skip spans whose context ends after this many tokens.")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS),
                        help="Numerical paths: the defaults, or inference_<8bit|bf16>"
                             "[_tileN][_chunkN] and replay_tileN.")
    return parser.parse_args()


def _load_records(root: Path) -> dict:
    records = {}
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            records[row["request_id"]] = row
    return records


def _select_spans(sequences, records, count, indices, max_context) -> list:
    """Pick sampled spans and verify the packed context equals the request prompt."""
    spans = []
    for index in indices:
        sequence = sequences[index]
        tokens = sequence["prompt_token_ids"] + sequence["sampled_tokens"]
        prompt_length = len(sequence["prompt_token_ids"])
        chosen = 0
        for request_id, start, stop in sequence["source_spans"]:
            if chosen == count or stop > max_context:
                break
            record = records[request_id]
            assert tokens[:start] == record["prompt_token_ids"]
            assert tokens[start:stop] == record["sampled_tokens"]
            behavior = sequence["behavior_logprobs"][start - prompt_length:
                                                     stop - prompt_length]
            assert behavior == record["sampled_logprobs"]
            spans.append({
                "sequence": index, "request_id": request_id, "start": start,
                "stop": stop, "prompt": tokens[:start], "targets": tokens[start:stop],
                "behavior": behavior, "thinking_tokens": record.get("thinking_tokens"),
            })
            chosen += 1
    return spans


def _write_identity(path: Path) -> None:
    pid = os.getpid()
    started = subprocess.check_output(["ps", "-o", "lstart=,command=", "-p", str(pid)],
                                      text=True).strip()
    _write_json(path, {"pid": pid, "ps_start_and_command": started,
                       "lease": str(LEASE)})


def _load_engine(model: Path):
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    from studio_compute.flash_model import QwenFlashModel
    from studio_compute.runtime import RuntimeEngine

    engine = RuntimeEngine(
        str(model), model_factory=QwenFlashModel, execution_profile="bf16-fast",
        activation_budget_gib=16, inference_concurrency=8, prefill_tokens=2048,
        segment_tokens=2048, head_rows=128,
    )
    # B = 0 makes the LoRA update exactly zero, as in the first r3 update.
    factors = engine.resident.adapter.initial(20261019)
    engine.mx.eval(factors)
    engine.policies["diagnostic"] = {"factors": factors, "revision": 0}
    engine.diagnostic_factors = factors
    return engine


def _run_variant(engine, span: dict, variant: str) -> list:
    """
    Purpose: Run one named numerical path for one sampled span.
    Input: The engine, the span, and a variant name.
    Output: Logprobs of the span's sampled tokens.
    Note: _tileN sets the expert tile height and _chunkN the prefill chunk size.
    """
    parts = variant.split("_")
    tile = next((int(p[4:]) for p in parts if p.startswith("tile")), 0)
    chunk = next((int(p[5:]) for p in parts if p.startswith("chunk")), None)
    if parts[0] == "inference":
        return _forced_inference(
            engine, span["prompt"], span["targets"],
            precision="8bit" if "8bit" in parts else "bf16",
            sort_rows="unsorted" not in parts, tile=tile, chunk=chunk,
        )
    assert parts[0] == "replay", variant
    return _forced_replay(engine, span["prompt"], span["targets"],
                          training=variant == "replay_unsorted" or tile > 0, tile=tile)


def _forced_inference(engine, prompt, targets, *, precision, sort_rows, tile=0,
                      chunk=None) -> list:
    """Run the sampler's prefill and decode code at batch one with forced tokens."""
    from studio_compute import inference
    from studio_compute.numerics import flash_ops

    body = {
        "prompt_token_ids": prompt, "target_token_ids": targets,
        "max_tokens": len(targets), "seed": 1, "revision": 0,
        "policy": "diagnostic", "cache_precision": precision,
        "use_prefix_cache": False,
    }
    original = flash_ops._moe
    default_chunk = engine.prefill_tokens
    engine.resident.prefill_moe_chunk = tile
    engine.prefill_tokens = chunk or default_chunk
    if not sort_rows:
        flash_ops._moe = lambda mixer, hidden, sort_rows=False: original(
            mixer, hidden, sort_rows=False)
    try:
        request = inference.create_request(engine, "diagnostic", body)
        try:
            while request.logits is None:
                inference.prefill_request(engine, request)
            while request.reason is None:
                if request.needs_decode:
                    inference.advance_requests(engine, [request])
                inference.sample_requests(engine, [request])
            return list(request.logprobs)
        finally:
            inference.release_request(engine, request)
    finally:
        flash_ops._moe = original
        engine.resident.prefill_moe_chunk = 0
        engine.prefill_tokens = default_chunk


def _forced_replay(engine, prompt, targets, *, training: bool, tile: int = 0) -> list:
    """Run the training replay forward with the update's segment layout."""
    from studio_compute.numerics.replay import Replay

    mx = engine.mx
    ids = mx.array([prompt + targets[:-1]], mx.int32)
    embeddings = engine.replay_ops.expand(
        engine.lm, engine.lm.model.embed_tokens(ids).astype(mx.bfloat16))
    positions = mx.broadcast_to(mx.arange(ids.shape[1], dtype=mx.int32)[None, None],
                                (3, 1, ids.shape[1]))
    engine.resident.select(engine.diagnostic_factors)
    engine.resident.mode(training)
    engine.replay_ops.training_moe_chunk = tile or None
    try:
        replay = Replay(engine.resident.model, checkpoint_layers=2,
                        segment_tokens=engine.segment_tokens,
                        head_rows=engine.head_rows,
                        execution_profile=engine.execution_profile,
                        model_ops=engine.replay_ops)
        work = replay.run_group([{
            "embeddings": embeddings, "positions": positions,
            "prefix_length": len(prompt), "tokens": targets, "objective": None,
            "input_ids": ids,
        }])
        while True:
            try:
                next(work)
            except StopIteration as stop:
                currents = stop.value[0]
                break
        return currents[0].astype(mx.float32).tolist()
    finally:
        engine.resident.mode(False)
        engine.replay_ops.training_moe_chunk = None


def _summarize(results: list, variants: list) -> dict:
    """Pairwise mean signed, mean absolute, and maximum logprob differences."""
    names = ["behavior", *variants]
    summary = {}
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            diffs = [a - b for row in results
                     for a, b in zip(row[first], row[second], strict=True)]
            first_token = [row[first][0] - row[second][0] for row in results]
            summary[f"{first}-{second}"] = {
                "tokens": len(diffs),
                "mean_signed": sum(diffs) / len(diffs),
                "mean_abs": sum(map(abs, diffs)) / len(diffs),
                "max_abs": max(map(abs, diffs)),
                "p99_abs": sorted(map(abs, diffs))[int(0.99 * (len(diffs) - 1))],
                "fraction_gt_1": sum(abs(x) > 1 for x in diffs) / len(diffs),
                "exact_fraction": sum(x == 0 for x in diffs) / len(diffs),
                "first_token_mean_abs": sum(map(abs, first_token)) / len(first_token),
            }
    return summary


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


if __name__ == "__main__":
    main()
