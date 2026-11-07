"""
Purpose: Measure the Flash n-gram PLE layer and a single-table lookup candidate.
Input: Reconstructed update sequences for real token IDs and the Flash model.
Output: JSON timings for PLE parts at decode and segment sizes, and a bitwise check.
Note: The script owns the GPU lease. It never opens a service state root.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import time
from pathlib import Path

LEASE = Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tokens = json.loads(args.sequences.read_text())["sequences"][0]
    tokens = tokens["prompt_token_ids"] + tokens["sampled_tokens"]
    lease = LEASE.open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    profiler = _load_profiler()
    profiler._write_identity(args.output / "process.json")
    import mlx.core as mx
    from studio_compute.numerics.flash_ple import JoinedNGramTable

    engine = profiler._load_engine(args.model)
    engine.resident.mode(False)
    layer = next(layer for layer in engine.lm.layers if "ple" in layer)
    embedding = layer.ple.ple_embedding
    started = time.perf_counter()
    joined = JoinedNGramTable(embedding.ngram_embedding)
    mx.eval(joined.parameters())
    report = {"join_seconds": time.perf_counter() - started, "cases": {}}
    for batch, length in ((1, 1), (8, 1), (1, 2048)):
        ids = mx.array([tokens[row * 64:row * 64 + length] for row in range(batch)],
                       mx.int32)
        hidden = mx.random.normal((batch, length, 10240)).astype(mx.bfloat16)
        ngram_ids = _ngram_ids(embedding, ids)
        mx.eval(ngram_ids, hidden)
        original = embedding.ngram_embedding(ngram_ids)
        candidate = joined(ngram_ids)
        mx.eval(original, candidate)
        case = {
            "bitwise_equal": bool(mx.array_equal(original, candidate).item()),
            "ngram_ids_seconds": _median(lambda: mx.eval(_ngram_ids(embedding, ids))),
            "sharded_lookup_seconds": _median(
                lambda: mx.eval(embedding.ngram_embedding(ngram_ids))),
            "joined_lookup_seconds": _median(lambda: mx.eval(joined(ngram_ids))),
            "ple_layer_seconds": _median(
                lambda: mx.eval(layer.ple(hidden, ids, None, None))),
        }
        report["cases"][f"{batch}x{length}"] = case
        print(json.dumps({"case": f"{batch}x{length}", **case}), flush=True)
        _write_json(args.output / "ple.json", report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, required=True,
                        help="Reconstructed update JSON.")
    parser.add_argument("--model", type=Path, required=True, help="Flash model path.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    return parser.parse_args()


def _load_profiler():
    path = Path(__file__).with_name("profile_flash_replay.py")
    spec = importlib.util.spec_from_file_location("profile_flash_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ngram_ids(embedding, ids):
    """Repeat the n-gram hashing part of Qwen4ExpNGramEmbedding.__call__."""
    import mlx.core as mx

    history = mx.concatenate([mx.full((ids.shape[0], embedding.context_len),
                                      embedding.eos_token_id, mx.int64),
                              ids.astype(mx.int64)], axis=-1)
    shifted = [embedding._shift_right_ignore_eos(history, shift)
               for shift in range(embedding.ngram_size)]
    blocks = []
    for ngram in range(2, embedding.ngram_size + 1):
        start = (ngram - 2) * embedding.heads_per_ngram
        end = start + embedding.heads_per_ngram
        mixed = shifted[0] * embedding.layer_multipliers[0]
        for position in range(1, ngram):
            mixed = mx.bitwise_xor(mixed, shifted[position]
                                   * embedding.layer_multipliers[position])
        sizes = embedding.ngram_heads_vocab_sizes[start:end]
        offsets = embedding.ngram_heads_offsets[start:end]
        blocks.append(mixed[..., None] % sizes[None, None] + offsets[None, None])
    return mx.concatenate(blocks, axis=-1)[:, -ids.shape[1]:]


def _median(function, repeats: int = 5) -> float:
    function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return sorted(samples)[len(samples) // 2]


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


if __name__ == "__main__":
    main()
