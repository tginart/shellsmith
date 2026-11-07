"""
Purpose: Measure Flash prompt-prefill chunk time by chunk size and expert layout.
Input: Reconstructed update sequences for real prompts, and the Flash model.
Output: JSON timings per (chunk tokens, layout) at a realistic context offset.
Method:
    1. Prefill a real prompt to the context offset with 2048-token chunks.
    2. For each layout and chunk size, fork the cache and time one chunk.
    3. Record the median of the repeats and the hidden difference to sorted rows.
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
    sequences = json.loads(args.sequences.read_text())["sequences"]
    lease = LEASE.open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from studio_compute.numerics import flash_ops

    # Time tiles at every size; the production threshold is chosen from this table.
    flash_ops.PREFILL_TILE_MIN_TOKENS = 1
    decode = _load_module("profile_flash_decode")
    decode._write_identity(args.output / "process.json")
    engine = decode._load_engine(args.model)
    tokens = max((s["prompt_token_ids"] + s["sampled_tokens"] for s in sequences),
                 key=len)
    assert len(tokens) >= args.offset + max(args.chunks) + 1
    report = {"offset": args.offset, "cases": {}}
    base = _prefilled(engine, tokens, offset=args.offset)
    for chunk in args.chunks:
        reference = None
        for tile in (0, *args.tiles):
            case = _time_chunk(engine, base, tokens, chunk=chunk, tile=tile,
                               repeats=args.repeats)
            hidden = case.pop("hidden")
            if reference is None:
                reference = hidden
            case["max_abs_vs_sorted"] = _max_abs(hidden, reference)
            name = f"{chunk}x{'sorted' if tile == 0 else f'tile{tile}'}"
            report["cases"][name] = case
            print(json.dumps({"case": name, **case}), flush=True)
            _write_json(args.output / "prefill.json", report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, required=True,
                        help="Reconstructed update JSON.")
    parser.add_argument("--model", type=Path, required=True, help="Flash model path.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--offset", type=int, default=16384,
                        help="Context tokens already in the cache.")
    parser.add_argument("--chunks", type=int, nargs="+",
                        default=[16, 32, 64, 128, 256, 512, 1024, 2048],
                        help="Prefill chunk sizes in tokens.")
    parser.add_argument("--tiles", type=int, nargs="+", default=[32],
                        help="Expert tile heights to compare with sorted rows.")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repeats.")
    return parser.parse_args()


def _load_module(name: str):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prefilled(engine, tokens: list, *, offset: int):
    """Prefill tokens[:offset] once with the production 2048-token chunks."""
    from studio_compute import inference

    body = {"prompt_token_ids": tokens[:offset], "max_tokens": 1, "seed": 0,
            "revision": 0, "policy": "profile", "cache_precision": "8bit",
            "use_prefix_cache": False}
    request = inference.create_request(engine, "prefill-base", body)
    while request.logits is None:
        inference.prefill_request(engine, request)
    return request


def _time_chunk(engine, base, tokens: list, *, chunk: int, tile: int,
                repeats: int) -> dict:
    """Time one chunk after the base context, on a fresh fork of its caches."""
    from studio_compute.mlx_request_cache import fork_cache

    mx = engine.mx
    offset = len(base.body["prompt_token_ids"])
    ids_BL = mx.array([tokens[offset:offset + chunk]], mx.int32)
    positions = mx.broadcast_to(
        mx.arange(offset, offset + chunk, dtype=mx.int32)[None, None], (3, 1, chunk))
    engine.resident.prefill_moe_chunk = tile
    engine.resident.select(base.factors)
    engine.resident.mode(False)
    samples, hidden = [], None
    try:
        for repeat in range(repeats + 1):
            caches = [fork_cache(item) for item in base.cache]
            mx.eval([item.state for item in caches])
            started = time.perf_counter()
            hidden = engine._decode_hidden(ids_BL, position_ids=positions, cache=caches)
            mx.eval(hidden, [item.state for item in caches])
            if repeat:
                samples.append(time.perf_counter() - started)
    finally:
        engine.resident.prefill_moe_chunk = 0
    seconds = sorted(samples)[len(samples) // 2]
    return {"chunk_tokens": chunk, "tile": tile, "seconds": seconds,
            "tokens_per_second": chunk / seconds, "hidden": hidden}


def _max_abs(left, right) -> float:
    import mlx.core as mx

    return float(mx.max(mx.abs(left.astype(mx.float32)
                               - right.astype(mx.float32))).item())


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


if __name__ == "__main__":
    main()
