"""
Purpose: Compare one-token decoder latency with identical real-model inputs and carries.
Input: The 27B model and preserved junior12 token path.
Output: Warm full-decoder/head times, logits diagnostics, peak memory, and frozen-base checks.
Note: Cache copies, prefill, and sampling/controller work remain outside the decode timer.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from mlx_hardware_roofline import _select_path, supervise


def run(args: argparse.Namespace) -> None:
    import mlx.core as mx

    from studio_compute.numerics import bf16
    from studio_compute.runtime import RuntimeEngine

    sequence = _select_path(args.batch)
    tokens = list(sequence.prompt_token_ids) + list(sequence.sampled_tokens)
    report = {
        "status": "running",
        "cases": [],
        "started_at": time.time(),
        "warmups": 2,
        "repetitions": 5,
        "scope": "One full decoder and vocabulary head; identical token and cloned incoming cache.",
        "excluded": "Model preparation, prefill, cache copies, sampling, and controller overhead.",
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    original = bf16._decode_gdn_token
    engine = None

    def chunk_token(layer, hidden, carries):
        x = mx.fast.rms_norm(
            hidden,
            layer.input_layernorm.weight.astype(mx.bfloat16),
            layer.input_layernorm.eps,
        )
        mixed, conv, state = bf16.gdn_forward(
            layer.linear_attn, x, carries, decode_one=False
        )
        middle = hidden + mixed
        normalized = mx.fast.rms_norm(
            middle,
            layer.post_attention_layernorm.weight.astype(mx.bfloat16),
            layer.post_attention_layernorm.eps,
        )
        return middle + layer.mlp(normalized), conv, state

    try:
        engine = RuntimeEngine(
            args.model, execution_profile="bf16-fast", prefix_cache_gib=0
        )
        engine.create("decode-benchmark", {"seed": 23})
        engine.resident.select(engine.clients["decode-benchmark"]["adapter"].state())
        engine.resident.mode(False)
        before = engine.resident.frozen_digest()
        report["runtime_identity"] = engine.identity
        for batch, length in ((1, 8192), (4, 8192), (1, 57344)):
            cache = engine.lm.make_cache()
            ids = mx.array([tokens[:length]] * batch, mx.int32)
            prefill_started = time.perf_counter()
            for start in range(0, length, 2048):
                stop = min(length, start + 2048)
                positions = mx.broadcast_to(
                    mx.arange(start, stop, dtype=mx.int32)[None, None],
                    (3, batch, stop - start),
                )
                hidden = engine._decode_hidden(
                    ids[:, start:stop], position_ids=positions, cache=cache
                )
                mx.eval(hidden, [x.state for x in cache])
            case = {
                "batch": batch,
                "context_tokens": length,
                "prefill_seconds": time.perf_counter() - prefill_started,
                "phases": {},
            }
            report["cases"].append(case)
            save()
            outputs = {}
            next_ids = mx.array([[tokens[length]]] * batch, mx.int32)
            positions = mx.full((3, batch, 1), length, mx.int32)
            for label, function in (
                ("padded_chunk", chunk_token),
                ("token_kernel", original),
            ):
                bf16._decode_gdn_token = function
                times = []
                copies = []
                mx.reset_peak_memory()
                for iteration in range(7):
                    started = time.perf_counter()
                    working = engine.prefixes._copy(cache)
                    copies.append(time.perf_counter() - started)
                    started = time.perf_counter()
                    result = engine._decode_logits(
                        next_ids, position_ids=positions, cache=working
                    )
                    mx.eval(result.logits, [x.state for x in working])
                    elapsed = time.perf_counter() - started
                    if iteration >= 2:
                        times.append(elapsed)
                    if iteration == 6:
                        outputs[label] = result.logits
                    del result, working
                case["phases"][label] = {
                    "seconds": times,
                    "median_seconds": statistics.median(times),
                    "cache_copy_seconds_excluded": copies,
                    "peak_mlx_bytes": mx.get_peak_memory(),
                }
                save()
            reference = outputs["padded_chunk"].astype(mx.float32)
            candidate = outputs["token_kernel"].astype(mx.float32)
            case["logit_relative_l2"] = float(
                (
                    mx.linalg.norm(candidate - reference) / mx.linalg.norm(reference)
                ).item()
            )
            case["argmax_equal"] = bool(
                mx.array_equal(
                    mx.argmax(reference, axis=-1), mx.argmax(candidate, axis=-1)
                ).item()
            )
            case["speedup"] = (
                case["phases"]["padded_chunk"]["median_seconds"]
                / case["phases"]["token_kernel"]["median_seconds"]
            )
            save()
            del cache, ids, hidden, outputs, reference, candidate
            mx.clear_cache()
        assert engine.clients["decode-benchmark"]["optimizer"].steps == 0
        report["frozen_base_unchanged"] = before == engine.resident.frozen_digest()
        assert report["frozen_base_unchanged"]
        report.update(status="passed", finished_at=time.time())
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        bf16._decode_gdn_token = original
        if engine is not None:
            engine.close()
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Approved existing 27B model assets."
    )
    parser.add_argument(
        "--batch", type=Path, required=True, help="Preserved junior12 batch."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh private evidence directory."
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=600, help="Hard limit, at most one hour."
    )
    parser.add_argument(
        "--lease",
        default=str(Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"),
        help="Existing exclusive Studio GPU lease.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    assert 0 < args.timeout_seconds <= 3600
    if args.worker:
        run(args)
    else:
        supervise(args, script=Path(__file__))
