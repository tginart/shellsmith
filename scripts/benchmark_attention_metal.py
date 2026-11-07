"""
Purpose: Measure fused D256 attention against the pinned fallback at actual shapes.
Input: BF16 GQA inputs and bounded long contexts.
Output: Separate forward, cached backward, full VJP, and peak-memory measurements.
Note: Full VJP includes primal reconstruction for both implementations.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from mlx_hardware_roofline import supervise


def run(args: argparse.Namespace) -> None:
    from functools import partial
    import mlx.core as mx
    from studio_compute.numerics.attention_metal import (
        attention_backward,
        attention_forward,
        fused_attention,
        _kernel,
        _parameters,
        _blocks,
    )

    mx.set_cache_limit(2 << 30)
    mx.set_memory_limit(32 << 30)
    mx.random.seed(23)
    report = {
        "status": "running",
        "cases": [],
        "device": mx.device_info(),
        "warmups": 2,
        "measurements": 5,
    }
    for keys in (57344, 262144):
        q = mx.random.normal((1, 24, 2048, 256)).astype(mx.bfloat16)
        k = mx.random.normal((1, 4, keys, 256)).astype(mx.bfloat16)
        v = mx.random.normal(k.shape).astype(mx.bfloat16)
        dy = mx.random.normal(q.shape).astype(mx.bfloat16)
        mx.eval(q, k, v, dy)
        reference = partial(
            mx.fast.scaled_dot_product_attention, scale=0.0625, mask="causal"
        )
        fused = partial(fused_attention, scale=0.0625, query_offset=keys - 2048)
        case = {"queries": 2048, "keys": keys, "phases": {}}
        report["cases"].append(case)

        def aligned_forward_control(q=q, k=k, v=v, keys=keys):
            return _kernel("forward")(
                inputs=[q, k, v, _parameters("forward", q.shape, k.shape, keys - 2048)],
                template=[("ALIGN_Q", True), ("ALIGN_K", True)],
                grid=(_blocks(2048) * 256, 24, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[q.shape, (*q.shape[:3], 2)],
                output_dtypes=[mx.bfloat16, mx.bfloat16],
            )

        for label, function in (
            ("reference_forward", partial(reference, q, k, v)),
            ("reference_vjp", partial(mx.vjp, reference, [q, k, v], [dy])),
            ("fused_forward", partial(fused, q, k, v)),
            ("aligned_forward_control", aligned_forward_control),
            ("fused_vjp", partial(mx.vjp, fused, [q, k, v], [dy])),
        ):
            mx.clear_cache()
            mx.reset_peak_memory()
            times = []
            for iteration in range(7):
                started = time.perf_counter()
                result = function()
                mx.eval(result)
                elapsed = time.perf_counter() - started
                if iteration >= 2:
                    times.append(elapsed)
                del result
            case["phases"][label] = {
                "seconds": times,
                "median_seconds": statistics.median(times),
                "peak_active_bytes": mx.get_peak_memory(),
            }
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
        del function
        times = []
        builds = []
        mx.reset_peak_memory()
        for iteration in range(7):
            started = time.perf_counter()
            primal, gradients = mx.vjp(reference, [q, k, v], [dy])
            builds.append(time.perf_counter() - started)
            mx.eval(primal)
            started = time.perf_counter()
            mx.eval(gradients)
            elapsed = time.perf_counter() - started
            if iteration >= 2:
                times.append(elapsed)
            del primal, gradients
        case["phases"]["reference_cached_backward"] = {
            "seconds": times,
            "median_seconds": statistics.median(times),
            "graph_build_seconds": builds,
            "peak_active_bytes": mx.get_peak_memory(),
            "scope": "Evaluate native primal first; time only its derivative graph.",
        }
        output, tape = attention_forward(q, k, v, 0.0625, keys - 2048)
        mx.eval(output, tape.statistics_BHQC)
        times = []
        mx.reset_peak_memory()
        for iteration in range(7):
            started = time.perf_counter()
            result = attention_backward(tape, dy)
            mx.eval(result)
            elapsed = time.perf_counter() - started
            if iteration >= 2:
                times.append(elapsed)
            del result
        case["phases"]["fused_cached_backward"] = {
            "seconds": times,
            "median_seconds": statistics.median(times),
            "peak_active_bytes": mx.get_peak_memory(),
        }
        case["full_vjp_speedup"] = (
            case["phases"]["reference_vjp"]["median_seconds"]
            / case["phases"]["fused_vjp"]["median_seconds"]
        )
        case["cached_backward_speedup"] = (
            case["phases"]["reference_cached_backward"]["median_seconds"]
            / case["phases"]["fused_cached_backward"]["median_seconds"]
        )
        del aligned_forward_control
        del tape, output, q, k, v, dy
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "passed"
    report["twofold_vjp_target_met"] = all(
        case["full_vjp_speedup"] >= 2 for case in report["cases"]
    )
    report["twofold_cached_backward_target_met"] = all(
        case["cached_backward_speedup"] >= 2 for case in report["cases"]
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh private output."
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=600, help="Hard limit at most one hour."
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
