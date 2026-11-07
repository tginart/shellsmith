"""
Purpose: Compare the BF16 reference recurrence with the Metal GDN pipeline.
Input: Actual Qwen recurrence dimensions and a fixed seed.
Output: Matched phase timings, dtype inventory, memory peaks, and derivative errors.
Note: This is a shape-matched component control. It does not load model weights.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mlx_hardware_roofline import supervise


def run(args: argparse.Namespace) -> None:
    import mlx.core as mx
    from studio_compute.numerics.ops import gated_delta_chunked

    mx.random.seed(23)
    mx.set_cache_limit(2 << 30)
    mx.set_memory_limit(64 << 30)
    shapes = [(1, args.tokens, 16, 128)] * 2 + [
        (1, args.tokens, 48, 128), (1, args.tokens, 48),
        (1, args.tokens, 48), (1, 48, 128, 128),
    ]
    inputs = [
        (mx.random.normal(shapes[0]) / 128).astype(mx.bfloat16),
        (mx.random.normal(shapes[1]) / 128 ** 0.5).astype(mx.bfloat16),
        mx.random.normal(shapes[2]).astype(mx.bfloat16),
        mx.random.uniform(0.9, 1.0, shapes[3]).astype(mx.bfloat16),
        mx.random.uniform(0.1, 0.9, shapes[4]).astype(mx.bfloat16),
        (mx.random.normal(shapes[5]) * 0.02).astype(mx.bfloat16),
    ]
    cotangents = [
        mx.random.normal(shapes[2]).astype(mx.bfloat16),
        mx.random.normal(shapes[5]).astype(mx.bfloat16),
    ]
    mx.eval(inputs, cotangents)
    report = {
        "status": "running", "scope": "shape-matched BF16 GDN component",
        "shapes": shapes, "tokens": args.tokens, "device": mx.device_info(),
        "warmups": 2, "repetitions": 7, "variants": {},
    }
    comparisons = []
    for backend in ("reference", "metal_solve", "metal_scan", "metal"):
        def forward(*values):
            return gated_delta_chunked(
                *values, compute_dtype=mx.bfloat16, backend=backend
            )

        outputs, gradients = mx.vjp(forward, inputs, cotangents)
        mx.eval(outputs, gradients)
        comparisons.append((outputs, gradients))
        measurements = {}
        mx.reset_peak_memory()
        for phase in ("forward", "vjp"):
            times = []
            for iteration in range(9):
                started = time.perf_counter()
                if phase == "forward":
                    values = forward(*inputs)
                else:
                    values = mx.vjp(forward, inputs, cotangents)
                mx.eval(values)
                seconds = time.perf_counter() - started
                if iteration >= 2:
                    times.append(seconds)
                del values
            measurements[phase] = times
        measurements["peak_mlx_bytes"] = mx.get_peak_memory()
        report["variants"][backend] = measurements
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for kind, index in (("outputs", 0), ("gradients", 1)):
        rows = []
        for actual, reference in zip(comparisons[-1][index], comparisons[0][index], strict=True):
            actual = actual.astype(mx.float32).reshape(-1)
            reference = reference.astype(mx.float32).reshape(-1)
            actual_norm = float(mx.linalg.norm(actual).item())
            reference_norm = float(mx.linalg.norm(reference).item())
            relative = float(mx.linalg.norm(actual - reference).item()) / max(reference_norm, 1e-12)
            cosine = float(mx.sum(actual * reference).item()) / max(actual_norm * reference_norm, 1e-12)
            rows.append({"relative_l2": relative, "cosine": cosine})
        report[kind] = rows
    report["status"] = "passed"
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh private output.")
    parser.add_argument("--tokens", type=int, default=2048, help="Actual segment length.")
    parser.add_argument("--lease", default=str(Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"),
                        help="Existing Studio exclusive GPU lease.")
    parser.add_argument("--timeout-seconds", type=int, default=600, help="Hard limit, at most one hour.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    assert 0 < args.timeout_seconds <= 3600
    if args.worker:
        run(args)
    else:
        supervise(args, script=Path(__file__))
