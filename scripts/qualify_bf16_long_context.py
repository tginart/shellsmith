"""
Purpose: Qualify complete-context BF16 attention and recurrent carry derivatives.
Input: Fixed BF16 inputs at production widths and maximum attention context.
Output: Independent FP64 error measurements and carried-state comparisons.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mlx_hardware_roofline import supervise


def run(args: argparse.Namespace) -> None:
    import mlx.core as mx
    import numpy as np
    from mlx_attention_qualify import numpy_attention
    from studio_compute.numerics.attention_metal import (
        attention_forward,
        attention_backward,
    )
    from studio_compute.numerics.ops import gated_delta_chunked

    mx.set_cache_limit(2 << 30)
    mx.set_memory_limit(32 << 30)
    report = {
        "status": "running",
        "attention": [],
        "gdn": [],
        "started_at": time.time(),
    }
    save = lambda: (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    plain = lambda x: np.asarray(x.astype(mx.float32), dtype=np.float64)
    relative = lambda x, y: float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-12))
    mx.random.seed(91)
    for queries in (1, 8):
        keys = 262144
        q = mx.random.normal((1, 24, queries, 256)).astype(mx.bfloat16)
        k = mx.random.normal((1, 4, keys, 256)).astype(mx.bfloat16)
        v = mx.random.normal(k.shape).astype(mx.bfloat16)
        dy = mx.random.normal(q.shape).astype(mx.bfloat16)
        started = time.perf_counter()
        out, tape = attention_forward(q, k, v, 0.0625, keys - queries)
        gradients = attention_backward(tape, dy)
        mx.eval(out, gradients)
        reference = numpy_attention(*[plain(x) for x in (q, k, v, dy)], scale=0.0625)
        errors = [
            relative(plain(x), y)
            for x, y in zip([out, *gradients], reference, strict=True)
        ]
        report["attention"].append(
            {
                "queries": queries,
                "keys": keys,
                "relative_l2": errors,
                "seconds": time.perf_counter() - started,
                "passed": max(errors) < 0.025,
            }
        )
        save()
        assert max(errors) < 0.025, errors
        del q, k, v, dy, out, tape, gradients, reference
        mx.clear_cache()
    length = 4097
    inputs = [
        (mx.random.normal((1, length, 1, 128)) * 0.02).astype(mx.bfloat16),
        (mx.random.normal((1, length, 1, 128)) * 0.005).astype(mx.bfloat16),
        (mx.random.normal((1, length, 2, 128)) * 0.1).astype(mx.bfloat16),
        mx.ones((1, length, 2), mx.bfloat16),
        mx.full((1, length, 2), 0.25, mx.bfloat16),
        (mx.random.normal((1, 2, 128, 128)) * 0.02).astype(mx.bfloat16),
    ]
    cotangents = [
        mx.random.normal(inputs[2].shape).astype(mx.bfloat16),
        mx.random.normal(inputs[-1].shape).astype(mx.bfloat16),
    ]
    function = lambda *x: gated_delta_chunked(
        *x, compute_dtype=mx.bfloat16, backend="metal"
    )
    started = time.perf_counter()
    actual = mx.vjp(function, inputs, cotangents)
    mx.eval(actual)
    expected = sequential([plain(x) for x in inputs], [plain(x) for x in cotangents])
    errors = [
        [relative(plain(x), y) for x, y in zip(a, e, strict=True)]
        for a, e in zip(actual, expected, strict=True)
    ]

    def carried(*values):
        prefix, state = function(*[x[:, :2048] for x in values[:-1]], values[-1])
        suffix, state = function(*[x[:, 2048:] for x in values[:-1]], state)
        return mx.concatenate([prefix, suffix], axis=1), state

    composed = mx.vjp(carried, inputs, cotangents)
    mx.eval(composed)
    carry_errors = [
        [relative(plain(x), plain(y)) for x, y in zip(a, e, strict=True)]
        for a, e in zip(composed, actual, strict=True)
    ]
    report["gdn"].append(
        {
            "tokens": length,
            "key_width": 128,
            "value_width": 128,
            "relative_l2": errors,
            "split_at": 2048,
            "carried_relative_l2": carry_errors,
            "seconds": time.perf_counter() - started,
        }
    )
    save()
    assert max(v for row in errors for v in row) < 0.1, errors
    assert max(v for row in carry_errors for v in row) < 0.1, carry_errors
    report.update(
        status="passed", finished_at=time.time(), peak_mlx_bytes=mx.get_peak_memory()
    )
    save()


def sequential(inputs: list, cotangents: list) -> tuple:
    """
    Purpose: Differentiate the token recurrence without chunk solves or autodiff.
    Input: FP64 tensors and cotangents. Output: Forward values and manual derivatives.
    """
    import numpy as np

    q, k, v, g, beta, initial = inputs
    groups = v.shape[2] // q.shape[2]
    q, k = [np.repeat(x, groups, axis=2) for x in (q, k)]
    state = initial.copy()
    output, tape = [], []
    for t in range(v.shape[1]):
        before = state
        decayed = g[:, t, :, None, None] * before
        difference = v[:, t] - np.einsum("bhvk,bhk->bhv", decayed, k[:, t])
        delta = beta[:, t, :, None] * difference
        state = decayed + delta[..., None] * k[:, t, :, None, :]
        output.append(np.einsum("bhvk,bhk->bhv", state, q[:, t]))
        tape.append((before, decayed, difference, delta, state))
    dy, dstate = cotangents
    dq, dk, dv, dg, db = [np.zeros_like(x) for x in (q, k, v, g, beta)]
    for t in reversed(range(v.shape[1])):
        before, decayed, difference, delta, current = tape[t]
        dq[:, t] = np.einsum("bhvk,bhv->bhk", current, dy[:, t])
        dstate = dstate + dy[:, t, :, :, None] * q[:, t, :, None, :]
        ddelta = np.einsum("bhvk,bhk->bhv", dstate, k[:, t])
        dk[:, t] = np.einsum("bhvk,bhv->bhk", dstate, delta)
        db[:, t] = np.sum(ddelta * difference, axis=-1)
        dv[:, t] = beta[:, t, :, None] * ddelta
        dk[:, t] -= np.einsum("bhvk,bhv->bhk", decayed, dv[:, t])
        ddecayed = dstate - dv[:, t, :, :, None] * k[:, t, :, None, :]
        dg[:, t] = np.sum(ddecayed * before, axis=(-1, -2))
        dstate = g[:, t, :, None, None] * ddecayed
    shape = (*inputs[0].shape[:3], groups, inputs[0].shape[-1])
    return [np.stack(output, axis=1), state], [
        dq.reshape(shape).sum(axis=3),
        dk.reshape(shape).sum(axis=3),
        dv,
        dg,
        db,
        dstate,
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh private report directory."
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=900, help="Hard limit, at most one hour."
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
