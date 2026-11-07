"""
Purpose: Qualify native attention derivatives against an independent FP64 oracle.
Input: A pinned runtime, exact BF16 inputs, private output, and the GPU lease.
Output: Per-case mathematical errors, native failures, and timing distributions.
Note: Diagnostic completion and numerical qualification have separate statuses.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
from typing import Any

from mlx_hardware_roofline import _measure, _save, supervise

# B: batch, H: query heads, J: KV heads, G: query heads per KV head.
# Q: query rows, K: key rows, D: head width.
BF16_UNIT_ROUNDOFF = 2**-8


def qualify(args: argparse.Namespace) -> None:
    """
    Purpose: Compare real native forward/VJP outputs with exact shared input facts.
    Input: An expected runtime and optional prior reference. Output: Qualification data.
    """
    assert os.environ.get("MLX_ENABLE_TF32") == "0"
    import mlx.core as mx
    import numpy as np

    assert importlib.metadata.version("mlx") == args.expected_version
    assert mx.default_device() == mx.gpu
    if args.package_root is not None:
        assert Path(mx.__file__).resolve().is_relative_to(args.package_root.resolve())
    mx.set_cache_limit(2 << 30)
    mx.set_memory_limit(32 << 30)
    output = Path(args.output)
    reference = _read(args.reference / "qualification.json") if args.reference else None
    if reference is not None:
        assert reference["status"] == "completed" and reference["qualification_passed"]
    report = {
        "status": "running",
        "qualification_passed": False,
        "started_at": time.time(),
        "runtime_version": args.expected_version,
        "runtime_module": mx.__file__,
        "short_query_fallback": args.short_query_fallback,
        "device": mx.device_info(),
        "numpy_version": np.__version__,
        "oracle": "Explicit NumPy FP64 GQA, causal suffix, and manual derivative.",
        "environment": {
            "MLX_ENABLE_TF32": "0",
            "MLX_SDPA_VJP_FALLBACK": os.environ.get("MLX_SDPA_VJP_FALLBACK", "default"),
        },
        "criteria": {
            "maximum_relative_l2": 0.02,
            "minimum_cosine": 0.9999,
            "candidate_vs_oracle_limit": "max(1.25 * pinned_error, 2 * BF16 unit roundoff)",
            "bf16_unit_roundoff": BF16_UNIT_ROUNDOFF,
            "note": "The roundoff allowance is an empirical gate, not a worst-case proof or a full-model qualification.",
        },
        "reference_report_sha256": _digest(args.reference / "qualification.json")
        if args.reference
        else None,
        "cases": [],
    }
    cases = [(1, 17), (5, 17), (8, 17), (9, 17), (16, 64), (33, 129)]
    for index, (queries, keys) in enumerate(cases):
        name = f"q{queries}-k{keys}"
        row = {
            "name": name,
            "queries": queries,
            "keys": keys,
            "heads": 24,
            "kv_heads": 4,
            "head_width": 256,
        }
        if reference is None:
            rng = np.random.default_rng(20261018 + index)
            inputs = [
                mx.array(rng.normal(size=shape).astype(np.float32)).astype(mx.bfloat16)
                for shape in (
                    (1, 24, queries, 256),
                    (1, 4, keys, 256),
                    (1, 4, keys, 256),
                    (1, 24, queries, 256),
                )
            ]
            mx.eval(inputs)
            plain = [np.asarray(value.astype(mx.float32)) for value in inputs]
            oracle = numpy_attention(*plain, scale=0.0625)
            arrays = dict(zip(("q", "k", "v", "dy"), plain, strict=True))
            arrays.update(
                dict(
                    zip(
                        ("oracle_output", "oracle_dq", "oracle_dk", "oracle_dv"),
                        oracle,
                        strict=True,
                    )
                )
            )
            np.savez(output / f"{name}.npz", **arrays)
            row["input_file_sha256"] = _digest(output / f"{name}.npz")
        else:
            prior = next(r for r in reference["cases"] if r["name"] == name)
            path = args.reference / f"{name}.npz"
            assert _digest(path) == prior["input_file_sha256"]
            with np.load(path, allow_pickle=False) as arrays:
                inputs = [
                    mx.array(arrays[key]).astype(mx.bfloat16)
                    for key in ("q", "k", "v", "dy")
                ]
                oracle = [
                    arrays[key]
                    for key in ("oracle_output", "oracle_dq", "oracle_dk", "oracle_dv")
                ]
            mx.eval(inputs)
            row["input_file_sha256"] = _digest(path)
        q_BHQD, k_BJKD, v_BJKD, dy_BHQD = inputs

        def attention(q_BHQD: Any, k_BJKD: Any, v_BJKD: Any) -> Any:
            if args.short_query_fallback and q_BHQD.shape[2] <= 8:
                return legacy_attention(mx, q_BHQD, k_BJKD, v_BJKD, scale=0.0625)
            return mx.fast.scaled_dot_product_attention(
                q_BHQD, k_BJKD, v_BJKD, scale=0.0625, mask="causal"
            )

        try:
            started = time.perf_counter()
            (actual_BHQD,), gradients = mx.vjp(attention, inputs[:3], [dy_BHQD])
            mx.eval(actual_BHQD, gradients)
            row["first_forward_vjp_seconds"] = time.perf_counter() - started
            actual = [
                np.asarray(value.astype(mx.float32))
                for value in [actual_BHQD, *gradients]
            ]
            errors = {
                key: compare(a, b)
                for key, a, b in zip(
                    ("output", "dq", "dk", "dv"), actual, oracle, strict=True
                )
            }
            row["oracle_errors"] = errors
            row["passed"] = all(
                e["relative_l2"] <= 0.02 and e["cosine"] >= 0.9999
                for e in errors.values()
            )
            if reference is not None:
                native_path = args.reference / f"{name}-native.npz"
                assert _digest(native_path) == prior["native_file_sha256"]
                with np.load(native_path, allow_pickle=False) as arrays:
                    pinned = [arrays[key] for key in ("output", "dq", "dk", "dv")]
                row["pinned_comparison"] = {
                    key: compare(a, b)
                    for key, a, b in zip(
                        ("output", "dq", "dk", "dv"), actual, pinned, strict=True
                    )
                }
                limits = {
                    key: max(
                        1.25 * prior["oracle_errors"][key]["relative_l2"],
                        2 * BF16_UNIT_ROUNDOFF,
                    )
                    for key in errors
                }
                row["oracle_relative_l2_limits"] = limits
                row["passed"] = row["passed"] and all(
                    errors[key]["relative_l2"] <= limits[key] for key in errors
                )
            else:
                np.savez(
                    output / f"{name}-native.npz",
                    **dict(zip(("output", "dq", "dk", "dv"), actual, strict=True)),
                )
            if reference is None:
                row["native_file_sha256"] = _digest(output / f"{name}-native.npz")
            del actual_BHQD, gradients

            def derivative(inputs: list = inputs) -> Any:
                return mx.vjp(attention, inputs[:3], [inputs[3]])[1]

            row["vjp_timing"] = _measure(mx, derivative, 1, 3)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            row.update(
                passed=False,
                native_error_type=type(error).__name__,
                native_error=str(error),
            )
        report["cases"].append(row)
        _save(output / "qualification.json", report)
        print(json.dumps(row), flush=True)
        del inputs, q_BHQD, k_BJKD, v_BJKD, dy_BHQD
        mx.clear_cache()
    report.update(
        status="completed",
        qualification_passed=all(r["passed"] for r in report["cases"]),
        finished_at=time.time(),
    )
    _save(output / "qualification.json", report)


def legacy_attention(
    mx: Any, q_BHQD: Any, k_BJKD: Any, v_BJKD: Any, *, scale: float
) -> Any:
    """
    Purpose: Preserve the pinned BF16 fallback for unsupported short native VJPs.
    Input: Complete causal suffix queries and GQA keys/values. Output: BF16 attention.
    Note: This function uses ordinary differentiable operations and retains every key.
    """
    batch, heads, queries, width = q_BHQD.shape
    kv_heads, keys = k_BJKD.shape[1:3]
    assert heads % kv_heads == 0 and queries <= keys
    assert q_BHQD.dtype == k_BJKD.dtype == v_BJKD.dtype
    assert k_BJKD.shape == v_BJKD.shape
    assert k_BJKD.shape[0] == batch and k_BJKD.shape[-1] == width
    group = heads // kv_heads
    q_BJGQD = (q_BHQD * mx.array(scale, dtype=q_BHQD.dtype)).reshape(
        batch, kv_heads, group, queries, width
    )
    k_BJ1KD = mx.expand_dims(k_BJKD, 2)
    v_BJ1KD = mx.expand_dims(v_BJKD, 2)
    scores_BJGQK = q_BJGQD @ mx.swapaxes(k_BJ1KD, -1, -2)
    mask_QK = mx.arange(keys - queries, keys)[:, None] >= mx.arange(keys)[None, :]
    scores_BJGQK = mx.where(mask_QK, scores_BJGQK, mx.finfo(scores_BJGQK.dtype).min)
    probabilities_BJGQK = mx.softmax(scores_BJGQK, axis=-1, precise=True)
    output_BJGQD = probabilities_BJGQK @ v_BJ1KD
    return output_BJGQD.reshape(batch, heads, queries, width)


def numpy_attention(
    q_BHQD: Any, k_BJKD: Any, v_BJKD: Any, dy_BHQD: Any, *, scale: float, query_offset: int | None = None
) -> list[Any]:
    """
    Purpose: Calculate independent causal GQA outputs and all three derivatives.
    Input: Stored BF16 values as NumPy arrays. Output: FP64 mathematical references.
    Method: Apply suffix masks per query head, then sum shared KV-head gradients.
    """
    import numpy as np

    q_BHQD, k_BJKD, v_BJKD, dy_BHQD = [
        np.asarray(x, dtype=np.float64) for x in (q_BHQD, k_BJKD, v_BJKD, dy_BHQD)
    ]
    batch, heads, queries, width = q_BHQD.shape
    kv_heads, keys = k_BJKD.shape[1:3]
    assert heads % kv_heads == 0 and queries <= keys
    assert v_BJKD.shape == k_BJKD.shape and dy_BHQD.shape == q_BHQD.shape
    assert k_BJKD.shape[0] == batch and k_BJKD.shape[-1] == width
    output_BHQD = np.empty_like(q_BHQD)
    dq_BHQD, dk_BJKD, dv_BJKD = [np.zeros_like(x) for x in (q_BHQD, k_BJKD, v_BJKD)]
    query_offset = keys - queries if query_offset is None else query_offset
    assert 0 <= query_offset <= keys - queries
    mask_QK = np.arange(keys)[None, :] <= (query_offset + np.arange(queries))[:, None]
    group = heads // kv_heads
    for b in range(batch):
        for h in range(heads):
            j = h // group
            scaled_QD = q_BHQD[b, h] * scale
            scores_QK = scaled_QD @ k_BJKD[b, j].T
            scores_QK = np.where(mask_QK, scores_QK, -np.inf)
            scores_QK -= np.max(scores_QK, axis=-1, keepdims=True)
            probabilities_QK = np.exp(scores_QK)
            probabilities_QK /= probabilities_QK.sum(axis=-1, keepdims=True)
            output_BHQD[b, h] = probabilities_QK @ v_BJKD[b, j]
            dp_QK = dy_BHQD[b, h] @ v_BJKD[b, j].T
            ds_QK = probabilities_QK * (
                dp_QK - (dp_QK * probabilities_QK).sum(axis=-1, keepdims=True)
            )
            dq_BHQD[b, h] = (ds_QK @ k_BJKD[b, j]) * scale
            dk_BJKD[b, j] += ds_QK.T @ scaled_QD
            dv_BJKD[b, j] += probabilities_QK.T @ dy_BHQD[b, h]
    return [output_BHQD, dq_BHQD, dk_BJKD, dv_BJKD]


def compare(actual: Any, expected: Any) -> dict[str, Any]:
    """Purpose: Calculate norm-scaled numerical errors without BF16 arithmetic."""
    import numpy as np

    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise FloatingPointError("Nonfinite attention output or derivative")
    difference = actual - expected
    norm = float(np.linalg.norm(expected.ravel()))
    actual_norm = float(np.linalg.norm(actual.ravel()))
    return {
        "raw_float64_bits_equal": bool(
            np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
        ),
        "relative_l2": float(np.linalg.norm(difference.ravel())) / max(norm, 1e-30),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "cosine": float(np.vdot(actual.ravel(), expected.ravel()))
        / max(norm * actual_norm, 1e-30),
    }


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh private evidence folder."
    )
    parser.add_argument(
        "--expected-version", required=True, help="Exact MLX version to qualify."
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        help="Required native module prefix for an isolated runtime.",
    )
    parser.add_argument(
        "--reference", type=Path, help="Pinned reference data and report directory."
    )
    parser.add_argument(
        "--lease",
        default=str(Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"),
        help="Existing exclusive GPU lease.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="Hard job limit, at most one hour.",
    )
    parser.add_argument(
        "--short-query-fallback",
        action="store_true",
        help="Use pinned differentiable math for at most eight query rows.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    assert 0 < args.timeout_seconds <= 3600
    if args.worker:
        qualify(args)
    else:
        supervise(args, script=Path(__file__))
