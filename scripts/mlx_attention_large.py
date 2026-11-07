"""
Purpose: Measure real attention shapes with matched BF16 data across MLX runtimes.
Input: A pinned reference, private evidence folder, runtime identity, and GPU lease.
Output: Complete-context timing, memory, and output/gradient comparisons.
Note: FP32 anchors verify long-key mathematics. Full model qualification remains separate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import mlx_attention_qualify as oracle_helper
from mlx_hardware_roofline import _measure, _save, supervise

# B: batch, H: query heads, J: KV heads, Q: query rows, K: key rows, D: head width.


def qualify(args: argparse.Namespace) -> None:
    """
    Purpose: Test long causal suffixes and the full 262,144-key context.
    Input: Exact pinned BF16 tensors or a fresh reference directory.
    Output: Timings and independent numerical gates, saved after each case.
    """
    assert os.environ.get("MLX_ENABLE_TF32") == "0"
    import mlx.core as mx

    assert importlib.metadata.version("mlx") == args.expected_version
    assert not args.whole_query or args.expected_version == "0.32.4"
    if args.package_root is not None:
        assert Path(mx.__file__).resolve().is_relative_to(args.package_root.resolve())
    assert mx.default_device() == mx.gpu
    mx.set_cache_limit(2 << 30)
    mx.set_memory_limit(192 << 30)
    root = Path(args.output)
    assert shutil.disk_usage(root).free >= 12 << 30
    helper_path = Path(oracle_helper.__file__)
    (root / "oracle-helper-source.py").write_bytes(helper_path.read_bytes())
    reference = (
        _read(args.reference / "large-attention.json") if args.reference else None
    )
    if reference is not None:
        assert reference["status"] == "completed" and reference["qualification_passed"]
    report = {
        "status": "running",
        "qualification_passed": False,
        "started_at": time.time(),
        "runtime": args.expected_version,
        "runtime_module": mx.__file__,
        "helper_source_sha256": _digest(helper_path),
        "device": mx.device_info(),
        "reference_report_sha256": _digest(args.reference / "large-attention.json")
        if args.reference
        else None,
        "environment": {
            "MLX_ENABLE_TF32": "0",
            **{
                key: os.environ.get(key, "default")
                for key in (
                    "MLX_SDPA_VJP_FALLBACK",
                    "MLX_SDPA_VJP_BLOCK",
                    "MLX_SDPA_VJP_BQ",
                    "MLX_SDPA_VJP_BK",
                )
            },
        },
        "tile_budget_bytes": None if args.whole_query else 2 << 30,
        "whole_query": args.whole_query,
        "short_query_fallback": True,
        "memory_scope": "One operation and its inputs; reference gradients load only after timed operations.",
        "criteria": {
            "maximum_relative_l2": 0.02,
            "minimum_cosine": 0.9999,
            "oracle_limit": "max(1.25 * pinned FP32-reference error, 2 * BF16 unit roundoff)",
            "component_only": True,
        },
        "cases": [],
    }
    for index, (queries, keys) in enumerate(
        ((17, 57344), (17, 262144), (2048, 57344), (2048, 262144))
    ):
        name = f"q{queries}-k{keys}"
        report["current_case"] = name
        _save(root / "large-attention.json", report)
        print(json.dumps({"case": name, "status": "started"}), flush=True)
        row = _case(mx, args, reference, index=index, queries=queries, keys=keys)
        report["cases"].append(row)
        report.pop("current_case")
        _save(root / "large-attention.json", report)
        print(json.dumps(row), flush=True)
        gc.collect()
        mx.clear_cache()
    report.update(
        status="completed",
        qualification_passed=all(r["passed"] for r in report["cases"]),
        finished_at=time.time(),
    )
    _save(root / "large-attention.json", report)


def _case(
    mx: Any,
    args: argparse.Namespace,
    reference: dict | None,
    *,
    index: int,
    queries: int,
    keys: int,
) -> dict:
    """
    Purpose: Time one exact shape before reference arrays can inflate its peak.
    Input: Case dimensions and reference identity. Output: Numerical and timing data.
    """

    root = Path(args.output)
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
        mx.random.seed(20261018 + index)
        q_BHQD = mx.random.normal((1, 24, queries, 256)).astype(mx.bfloat16)
        k_BJKD = mx.random.normal((1, 4, keys, 256)).astype(mx.bfloat16)
        v_BJKD = mx.random.normal(k_BJKD.shape).astype(mx.bfloat16)
        dy_BHQD = mx.random.normal(q_BHQD.shape).astype(mx.bfloat16)
        inputs = [q_BHQD, k_BJKD, v_BJKD, dy_BHQD]
        mx.eval(inputs)
        path = root / f"{name}-inputs.safetensors"
        mx.save_safetensors(
            str(path), dict(zip(("q", "k", "v", "dy"), inputs, strict=True))
        )
        row["input_sha256"] = _digest(path)
    else:
        prior = next(r for r in reference["cases"] if r["name"] == name)
        path = args.reference / f"{name}-inputs.safetensors"
        assert _digest(path) == prior["input_sha256"]
        data = mx.load(str(path))
        inputs = [data[k] for k in ("q", "k", "v", "dy")]
        mx.eval(inputs)
        row["input_sha256"] = _digest(path)
    assert all(v.dtype == mx.bfloat16 for v in inputs)
    assert inputs[0].shape == (1, 24, queries, 256) and inputs[1].shape == (
        1,
        4,
        keys,
        256,
    )

    def attention(q_BHQD: Any, k_BJKD: Any, v_BJKD: Any) -> Any:
        if args.whole_query:
            if q_BHQD.shape[2] <= 8:
                return oracle_helper.legacy_attention(
                    mx, q_BHQD, k_BJKD, v_BJKD, scale=0.0625
                )
            return mx.fast.scaled_dot_product_attention(
                q_BHQD, k_BJKD, v_BJKD, scale=0.0625, mask="causal"
            )
        return tiled_attention(mx, q_BHQD, k_BJKD, v_BJKD)

    def derivative() -> list:
        return mx.vjp(attention, inputs[:3], [inputs[3]])[1]

    row["vjp_timing"] = _measure(mx, derivative, 1, 3)
    (out_BHQD,), gradients = mx.vjp(attention, inputs[:3], [inputs[3]])
    mx.eval(out_BHQD, gradients)
    values = dict(
        zip(("output", "dq", "dk", "dv"), [out_BHQD, *gradients], strict=True)
    )
    assert all(bool(mx.isfinite(v).all().item()) for v in values.values())
    row["passed"] = True
    if reference is None:
        path = root / f"{name}-native.safetensors"
        mx.save_safetensors(str(path), values)
        row["native_sha256"] = _digest(path)
        if queries <= 32:
            fp32_inputs = [x.astype(mx.float32) for x in inputs]

            def plain(q_BHQD: Any, k_BJKD: Any, v_BJKD: Any) -> Any:
                return oracle_helper.legacy_attention(
                    mx, q_BHQD, k_BJKD, v_BJKD, scale=0.0625
                )

            (fp32_out_BHQD,), fp32_gradients = mx.vjp(
                plain, fp32_inputs[:3], [fp32_inputs[3]]
            )
            mx.eval(fp32_out_BHQD, fp32_gradients)
            expected = dict(zip(values, [fp32_out_BHQD, *fp32_gradients], strict=True))
            path = root / f"{name}-fp32.safetensors"
            mx.save_safetensors(str(path), expected)
            row["fp32_sha256"] = _digest(path)
            row["fp32_errors"] = _compare_maps(mx, values, expected)
            row["passed"] = _errors_pass(row["fp32_errors"])
    else:
        path = args.reference / f"{name}-native.safetensors"
        assert _digest(path) == prior["native_sha256"]
        expected = mx.load(str(path))
        mx.eval(expected)
        row["pinned_errors"] = _compare_maps(mx, values, expected)
        row["passed"] = _errors_pass(row["pinned_errors"])
        if queries <= 32:
            path = args.reference / f"{name}-fp32.safetensors"
            assert _digest(path) == prior["fp32_sha256"]
            expected = mx.load(str(path))
            mx.eval(expected)
            row["fp32_errors"] = _compare_maps(mx, values, expected)
            limits = {
                k: max(
                    1.25 * prior["fp32_errors"][k]["relative_l2"],
                    2 * oracle_helper.BF16_UNIT_ROUNDOFF,
                )
                for k in values
            }
            row["fp32_relative_l2_limits"] = limits
            row["passed"] = (
                row["passed"]
                and _errors_pass(row["fp32_errors"])
                and all(
                    row["fp32_errors"][k]["relative_l2"] <= limits[k] for k in values
                )
            )
    return row


def tiled_attention(mx: Any, q_BHQD: Any, k_BJKD: Any, v_BJKD: Any) -> Any:
    """
    Purpose: Preserve production query partitions with a safe short-query fallback.
    Input: BF16 causal suffix queries and every KV row. Output: Complete attention.
    """
    batch, heads, queries, _ = q_BHQD.shape
    keys = k_BJKD.shape[2]
    tile = max(16, min(queries, (2 << 30) // (batch * heads * keys * 4)))
    outputs_BHQD = []
    for start in range(0, queries, tile):
        stop = min(queries, start + tile)
        prefix = keys - queries + stop
        q_tile_BHQD = q_BHQD[:, :, start:stop]
        k_prefix_BJKD = k_BJKD[:, :, :prefix]
        v_prefix_BJKD = v_BJKD[:, :, :prefix]
        if stop - start <= 8:
            out_BHQD = oracle_helper.legacy_attention(
                mx, q_tile_BHQD, k_prefix_BJKD, v_prefix_BJKD, scale=0.0625
            )
        else:
            out_BHQD = mx.fast.scaled_dot_product_attention(
                q_tile_BHQD, k_prefix_BJKD, v_prefix_BJKD, scale=0.0625, mask="causal"
            )
        outputs_BHQD.append(out_BHQD)
    return (
        mx.concatenate(outputs_BHQD, axis=2)
        if len(outputs_BHQD) > 1
        else outputs_BHQD[0]
    )


def _compare_maps(mx: Any, actual: dict, expected: dict) -> dict:
    import numpy as np

    assert actual.keys() == expected.keys()
    return {
        k: oracle_helper.compare(
            np.asarray(actual[k].astype(mx.float32)),
            np.asarray(expected[k].astype(mx.float32)),
        )
        for k in actual
    }


def _errors_pass(errors: dict) -> bool:
    return all(
        v["relative_l2"] <= 0.02 and v["cosine"] >= 0.9999 for v in errors.values()
    )


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
        "--reference", type=Path, help="Pinned large-case reference directory."
    )
    parser.add_argument(
        "--lease",
        default=str(Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"),
        help="Existing exclusive GPU lease.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=900,
        help="Hard job limit, at most one hour.",
    )
    parser.add_argument(
        "--whole-query",
        action="store_true",
        help="Use the bounded native backward on the complete replay query chunk.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    assert 0 < args.timeout_seconds <= 3600
    if args.worker:
        qualify(args)
    else:
        supervise(args, script=Path(__file__))
