"""
Purpose: Measure empirical MLX compute and memory bandwidth limits on one GPU.
Input: An output folder, a hard time limit, and the existing GPU lease.
Output: Raw timings, useful operation counts, device facts, and a process receipt.
Note: FLOP counts use two operations per multiply-add. Bytes count useful traffic.
"""

from __future__ import annotations

import argparse
import fcntl
from functools import partial
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

# M: token rows, K: input width, N: output width, E: vector elements.


def supervise(args: argparse.Namespace, *, script: Path | None = None) -> None:
    """
    Purpose: Hold the GPU lease and bound one owned worker.
    Input: Validated CLI arguments. Output: An execution receipt and worker log.
    Note: The supervisor terminates only the process group that it creates.
    """
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve() if script is None else script.resolve()
    receipt = {
        "status": "running",
        "started_at": time.time(),
        "timeout_seconds": args.timeout_seconds,
        "supervisor_pid": os.getpid(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "supervisor_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "swap_before": _command(["sysctl", "vm.swapusage"]),
        "thermal_before": _command(["pmset", "-g", "therm"]),
    }
    (output / "source.py").write_bytes(source.read_bytes())
    (output / "supervisor-source.py").write_bytes(Path(__file__).read_bytes())
    with Path(args.lease).open("r") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command = [sys.executable, "-u", str(source), *sys.argv[1:], "--worker"]
        with (output / "worker.log").open("w") as log:
            child = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            receipt["worker_pid"] = child.pid
            _save(output / "execution.json", receipt)
            try:
                child.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                receipt["timed_out"] = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
            receipt.update(
                status="passed" if child.returncode == 0 else "failed",
                returncode=child.returncode,
                finished_at=time.time(),
                swap_after=_command(["sysctl", "vm.swapusage"]),
                thermal_after=_command(["pmset", "-g", "therm"]),
            )
            _save(output / "execution.json", receipt)
            if child.returncode != 0:
                raise SystemExit(124 if receipt.get("timed_out") else max(1, child.returncode))


def calibrate(args: argparse.Namespace) -> None:
    """
    Purpose: Measure streaming traffic, dense arithmetic, and quantized projections.
    Input: Qualified MLX runtime and actual production projection dimensions.
    Output: Warm per-operation timing distributions and memory peaks.
    Method:
        1. Materialize all inputs before each timed case.
        2. Warm each case, then measure separate evaluated operations.
        3. Save each result before the next case starts.
    """
    import mlx.core as mx

    output = Path(args.output)
    mx.set_cache_limit(2 << 30)
    mx.set_memory_limit(64 << 30)
    mx.random.seed(20261018)
    report: dict[str, Any] = {
        "status": "running",
        "device": mx.device_info(),
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("mlx", "mlx-lm", "mlx-vlm")
        },
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "host": _command(["hostname"]),
        "os": _command(["sw_vers"]),
        "gpu": _command(["system_profiler", "SPDisplaysDataType"]),
        "byte_scope": "Useful loads and stores; no hardware DRAM counters.",
        "flop_scope": "Dense-equivalent GEMM operations; quantization excluded.",
        "timing_scope": "Host graph construction plus synchronized evaluation.",
        "environment": {
            "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32", "1 (MLX default)")
        },
        "cases": [],
    }

    def record(name: str, function: Callable, **counts: Any) -> None:
        row = _measure(mx, function, args.warmups, args.repetitions)
        row.update(name=name, **counts)
        if "useful_bytes" in row:
            row["useful_gb_s"] = row["useful_bytes"] / row["median_seconds"] / 1e9
        if "flops" in row:
            row["dense_equivalent_tflop_s"] = (
                row["flops"] / row["median_seconds"] / 1e12
            )
        report["cases"].append(row)
        _save(output / "calibration.json", report)
        print(json.dumps(row), flush=True)

    copy = mx.fast.metal_kernel(
        name="roofline_copy",
        input_names=["x"],
        output_names=["out"],
        source="uint i = thread_position_in_grid.x; out[i] = x[i];",
    )
    triad = mx.fast.metal_kernel(
        name="roofline_triad",
        input_names=["x", "y"],
        output_names=["out"],
        source="uint i = thread_position_in_grid.x; out[i] = x[i] + 0.5f * y[i];",
    )
    sizes_bytes = (1 << 20,) if args.smoke else (256 << 20, 1 << 30, 4 << 30)
    for size_bytes in sizes_bytes:
        count = size_bytes // 4
        x_E = mx.random.uniform(shape=(count,), dtype=mx.float32)
        y_E = mx.random.uniform(shape=(count,), dtype=mx.float32)
        mx.eval(x_E, y_E)
        for threads in (128, 256, 512):

            def run_copy(x_E=x_E, count=count, threads=threads):
                return copy(
                    inputs=[x_E],
                    grid=(count, 1, 1),
                    threadgroup=(threads, 1, 1),
                    output_shapes=[x_E.shape],
                    output_dtypes=[mx.float32],
                )[0]

            def run_triad(x_E=x_E, y_E=y_E, count=count, threads=threads):
                return triad(
                    inputs=[x_E, y_E],
                    grid=(count, 1, 1),
                    threadgroup=(threads, 1, 1),
                    output_shapes=[x_E.shape],
                    output_dtypes=[mx.float32],
                )[0]

            actual_E = run_copy()
            mx.eval(actual_E)
            assert bool(mx.array_equal(actual_E, x_E).item())
            del actual_E
            record(
                f"copy-{size_bytes}-{threads}",
                run_copy,
                useful_bytes=size_bytes * 2,
                threadgroup=threads,
            )
            actual_E = run_triad()
            mx.eval(actual_E)
            assert bool(mx.allclose(actual_E, x_E + 0.5 * y_E).item())
            del actual_E
            record(
                f"triad-{size_bytes}-{threads}",
                run_triad,
                useful_bytes=size_bytes * 3,
                flops=count * 2,
                threadgroup=threads,
            )
        del run_copy, run_triad
        del x_E, y_E
        gc.collect()
        mx.clear_cache()

    sizes = (128,) if args.smoke else (2048, 4096, 8192)
    for dtype in (mx.float32, mx.float16, mx.bfloat16):
        for size in sizes:
            x_MK = (mx.random.normal((size, size)) * 0.01).astype(dtype)
            w_KN = (mx.random.normal((size, size)) * 0.01).astype(dtype)
            mx.eval(x_MK, w_KN)
            record(
                f"gemm-{dtype}-{size}",
                lambda x_MK=x_MK, w_KN=w_KN: x_MK @ w_KN,
                flops=2 * size**3,
                useful_bytes=3 * x_MK.nbytes,
                shape=[size, size, size],
                dtype=str(dtype),
            )
            del x_MK, w_KN
            mx.clear_cache()

    projections = (
        ((128, 128),)
        if args.smoke
        else (
            (5120, 17408),
            (17408, 5120),
            (5120, 12288),
        )
    )
    for width_in, width_out in projections:
        w_NK = (mx.random.normal((width_out, width_in)) * 0.01).astype(mx.bfloat16)
        quantized_NK, scales_NG, biases_NG = mx.quantize(
            w_NK, group_size=64, bits=8, mode="affine"
        )
        mx.eval(quantized_NK, scales_NG, biases_NG)
        weight_bytes = sum(v.nbytes for v in (quantized_NK, scales_NG, biases_NG))
        rows = (1, 32) if args.smoke else (1, 4, 16, 128, 2048)
        for length in rows:
            x_MK = mx.random.normal((length, width_in)).astype(mx.bfloat16)
            mx.eval(x_MK)

            qmm = partial(
                mx.quantized_matmul,
                x_MK,
                quantized_NK,
                scales_NG,
                biases_NG,
                transpose=True,
                group_size=64,
                bits=8,
                mode="affine",
            )

            record(
                f"qmm8-{length}-{width_in}-{width_out}",
                qmm,
                flops=2 * length * width_in * width_out,
                useful_bytes=weight_bytes + x_MK.nbytes + length * width_out * 2,
                weight_bytes=weight_bytes,
                shape=[length, width_in, width_out],
            )
            del x_MK, qmm
        del w_NK, quantized_NK, scales_NG, biases_NG
        gc.collect()
        mx.clear_cache()

    report.update(status="passed", finished_at=time.time())
    _save(output / "calibration.json", report)


def inference(args: argparse.Namespace) -> None:
    """
    Purpose: Measure production inference on exact preserved junior12 token paths.
    Input: A read-only batch and model. Output: Matched inference timings and costs.
    Note: Each case starts with a fresh BF16 cache and fixed teacher-forced tokens.
    """
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    import mlx.core as mx

    from studio_compute.runtime import RuntimeEngine

    output = Path(args.output)
    if args.batch is None:
        raise SystemExit("--batch is required for inference modes.")
    sequence = _select_path(Path(args.batch))
    path_tokens = list(sequence.prompt_token_ids) + list(sequence.sampled_tokens)
    engine = RuntimeEngine(args.model, prefix_cache_gib=0, execution_profile="legacy")
    engine.create("roofline", {"seed": 23})
    factors = engine.clients["roofline"]["adapter"].state()
    engine.policies["roofline"] = {"factors": factors, "revision": 0}
    before = engine.resident.frozen_digest()
    inventory = []
    for name, module in engine.resident.adapter.modules.items():
        base = module.base
        weight_NK = base.weight
        quantized = hasattr(base, "bits")
        width_in = (
            weight_NK.shape[1] * 32 // base.bits if quantized else weight_NK.shape[1]
        )
        stored_bytes = weight_NK.nbytes
        if quantized:
            stored_bytes += base.scales.nbytes + base.biases.nbytes
        inventory.append(
            {
                "name": name,
                "width_in": width_in,
                "width_out": weight_NK.shape[0],
                "dense_parameters": width_in * weight_NK.shape[0],
                "stored_bytes": stored_bytes,
                "quantized": quantized,
                "lora_rank": module.a.shape[0],
            }
        )
    report = {
        "status": "running",
        "device": mx.device_info(),
        "source_path_sha256": hashlib.sha256(
            json.dumps(path_tokens).encode()
        ).hexdigest(),
        "frozen_base_before": before,
        "projection_inventory": inventory,
        "model_config": vars(engine.lm.args),
        "cases": [],
        "timing_scope": "Production fixed-token inference, cache hits disabled, warmed kernels.",
        "head_weight_bytes": sum(
            value.nbytes for value in engine.lm.lm_head.parameters().values()
        ),
        "note": "Projection bytes omit head, embedding lookup, norms, GDN recurrence and KV traffic.",
        "environment": {
            key: os.environ.get(key, "MLX default")
            for key in (
                "MLX_ENABLE_TF32",
                "MLX_MAX_OPS_PER_BUFFER",
                "MLX_MAX_MB_PER_BUFFER",
                "MLX_METAL_FAST_SYNCH",
            )
        },
    }
    cases = (
        [tuple(map(int, item.split(":"))) for item in args.case]
        if args.case
        else [(2048, 1), (8192, 1), (32768, 1), (8192, 4)]
    )
    for length, batch in cases:
        assert 0 < length <= len(path_tokens) - 64 and 0 < batch <= 4
        prompt = path_tokens[:length]
        target = path_tokens[length : length + 64]
        for trial in range(3):
            bodies = [
                {
                    "policy": "roofline",
                    "revision": 0,
                    "prompt_token_ids": prompt,
                    "target_token_ids": target,
                    "max_tokens": len(target),
                    "seed": 23 + index,
                    "cache_precision": "bf16",
                    "use_prefix_cache": False,
                }
                for index in range(batch)
            ]
            mx.reset_peak_memory()
            started = time.monotonic()
            work = (
                engine._sample(bodies[0], started)
                if batch == 1
                else engine.sample_batch(bodies)
            )
            events = []
            while True:
                try:
                    event = next(work)
                    if event["phase"] in ("decode", "batched_decode"):
                        events.append(time.monotonic())
                except StopIteration as stopped:
                    results = [stopped.value] if batch == 1 else stopped.value
                    break
            elapsed = time.monotonic() - started
            intervals = [b - a for a, b in zip(events, events[1:])]
            row = {
                "prompt_tokens": length,
                "batch": batch,
                "trial": trial,
                "warmup": trial == 0,
                "completion_tokens_each": len(target),
                "elapsed_seconds": elapsed,
                "prefill_seconds": results[0]["prefill_seconds"],
                "decode_seconds": results[0]["decode_seconds"],
                "decode_median_interval_seconds": statistics.median(intervals),
                "decode_intervals_seconds": intervals,
                "ttft_prefill_seconds": results[0]["prefill_seconds"],
                "first_generator_yield_seconds": events[0] - started,
                "prefill_tokens_per_second": batch
                * length
                / results[0]["prefill_seconds"],
                "decode_tokens_per_second": batch
                * (len(target) - 1)
                / results[0]["decode_seconds"],
                "peak_active_memory_bytes": mx.get_peak_memory(),
                "logprobs": [r["behavior_logprobs"] for r in results],
            }
            report["cases"].append(row)
            _save(output / "inference.json", report)
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in ("logprobs", "decode_intervals_seconds")
                    }
                ),
                flush=True,
            )
            del work, results
            gc.collect()
            mx.clear_cache()
    report["frozen_base_after"] = engine.resident.frozen_digest()
    assert report["frozen_base_after"] == before
    report.update(status="passed", finished_at=time.time())
    _save(output / "inference.json", report)
    engine.close()


def operators(args: argparse.Namespace) -> None:
    """
    Purpose: Test compile effects on the differentiable GDN recurrence.
    Input: Actual model dimensions with nonzero inputs and state.
    Output: Operation timings and errors against the existing recurrence.
    Note: This diagnostic does not select a production kernel.
    """
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    import mlx.core as mx

    from studio_compute.numerics.ops import gated_delta_chunked

    output = Path(args.output)
    mx.set_cache_limit(2 << 30)
    mx.set_memory_limit(64 << 30)
    mx.random.seed(20261018)
    # B: batch, T: tokens, H/J: value/key heads, D: head width.
    tokens, heads, key_heads, width = 2048, 48, 16, 128
    q_BTJD = (mx.random.normal((1, tokens, key_heads, width)) * width**-0.5).astype(
        mx.bfloat16
    )
    k_BTJD = (mx.random.normal(q_BTJD.shape) * width**-0.5).astype(mx.bfloat16)
    v_BTHD = mx.random.normal((1, tokens, heads, width)).astype(mx.bfloat16)
    g_BTH = mx.random.uniform(0.9, 1.0, (1, tokens, heads))
    beta_BTH = mx.random.uniform(0.0, 1.0, g_BTH.shape).astype(mx.bfloat16)
    state_BHDD = mx.random.normal((1, heads, width, width)) * 0.01
    dy_BTHD = mx.random.normal(v_BTHD.shape).astype(mx.bfloat16)
    ds_BHDD = mx.random.normal(state_BHDD.shape)
    inputs = [q_BTJD, k_BTJD, v_BTHD, g_BTH, beta_BTH, state_BHDD]
    cotangents = [dy_BTHD, ds_BHDD]
    mx.eval(inputs, cotangents)
    compiled = mx.compile(gated_delta_chunked)
    report = {
        "status": "running",
        "device": mx.device_info(),
        "environment": {"MLX_ENABLE_TF32": os.environ["MLX_ENABLE_TF32"]},
        "shape": {
            "tokens": tokens,
            "value_heads": heads,
            "key_heads": key_heads,
            "width": width,
        },
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "comparisons": {},
        "operations": {},
    }
    for name, function in (("existing", gated_delta_chunked), ("compiled", compiled)):
        for stage in ("forward", "vjp"):
            call = (
                partial(function, *inputs)
                if stage == "forward"
                else partial(mx.vjp, function, inputs, cotangents)
            )
            report["operations"][f"{name}-{stage}"] = _measure(
                mx, call, args.warmups, args.repetitions
            )
            _save(output / "operators.json", report)

    def differentiate(*tensors):
        return mx.vjp(gated_delta_chunked, tensors[:6], tensors[6:])

    compiled_vjp = mx.compile(differentiate)
    call = partial(compiled_vjp, *inputs, *cotangents)
    report["operations"]["compiled-entire-vjp"] = _measure(
        mx, call, args.warmups, args.repetitions
    )
    _save(output / "operators.json", report)

    references = list(gated_delta_chunked(*inputs)) + list(
        mx.vjp(gated_delta_chunked, inputs, cotangents)[1]
    )
    candidates = list(compiled(*inputs)) + list(mx.vjp(compiled, inputs, cotangents)[1])
    mx.eval(references, candidates)
    for name, actual, reference in zip(
        ("output", "state", "dq", "dk", "dv", "dg", "dbeta", "dstate"),
        candidates,
        references,
        strict=True,
    ):
        actual_F = actual.astype(mx.float32).flatten()
        reference_F = reference.astype(mx.float32).flatten()
        error = float(mx.sum((actual_F - reference_F) ** 2).item())
        norm = float(mx.sum(reference_F**2).item())
        report["comparisons"][name] = {
            "relative_l2": (error / max(norm, 1e-30)) ** 0.5,
            "maximum_absolute_error": float(
                mx.max(mx.abs(actual_F - reference_F)).item()
            ),
        }
    compiled_values, compiled_gradients = compiled_vjp(*inputs, *cotangents)
    mx.eval(compiled_values, compiled_gradients)
    report["entire_vjp_comparisons"] = []
    for actual, reference in zip(
        [*compiled_values, *compiled_gradients], references, strict=True
    ):
        actual_F = actual.astype(mx.float32).flatten()
        reference_F = reference.astype(mx.float32).flatten()
        error = float(mx.sum((actual_F - reference_F) ** 2).item())
        norm = float(mx.sum(reference_F**2).item())
        report["entire_vjp_comparisons"].append(
            {
                "relative_l2": (error / max(norm, 1e-30)) ** 0.5,
                "maximum_absolute_error": float(
                    mx.max(mx.abs(actual_F - reference_F)).item()
                ),
            }
        )
    report.update(status="passed", finished_at=time.time())
    _save(output / "operators.json", report)


def attention_controls(args: argparse.Namespace) -> None:
    """
    Purpose: Measure attention dispatch against shape-matched matrix operations.
    Input: Actual long-prefix attention dimensions and strict precision.
    Output: Timings, compilation effects, and numerical comparison results.
    Note: These controls do not change the production attention implementation.
    """
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    import mlx.core as mx

    from studio_compute.numerics.ops import ATTENTION_TILE_BYTES, tiled_causal_attention

    output = Path(args.output)
    mx.set_memory_limit(64 << 30)
    mx.set_cache_limit(2 << 30)
    mx.random.seed(20261018)
    # B: batch, H/J: query/key heads, Q/K: query/key tokens, D: head width.
    queries, keys, heads, key_heads, width = 2048, 57344, 24, 4, 256
    q_BHQD = mx.random.normal((1, heads, queries, width)).astype(mx.bfloat16)
    k_BJKD = mx.random.normal((1, key_heads, keys, width)).astype(mx.bfloat16)
    v_BJKD = mx.random.normal(k_BJKD.shape).astype(mx.bfloat16)
    dy_BHQD = mx.random.normal(q_BHQD.shape).astype(mx.bfloat16)
    inputs = [q_BHQD, k_BJKD, v_BJKD]
    mx.eval(inputs, dy_BHQD)

    def attend(q_BHQD, k_BJKD, v_BJKD):
        return tiled_causal_attention(
            q_BHQD, k_BJKD, v_BJKD, width**-0.5, keys - queries
        )

    def differentiate(q_BHQD, k_BJKD, v_BJKD, dy_BHQD):
        return mx.vjp(attend, [q_BHQD, k_BJKD, v_BJKD], [dy_BHQD])[1]

    compiled = mx.compile(attend)
    compiled_vjp = mx.compile(differentiate)
    report = {
        "status": "running",
        "device": mx.device_info(),
        "environment": {"MLX_ENABLE_TF32": os.environ["MLX_ENABLE_TF32"]},
        "shape": {
            "queries": queries,
            "keys": keys,
            "heads": heads,
            "key_heads": key_heads,
            "width": width,
        },
        "operations": {},
        "gradient_errors": [],
    }
    tile = max(16, min(queries, ATTENTION_TILE_BYTES // (heads * keys * 4)))
    q_BJRQD = q_BHQD[:, :, :tile].reshape(1, key_heads, heads // key_heads, tile, width)
    k_BJRDK = mx.swapaxes(k_BJKD[:, :, None], -1, -2)
    probabilities_BJRQK = mx.random.uniform(
        shape=(1, key_heads, heads // key_heads, tile, keys)
    ).astype(mx.bfloat16)
    v_BJRKD = v_BJKD[:, :, None]
    mx.eval(q_BJRQD, k_BJRDK, probabilities_BJRQK, v_BJRKD)
    functions = {
        "existing-forward": partial(attend, *inputs),
        "compiled-forward": partial(compiled, *inputs),
        "existing-vjp": partial(differentiate, *inputs, dy_BHQD),
        "compiled-entire-vjp": partial(compiled_vjp, *inputs, dy_BHQD),
        "tile-qk-gemm": lambda: q_BJRQD @ k_BJRDK,
        "tile-pv-gemm": lambda: probabilities_BJRQK @ v_BJRKD,
    }
    for name, function in functions.items():
        row = _measure(mx, function, args.warmups, args.repetitions)
        if name.startswith("tile-"):
            row["flops"] = 2 * heads * tile * keys * width
            row["tflop_s"] = row["flops"] / row["median_seconds"] / 1e12
        report["operations"][name] = row
        _save(output / "attention.json", report)
    reference = differentiate(*inputs, dy_BHQD)
    candidate = compiled_vjp(*inputs, dy_BHQD)
    mx.eval(reference, candidate)
    for actual, expected in zip(candidate, reference, strict=True):
        actual_F = actual.astype(mx.float32).flatten()
        expected_F = expected.astype(mx.float32).flatten()
        error = float(mx.sum((actual_F - expected_F) ** 2).item())
        norm = float(mx.sum(expected_F**2).item())
        report["gradient_errors"].append(
            {
                "relative_l2": (error / max(norm, 1e-30)) ** 0.5,
                "maximum_absolute_error": float(
                    mx.max(mx.abs(actual_F - expected_F)).item()
                ),
            }
        )
    report.update(status="passed", finished_at=time.time())
    _save(output / "attention.json", report)


def shader_controls(args: argparse.Namespace) -> None:
    """
    Purpose: Measure FP32 arithmetic with independent register FMA chains.
    Input: A Metal GPU and fixed loop sizes. Output: Warm useful FLOP rates.
    Note: CPU samples verify the result with one round per fused multiply-add.
    """
    import mlx.core as mx
    import numpy as np

    mx.random.seed(20261018)
    elements, steps = 262144, 16384
    x_E = mx.random.uniform(0.999999, 1.000001, (elements,))
    y_E = mx.random.uniform(-0.001, 0.001, (elements,))
    mx.eval(x_E, y_E)
    kernel = mx.fast.metal_kernel(
        name="roofline_fp32_fma",
        input_names=["x", "y"],
        output_names=["out"],
        source="""
            uint i = thread_position_in_grid.x;
            float factor = x[i];
            float addend = y[i];
            float accumulators[CHAINS];
            #pragma clang loop unroll(full)
            for (int j = 0; j < CHAINS; ++j) {
                accumulators[j] = factor + float(j) * 0.001f;
            }
            #pragma clang loop unroll(disable)
            for (int step = 0; step < STEPS; ++step) {
                #pragma clang loop unroll(full)
                for (int j = 0; j < CHAINS; ++j) {
                    accumulators[j] = metal::fma(accumulators[j], factor, addend);
                }
            }
            float total = 0.0f;
            #pragma clang loop unroll(full)
            for (int j = 0; j < CHAINS; ++j) {
                total += accumulators[j];
            }
            out[i] = total;
        """,
        compile_options={"math_mode": "safe"},
    )
    report = {
        "status": "running",
        "device": mx.device_info(),
        "elements": elements,
        "steps": steps,
        "cases": [],
        "flop_scope": "Two FP32 operations per FMA; setup and final sum excluded.",
        "scope": "Warm shader arithmetic reference, not a proof of physical maximum.",
    }
    for chains in (4, 8, 16, 32):
        reference_SC = np.asarray(x_E[:3])[:, None] + np.arange(
            chains, dtype=np.float32
        )[None] * np.float32(0.001)
        factor_S1 = np.asarray(x_E[:3], dtype=np.float64)[:, None]
        addend_S1 = np.asarray(y_E[:3], dtype=np.float64)[:, None]
        for _ in range(steps):
            reference_SC = (
                reference_SC.astype(np.float64) * factor_S1 + addend_S1
            ).astype(np.float32)
        expected_S = np.zeros(3, dtype=np.float32)
        for index in range(chains):
            expected_S = (expected_S + reference_SC[:, index]).astype(np.float32)
        for threads in (128, 256, 512):
            call = partial(
                kernel,
                inputs=[x_E, y_E],
                grid=(elements, 1, 1),
                threadgroup=(threads, 1, 1),
                template=[("CHAINS", chains), ("STEPS", steps)],
                output_shapes=[x_E.shape],
                output_dtypes=[mx.float32],
            )
            row = _measure(mx, call, args.warmups, args.repetitions)
            actual_E = call()[0]
            mx.eval(actual_E)
            assert bool(mx.isfinite(actual_E).all().item())
            maximum_error = float(np.max(np.abs(np.asarray(actual_E[:3]) - expected_S)))
            assert maximum_error <= 1e-5
            flops = 2 * elements * steps * chains
            row.update(
                chains=chains,
                threadgroup=threads,
                flops=flops,
                tflop_s=flops / row["median_seconds"] / 1e12,
                cpu_sample_maximum_absolute_error=maximum_error,
            )
            report["cases"].append(row)
            _save(Path(args.output) / "shader.json", report)
    report.update(status="passed", finished_at=time.time())
    _save(Path(args.output) / "shader.json", report)


def attention_checkpoint_controls(args: argparse.Namespace) -> None:
    """
    Purpose: Check redundant attention checkpoints through the full context limit.
    Input: Actual head dimensions and matched random tensors.
    Output: Forward and derivative times, memory peaks, and gradient byte hashes.
    Note: Each case releases diagnostic gradients before the next measurement.
    """
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    import mlx.core as mx
    import numpy as np

    from studio_compute.numerics.ops import ATTENTION_TILE_BYTES, tiled_causal_attention

    output = Path(args.output)
    mx.set_memory_limit(192 << 30)
    mx.set_cache_limit(2 << 30)
    mx.random.seed(20261018)
    report = {
        "status": "running",
        "device": mx.device_info(),
        "cases": [],
        "environment": {"MLX_ENABLE_TF32": os.environ["MLX_ENABLE_TF32"]},
        "memory_scope": "MLX active peak for one operation; no model or retained reference gradients.",
        "warmups": args.warmups,
        "repetitions": args.repetitions,
    }
    for queries, keys in ((2048, 57344), (2048, 262144)):
        q_BHQD = mx.random.normal((1, 24, queries, 256)).astype(mx.bfloat16)
        k_BJKD = mx.random.normal((1, 4, keys, 256)).astype(mx.bfloat16)
        v_BJKD = mx.random.normal(k_BJKD.shape).astype(mx.bfloat16)
        dy_BHQD = mx.random.normal(q_BHQD.shape).astype(mx.bfloat16)
        inputs = [q_BHQD, k_BJKD, v_BJKD]
        mx.eval(inputs, dy_BHQD)

        def existing(q_BHQD, k_BJKD, v_BJKD):
            return tiled_causal_attention(
                q_BHQD, k_BJKD, v_BJKD, 0.0625, keys - queries
            )

        def direct(q_BHQD, k_BJKD, v_BJKD):
            tile = max(16, min(queries, ATTENTION_TILE_BYTES // (24 * keys * 4)))
            outputs_BHQD = []
            for start in range(0, queries, tile):
                stop = min(queries, start + tile)
                prefix = keys - queries + stop
                outputs_BHQD.append(
                    mx.fast.scaled_dot_product_attention(
                        q_BHQD[:, :, start:stop],
                        k_BJKD[:, :, :prefix],
                        v_BJKD[:, :, :prefix],
                        scale=0.0625,
                        mask="causal",
                    )
                )
            return (
                mx.concatenate(outputs_BHQD, axis=2)
                if len(outputs_BHQD) > 1
                else outputs_BHQD[0]
            )

        row = {"queries": queries, "keys": keys, "operations": {}}
        for name, function in (("checkpointed", existing), ("direct", direct)):
            for stage in ("forward", "vjp"):

                def derivative(function=function, inputs=inputs, dy_BHQD=dy_BHQD):
                    return mx.vjp(function, inputs, [dy_BHQD])[1]

                call = partial(function, *inputs) if stage == "forward" else derivative
                gc.collect()
                mx.clear_cache()
                timing = _measure(mx, call, args.warmups, args.repetitions)
                row["operations"][f"{name}-{stage}"] = timing
                _save(
                    output / "attention-checkpoints.json",
                    report | {"current_case": row},
                )
                if stage == "vjp":
                    gradients = derivative()
                    mx.eval(gradients)
                    digest = hashlib.sha256()
                    for gradient in gradients:
                        digest.update(np.asarray(gradient.view(mx.uint16)).tobytes())
                    timing["gradient_raw_bytes_sha256"] = digest.hexdigest()
                    del gradients, gradient
        row["gradient_hashes_equal"] = (
            row["operations"]["checkpointed-vjp"]["gradient_raw_bytes_sha256"]
            == row["operations"]["direct-vjp"]["gradient_raw_bytes_sha256"]
        )
        assert row["gradient_hashes_equal"]
        report["cases"].append(row)
        _save(output / "attention-checkpoints.json", report)
        del derivative, call
        del inputs, q_BHQD, k_BJKD, v_BJKD, dy_BHQD
        gc.collect()
        mx.clear_cache()
    report.update(status="passed", finished_at=time.time())
    _save(output / "attention-checkpoints.json", report)


def _select_path(batch: Path) -> Any:
    """
    Purpose: Select the longest exact causal path from the preserved junior12 batch.
    Input: The audited compressed batch. Output: A packed token sequence.
    Note: This function verifies the source bytes and does not change credit.
    """
    import gzip

    from shellsmith.train.backend import TokenSequence
    from shellsmith.train.segments import pack_rollout_segments

    raw = batch.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "7621071b492b91be50e5f37bee6f3e618aefc9cf707f79853fd539fb70710d9d"
    )
    payload = json.loads(gzip.decompress(raw))
    original = {}
    for row in payload["body"]["sequences"]:
        original[row["request_id"]] = TokenSequence(
            **{
                key: tuple(value) if isinstance(value, list) else value
                for key, value in row.items()
            }
        )
    paths = []
    for capture in payload["captures"]:
        turns = [
            (turn["sequence_index"], original[turn["request_id"]])
            for turn in capture["turns"]
        ]
        paths.extend(pack_rollout_segments(turns).sequences)
    sequence = max(
        paths, key=lambda item: len(item.prompt_token_ids) + len(item.sampled_tokens)
    )
    assert len(sequence.prompt_token_ids) + len(sequence.sampled_tokens) - 1 == 57538
    assert sum(sequence.loss_mask) == 21675 and len(sequence.source_spans) == 29
    return sequence


def _measure(mx: Any, function: Callable, warmups: int, repeats: int) -> dict:
    """
    Purpose: Time fresh operation graphs after the warmup.
    Input: An MLX callable and repetition counts. Output: Wall and build samples.
    Note: Each output is evaluated and released before the next repetition.
    """
    for _ in range(warmups):
        result = function()
        mx.eval(result)
        del result
    mx.synchronize()
    mx.reset_peak_memory()
    samples, builds = [], []
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        built = time.perf_counter()
        mx.eval(result)
        stopped = time.perf_counter()
        samples.append(stopped - started)
        builds.append(built - started)
        del result
    return {
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "samples_seconds": samples,
        "median_build_seconds": statistics.median(builds),
        "build_samples_seconds": builds,
        "peak_active_memory_bytes": mx.get_peak_memory(),
    }


def _command(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def _save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Private evidence folder.")
    parser.add_argument(
        "--lease",
        default=str(Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"),
        help="Existing exclusive GPU lease file.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=600,
        help="Hard worker time limit, at most 3600 seconds.",
    )
    parser.add_argument(
        "--warmups", type=int, default=3, help="Warm repetitions per case."
    )
    parser.add_argument(
        "--repetitions", type=int, default=9, help="Measured repetitions per case."
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Use small tensors for harness validation."
    )
    parser.add_argument(
        "--mode",
        choices=(
            "calibration",
            "inference",
            "operators",
            "attention",
            "shader",
            "attention-checkpoints",
        ),
        default="calibration",
        help="Hardware controls or full model inference.",
    )
    parser.add_argument(
        "--model",
        default=str(Path.home() / "models/Qwen3.8-27B-8bit"),
        help="Read-only qualified model folder.",
    )
    parser.add_argument(
        "--batch",
        help="Read-only preserved junior12 batch (failed-batch-reconstructed.json.gz). "
        "Required for inference modes.",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Inference prompt length and batch, such as 8192:1. Repeat to select cases.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    assert 0 < args.timeout_seconds <= 3600
    assert args.warmups >= 1 and args.repetitions >= 3
    if args.worker:
        if args.mode == "inference":
            inference(args)
        elif args.mode == "operators":
            operators(args)
        elif args.mode == "attention":
            attention_controls(args)
        elif args.mode == "shader":
            shader_controls(args)
        elif args.mode == "attention-checkpoints":
            attention_checkpoint_controls(args)
        else:
            calibrate(args)
    else:
        supervise(args)


if __name__ == "__main__":
    main()
