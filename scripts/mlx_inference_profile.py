"""
Purpose: Measure real sampled inference, prefix reuse, and a weight-stream reference.
Input: A preserved token path, qualified model, private output, and GPU lease.
Output: True first-token times, request traces, memory, and matched cache evidence.
Note: The disposable engine never publishes policies or writes model checkpoints.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import struct
import time
from pathlib import Path
from typing import Any

from mlx_hardware_roofline import _measure, _save, _select_path, supervise

# B: batch, T: token rows, K: projection input width.


def profile(args: argparse.Namespace) -> None:
    """
    Purpose: Observe materialized tokens and compare exact and extended prefix hits.
    Input: Read-only artifacts and qualified precision. Output: Incremental reports.
    """
    assert os.environ.get("MLX_ENABLE_TF32") == "0"
    import mlx.core as mx

    from studio_compute.runtime import RuntimeEngine

    assert mx.default_device() == mx.gpu
    output = Path(args.output)
    sequence = _select_path(Path(args.batch))
    tokens = list(sequence.prompt_token_ids) + list(sequence.sampled_tokens)
    started = time.monotonic()
    engine = RuntimeEngine(args.model, execution_profile="legacy")
    engine.create("profile", {"seed": 23})
    factors = engine.clients["profile"]["adapter"].state()
    engine.policies["profile"] = {"factors": factors, "revision": 0}
    initialized = time.monotonic()
    before = engine.resident.frozen_digest()
    observations: dict[str, list[float]] = {}

    def observe(request_id: str, count: int, *, finished: bool = False) -> None:
        assert count == len(observations[request_id]) + 1
        observations[request_id].append(time.monotonic())

    engine.set_inference_observer(observe)
    report = {
        "status": "running",
        "started_at": time.time(),
        "initialization_seconds": initialized - started,
        "initial_frozen_digest_seconds": time.monotonic() - initialized,
        "device": mx.device_info(),
        "identity": engine.identity,
        "capabilities": engine.capabilities,
        "environment": {"MLX_ENABLE_TF32": "0"},
        "frozen_base_before": before,
        "cases": [],
        "projection_stream": [],
        "scope": "In-process production engine; actual categorical samples, empty stop set, 64 outputs per request.",
        "ttft_scope": "Owner-thread observer at the first materialized token, before the next decoder advance.",
        "memory_scope": "Per-case MLX peak and process-lifetime OS physical-footprint high-water mark.",
    }
    try:
        for length, batch in ((2048, 1), (8192, 1), (32768, 1), (8192, 4)):
            for trial in range(3):
                engine.prefixes.clear()
                gc.collect()
                mx.clear_cache()
                _request_case(
                    engine,
                    observations,
                    report,
                    output,
                    name=f"sample-{length}-{batch}-{trial}",
                    prompt=tokens[:length],
                    batch=batch,
                    warmup=trial == 0,
                    use_cache=False,
                )
        engine.prefixes.clear()
        cold = _request_case(
            engine,
            observations,
            report,
            output,
            name="prefix-cold-10240",
            prompt=tokens[:10240],
            batch=1,
            warmup=False,
            use_cache=False,
        )
        build = _request_case(
            engine,
            observations,
            report,
            output,
            name="prefix-build-8192",
            prompt=tokens[:8192],
            batch=1,
            warmup=False,
            use_cache=True,
        )
        exact = _request_case(
            engine,
            observations,
            report,
            output,
            name="prefix-exact-8192",
            prompt=tokens[:8192],
            batch=1,
            warmup=False,
            use_cache=True,
        )
        extended = _request_case(
            engine,
            observations,
            report,
            output,
            name="prefix-extend-10240",
            prompt=tokens[:10240],
            batch=1,
            warmup=False,
            use_cache=True,
        )
        assert exact["prefix_cache_tokens"] == [8192]
        assert extended["prefix_cache_tokens"] == [8192]
        report["cache_comparisons"] = [
            _compare_requests(build, exact),
            _compare_requests(cold, extended),
        ]
        _save(output / "inference-profile.json", report)
        engine.prefixes.clear()
        gc.collect()
        mx.clear_cache()
        _projection_stream(engine, report, output)
        report["frozen_base_after"] = engine.resident.frozen_digest()
        assert report["frozen_base_after"] == before
        report.update(
            status="passed", finished_at=time.time(), process_memory=_footprint()
        )
        _save(output / "inference-profile.json", report)
    except BaseException as error:
        report.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            finished_at=time.time(),
        )
        _save(output / "inference-profile.json", report)
        raise
    finally:
        engine.close()


def _request_case(
    engine: Any,
    observations: dict,
    report: dict,
    output: Path,
    *,
    name: str,
    prompt: list[int],
    batch: int,
    warmup: bool,
    use_cache: bool,
) -> dict:
    """
    Purpose: Time one complete request group with true first-token observations.
    Input: An engine, exact prompt, and cache condition. Output: Raw timing evidence.
    """
    bodies = [
        {
            "policy": "profile",
            "revision": 0,
            "prompt_token_ids": prompt,
            "max_tokens": 64,
            "seed": 23 + index,
            "cache_precision": "bf16",
            "use_prefix_cache": use_cache,
            "stop_token_ids": [],
            "operation_id": f"{name}-{index}",
        }
        for index in range(batch)
    ]
    for body in bodies:
        observations[body["operation_id"]] = []
    engine.mx.reset_peak_memory()
    started = time.monotonic()
    work = (
        engine._sample(bodies[0], started)
        if batch == 1
        else engine.sample_batch(bodies)
    )
    first_yield = None
    while True:
        try:
            event = next(work)
            if event["phase"] in ("decode", "batched_decode") and first_yield is None:
                first_yield = time.monotonic() - started
        except StopIteration as stopped:
            results = [stopped.value] if batch == 1 else stopped.value
            break
    elapsed = time.monotonic() - started
    timelines = [observations.pop(body["operation_id"]) for body in bodies]
    assert all(
        len(timeline) == len(result["sampled_tokens"]) == 64
        for timeline, result in zip(timelines, results, strict=True)
    )
    last_tokens = max(timeline[-1] for timeline in timelines)
    first_tokens = max(timeline[0] for timeline in timelines)
    row = {
        "name": name,
        "prompt_tokens": len(prompt),
        "batch": batch,
        "warmup": warmup,
        "use_prefix_cache": use_cache,
        "elapsed_seconds": elapsed,
        "ttft_seconds_each": [timeline[0] - started for timeline in timelines],
        "first_decode_generator_yield_seconds": first_yield,
        "token_observation_seconds_each": [
            [t - started for t in timeline] for timeline in timelines
        ],
        "steady_decode_aggregate_tokens_s": batch * 63 / (last_tokens - first_tokens),
        "prefill_seconds": results[0]["prefill_seconds"],
        "decode_seconds": results[0]["decode_seconds"],
        "sampled_tokens": [result["sampled_tokens"] for result in results],
        "behavior_logprobs": [result["behavior_logprobs"] for result in results],
        "prefix_cache_tokens": [
            result.get("prefix_cache_tokens", 0) for result in results
        ],
        "peak_active_memory_bytes": engine.mx.get_peak_memory(),
        "process_memory": _footprint(),
    }
    assert math.isfinite(row["steady_decode_aggregate_tokens_s"])
    report["cases"].append(row)
    _save(output / "inference-profile.json", report)
    print(
        json.dumps(
            {
                k: v
                for k, v in row.items()
                if k
                not in (
                    "sampled_tokens",
                    "behavior_logprobs",
                    "token_observation_seconds_each",
                )
            }
        ),
        flush=True,
    )
    return row


def _projection_stream(engine: Any, report: dict, output: Path) -> None:
    """
    Purpose: Measure all frozen projection reads with one synchronization per pass.
    Input: The actual resident frozen weights. Output: An empirical quantized reference.
    Note: Independent operations omit layer dependencies, adapters, norms, and caches.
    """
    mx = engine.mx
    assert not engine.lm.args.tie_word_embeddings
    bases = [module.base for module in engine.resident.adapter.modules.values()]
    bases.append(engine.lm.lm_head)
    weight_bytes = sum(
        value.nbytes for base in bases for value in base.parameters().values()
    )
    assert len(bases) == 497
    for batch in (1, 4):
        inputs_BTK = []
        for base in bases:
            assert base.bits == 8 and base.group_size == 64 and base.mode == "affine"
            width_in = base.weight.shape[1] * 4
            inputs_BTK.append(
                mx.random.normal((batch, 1, width_in)).astype(mx.bfloat16)
            )
        mx.eval(inputs_BTK)

        def project_all(bases: list = bases, inputs_BTK: list = inputs_BTK) -> list:
            return [base(x_BTK) for base, x_BTK in zip(bases, inputs_BTK, strict=True)]

        timing = _measure(mx, project_all, 2, 5)
        row = {
            "batch": batch,
            "frozen_projection_calls": len(bases),
            "stored_weight_bytes": weight_bytes,
            "estimated_logical_weight_gb_s": weight_bytes
            / timing["median_seconds"]
            / 1e9,
            "scope": "Whole-weight-set independent native projections; useful logical bytes, not DRAM counters.",
            **timing,
        }
        report["projection_stream"].append(row)
        _save(output / "inference-profile.json", report)
        print(json.dumps(row), flush=True)
        del project_all, inputs_BTK
        gc.collect()
        mx.clear_cache()


def _compare_requests(reference: dict, actual: dict) -> dict:
    """Purpose: Verify prefix reuse preserves sampled tokens and behavior logprobs."""
    assert actual["sampled_tokens"] == reference["sampled_tokens"]
    error = max(
        abs(a - b)
        for x, y in zip(
            reference["behavior_logprobs"], actual["behavior_logprobs"], strict=True
        )
        for a, b in zip(x, y, strict=True)
    )
    assert error <= 1e-6
    return {
        "reference": reference["name"],
        "actual": actual["name"],
        "sampled_tokens_equal": True,
        "maximum_logprob_error": error,
        "ttft_speedup": reference["ttft_seconds_each"][0]
        / actual["ttft_seconds_each"][0],
    }


def _footprint() -> dict[str, int]:
    """
    Purpose: Read this process's current and lifetime peak physical footprint.
    Input: The running macOS process. Output: Native RUSAGE_INFO_V4 byte counters.
    """
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    library.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    library.proc_pid_rusage.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(1024)
    if library.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buffer)) != 0:
        raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
    fields = struct.unpack_from("35Q", buffer.raw, 16)
    return {
        "physical_footprint_bytes": fields[7],
        "lifetime_max_physical_footprint_bytes": fields[28],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh private evidence folder."
    )
    parser.add_argument(
        "--lease",
        default=str(Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"),
        help="Existing exclusive GPU lease.",
    )
    parser.add_argument(
        "--model",
        default=str(Path.home() / "models/Qwen3.8-27B-8bit"),
        help="Read-only qualified model folder.",
    )
    parser.add_argument(
        "--batch",
        required=True,
        help="Read-only preserved junior12 batch (failed-batch-reconstructed.json.gz).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=600,
        help="Hard job limit, at most one hour.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    assert 0 < args.timeout_seconds <= 3600
    if args.worker:
        profile(args)
    else:
        supervise(args, script=Path(__file__))
