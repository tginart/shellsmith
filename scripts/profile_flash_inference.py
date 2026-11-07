"""
Purpose: Compare frozen Flash inference, transient progress, and batched attention.
Input: Real rollout token records, a baseline source snapshot, and a candidate runtime.
Output: Atomic timings, profiles, fixed-path logprob comparisons, and native references.
Note: The benchmark owns the GPU lease and never opens a hero service state root.
"""

from __future__ import annotations

import argparse
import cProfile
import fcntl
import importlib.util
import json
import pstats
import statistics
import time
from collections import Counter
from pathlib import Path

from studio_compute.storage import Store, atomic_json

VARIANTS = ("baseline", "sqlite_only", "batched_attention", "sorted_experts")


def load_records(root: Path, *, count: int, tokens: int) -> list[dict]:
    """
    Purpose: Select distinct recorded conversations with full, unmodified prompt tokens.
    Input: A rollout tree, required conversation count, and continuation length.
    Output: Fixed-path bodies. Each body records its source file and line number.
    """
    selected = []
    for path in sorted(root.rglob("player_records/*.jsonl")):
        candidates = []
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            row = json.loads(line)
            prompt, target = (
                row.get("prompt_token_ids", []),
                row.get("sampled_tokens", []),
            )
            if len(prompt) >= 1024 and len(target) >= tokens:
                candidates.append((len(prompt), line_number, prompt, target))
        if not candidates:
            continue
        _, line_number, prompt, target = max(candidates)
        selected.append(
            {
                "prompt_token_ids": prompt,
                "target_token_ids": target[:tokens],
                "max_tokens": tokens,
                "seed": 17 + len(selected),
                "revision": 0,
                "policy": "benchmark",
                "cache_precision": "8bit",
                "use_prefix_cache": True,
                "source": str(path),
                "line": line_number,
            }
        )
        if len(selected) == count:
            return selected
    raise ValueError(
        f"Need {count} distinct recorded conversations with {tokens} target tokens."
    )


def load_baseline(source: Path):
    name = "studio_compute.numerics.flash_ops_baseline"
    spec = importlib.util.spec_from_file_location(
        name, source / "src/studio_compute/numerics/flash_ops.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_case(
    engine, bodies: list[dict], *, variant: str, folder: Path
) -> tuple[dict, list[list[float]]]:
    """
    Purpose: Time prompt admission and fixed-width decode with the real journal protocol.
    Input: Ready runtime, fixed-path bodies, and one progress-persistence variant.
    Output: Separate prefill, GPU evaluation, CPU, journal, and steady decode timings.
    Note: All requests have independent cache objects and identical continuation lengths.
    """
    from studio_compute import inference

    mx = engine.mx
    store = Store(folder)
    statements = Counter()
    store.db.set_trace_callback(
        lambda statement: statements.update([statement.split()[0]])
    )
    requests = []
    prefill_started = time.perf_counter()
    prefill_work_seconds = 0.0
    peak_memory_before = mx.get_peak_memory()
    try:
        for index, body in enumerate(bodies):
            name = str(index)
            store.submit(name, "benchmark", "logprobs", body)
            request = inference.create_request(engine, name, body)
            if request is None:
                raise MemoryError("The benchmark must admit every requested slot.")
            requests.append(request)
            store.running(name)
            while request.logits is None:
                tick = time.perf_counter()
                inference.prefill_request(engine, request)
                prefill_work_seconds += time.perf_counter() - tick
        mx.synchronize()
        prefill_seconds = time.perf_counter() - prefill_started
        steps = len(bodies[0]["target_token_ids"])
        assert all(len(body["target_token_ids"]) == steps for body in bodies)
        timers = {
            "advance_seconds": 0.0,
            "sampling_seconds": 0.0,
            "journal_seconds": 0.0,
            "eval_seconds": 0.0,
        }
        original_eval = mx.eval

        def evaluate(*args, **kwargs):
            tick = time.perf_counter()
            try:
                return original_eval(*args, **kwargs)
            finally:
                timers["eval_seconds"] += time.perf_counter() - tick

        mx.eval = evaluate
        step_seconds = []
        commits_before = statements["COMMIT"]
        decode_started = time.perf_counter()
        try:
            for step in range(steps):
                boundary = time.perf_counter()
                if step:
                    tick = time.perf_counter()
                    inference.advance_requests(engine, requests)
                    timers["advance_seconds"] += time.perf_counter() - tick
                tick = time.perf_counter()
                inference.sample_requests(engine, requests)
                timers["sampling_seconds"] += time.perf_counter() - tick
                tick = time.perf_counter()
                for request in requests:
                    if request.reason is None:
                        progress = {"phase": "decode", "tokens": len(request.tokens)}
                        if variant == "baseline":
                            store.running(request.request_id, progress)
                        else:
                            store.observe_progress(request.request_id, progress)
                timers["journal_seconds"] += time.perf_counter() - tick
                step_seconds.append(time.perf_counter() - boundary)
            mx.synchronize()
            decode_seconds = time.perf_counter() - decode_started
        finally:
            mx.eval = original_eval
        decode_commits = statements["COMMIT"] - commits_before
        probabilities = [list(request.logprobs) for request in requests]
        for request in requests:
            store.finish(request.request_id, inference.request_result(engine, request))
        assert all(
            request.tokens == body["target_token_ids"]
            for request, body in zip(requests, bodies, strict=True)
        )
        tokens = steps * len(requests)
        return {
            "variant": variant,
            "concurrency": len(requests),
            "tokens": tokens,
            "prompt_tokens": [len(body["prompt_token_ids"]) for body in bodies],
            "cached_tokens": [request.cached_tokens for request in requests],
            "prefill_seconds": prefill_seconds,
            "prefill_work_seconds": prefill_work_seconds,
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second": tokens / decode_seconds,
            "total_tokens_per_second": tokens / (decode_seconds + prefill_seconds),
            "median_step_ms": statistics.median(step_seconds) * 1000,
            "p95_step_ms": sorted(step_seconds)[int(0.95 * len(step_seconds))] * 1000,
            "decode_progress_commits": decode_commits,
            "terminal_receipts": len(requests),
            "timers": timers,
            "host_graph_and_journal_seconds": decode_seconds - timers["eval_seconds"],
            "peak_memory_before": peak_memory_before,
            "peak_memory_bytes": mx.get_peak_memory(),
        }, probabilities
    finally:
        for request in requests:
            inference.release_request(engine, request)
        store.close()
        assert not engine.reserved and not engine.policy_users


def verify_distributions(
    engine, bodies: list[dict], baseline_decode, candidate_decode
) -> dict:
    """
    Purpose: Compare full-vocabulary distributions with identical private cache histories.
    Input: Eight real ragged prompts and the original and candidate decode functions.
    Output: Fixed-target error, FP32 diagnostic KL, greedy agreement, and cache invariants.
    Note: Diagnostic normalization does not change production BF16 logits or sampling.
    """
    from types import SimpleNamespace

    from studio_compute import inference
    from studio_compute.mlx_request_cache import fork_cache

    mx = engine.mx
    requests, references = [], []
    target_errors, divergences, logit_errors = [], [], []
    greedy_matches = 0
    engine.prefixes.clear()
    try:
        for index, body in enumerate(bodies):
            print(
                json.dumps(
                    {
                        "stage": "distribution_prefill",
                        "row": index,
                        "prompt_tokens": len(body["prompt_token_ids"]),
                    }
                ),
                flush=True,
            )
            request = inference.create_request(engine, f"parity-{index}", body)
            if request is None:
                raise MemoryError("Parity must admit every independent request.")
            requests.append(request)
            while request.logits is None:
                inference.prefill_request(engine, request)
            references.append(
                SimpleNamespace(
                    cache=[fork_cache(cache) for cache in request.cache],
                    tokens=[],
                    body=body,
                    needs_decode=True,
                )
            )
        for step in range(8):
            for request, reference, body in zip(
                requests, references, bodies, strict=True
            ):
                token = body["target_token_ids"][step]
                request.tokens.append(token)
                reference.tokens.append(token)
            baseline_decode(engine, references)
            candidate_decode(engine, requests)
            for request, reference, body in zip(
                requests, references, bodies, strict=True
            ):
                actual_BV, expected_BV = request.logits, reference.logits
                actual_fp_BV = actual_BV.astype(mx.float32)
                expected_fp_BV = expected_BV.astype(mx.float32)
                actual_lp_BV = actual_fp_BV - mx.logsumexp(
                    actual_fp_BV, axis=-1, keepdims=True
                )
                expected_lp_BV = expected_fp_BV - mx.logsumexp(
                    expected_fp_BV, axis=-1, keepdims=True
                )
                kl_S = mx.sum(mx.exp(expected_lp_BV) * (expected_lp_BV - actual_lp_BV))
                target = body["target_token_ids"][step + 1]
                actual_S = (
                    actual_BV - mx.logsumexp(actual_BV, axis=-1, keepdims=True)
                )[0, target]
                expected_S = (
                    expected_BV - mx.logsumexp(expected_BV, axis=-1, keepdims=True)
                )[0, target]
                target_errors.append(
                    abs(float(actual_S.item()) - float(expected_S.item()))
                )
                divergences.append(max(0.0, float(kl_S.item())))
                logit_errors.append(
                    float(mx.max(mx.abs(actual_fp_BV - expected_fp_BV)).item())
                )
                greedy_matches += int(
                    mx.argmax(actual_BV).item() == mx.argmax(expected_BV).item()
                )
                for actual_cache, reference_cache in zip(
                    request.cache, reference.cache, strict=True
                ):
                    if hasattr(actual_cache, "offset"):
                        assert actual_cache.offset == reference_cache.offset
                    if hasattr(actual_cache, "bits"):
                        assert actual_cache.bits == 8 and actual_cache.group_size == 64
        result = {
            "rows": len(requests),
            "decode_steps": 8,
            "max_target_logprob_error_nats": max(target_errors),
            "max_distribution_kl_nats": max(divergences),
            "mean_distribution_kl_nats": statistics.mean(divergences),
            "max_logit_error": max(logit_errors),
            "greedy_agreement": greedy_matches / len(target_errors),
            "limits": {
                "target_logprob_error_nats": 0.125,
                "distribution_kl_nats": 0.001,
            },
        }
        if (
            result["max_target_logprob_error_nats"] > 0.125
            or result["max_distribution_kl_nats"] > 0.001
        ):
            raise AssertionError(
                "Full-model attention parity failed: " + json.dumps(result)
            )
        return result
    finally:
        for request in requests:
            inference.release_request(engine, request)
        for reference in references:
            reference.cache.clear()
        assert not engine.reserved and not engine.policy_users


def cold_prefill_probe(engine, body: dict) -> tuple[dict, object]:
    """
    Purpose: Measure an uncached recorded prompt separately from token generation.
    Input: One unmodified long prompt. Output: Prefill timing and its next-token logits.
    """
    from studio_compute import inference

    engine.prefixes.clear()
    started = time.perf_counter()
    request = inference.create_request(
        engine, "cold-prefill", {**body, "use_prefix_cache": False}
    )
    if request is None:
        raise MemoryError("The cold prompt must fit independently.")
    try:
        while request.logits is None:
            inference.prefill_request(engine, request)
        engine.mx.synchronize()
        elapsed = time.perf_counter() - started
        return {
            "prompt_tokens": len(body["prompt_token_ids"]),
            "seconds": elapsed,
            "prompt_tokens_per_second": len(body["prompt_token_ids"]) / elapsed,
        }, request.logits
    finally:
        inference.release_request(engine, request)


def native_reference(engine, body: dict, *, batch: int, tokens: int) -> dict:
    """
    Purpose: Measure the pinned native model as an independent uniform-batch reference.
    Input: The same resident LoRA modules, prompt, continuation, and 8-bit KV format.
    Output: Native prefill/decode timings and actual cache-state dtypes.
    Note: Report precision differences explicitly; this reference does not replace parity checks.
    """
    mx, lm = engine.mx, engine.lm
    engine.prefixes.clear()
    mx.clear_cache()
    engine.resident.select(engine.policies["benchmark"]["factors"])
    engine.resident.mode(False)
    caches = [
        cache.to_quantized(bits=8, group_size=64)
        if hasattr(cache, "to_quantized")
        else cache
        for cache in lm.make_cache()
    ]
    prompt = body["prompt_token_ids"]
    started = time.perf_counter()
    for start in range(0, len(prompt), 2048):
        stop = min(len(prompt), start + 2048)
        positions = mx.broadcast_to(
            mx.arange(start, stop)[None, None], (3, batch, stop - start)
        )
        output = lm(
            mx.array([prompt[start:stop]] * batch, mx.int32),
            cache=caches,
            position_ids=positions,
        )
        logits_BV = output.logits[:, -1].astype(mx.bfloat16)
        mx.eval(logits_BV, [cache.state for cache in caches])
    mx.synchronize()
    prefill_seconds = time.perf_counter() - started
    probabilities = []
    started = time.perf_counter()
    for index, token in enumerate(body["target_token_ids"][:tokens]):
        lp_BV = logits_BV - mx.logsumexp(logits_BV, axis=-1, keepdims=True)
        probabilities.append(lp_BV[:, token])
        if index + 1 < tokens:
            output = lm(
                mx.full((batch, 1), token, mx.int32),
                cache=caches,
                position_ids=mx.full((3, batch, 1), len(prompt) + index, mx.int32),
            )
            logits_BV = output.logits[:, -1].astype(mx.bfloat16)
        mx.eval(logits_BV, probabilities[-1], [cache.state for cache in caches])
    mx.synchronize()
    elapsed = time.perf_counter() - started
    from mlx.utils import tree_flatten

    dtypes = Counter(
        str(value.dtype)
        for _, value in tree_flatten([cache.state for cache in caches])
        if hasattr(value, "dtype")
    )
    return {
        "batch": batch,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": elapsed,
        "tokens": batch * tokens,
        "decode_tokens_per_second": batch * tokens / elapsed,
        "cache_dtypes": dict(dtypes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="The isolated benchmark receipt root."
    )
    parser.add_argument(
        "--baseline-source",
        type=Path,
        required=True,
        help="The immutable stopped-run source.",
    )
    parser.add_argument(
        "--records-root",
        type=Path,
        required=True,
        help="The stopped hero rollout log tree.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home() / "models/Qwen3.8-Flash-Next-4e8d8g",
        help="The approved Flash model.",
    )
    parser.add_argument(
        "--tokens", type=int, default=128, help="Fixed continuation tokens per request."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Timed repetitions after one warmup per variant.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="Independent request counts.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = load_records(
        args.records_root, count=max(args.concurrency), tokens=args.tokens
    )
    baseline = load_baseline(args.baseline_source)
    from studio_compute.numerics import flash_ops

    candidate_decode = flash_ops.decode_rows
    candidate_moe = flash_ops._moe
    lease_path = Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"
    lease = lease_path.open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from studio_compute.flash_model import QwenFlashModel
    from studio_compute.runtime import RuntimeEngine

    engine = None
    report = {
        "status": "loading",
        "records": [
            {
                k: v
                for k, v in body.items()
                if k not in ("prompt_token_ids", "target_token_ids")
            }
            for body in records
        ],
        "measurements": [],
        "native_references": [],
        "parity": [],
        "baseline_source": str(args.baseline_source),
        "warmups": 1,
        "timed_trials": args.trials,
        "target_tokens_per_request": args.tokens,
        "lora_seed": 20261019,
        "nonzero_b_offset": 0.0001,
    }
    atomic_json(root / "benchmark.json", report)
    try:
        engine = RuntimeEngine(
            args.model,
            model_factory=QwenFlashModel,
            execution_profile="bf16-fast",
            activation_budget_gib=16,
            inference_concurrency=max(args.concurrency),
            prefill_tokens=2048,
            segment_tokens=2048,
            head_rows=128,
        )
        factors = {
            name: value + 0.0001 if name.endswith(".b") else value
            for name, value in engine.resident.adapter.initial(20261019).items()
        }
        engine.mx.eval(factors)
        engine.policies["benchmark"] = {"factors": factors, "revision": 0}
        engine.resident.select(factors)
        engine.resident.mode(False)
        report.update(
            status="running", identity=engine.identity, capabilities=engine.capabilities
        )
        atomic_json(
            root / "ready.json", {"ready": True, "pid": __import__("os").getpid()}
        )
        report["distribution_parity"] = verify_distributions(
            engine, records, baseline.decode_rows, candidate_decode
        )
        atomic_json(root / "benchmark.json", report)
        print(
            json.dumps({"distribution_parity": report["distribution_parity"]}),
            flush=True,
        )
        expected = {}
        for workload in ("recorded_ragged", "uniform_2k"):
            bodies = (
                [
                    {
                        **records[0],
                        "prompt_token_ids": records[0]["prompt_token_ids"][:2048],
                        "seed": 17 + row,
                    }
                    for row in range(max(args.concurrency))
                ]
                if workload == "uniform_2k"
                else records
            )
            for concurrency in args.concurrency:
                for variant in VARIANTS:
                    flash_ops.decode_rows = (
                        baseline.decode_rows
                        if variant in ("baseline", "sqlite_only")
                        else candidate_decode
                    )
                    flash_ops._moe = (
                        candidate_moe
                        if variant == "sorted_experts"
                        else lambda mixer, hidden, **kwargs: baseline._moe(
                            mixer, hidden
                        )
                    )
                    for trial in range(args.trials + 1):
                        name = f"{workload}-b{concurrency}-{variant}-{trial}"
                        profile = cProfile.Profile()
                        if trial == 1 and concurrency == max(args.concurrency):
                            profile.enable()
                        row, probabilities = run_case(
                            engine,
                            bodies[:concurrency],
                            variant=variant,
                            folder=root / name,
                        )
                        if trial == 1 and concurrency == max(args.concurrency):
                            profile.disable()
                            profile.dump_stats(str(root / (name + ".prof")))
                            with (root / (name + "-profile.txt")).open("w") as stream:
                                pstats.Stats(profile, stream=stream).sort_stats(
                                    "cumulative"
                                ).print_stats(40)
                        row.update(workload=workload, trial=trial, warmup=trial == 0)
                        report["measurements"].append(row)
                        key = (workload, concurrency)
                        if variant == "baseline" and trial == 0:
                            expected[key] = probabilities
                        differences = [
                            abs(actual - prior)
                            for current, reference in zip(
                                probabilities, expected[key], strict=True
                            )
                            for actual, prior in zip(current, reference, strict=True)
                        ]
                        maximum = max(differences)
                        rms = (
                            sum(value * value for value in differences)
                            / len(differences)
                        ) ** 0.5
                        report["parity"].append(
                            {
                                "workload": workload,
                                "concurrency": concurrency,
                                "variant": variant,
                                "trial": trial,
                                "max_logprob_error_nats": maximum,
                                "rms_logprob_error_nats": rms,
                            }
                        )
                        atomic_json(root / "benchmark.json", report)
                        print(
                            json.dumps(
                                {
                                    **row,
                                    "max_logprob_error_nats": maximum,
                                    "rms_logprob_error_nats": rms,
                                }
                            ),
                            flush=True,
                        )
                        if variant == "sqlite_only" and maximum != 0:
                            raise AssertionError(
                                "Progress persistence changed fixed-path numerics."
                            )
                        if maximum > 0.125:
                            raise AssertionError(
                                f"Batched projection logprob error exceeds 0.125 nats: {maximum}"
                            )
        flash_ops.decode_rows = candidate_decode
        cold_body = max(records, key=lambda body: len(body["prompt_token_ids"]))
        cold_expected = None
        report["cold_prefill"] = []
        for variant in ("baseline", "sorted_experts"):
            flash_ops._moe = (
                candidate_moe
                if variant == "sorted_experts"
                else lambda mixer, hidden, **kwargs: baseline._moe(mixer, hidden)
            )
            for trial in range(2):
                row, logits_BV = cold_prefill_probe(engine, cold_body)
                if cold_expected is None:
                    cold_expected = logits_BV
                error = float(
                    engine.mx.max(
                        engine.mx.abs(
                            logits_BV.astype(engine.mx.float32)
                            - cold_expected.astype(engine.mx.float32)
                        )
                    ).item()
                )
                row.update(variant=variant, trial=trial, max_logit_error=error)
                report["cold_prefill"].append(row)
                atomic_json(root / "benchmark.json", report)
                print(json.dumps({"cold_prefill": row}), flush=True)
                if error > 0.125:
                    raise AssertionError(
                        "Expert locality changed cold-prompt logits: " + json.dumps(row)
                    )
        flash_ops._moe = candidate_moe
        native_body = {
            **records[0],
            "prompt_token_ids": records[0]["prompt_token_ids"][:2048],
        }
        for trial in range(args.trials + 1):
            row = native_reference(
                engine, native_body, batch=max(args.concurrency), tokens=args.tokens
            )
            row.update(trial=trial, warmup=trial == 0)
            report["native_references"].append(row)
            atomic_json(root / "benchmark.json", report)
            print(json.dumps({"native_reference": row}), flush=True)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        flash_ops.decode_rows = candidate_decode
        flash_ops._moe = candidate_moe
        if engine is not None:
            report["final_memory"] = engine.metrics(time.monotonic())
            engine.close()
        atomic_json(root / "benchmark.json", report)
        lease.close()


if __name__ == "__main__":
    main()
