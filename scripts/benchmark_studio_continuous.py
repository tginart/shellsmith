"""
Purpose: Measure continuous inference with recorded agent conversations on real Metal hardware.
Input: Pilot player_records JSONL files and the qualified Qwen model.
Output: An atomic JSON report with parity, refill, cache, memory, and throughput evidence.
Note: Teacher-forced replay fixes token paths across concurrency settings.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from studio_compute import inference
from studio_compute.runtime import RuntimeEngine
from studio_compute.storage import atomic_json


def load_conversations(
    root: Path, *, sessions: int, turns: int, max_completion_tokens: int
) -> list[list[dict]]:
    """
    Purpose: Select real multi-turn prompts without padding or prompt truncation.
    Input: A directory with recorded player JSONL files. Output: Bounded fixed-path requests.
    Note: Each later turn enters the queue only after its prior turn completes.
    """
    conversations = []
    for path in sorted(root.rglob("player_records/*.jsonl")):
        selected = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if not row.get("prompt_token_ids") or not row.get("sampled_tokens"):
                continue
            target = row["sampled_tokens"][:max_completion_tokens]
            selected.append(
                {
                    "prompt_token_ids": row["prompt_token_ids"],
                    "target_token_ids": target,
                    "max_tokens": len(target),
                    "seed": 17 + len(conversations),
                    "revision": 0,
                    "policy": "benchmark",
                    "source_record_sha256": hashlib.sha256(line.encode()).hexdigest(),
                }
            )
            if len(selected) == turns:
                break
        if len(selected) == turns:
            conversations.append(selected)
        if len(conversations) == sessions:
            return conversations
    raise ValueError(f"Need {sessions} conversations with {turns} recorded turns each.")


def drain(work):
    while True:
        try:
            next(work)
        except StopIteration as done:
            return done.value


def measure(
    engine: RuntimeEngine, conversations: list[list[dict]], *, concurrency: int | None
) -> tuple[dict, dict]:
    """
    Purpose: Measure the same queued agent workload with native singleton or persistent decode.
    Input: Recorded conversations and an inference limit. Output: Metrics and logprob records.
    """
    mx = engine.mx
    engine.prefixes.clear()
    mx.clear_cache()
    mx.reset_peak_memory()
    started = time.monotonic()
    initial = engine.metrics(started)
    results, receipts, errors = {}, {}, {}
    histogram = Counter()
    refills = decode_boundaries = queued_boundaries = 0
    peak_slots = peak_pending = peak_prefix_bytes = 0
    overlap_refills = 0

    def record(name, result):
        nonlocal peak_prefix_bytes
        results[name] = result["behavior_logprobs"]
        receipts[name] = {
            k: result[k]
            for k in (
                "prompt_tokens",
                "completion_tokens",
                "prefix_cache_tokens",
                "prefill_seconds",
                "decode_seconds",
            )
        }
        peak_prefix_bytes = max(peak_prefix_bytes, engine.prefixes.bytes)

    if concurrency is None:
        for turn in range(len(conversations[0])):
            for session, conversation in enumerate(conversations):
                name = f"{session}:{turn}"
                record(
                    name, drain(engine._sample(conversation[turn], time.monotonic()))
                )
    else:
        engine.inference_concurrency = concurrency

        def create(name, body):
            nonlocal refills, overlap_refills
            request = inference.create_request(engine, name, body)
            if request is not None:
                refills += 1
                overlap_refills += any(r.tokens for r in scheduler.active.values())
            return request

        def finish(request):
            record(request.request_id, inference.request_result(engine, request))
            session, turn = map(int, request.request_id.split(":"))
            if turn + 1 < len(conversations[session]):
                scheduler.enqueue(
                    f"{session}:{turn + 1}", conversations[session][turn + 1]
                )

        scheduler = engine.inference_scheduler(
            create=create,
            finish=finish,
            fail=lambda name, error: errors.update(
                {name: f"{type(error).__name__}: {error}"}
            ),
        )
        for session, conversation in enumerate(conversations):
            scheduler.enqueue(f"{session}:0", conversation[0])
        try:
            while scheduler.pending or scheduler.active:
                peak_pending = max(peak_pending, len(scheduler.pending))
                row = scheduler.step()
                peak_slots = max(peak_slots, len(scheduler.active))
                histogram.update(row["decode_batch_sizes"])
                decode_boundaries += bool(row["decode_batch_sizes"])
                queued_boundaries += bool(
                    scheduler.pending and row["decode_batch_sizes"]
                )
                if errors:
                    raise RuntimeError(errors)
        finally:
            scheduler.close()
    elapsed = time.monotonic() - started
    tokens = sum(row["completion_tokens"] for row in receipts.values())
    final = engine.metrics(started)
    assert not engine.reserved and not engine.policy_users
    measurement = {
        "concurrency": concurrency,
        "mode": "singleton" if concurrency is None else "continuous",
        "seconds": elapsed,
        "completion_tokens": tokens,
        "tokens_per_second": tokens / elapsed,
        "requests": len(receipts),
        "receipts": receipts,
        "decode_batch_histogram": dict(sorted(histogram.items())),
        "peak_active_slots_after_retirement": peak_slots,
        "peak_queued_requests": peak_pending,
        "admissions": refills,
        "admissions_with_live_survivors": overlap_refills,
        "decode_boundaries": decode_boundaries,
        "decode_boundaries_with_queue": queued_boundaries,
        "cached_requests": sum(
            row["prefix_cache_tokens"] > 0 for row in receipts.values()
        ),
        "cached_tokens": sum(row["prefix_cache_tokens"] for row in receipts.values()),
        "prefix_cache_peak_bytes": peak_prefix_bytes,
        "prefix_cache": engine.prefixes.metrics(),
        "memory_before": initial,
        "memory_after": final,
        "swap_growth_bytes": final["swap_used_bytes"] - initial["swap_used_bytes"],
        "admission_budget_bytes": engine.admission_budget_bytes(),
        "reservations_after": len(engine.reserved),
        "policy_users_after": dict(engine.policy_users),
    }
    return measurement, results


def validate_distribution_targets(
    conversations: list[list[dict]], *, concurrency: int
) -> None:
    """
    Purpose: Reject short diagnostic targets before model or cache allocation.
    Input: The selected conversations and row limit. Output: An error for insufficient targets.
    """
    selected = conversations[:concurrency]
    if not selected or any(
        len(conversation[0]["target_token_ids"]) < 9 for conversation in selected
    ):
        raise ValueError(
            "Distribution parity requires at least nine completion tokens "
            "in each selected first turn."
        )


def verify_distributions(
    engine: RuntimeEngine, conversations: list[list[dict]], *, concurrency: int
) -> dict:
    """
    Purpose: Separate BF16 logprob record roundoff from categorical distribution error.
    Input: Mixed real prompts and fixed continuation tokens. Output: Full-vocabulary KL bounds.
    Note: FP32 normalization serves diagnostics. Production sampling arithmetic stays unchanged.
    """
    validate_distribution_targets(conversations, concurrency=concurrency)
    mx = engine.mx
    requests, references, divergences, logprob_errors = [], [], [], []
    engine.prefixes.clear()
    try:
        for i, conversation in enumerate(conversations[:concurrency]):
            request = inference.create_request(
                engine, f"distribution-{i}", conversation[0]
            )
            assert request is not None
            requests.append(request)
            while request.logits is None:
                inference.prefill_request(engine, request)
            references.append(engine.prefixes._copy(request.cache))
        for index in range(8):
            expected = []
            for request, cache in zip(requests, references, strict=True):
                token = request.body["target_token_ids"][index]
                request.tokens.append(token)
                engine.resident.select(request.factors)
                engine.resident.mode(False)
                out = engine._decode_logits(
                    mx.array([[token]], mx.int32),
                    cache=cache,
                    position_ids=mx.full(
                        (3, 1, 1),
                        len(request.body["prompt_token_ids"]) + index,
                        mx.int32,
                    ),
                )
                logits_BV = out.logits[:, -1].astype(engine.compute_dtype)
                mx.eval(logits_BV, [item.state for item in cache])
                expected.append(logits_BV)
            inference.advance_requests(engine, requests)
            for request, logits_BV in zip(requests, expected, strict=True):
                reference_BV = logits_BV.astype(mx.float32)
                actual_BV = request.logits.astype(mx.float32)
                reference_BV -= mx.logsumexp(reference_BV, -1, keepdims=True)
                actual_BV -= mx.logsumexp(actual_BV, -1, keepdims=True)
                divergence_S = mx.sum(mx.exp(reference_BV) * (reference_BV - actual_BV))
                divergences.append(max(0.0, float(divergence_S.item())))
                reference_record_BV = logits_BV - mx.logsumexp(
                    logits_BV, -1, keepdims=True
                )
                actual_record_BV = request.logits - mx.logsumexp(
                    request.logits, -1, keepdims=True
                )
                token = request.body["target_token_ids"][index + 1]
                logprob_errors.append(
                    abs(
                        float(reference_record_BV[0, token].item())
                        - float(actual_record_BV[0, token].item())
                    )
                )
        result = {
            "rows": len(requests),
            "decode_steps": 8,
            "max_kl_nats": max(divergences),
            "mean_kl_nats": sum(divergences) / len(divergences),
            "max_record_logprob_error": max(logprob_errors),
            "kl_limit_nats": 0.005,
            "normalization": "FP32 full vocabulary; production BF16 records unchanged",
        }
        assert result["max_kl_nats"] <= result["kl_limit_nats"], result
        return result
    finally:
        for request in requests:
            inference.release_request(engine, request)


def benchmark(args: argparse.Namespace) -> None:
    conversations = load_conversations(
        args.records_root,
        sessions=args.sessions,
        turns=args.turns,
        max_completion_tokens=args.max_completion_tokens,
    )
    if args.distribution_parity:
        validate_distribution_targets(conversations, concurrency=max(args.concurrency))
    import mlx.core as mx

    for conversation in conversations:
        for body in conversation:
            body["cache_precision"] = args.cache_precision
    lease_path = args.gpu_lease_root.expanduser()
    lease_path.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "workload": {
            "sessions": args.sessions,
            "turns": args.turns,
            "prompt_lengths": [
                [len(b["prompt_token_ids"]) for b in c] for c in conversations
            ],
            "completion_lengths": [[b["max_tokens"] for b in c] for c in conversations],
            "source_record_sha256": [
                [b.pop("source_record_sha256") for b in c] for c in conversations
            ],
            "description": "Unpadded recorded agent prompts; live multi-turn arrivals; fixed recorded completion paths",
        },
        "measurements": [],
        "validation_gaps": [],
        "tolerance": {
            "max_logprob_absolute_error": args.logprob_tolerance,
            "reason": "BF16 projection reductions can differ with row count; logprob records retain BF16 rounding.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (lease_path / "owner.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        engine = RuntimeEngine(
            str(args.model),
            execution_profile=args.execution_profile,
            inference_concurrency=max(args.concurrency),
            prefill_tokens=args.prefill_tokens,
            allocator_gib=args.allocator_gib,
        )
        try:
            engine.policies["benchmark"] = {
                "factors": engine.idle_factors,
                "revision": 0,
            }
            report.update(
                identity=engine.identity,
                capabilities=engine.capabilities,
                device=mx.device_info(),
            )
            atomic_json(args.output, report)
            if args.distribution_parity:
                report["distribution_parity"] = verify_distributions(
                    engine, conversations, concurrency=max(args.concurrency)
                )
                atomic_json(args.output, report)
            warmup = {
                **conversations[0][0],
                "prompt_token_ids": conversations[0][0]["prompt_token_ids"][:128],
                "target_token_ids": conversations[0][0]["target_token_ids"][:4],
                "max_tokens": 4,
                "use_prefix_cache": False,
            }
            drain(engine._sample(warmup, time.monotonic()))
            baseline, reference = measure(engine, conversations, concurrency=None)
            report["measurements"].append({**baseline, "trial": 0})
            atomic_json(args.output, report)
            print(
                json.dumps(
                    {k: baseline[k] for k in ("mode", "seconds", "tokens_per_second")}
                ),
                flush=True,
            )
            for trial in range(args.repeats):
                for concurrency in args.concurrency:
                    measurement, actual = measure(
                        engine, conversations, concurrency=concurrency
                    )
                    differences = [
                        abs(a - b)
                        for name, values in actual.items()
                        for a, b in zip(values, reference[name], strict=True)
                    ]
                    assert actual.keys() == reference.keys()
                    differences.sort()
                    measurement.update(
                        trial=trial,
                        max_logprob_error=max(differences),
                        p99_logprob_error=differences[
                            int(0.99 * (len(differences) - 1))
                        ],
                        mean_logprob_error=sum(differences) / len(differences),
                        speedup_over_singleton=measurement["tokens_per_second"]
                        / baseline["tokens_per_second"],
                    )
                    report["measurements"].append(measurement)
                    atomic_json(args.output, report)
                    print(
                        json.dumps(
                            {
                                k: measurement[k]
                                for k in (
                                    "concurrency",
                                    "trial",
                                    "seconds",
                                    "tokens_per_second",
                                    "max_logprob_error",
                                    "speedup_over_singleton",
                                    "cached_requests",
                                    "admissions_with_live_survivors",
                                    "decode_batch_histogram",
                                    "swap_growth_bytes",
                                )
                            }
                        ),
                        flush=True,
                    )
                    assert max(differences) <= args.logprob_tolerance, measurement[
                        "max_logprob_error"
                    ]
                    assert measurement["swap_growth_bytes"] <= 0
                    assert (
                        measurement["memory_after"]["allocator_peak_bytes"]
                        <= measurement["admission_budget_bytes"]
                    )
            report["status"] = "passed"
        except BaseException as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            engine.close()
            atomic_json(args.output, report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Local qualified model path."
    )
    parser.add_argument(
        "--records-root",
        type=Path,
        required=True,
        help="Pilot train directory with player_records JSONL files.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Atomic evidence JSON path."
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=12,
        help="Concurrent conversations, including queued work.",
    )
    parser.add_argument("--turns", type=int, default=3, help="Turns per conversation.")
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=96,
        help="Maximum recorded completion path length.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Independent inference concurrency settings.",
    )
    parser.add_argument(
        "--repeats", type=int, default=2, help="Timed repeats per concurrency setting."
    )
    parser.add_argument(
        "--prefill-tokens",
        type=int,
        default=512,
        help="Global prefill chunk budget per scheduler boundary.",
    )
    parser.add_argument(
        "--execution-profile",
        choices=["bf16-fast", "legacy"],
        default="bf16-fast",
        help="Qualified arithmetic profile.",
    )
    parser.add_argument(
        "--distribution-parity",
        action="store_true",
        help="Compare eight decode steps against native singleton full-vocabulary distributions.",
    )
    parser.add_argument(
        "--cache-precision",
        choices=["bf16", "8bit"],
        default="bf16",
        help="Native inference cache precision.",
    )
    parser.add_argument(
        "--allocator-gib",
        type=float,
        help="Optional allocator cap for memory-pressure runs.",
    )
    parser.add_argument(
        "--logprob-tolerance",
        type=float,
        default=0.25,
        help="Predeclared BF16 absolute logprob error bound.",
    )
    parser.add_argument(
        "--gpu-lease-root",
        type=Path,
        default=Path("~/.local/state/shellsmith/gpu-lease"),
        help="Shared GPU ownership lease directory.",
    )
    args = parser.parse_args()
    if (
        min(
            args.sessions,
            args.turns,
            args.max_completion_tokens,
            args.repeats,
            *args.concurrency,
        )
        < 1
    ):
        parser.error("Workload sizes, repeats, and concurrency must be positive.")
    return args


if __name__ == "__main__":
    benchmark(parse_args())
