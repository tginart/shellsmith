#!/usr/bin/env python3
"""Light, matched native/High Command comparison on one resident model.

One warmup, two measurements; fixed token paths, BF16 factors/activations,
8-bit base, FP32 Adam (.9/.95/1e-8), no clipping or token averaging.
Run on Studio before the direct context qualification. No percentage gate.
"""

from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from studio_compute.runtime import RuntimeEngine
from studio_compute.storage import atomic_json


def drain(work):
    while True:
        try:
            next(work)
        except StopIteration as result:
            return result.value


def benchmark(args):
    import mlx.core as mx
    from studio_compute.numerics.cispo import cispo_loss
    from high_command_342d.mlx_group_replay import Replay as Reference
    from transformers import AutoTokenizer

    lease_path = Path(args.gpu_lease_root).expanduser()
    lease_path.mkdir(parents=True, exist_ok=True)
    lease = (lease_path / "owner.lock").open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    engine = RuntimeEngine(args.model, execution_profile="legacy")
    engine.create("benchmark", {"seed": 17})
    factors = engine.clients["benchmark"]["adapter"].state()
    engine.policies["benchmark"] = {"factors": factors, "revision": 0}
    if args.ready_file:
        atomic_json(Path(args.ready_file), {"ready": True, "pid": __import__("os").getpid()})
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    unit = tokenizer.encode(" item", add_special_tokens=False)
    prompt = (unit * args.prompt_tokens)[:args.prompt_tokens]
    target = (
        tokenizer.encode(
            " The result is a concise factual answer.", add_special_tokens=False
        )
        * args.completion_tokens
    )[:args.completion_tokens]
    report = {
        "scope": "light_performance",
        "status": "running",
        "identity": engine.identity,
        "capabilities": engine.capabilities,
        "measurements": [],
        "warmups": 1,
        "timed_measurements": 2,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    def native(bodies):
        started = time.monotonic()
        engine.resident.select(factors)
        engine.resident.mode(False)
        lm = engine.lm
        batch = len(bodies)
        cache = lm.make_cache()
        ids = mx.array([prompt] * batch, mx.int32)
        for start in range(0, len(prompt), 2048):
            stop = min(len(prompt), start + 2048)
            positions = mx.broadcast_to(
                mx.arange(start, stop, dtype=mx.int32)[None, None],
                (3, batch, stop - start),
            )
            hidden = lm.model(ids[:, start:stop], position_ids=positions, cache=cache)
            mx.eval(hidden, [x.state for x in cache])
        head = (
            lm.model.embed_tokens.as_linear
            if lm.args.tie_word_embeddings
            else lm.lm_head
        )
        logits = head(hidden[:, -1:])[:, -1].astype(mx.float32)
        mx.eval(logits)
        prefill = time.monotonic() - started
        probabilities = []
        decode_start = time.monotonic()
        for index, token in enumerate(target):
            lp = logits - mx.logsumexp(logits, -1, keepdims=True)
            probabilities.append(lp[:, token])
            mx.eval(probabilities[-1])
            if index + 1 < len(target):
                out = lm(
                    mx.full((batch, 1), token, mx.int32),
                    cache=cache,
                    position_ids=mx.full((3, batch, 1), len(prompt) + index, mx.int32),
                )
                logits = out.logits[:, -1].astype(mx.float32)
                mx.eval(logits, [x.state for x in cache])
        return {
            "seconds": time.monotonic() - started,
            "prefill_seconds": prefill,
            "decode_seconds": time.monotonic() - decode_start,
            "logprobs": mx.stack(probabilities, axis=1).tolist(),
        }

    try:
        for batch_size in (1, 4):
            bodies = [
                {
                    "policy": "benchmark",
                    "revision": 0,
                    "prompt_token_ids": prompt,
                    "max_tokens": len(target),
                    "target_token_ids": target,
                    "seed": 17 + row,
                    "cache_precision": "bf16",
                    "use_prefix_cache": False,
                }
                for row in range(batch_size)
            ]
            for trial in range(3):
                mx.reset_peak_memory()
                reference = native(bodies)
                started = time.monotonic()
                results = (
                    [drain(engine._sample(bodies[0], started))]
                    if batch_size == 1
                    else drain(engine.sample_batch(bodies))
                )
                elapsed = time.monotonic() - started
                errors = [
                    max(
                        abs(a - b)
                        for a, b in zip(
                            result["behavior_logprobs"], reference["logprobs"][row]
                        )
                    )
                    for row, result in enumerate(results)
                ]
                if max(errors) > 1e-5:
                    raise AssertionError(f"Matched native inference differs: {errors}")
                row = {
                    "kind": "inference",
                    "batch_size": batch_size,
                    "trial": trial,
                    "warmup": trial == 0,
                    "prompt_tokens_per_request": len(prompt),
                    "decode_tokens_per_request": len(target),
                    "native": reference,
                    "service_seconds": elapsed,
                    "service_tokens_per_second": batch_size * len(target) / elapsed,
                    "service_prefill_seconds": results[0]["prefill_seconds"],
                    "service_decode_seconds": results[0]["decode_seconds"],
                    "max_logprob_error": max(errors),
                    "metrics": engine.metrics(started),
                }
                report["measurements"].append(row)
                atomic_json(output, report)
                print(
                    json.dumps(
                        {k: v for k, v in row.items() if k not in ("native", "metrics")}
                    ),
                    flush=True,
                )
        prompt = prompt[:args.training_prompt_tokens]
        target = target[:args.training_completion_tokens]
        sequences = [
            {
                "prompt_token_ids": prompt,
                "sampled_tokens": target,
                "behavior_logprobs": [-5.0] * len(target),
                "advantages": [0.1] * len(target),
                "loss_mask": [1.0] * len(target),
            }
            for _ in range(2)
        ]
        for trial in range(3):
            entry = engine.clients["benchmark"]
            engine.resident.select(factors)
            engine.resident.mode(True)
            branches = []
            for seq in sequences:
                ids = mx.array([tuple(prompt) + tuple(target[:-1])], mx.int32)
                values = engine.lm.model.embed_tokens(ids).astype(mx.bfloat16)
                branches.append(
                    {
                        "embeddings": values,
                        "positions": mx.broadcast_to(
                            mx.arange(ids.shape[1], dtype=mx.int32)[None, None],
                            (3, 1, ids.shape[1]),
                        ),
                        "prefix_length": len(prompt),
                        "tokens": target,
                        "objective": lambda current: cispo_loss(
                            current, [-5.0] * len(target), [0.1] * len(target)
                        ),
                    }
                )
            started = time.monotonic()
            mx.reset_peak_memory()
            reference = Reference(engine.resident.model, checkpoint_layers=8).run_group(
                branches
            )
            native_seconds = time.monotonic() - started
            expected = {k: mx.array(v) for k, v in reference[2].items()}
            mx.eval(expected)
            started = time.monotonic()
            entry["backward"] = None
            drain(
                engine.work(
                    "benchmark",
                    "forward_backward",
                    {
                        "sequences": sequences,
                        "operation_id": f"bench-{trial}",
                        "loss": "cispo",
                    },
                )
            )
            elapsed = time.monotonic() - started
            actual = entry["optimizer"].accumulated
            error = (
                sum(
                    float(mx.sum((actual[k] - expected[k]) ** 2).item()) for k in actual
                )
                ** 0.5
            )
            scale = max(
                1e-12,
                sum(float(mx.sum(value**2).item()) for value in expected.values())
                ** 0.5,
            )
            if error / scale > 0.01:
                raise AssertionError(f"Matched grouped replay differs: {error / scale}")
            row = {
                "kind": "grouped_training",
                "trial": trial,
                "warmup": trial == 0,
                "prompt_tokens": len(prompt),
                "targets_per_completion": len(target),
                "completions": 2,
                "reference_seconds": native_seconds,
                "service_seconds": elapsed,
                "gradient_relative_error": error / scale,
                "optimizer_semantics": {
                    "beta1": 0.9,
                    "beta2": 0.95,
                    "eps": 1e-8,
                    "reduction": "sum",
                    "weight_decay": 0.0,
                    "gradient_clipping": False,
                },
                "metrics": engine.metrics(started),
            }
            report["measurements"].append(row)
            atomic_json(output, report)
            engine.abort("benchmark")
            print(json.dumps(row), flush=True)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(output, report)
        engine.close()
        lease.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--completion-tokens", type=int, default=32)
    parser.add_argument("--training-prompt-tokens", type=int, default=4096)
    parser.add_argument("--training-completion-tokens", type=int, default=32)
    parser.add_argument("--ready-file")
    parser.add_argument(
        "--gpu-lease-root", default="~/.local/state/shellsmith/gpu-lease"
    )
    benchmark(parser.parse_args())
