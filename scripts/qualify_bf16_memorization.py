"""
Purpose: Qualify ten-update SFT through the resident training and inference engine.
Input: Local model assets and a fresh disposable output folder.
Output: Every learning attempt, greedy answer, loss, and checkpoint comparison.
Note: The selector uses argmax. Model execution uses the production inference path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

PROMPT = "Question: What is the secret phrase for Studio check Z7K4?\nAnswer:"
ANSWER = " violet otter 7391 cedar"
SEEDS = (23, 91)
LEARNING_RATES = (3e-4, 1e-3, 3e-3)
UPDATES = 10


def run(args: argparse.Namespace) -> None:
    """
    Purpose: Compare fresh adapters under one fixed memorization fixture.
    Input: A legacy calibration or its frozen rate. Output: An atomic report.
    """
    import mlx.core as mx
    from transformers import AutoTokenizer

    from studio_compute.runtime import RuntimeEngine

    class GreedyEngine(RuntimeEngine):
        def _draw_token(self, logprobs_BV: Any, key: Any) -> Any:
            return mx.argmax(logprobs_BV, axis=-1)

    options = dict(checkpoint_layers=1, prefix_cache_gib=0,
                   execution_profile=args.execution_profile)
    engine = GreedyEngine(args.model, **options)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    prompt = tokenizer.encode(PROMPT, add_special_tokens=False)
    targets = tokenizer.encode(ANSWER, add_special_tokens=False)
    targets.append(tokenizer.eos_token_id)
    fixture = {"prompt": PROMPT, "answer": ANSWER, "prompt_ids": prompt,
               "target_ids": targets, "updates": UPDATES, "seeds": list(SEEDS)}
    fixture_hash = hashlib.sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest()
    report = {
        "status": "running", "fixture": fixture, "fixture_sha256": fixture_hash,
        "execution_profile": args.execution_profile, "identity": engine.identity,
        "capabilities": engine.capabilities, "attempts": [],
        "selector": "argmax through production inference; no forced tokens",
        "started_at": time.time(),
    }
    base_hash = engine.resident.frozen_digest()
    if args.calibration:
        calibration = json.loads(args.calibration.read_text())
        assert calibration["status"] == "passed"
        assert calibration["fixture_sha256"] == fixture_hash
        assert calibration["execution_profile"] == "legacy"
        rates = [calibration["selected_learning_rate"]]
        report["calibration_sha256"] = hashlib.sha256(
            args.calibration.read_bytes()
        ).hexdigest()
    else:
        assert args.execution_profile == "legacy", "BF16 requires frozen calibration."
        rates = LEARNING_RATES
    _save(args.output / "report.json", report)

    def work(kind: str, body: dict) -> dict:
        return _drain(engine.work("memorize", kind, body))

    sequence = {
        "prompt_token_ids": prompt, "sampled_tokens": targets,
        "weights": [1.0 / len(targets)] * len(targets),
        "loss_mask": [1.0] * len(targets),
    }

    def evaluate() -> dict:
        result = work("forward", {
            "sequences": [sequence], "loss": "weighted_cross_entropy",
            "operation_id": "evaluate",
        })
        current = result["logprobs"][0]
        body = {
            "prompt_token_ids": prompt, "max_tokens": len(targets),
            "seed": 0, "stop_token_ids": [tokenizer.eos_token_id],
            "use_prefix_cache": False,
            "policy": "disposable-memorization",
            "revision": engine.clients["memorize"]["optimizer"].steps,
        }
        generation = _drain(engine._sample_policy_units(
            body, time.monotonic(), engine.clients["memorize"]["adapter"].state()
        ))
        sampled = generation["sampled_tokens"]
        return {
            "mean_nll": -sum(current) / len(current),
            "generated_tokens": sampled,
            "generated_text": tokenizer.decode(sampled, skip_special_tokens=True),
            "exact_answer": sampled == targets,
        }

    def update(rate: float, number: int) -> dict:
        operation = f"backward-{number}"
        before = engine.clients["memorize"]["adapter"].state()
        started = time.perf_counter()
        backward = work("forward_backward", {
            "sequences": [sequence], "loss": "weighted_cross_entropy",
            "operation_id": operation,
        })
        step = work("optim_step", {
            "learning_rate": rate, "backward_operation": operation,
            "operation_id": f"update-{number}",
        })
        after = engine.clients["memorize"]["adapter"].state()
        squared = sum(float(mx.sum(
            (after[name].astype(mx.float32) - value.astype(mx.float32)) ** 2
        ).item()) for name, value in before.items())
        return {
            "step": number, "seconds": time.perf_counter() - started,
            "adapter_delta_l2": squared ** 0.5,
            "optim_metrics": step["optim_metrics"],
            "loss": backward["forward_metrics"]["loss"], "peak_mlx_bytes": mx.get_peak_memory(),
        }

    try:
        for rate in rates:
            passed = []
            for seed in SEEDS:
                engine.create("memorize", {"seed": seed, "learning_rate": rate})
                attempt = {"learning_rate": rate, "seed": seed, "steps": [],
                           "status": "running"}
                report["attempts"].append(attempt)
                attempt["initial"] = evaluate()
                assert not attempt["initial"]["exact_answer"], "Fixture is already memorized."
                for number in range(1, UPDATES + 1):
                    row = update(rate, number)
                    attempt["steps"].append(row)
                    if number == 5:
                        checkpoint = args.output / f"checkpoint-{rate}-{seed}"
                        checkpoint.mkdir()
                        engine.write_state("memorize", checkpoint)
                    _save(args.output / "report.json", report)
                attempt["final"] = evaluate()
                uninterrupted = _state_hash(engine)
                engine.close_client("memorize")
                engine.restore("memorize", checkpoint, {"seed": seed, "revision": 5})
                replayed = [update(rate, number) for number in range(6, UPDATES + 1)]
                attempt["restored_steps"] = replayed
                attempt["restore_exact"] = _state_hash(engine) == uninterrupted
                final = attempt["final"]
                success = (
                    final["exact_answer"]
                    and final["mean_nll"] <= 0.1 * attempt["initial"]["mean_nll"]
                    and attempt["restore_exact"]
                    and any(row["optim_metrics"]["gradient_norm"] > 0
                            and row["adapter_delta_l2"] > 0 for row in attempt["steps"])
                )
                attempt["status"] = "passed" if success else "failed"
                passed.append(success)
                engine.close_client("memorize")
                _save(args.output / "report.json", report)
            if all(passed) and "selected_learning_rate" not in report:
                report["selected_learning_rate"] = rate
            # Complete the prescribed legacy grid. BF16 uses only the frozen rate.
            if args.calibration:
                break
        if "selected_learning_rate" not in report:
            report["status"] = "failed"
        else:
            rate = report["selected_learning_rate"]
            engine.create("memorize", {"seed": SEEDS[0], "learning_rate": rate})
            initial = evaluate()
            control = [update(0.0, number) for number in range(1, UPDATES + 1)]
            final = evaluate()
            report["zero_rate_control"] = {
                "initial": initial, "final": final, "steps": control,
                "passed": initial == final
                and all(row["adapter_delta_l2"] == 0 for row in control)
                and not final["exact_answer"],
            }
            engine.close_client("memorize")
            report["status"] = "passed" if report["zero_rate_control"]["passed"] else "failed"
        report["frozen_base_unchanged"] = engine.resident.frozen_digest() == base_hash
        assert report["frozen_base_unchanged"]
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_at"] = time.time()
        _save(args.output / "report.json", report)
        engine.close()
    if report["status"] != "passed":
        raise RuntimeError("Ten-update memorization qualification failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Approved local 27B model assets.")
    parser.add_argument("--output", type=Path, required=True, help="Fresh private evidence folder.")
    parser.add_argument("--execution-profile", choices=("legacy", "bf16-fast"),
                        default="legacy", help="Numerical implementation to qualify.")
    parser.add_argument("--calibration", type=Path, help="Passed legacy report that freezes the rate.")
    parser.add_argument("--timeout-seconds", type=int, default=3600,
                        help="Hard process deadline, at most 3600 seconds.")
    parser.add_argument("--lease", default=str(Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"),
                        help="Existing exclusive Studio GPU lease.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.timeout_seconds <= 3600:
        parser.error("timeout-seconds must be in [1, 3600].")
    if args.worker:
        run(args)
    else:
        from mlx_hardware_roofline import supervise
        supervise(args, script=Path(__file__))


def _drain(generator: Any) -> Any:
    while True:
        try:
            next(generator)
        except StopIteration as result:
            return result.value


def _state_hash(engine: Any) -> str:
    entry = engine.clients["memorize"]
    tensors, metadata = entry["optimizer"].state()
    return hashlib.sha256(json.dumps({
        "adapter": engine.resident.tensor_digest(entry["adapter"].state()),
        "optimizer": engine.resident.tensor_digest(tensors),
        "metadata": metadata,
    }, sort_keys=True).encode()).hexdigest()


def _save(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
