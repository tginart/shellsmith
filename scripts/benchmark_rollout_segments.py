#!/usr/bin/env python3
"""Bounded, matched replay comparison; no service, campaign, or saved policy.

Usage: PYTHONPATH=src python scripts/benchmark_rollout_segments.py --output report.json
Add --model /path/to/approved/model for a resident 27B check on an idle Studio.
The default is the same quantized BF16 hybrid tiny fixture used by numerical tests.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

from shellsmith.train.backend import TokenSequence
from shellsmith.train.segments import pack_rollout_segments


def drain(work):
    while True:
        try:
            next(work)
        except StopIteration as result:
            return result.value


def benchmark(args):
    import mlx.core as mx

    from studio_compute.engine import ResidentEngine
    from studio_compute.numerics import replay as replay_module
    from studio_compute.numerics.optimizer import MasterAdam

    native_replay = replay_module.Replay
    reference_replay = native_replay
    reference_hash = None
    if args.reference_replay:
        reference_path = Path(args.reference_replay).resolve()
        reference_hash = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        spec = importlib.util.spec_from_file_location(
            "studio_compute.numerics._benchmark_reference", reference_path
        )
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        reference_replay = reference.Replay

    lease = None
    if args.model:
        lease_path = Path(args.gpu_lease_root).expanduser() / "owner.lock"
        lease_path.parent.mkdir(parents=True, exist_ok=True)
        lease = lease_path.open("a")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        engine = ResidentEngine(args.model, checkpoint_layers=8, execution_profile="legacy")
    else:
        # Keep the test architecture in one place. This does not fetch weights.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        from test_studio_numerics import TinyResident

        engine = ResidentEngine(
            "tiny",
            model_factory=TinyResident,
            execution_profile="legacy",
            checkpoint_layers=2,
            segment_tokens=64,
            head_rows=8,
        )
    engine.create("comparison", {"seed": 23})
    entry = engine.clients["comparison"]
    initial = entry["adapter"].state()
    frozen_before = engine.resident.frozen_digest()
    prompt = [1 + (i % 60) for i in range(256)]
    responses = []
    for turn in range(4):
        tokens = tuple(1 + ((turn * 8 + i) % 60) for i in range(8))
        mask = (1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0)  # injected control token
        responses.append(
            TokenSequence(
                tuple(prompt),
                tokens,
                (-4.0,) * 8,
                tuple(
                    (0.02 if (turn + i) % 2 else -0.03) * m / 28
                    for i, m in enumerate(mask)
                ),
                mask,
                f"turn-{turn}",
            )
        )
        prompt += list(tokens) + [61, 62, 63] * 5
    packed = pack_rollout_segments([(0, s) for s in responses]).sequences
    report = {
        "schema": "shellsmith.segment_benchmark.v1",
        "reference_replay_sha256": reference_hash,
        "host": platform.node(),
        "model": args.model or "tiny-4-layer-hybrid-8bit",
        "precision": "BF16 execution/LoRA; FP32 gradients/master/Adam",
        "optimizer": {
            "learning_rate": 5e-5,
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1e-8,
            "steps_per_trial": 1,
        },
        "context_window": engine.capabilities["context_window"],
        "warmups": args.warmups,
        "trials": [],
        "supervised_tokens": 28,
        "responses": len(responses),
        "packed_segments": len(packed),
        "old_replay_tokens": sum(
            len(s.prompt_token_ids) + len(s.sampled_tokens) - 1 for s in responses
        ),
        "packed_replay_tokens": sum(
            len(s.prompt_token_ids) + len(s.sampled_tokens) - 1 for s in packed
        ),
    }
    try:
        for trial in range(args.warmups + args.trials):
            snapshots = {}
            for name, sequences in (
                (("responses", responses), ("packed", packed))
                if trial % 2 == 0
                else (("packed", packed), ("responses", responses))
            ):
                replay_module.Replay = (
                    reference_replay if name == "responses" else native_replay
                )
                entry["adapter"].load(initial)
                entry["optimizer"] = MasterAdam(entry["adapter"], 5e-5)
                mx.eval(
                    entry["adapter"].state(),
                    entry["optimizer"].master,
                    entry["optimizer"].first,
                    entry["optimizer"].second,
                )
                engine.resident.select(entry["adapter"].state())
                mx.clear_cache()
                mx.reset_peak_memory()
                started = time.monotonic()
                result = drain(
                    engine.work(
                        "comparison",
                        "forward_backward",
                        {
                            "operation_id": f"{name}-{trial}",
                            "loss": "cispo",
                            "sequences": [asdict(s) for s in sequences],
                        },
                    )
                )
                backward = time.monotonic() - started
                gradients = {
                    k: mx.array(v) for k, v in entry["optimizer"].accumulated.items()
                }
                mx.eval(gradients)
                optimized = drain(
                    engine.work(
                        "comparison",
                        "optim_step",
                        {
                            "backward_operation": f"{name}-{trial}",
                            "learning_rate": 5e-5,
                        },
                    )
                )
                snapshots[name] = {
                    "gradients": gradients,
                    "adapter": dict(entry["adapter"].state()),
                    "loss": result["forward_metrics"]["cispo_loss"],
                    "backward_seconds": backward,
                    "whole_update_seconds": time.monotonic() - started,
                    "peak_memory_bytes": mx.get_peak_memory(),
                    "phase_seconds": result["forward_metrics"]["phase_seconds"],
                    "optimizer_committed": optimized["optim_metrics"][
                        "optimizer_committed"
                    ],
                }

            def flat(d):
                return mx.concatenate(
                    [d[k].astype(mx.float32).reshape(-1) for k in sorted(d)]
                )

            a, b = (flat(snapshots[n]["gradients"]) for n in ("responses", "packed"))
            relative = float(
                (mx.linalg.norm(a - b) / mx.maximum(mx.linalg.norm(a), 1e-8)).item()
            )
            cosine = float(
                (mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b))).item()
            )
            da, db = (flat(snapshots[n]["adapter"]) for n in ("responses", "packed"))
            row = {
                "trial": trial,
                "warmup": trial < args.warmups,
                "gradient_relative_l2": relative,
                "gradient_cosine": cosine,
                "adapter_max_abs_error": float(mx.max(mx.abs(da - db)).item()),
                **{
                    n: {k: v for k, v in s.items() if k not in ("gradients", "adapter")}
                    for n, s in snapshots.items()
                },
            }
            assert relative < 0.05 and cosine > 0.998, row
            assert (
                abs(snapshots["responses"]["loss"] - snapshots["packed"]["loss"]) < 0.02
            ), row
            report["trials"].append(row)
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        report["frozen_base_unchanged"] = (
            frozen_before == engine.resident.frozen_digest()
        )
        assert report["frozen_base_unchanged"]
        report["status"] = "passed"
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        replay_module.Replay = native_replay
        engine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument(
        "--reference-replay",
        help="Preserved pre-change replay.py; executed only for response trials",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument(
        "--gpu-lease-root", default="~/.local/state/shellsmith/gpu-lease"
    )
    args = parser.parse_args()
    if not 0 <= args.warmups <= 1 or not 1 <= args.trials <= 3:
        parser.error("Bounded to one warmup and 1-3 trials")
    benchmark(args)
