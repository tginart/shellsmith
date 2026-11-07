"""
Purpose: Run matched disposable update trials with one leased GPU worker at a time.
Input: The preserved junior12 batch, approved model, and immutable source checkout.
Output: Three longest-path trials per configuration or two full-batch trials per profile.
Note: Each subprocess has its own one-hour hard limit. The command creates no monitor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

CONFIGURATIONS = (
    ("legacy", "legacy", "production"),
    ("bf16-reference", "bf16-fast", "bf16-reference"),
    ("bf16-gdn", "bf16-fast", "bf16-gdn"),
    ("bf16-attention", "bf16-fast", "bf16-attention"),
    ("bf16-final", "bf16-fast", "production"),
)


def run(args: argparse.Namespace) -> None:
    """
    Purpose: Preserve each measured trial and stop on any failed correctness gate.
    Input: Fixed experiment facts. Output: Atomic aggregate progress and raw trial folders.
    """
    script = args.source / "scripts/benchmark_training_update.py"
    identity = {
        "batch_sha256": hashlib.sha256(args.batch.read_bytes()).hexdigest(),
        "benchmark_source_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "source": str(args.source.resolve()),
        "model": str(args.model.resolve()),
        "suite": args.suite,
        "configurations": args.configurations,
        "timeout_seconds_per_trial": 3600,
        "warmup": "2048-row synthetic backward, discarded without an optimizer update",
        "compile_scope": "Representative warmup includes first-call compilation; new long-context shapes can still compile within measured updates.",
    }
    output = args.output / "qualification.json"
    if args.resume:
        report = json.loads(output.read_text())
        assert report["identity"] == identity, (
            "A resume requires identical source and input facts."
        )
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        report = {
            "status": "running",
            "identity": identity,
            "runs": [],
            "started_at": time.time(),
        }
    selected = (
        CONFIGURATIONS
        if args.suite == "longest"
        else (CONFIGURATIONS[0], CONFIGURATIONS[-1])
    )
    if args.configurations:
        requested = set(args.configurations)
        selected = tuple(row for row in selected if row[0] in requested)
        assert {row[0] for row in selected} == requested, (
            "A batch suite supports legacy and bf16-final only."
        )
    trials = 3 if args.suite == "longest" else 2
    completed = {(row["configuration"], row["trial"]) for row in report["runs"]}
    try:
        for trial in range(1, trials + 1):
            # Alternate order to expose cache and thermal order effects.
            order = selected if trial % 2 else tuple(reversed(selected))
            for label, profile, implementation in order:
                if (label, trial) in completed:
                    continue
                folder = args.output / f"{label}-trial{trial}"
                assert not folder.exists(), (
                    "Preserve a failed trial; use a fresh matrix for changed sources."
                )
                command = [
                    sys.executable,
                    str(script),
                    "--model",
                    str(args.model),
                    "--batch",
                    str(args.batch),
                    "--output",
                    str(folder),
                    "--scope",
                    args.suite,
                    "--execution-profile",
                    profile,
                    "--implementation",
                    implementation,
                    "--checkpoint-layers",
                    "1",
                    "--timeout-seconds",
                    "3600",
                ]
                report.update(
                    status="running",
                    active={
                        "configuration": label,
                        "trial": trial,
                        "output": str(folder),
                        "started_at": time.time(),
                    },
                )
                save(output, report)
                print(json.dumps(report["active"]), flush=True)
                subprocess.run(command, check=True, cwd=args.source)
                receipt = json.loads((folder / "execution.json").read_text())
                result = json.loads((folder / "report.json").read_text())
                assert receipt["status"] == result["status"] == "passed"
                assert result["frozen_base_unchanged"] and result["optimizer_committed"]
                assert result["optim_metrics"]["optimizer_steps"] == 1
                assert result["measured_decoder_tokens"] == (
                    57538 if args.suite == "longest" else 444145
                )
                assert result["measured_supervised_tokens"] == (
                    21675 if args.suite == "longest" else 156208
                )
                if report["runs"]:
                    previous = report["runs"][0]
                    assert (
                        previous["initial_adapter_hash"]
                        == result["initial_adapter_hash"]
                    )
                    assert (
                        previous["packed_sequences_sha256"]
                        == result["packed_sequences_sha256"]
                    )
                row = {
                    "configuration": label,
                    "trial": trial,
                    "output": str(folder),
                    "execution_receipt": receipt,
                    **{
                        name: result[name]
                        for name in (
                            "whole_update_seconds",
                            "first_update_ready_including_benchmark_preparation_seconds",
                            "update_peak_active_memory_bytes",
                            "process_memory_at_update_boundary",
                            "initial_adapter_hash",
                            "packed_sequences_sha256",
                            "gradient_hash",
                            "updated_adapter_hash",
                            "runtime_identity",
                            "profile",
                            "warmup",
                            "forward_metrics",
                            "execution_metrics",
                        )
                    },
                }
                report["runs"].append(row)
                save(output, report)
                print(
                    json.dumps(
                        {
                            "configuration": label,
                            "trial": trial,
                            "whole_update_seconds": row["whole_update_seconds"],
                        }
                    ),
                    flush=True,
                )
        summaries = {}
        for label, _, _ in selected:
            rows = [row for row in report["runs"] if row["configuration"] == label]
            assert len(rows) == trials
            seconds = [row["whole_update_seconds"] for row in rows]
            summaries[label] = {
                "seconds": seconds,
                "median_seconds": statistics.median(seconds),
                "minimum_seconds": min(seconds),
                "maximum_seconds": max(seconds),
                "repeated_gradient_hashes_equal": len(
                    {row["gradient_hash"] for row in rows}
                )
                == 1,
                "repeated_updated_adapter_hashes_equal": len(
                    {row["updated_adapter_hash"] for row in rows}
                )
                == 1,
                "maximum_mlx_bytes": max(
                    row["update_peak_active_memory_bytes"] for row in rows
                ),
            }
        baseline = summaries.get("legacy", {}).get("median_seconds")
        final = summaries.get("bf16-final", {}).get("median_seconds")
        report.update(
            status="passed",
            summary=summaries,
            speedup=(
                baseline / final if baseline is not None and final is not None else None
            ),
            complete_update_1_5x_target_met=(
                baseline / final >= 1.5
                if baseline is not None and final is not None
                else None
            ),
            absolute_time_target_seconds=210 if args.suite == "longest" else 1200,
            absolute_time_target_met=(
                final <= (210 if args.suite == "longest" else 1200)
                if final is not None
                else None
            ),
            finished_at=time.time(),
        )
        report.pop("active", None)
    except BaseException as error:
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            finished_at=time.time(),
        )
        raise
    finally:
        save(output, report)


def save(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Immutable checkout with the benchmark and production source.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Approved existing 27B model directory.",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        required=True,
        help="Preserved audited junior12 gzip JSON.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh private evidence directory."
    )
    parser.add_argument(
        "--suite",
        choices=("longest", "batch"),
        required=True,
        help="Three trials of each longest-path ablation, or two matched full batches.",
    )
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=[row[0] for row in CONFIGURATIONS],
        help="Run a named subset for a later, separately identified implementation stage.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue completed trials with identical facts.",
    )
    run(parser.parse_args())
