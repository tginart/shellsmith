"""
Purpose: Measure one disposable update from the complete preserved junior12 batch.
Input: The audited batch, approved model, and a private output directory.
Output: A report and progress records. The script saves only disposable gradient and optimizer diagnostics.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mlx_inference_profile import _footprint
from analyze_rollout_segments import analyze
from shellsmith.train.backend import TokenSequence
from shellsmith.train.segments import pack_rollout_segments

BATCH_SHA256 = "7621071b492b91be50e5f37bee6f3e618aefc9cf707f79853fd539fb70710d9d"


class Timings:
    """
    Purpose: Aggregate layer times at the existing evaluation barriers.
    Input: Phase, layer, and duration. Output: Disjoint time totals.
    """

    def __init__(self) -> None:
        self.rows = {}

    def record(self, phase: str, layer: int, is_linear: bool, seconds: float) -> None:
        key = phase + (":gdn" if is_linear else ":attention")
        row = self.rows.setdefault(key, {"seconds": 0.0, "calls": 0})
        row["seconds"] += seconds
        row["calls"] += 1


def run(args: argparse.Namespace) -> None:
    """
    Purpose: Measure the full backward pass and one in-memory Adam update.
    Input: Parsed arguments with the batch, model, and output paths.
    Output: Atomic progress snapshots, a JSON report, and no persistent model state.
    Method:
        1. Verify exact batch coverage before model allocation.
        2. Accumulate every packed path with the production runtime.
        3. Apply one Adam step and verify the frozen base hash.
    """
    assert os.environ.get("MLX_ENABLE_TF32") == "0"
    assert importlib.metadata.version("mlx") == "0.32.3"
    run_started = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": "shellsmith.full_batch_benchmark.v1",
        "host": platform.node(),
        "pid": os.getpid(),
        "status": "preparing",
        "started_at": time.time(),
        "optimizer_committed": False,
        "saved_checkpoint": False,
        "published_policy": False,
        "historical_experiment_restarted": False,
    }
    save_json(output / "report.json", report)
    raw = Path(args.batch).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BATCH_SHA256
    audit = analyze(args.batch)
    assert audit["responses"] == 329 and audit["packed_segments"] == 22
    assert audit["packed_replay_tokens"] == 444145
    assert audit["supervised_tokens_after"] == 156208
    assert audit["causal_prefixes_targets_credit_and_logprobs_exact"]
    payload = json.loads(gzip.decompress(raw))
    original = {}
    for row in payload["body"]["sequences"]:
        sequence = TokenSequence(
            **{
                key: tuple(value) if isinstance(value, list) else value
                for key, value in row.items()
            }
        )
        original[sequence.request_id] = sequence
    packed = []
    for capture in payload["captures"]:
        turns = [
            (turn["sequence_index"], original[turn["request_id"]])
            for turn in capture["turns"]
        ]
        packed.extend(pack_rollout_segments(turns).sequences)
    if args.scope == "longest":
        packed = [
            max(
                packed,
                key=lambda value: (
                    len(value.prompt_token_ids) + len(value.sampled_tokens)
                ),
            )
        ]
        assert (
            len(packed[0].prompt_token_ids) + len(packed[0].sampled_tokens) - 1 == 57538
        )
        assert sum(packed[0].loss_mask) == 21675
    sequences = [asdict(sequence) for sequence in packed]
    report["execution_profile"] = args.execution_profile
    report["measured_scope"] = args.scope
    report["measured_decoder_tokens"] = sum(
        len(s.prompt_token_ids) + len(s.sampled_tokens) - 1 for s in packed
    )
    report["measured_supervised_tokens"] = sum(sum(s.loss_mask) for s in packed)

    report.update(
        batch=audit,
        paths=[
            {
                "decoder_tokens": len(s.prompt_token_ids) + len(s.sampled_tokens) - 1,
                "supervised_tokens": sum(s.loss_mask),
                "source_responses": len(s.source_spans),
            }
            for s in packed
        ],
    )
    del original, payload, raw, packed
    report["batch_preparation_seconds"] = time.monotonic() - run_started
    report["packed_sequences_sha256"] = hashlib.sha256(
        json.dumps(sequences, sort_keys=True).encode()
    ).hexdigest()
    save_json(output / "report.json", report)

    import mlx.core as mx
    from studio_compute.runtime import RuntimeEngine

    from studio_compute.numerics import replay

    native_replay = replay.Replay
    timings = Timings()

    def measured_replay(*values, **options):
        if args.implementation != "production":
            options["saved_activations"] = False
        return native_replay(*values, profile=timings, **options)

    replay.Replay = measured_replay
    from studio_compute.numerics import ops

    original_gdn = ops.gated_delta_chunked
    original_attention = ops.tiled_causal_attention
    if args.implementation == "bf16-reference":

        def reference_gdn(*values, **options):
            options["backend"] = "reference"
            return original_gdn(*values, **options)

        ops.gated_delta_chunked = reference_gdn
    if args.implementation == "bf16-attention":
        from studio_compute.numerics.attention_metal import fused_attention

        ops.tiled_causal_attention = fused_attention
    report["implementation"] = args.implementation
    engine = None
    try:
        load_started = time.monotonic()
        engine = RuntimeEngine(
            args.model,
            execution_profile=args.execution_profile,
            checkpoint_layers=args.checkpoint_layers,
            segment_tokens=2048,
            head_rows=128,
        )
        engine.create("full-batch-benchmark", {"seed": 23})
        entry = engine.clients["full-batch-benchmark"]
        initial = entry["adapter"].state()
        report["initial_adapter_hash"] = engine.resident.tensor_digest(initial)
        initialized = time.monotonic()
        frozen_before = engine.resident.frozen_digest()
        report["initial_frozen_audit_seconds"] = time.monotonic() - initialized
        report["model_and_adapter_initialization_seconds"] = initialized - load_started
        report.update(
            model=args.model,
            runtime_identity=engine.identity,
            capabilities=engine.capabilities,
            precision=engine.capabilities["precision"],
            initialization_seconds=time.monotonic() - load_started,
            frozen_base_hash_before=frozen_before,
            optimizer={
                "learning_rate": args.learning_rate,
                "beta1": 0.9,
                "beta2": 0.95,
                "epsilon": 1e-8,
            },
            status="running",
            update_started_at=time.time(),
            initial_active_memory_bytes=mx.get_active_memory(),
        )
        save_json(output / "report.json", report)
        if args.cost_model_only:
            from studio_compute.numerics.retention import plan_retention

            groups = {}
            for sequence in sequences:
                groups.setdefault(tuple(sequence["prompt_token_ids"]), []).append(
                    sequence
                )
            future = sum(x.nbytes for x in entry["optimizer"].master.values())
            interval, reservation, budgets = engine._saved_admission(groups, future)
            plans = []
            for prompt, group in groups.items():
                length = max(
                    len(x["prompt_token_ids"]) + len(x["sampled_tokens"]) - 1
                    for x in group
                )
                _, plan = plan_retention(
                    engine.lm.layers,
                    length,
                    2048,
                    budgets[prompt],
                    engine.projection_costs,
                    interval,
                )
                plans.append({"decoder_rows": length, **plan})
            baseline = json.loads(args.baseline_report.read_text())
            attention = json.loads(args.attention_report.read_text())["cases"][0]
            baseline_rows = baseline.get(
                "measured_decoder_tokens",
                baseline.get("batch", {}).get("packed_replay_tokens"),
            )
            assert baseline_rows == report["measured_decoder_tokens"]
            assert (
                baseline["packed_sequences_sha256"] == report["packed_sequences_sha256"]
            )
            stage_seconds = baseline["whole_update_seconds"]
            phase_projection = None
            if args.phase_reference_reports:
                legacy_path, gdn_path = args.phase_reference_reports
                legacy = json.loads(legacy_path.read_text())
                gdn = json.loads(gdn_path.read_text())
                assert legacy["status"] == gdn["status"] == "passed"
                baseline_phases = baseline["profile"].get("totals", baseline["profile"])
                keys = (
                    "forward:gdn",
                    "forward:attention",
                    "vjp_total:gdn",
                    "vjp_total:attention",
                )
                ratios = {
                    key: gdn["profile"][key]["seconds"]
                    / legacy["profile"][key]["seconds"]
                    for key in keys
                }
                stage_seconds += sum(
                    baseline_phases[key]["seconds"] * (ratios[key] - 1) for key in keys
                )
                phase_projection = {
                    "legacy_longest_report": str(legacy_path),
                    "gdn_longest_report": str(gdn_path),
                    "phase_ratios": ratios,
                    "projected_stage_seconds": stage_seconds,
                    "scope": "Forecast from measured phase ratios; this is not a measured full-batch BF16/GDN result.",
                }
            phases = attention["phases"]
            seconds = lambda name: phases[name]["median_seconds"]
            # Q*K weights model dense work. Causal early segments make this conservative.
            weight = sum(
                min(2048, length - start)
                * (start + min(2048, length - start))
                / (2048 * 57344)
                for length in [p["decoder_rows"] for p in plans]
                for start in range(0, length, 2048)
            )
            heads = sum(not layer.is_linear for layer in engine.lm.layers)
            attention_change = (
                heads
                * weight
                * (
                    seconds("fused_forward")
                    - seconds("reference_forward")
                    + seconds("fused_cached_backward")
                    - seconds("reference_vjp")
                )
            )
            retention_benefit = (
                sum(p["estimated_recomputation_ns_saved"] for p in plans) * 1e-9
            )
            optimistic = stage_seconds + attention_change - retention_benefit
            conservative = stage_seconds + attention_change
            report["cost_model"] = {
                "baseline_report": str(args.baseline_report),
                "attention_report": str(args.attention_report),
                "baseline_seconds": baseline["whole_update_seconds"],
                "phase_projection": phase_projection,
                "attention_component_change_seconds": attention_change,
                "isolated_optional_retention_benefit_seconds": retention_benefit,
                "forecast_range_seconds": [max(0, optimistic), conservative],
                "working_target_seconds": 210 if args.scope == "longest" else 1200,
                "target_in_range": optimistic
                <= (210 if args.scope == "longest" else 1200),
                "checkpoint_interval": interval,
                "reserved_bytes": reservation,
                "plans": plans,
                "limitations": [
                    "Q*K interpolation omits causal tile and dispatch differences.",
                    "Retention benefits overlap; the optimistic endpoint is not a promise.",
                    "No assumed benefit from explicit derivative projection scheduling.",
                    "The forecast excludes preparation and compilation.",
                ],
            }
            report.update(status="cost_model_complete", finished_at=time.time())
            save_json(output / "report.json", report)
            return
        if args.warmup_backwards:
            before = engine.resident.tensor_digest(entry["adapter"].state())
            rng_before = entry["rng"].tolist()
            tokens = max(
                sequences,
                key=lambda x: len(x["prompt_token_ids"]) + len(x["sampled_tokens"]),
            )
            flat = tokens["prompt_token_ids"] + tokens["sampled_tokens"]
            assert len(flat) >= 2049
            warm_sequence = {
                "prompt_token_ids": flat[:1921],
                "sampled_tokens": flat[1921:2049],
                "behavior_logprobs": [0.0] * 128,
                "advantages": [1.0] * 128,
                "loss_mask": [1.0] * 128,
                "weights": [1.0 / 128] * 128,
                "request_id": "compile-warmup",
                "source_spans": [],
            }
            warm_started = time.monotonic()
            warm_result = drain(
                engine.work(
                    "full-batch-benchmark",
                    "forward_backward",
                    {
                        "operation_id": "compile-warmup",
                        "loss": "weighted_cross_entropy",
                        "sequences": [warm_sequence],
                    },
                )
            )
            report["warmup"] = {
                "decoder_rows": 2048,
                "supervised_rows": 128,
                "seconds_including_first_call_compilation": time.monotonic()
                - warm_started,
                "execution": warm_result["forward_metrics"],
                "optimizer_update": False,
            }
            engine.abort("full-batch-benchmark", "compile-warmup")
            assert entry["optimizer"].steps == 0
            assert entry["rng"].tolist() == rng_before
            assert engine.resident.tensor_digest(entry["adapter"].state()) == before
            timings.rows.clear()
            del warm_result, warm_sequence
        else:
            report["warmup"] = {"decoder_rows": 0, "optimizer_update": False}
        save_json(output / "report.json", report)
        mx.clear_cache()
        mx.reset_peak_memory()
        update_started = time.monotonic()
        work = engine.work(
            "full-batch-benchmark",
            "forward_backward",
            {
                "operation_id": "full-batch-backward",
                "loss": "cispo",
                "sequences": sequences,
            },
        )
        last_recorded = 0.0
        with (output / "progress.jsonl").open("a", buffering=1) as stream:
            while True:
                try:
                    progress = next(work)
                except StopIteration as done:
                    backward_result = done.value
                    break
                now = time.monotonic()
                if now - last_recorded >= 5 or progress["phase"] == "segment_complete":
                    row = {
                        **progress,
                        "observed_at": time.time(),
                        "update_elapsed_seconds": now - update_started,
                        "process_memory": _footprint(),
                        "rss_peak_bytes": resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss,
                    }
                    stream.write(json.dumps(row) + "\n")
                    save_json(output / "progress.json", row)
                    print(json.dumps(row), flush=True)
                    last_recorded = now
        backward_seconds = time.monotonic() - update_started
        report.update(
            status="optimizer",
            backward_seconds=backward_seconds,
            forward_metrics=backward_result["forward_metrics"],
            replay_groups=backward_result["groups"],
            backward_peak_memory_bytes=mx.get_peak_memory(),
        )
        save_json(output / "report.json", report)
        counters = {}
        for group in backward_result["groups"]:
            for name, count in group.get("projection_counts", {}).items():
                counters[name] = counters.get(name, 0) + count
        if "reconstructed.vocabulary_head" in counters:
            counters["backward.vocabulary_head"] = counters[
                "reconstructed.vocabulary_head"
            ]
        report["projection_counts"] = counters
        report["projection_count_scope"] = (
            "Logical frozen-projection and nonlinear reconstruction calls; compiler pruning and fusion are not hardware counters."
        )
        report["optional_retained_activation_bytes_peak"] = max(
            (
                group.get("activation_retention", {}).get("retained_bytes", 0)
                for group in backward_result["groups"]
            ),
            default=0,
        )
        if (
            args.execution_profile == "bf16-fast"
            and args.implementation == "production"
        ):
            from studio_compute.numerics.retention import mandatory_saved_bytes

            report["mandatory_saved_activation_bytes_peak"] = max(
                (
                    mandatory_saved_bytes(
                        engine.lm.layers, max(group["branch_sequence_tokens"])
                    )
                    for group in backward_result["groups"]
                ),
                default=0,
            )
        gradients = entry["optimizer"].accumulated
        assert len(gradients) == 992
        assert all(
            v.dtype == engine.compute_dtype and v.ndim == 2 for v in gradients.values()
        )
        optimizer_started = time.monotonic()
        step = drain(
            engine.work(
                "full-batch-benchmark",
                "optim_step",
                {
                    "backward_operation": "full-batch-backward",
                    "learning_rate": args.learning_rate,
                },
            )
        )
        report.update(
            status="verifying",
            profile=timings.rows,
            optimizer_seconds=time.monotonic() - optimizer_started,
            whole_update_seconds=time.monotonic() - update_started,
            optimizer_committed=step["optim_metrics"]["optimizer_committed"],
            optim_metrics=step["optim_metrics"],
            update_peak_active_memory_bytes=mx.get_peak_memory(),
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            execution_metrics=step["execution_metrics"],
            update_finished_at=time.time(),
            first_update_ready_including_benchmark_preparation_seconds=time.monotonic()
            - run_started,
            process_memory_at_update_boundary=_footprint(),
        )
        save_json(output / "report.json", report)
        mx.save_safetensors(str(output / "gradients.safetensors"), gradients)
        report["gradient_hash"] = engine.resident.tensor_digest(gradients)
        report["updated_adapter_hash"] = engine.resident.tensor_digest(
            entry["adapter"].state()
        )
        tensors, metadata = entry["optimizer"].state()
        tensors.update({"adapter." + k: v for k, v in entry["adapter"].state().items()})
        tensors.update({"initial." + k: v for k, v in initial.items()})
        mx.save_safetensors(str(output / "diagnostic-adam.safetensors"), tensors)
        report["diagnostic_optimizer_metadata"] = metadata
        report["diagnostic_scope"] = (
            "After update timer; private comparison tensors, no historical checkpoint or published policy."
        )
        frozen_after = engine.resident.frozen_digest()
        report["frozen_base_hash_after"] = frozen_after
        report["frozen_base_unchanged"] = frozen_before == frozen_after
        assert report["frozen_base_unchanged"]
        assert report["optim_metrics"]["optimizer_steps"] == 1
        report.update(status="passed", finished_at=time.time())
        save_json(output / "report.json", report)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "whole_update_seconds": report["whole_update_seconds"],
                }
            ),
            flush=True,
        )
    except BaseException as error:
        report.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
            finished_at=time.time(),
        )
        if engine is not None:
            report["observed_peak_active_memory_bytes"] = mx.get_peak_memory()
        save_json(output / "report.json", report)
        raise
    finally:
        replay.Replay = native_replay
        ops.gated_delta_chunked = original_gdn
        ops.tiled_causal_attention = original_attention
        if engine is not None:
            engine.close()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def drain(work: Any) -> Any:
    while True:
        try:
            next(work)
        except StopIteration as result:
            return result.value


if __name__ == "__main__":
    from studio_compute.cli import checkpoint_limit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch", required=True, help="Audited preserved batch in gzip JSON format."
    )
    parser.add_argument(
        "--model", required=True, help="Existing approved frozen 27B model directory."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Private directory for benchmark evidence.",
    )
    parser.add_argument(
        "--checkpoint-layers",
        type=checkpoint_limit,
        default=1,
        metavar="auto|N",
        help="Matched explicit interval, or the production automatic admission policy.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for the single disposable Adam step.",
    )
    parser.add_argument(
        "--scope",
        choices=("batch", "longest"),
        default="longest",
        help="The complete preserved batch or its longest packed path.",
    )
    parser.add_argument(
        "--execution-profile",
        choices=("legacy", "bf16-fast"),
        default="legacy",
        help="Numerical implementation to measure.",
    )
    parser.add_argument(
        "--cost-model-only",
        action="store_true",
        help="Measure retention costs and publish a forecast without an optimizer update.",
    )
    parser.add_argument(
        "--baseline-report", type=Path, help="Matched GDN-only baseline report."
    )
    parser.add_argument(
        "--phase-reference-reports",
        type=Path,
        nargs=2,
        help="Actual legacy and BF16/GDN longest-path reports for a full-batch phase forecast.",
    )
    parser.add_argument(
        "--attention-report", type=Path, help="Matched attention component report."
    )
    parser.add_argument(
        "--warmup-backwards",
        type=int,
        choices=(0, 1),
        default=1,
        help="Discard one synthetic backward before the measured single update.",
    )
    parser.add_argument(
        "--implementation",
        choices=("production", "bf16-reference", "bf16-gdn", "bf16-attention"),
        default="production",
        help="Production path or a disposable BF16 component ablation.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=3600,
        help="Hard process limit, at most one hour.",
    )
    parser.add_argument(
        "--lease",
        default=str(Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"),
        help="Existing exclusive Studio GPU lease.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.timeout_seconds <= 3600:
        parser.error("timeout-seconds must be in [1, 3600].")
    if args.cost_model_only and (
        args.execution_profile != "bf16-fast"
        or args.baseline_report is None
        or args.attention_report is None
    ):
        parser.error("The cost model requires BF16 and both component reports.")
    if args.implementation != "production" and args.execution_profile != "bf16-fast":
        parser.error(
            "BF16 component ablations require the bf16-fast execution profile."
        )
    if args.worker:
        run(args)
    else:
        from mlx_hardware_roofline import supervise

        supervise(args, script=Path(__file__))
