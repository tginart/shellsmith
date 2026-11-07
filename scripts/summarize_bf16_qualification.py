"""
Purpose: Publish matched BF16 training times, memory, coverage, and remaining reconstruction.
Input: Completed ablation/final/batch matrices and one instrumented final-path report.
Output: Reproducible JSON and Markdown tables with explicit unmet objectives.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from pathlib import Path

GIB = 1 << 30


def summarize(args: argparse.Namespace) -> dict:
    """
    Purpose: Reduce complete trials without mixing layer VJP totals and nested build times.
    Input: Passed matrix files. Output: Stage medians and explicit performance gates.
    """
    ablations, final, batch = [
        _read(path) for path in (args.ablations, args.final, args.batch)
    ]
    assert all(report["status"] == "passed" for report in (ablations, final, batch))
    stages = {
        label: [
            row for row in ablations["runs"] if row["configuration"] == configuration
        ]
        for label, configuration in (
            ("Legacy", "legacy"),
            ("BF16 reference", "bf16-reference"),
            ("BF16 + GDN", "bf16-gdn"),
            ("BF16 + fused attention", "bf16-attention"),
            ("Saved projections", "bf16-final"),
        )
    }
    stages["Saved statistics (final)"] = final["runs"]
    longest = {name: _reduce(rows, 3) for name, rows in stages.items()}
    full_batch = {
        name: _reduce(
            [row for row in batch["runs"] if row["configuration"] == configuration], 2
        )
        for name, configuration in (("Legacy", "legacy"), ("BF16 final", "bf16-final"))
    }
    for table, expected_rows, expected_targets in (
        (longest, 57538, 21675),
        (full_batch, 444145, 156208),
    ):
        assert len({row["initial_adapter_hash"] for row in table.values()}) == 1
        assert len({row["packed_sequences_sha256"] for row in table.values()}) == 1
        assert all(
            row["paths"] == (1 if expected_rows == 57538 else 22)
            for row in table.values()
        )
        assert all(row["decoder_rows"] == expected_rows for row in table.values())
        assert all(
            row["supervised_tokens"] == expected_targets for row in table.values()
        )
    attention = _read(args.attention)
    assert attention["status"] == "passed"
    attention_cases = [
        {
            "queries": case["queries"],
            "keys": case["keys"],
            "cached_backward_speedup": case["cached_backward_speedup"],
            "complete_vjp_speedup": case["full_vjp_speedup"],
            "twofold_backward_target_met": case["cached_backward_speedup"] >= 2,
        }
        for case in attention["cases"]
    ]
    counter = _read(args.counters)
    assert counter["status"] == "passed" and counter["frozen_base_unchanged"]
    assert (
        counter["measured_decoder_tokens"] == 57538
        and counter["measured_supervised_tokens"] == 21675
    )
    assert counter["projection_counts"]
    assert counter["initial_adapter_hash"] == longest["Legacy"]["initial_adapter_hash"]
    assert (
        counter["packed_sequences_sha256"]
        == longest["Legacy"]["packed_sequences_sha256"]
    )
    same_build = (
        counter["runtime_identity"]["kernel_build_id"]
        in longest["Saved statistics (final)"]["kernel_build_ids"]
    )
    if not same_build:
        assert args.release_equivalence is not None, (
            "A new build requires final qualification evidence."
        )
        release = _read(args.release_equivalence)
        assert (
            release["status"] == "passed"
            and release["matched_final_gradients_and_adapter_exact"]
        )
        assert all(
            counter[name] == row[name]
            for row in final["runs"]
            for name in (
                "initial_adapter_hash",
                "packed_sequences_sha256",
                "gradient_hash",
                "updated_adapter_hash",
            )
        ), "The release counter run must reproduce all timed updates exactly."
    assert all(
        group["saved_activations"] and group["hidden_checkpoint_layers"] == 1
        for group in counter["replay_groups"]
    )
    baseline, candidate = longest["Legacy"], longest["Saved statistics (final)"]
    batch_baseline, batch_candidate = full_batch["Legacy"], full_batch["BF16 final"]
    diagnostics = {}
    for name in ("diagnostics", "ablation_diagnostics"):
        path = getattr(args, name, None)
        if path is not None:
            diagnostics[name] = _read(path)
            assert diagnostics[name]["status"] == "passed"
    return {
        "schema": "shellsmith.bf16_qualification.v1",
        **diagnostics,
        "longest_path": longest,
        "full_batch": full_batch,
        "coverage": {
            "paths": 22,
            "decoder_rows": 444145,
            "supervised_tokens": 156208,
            "source_responses": 329,
            "rollouts": 15,
        },
        "longest_complete_update_speedup": baseline["median_update_seconds"]
        / candidate["median_update_seconds"],
        "longest_layer_backward_speedup": baseline["median_layer_backward_seconds"]
        / candidate["median_layer_backward_seconds"],
        "longest_total_backward_speedup": baseline["median_total_backward_seconds"]
        / candidate["median_total_backward_seconds"],
        "full_batch_total_backward_speedup": batch_baseline[
            "median_total_backward_seconds"
        ]
        / batch_candidate["median_total_backward_seconds"],
        "attention_primitives": attention_cases,
        "full_batch_complete_update_speedup": batch_baseline["median_update_seconds"]
        / batch_candidate["median_update_seconds"],
        "targets": {
            "longest_210_seconds_met": candidate["median_update_seconds"] <= 210,
            "full_batch_1200_seconds_met": batch_candidate["median_update_seconds"]
            <= 1200,
            "longest_1_5x_update_met": baseline["median_update_seconds"]
            / candidate["median_update_seconds"]
            >= 1.5,
            "full_batch_1_5x_update_met": batch_baseline["median_update_seconds"]
            / batch_candidate["median_update_seconds"]
            >= 1.5,
            "longest_2x_backward_met": baseline["median_total_backward_seconds"]
            / candidate["median_total_backward_seconds"]
            >= 2,
            "full_batch_2x_backward_met": batch_baseline[
                "median_total_backward_seconds"
            ]
            / batch_candidate["median_total_backward_seconds"]
            >= 2,
            "all_attention_primitives_2x_backward_met": all(
                case["twofold_backward_target_met"] for case in attention_cases
            ),
        },
        "counter_trial": {
            "update_seconds": counter["whole_update_seconds"],
            "build_matches_timed_trials": same_build,
            "release_equivalence_report": str(args.release_equivalence)
            if args.release_equivalence
            else None,
            "kernel_build_id": counter["runtime_identity"]["kernel_build_id"],
            "optional_retained_activation_bytes_peak": counter[
                "optional_retained_activation_bytes_peak"
            ],
            "mandatory_saved_activation_bytes_peak": counter[
                "mandatory_saved_activation_bytes_peak"
            ],
            "projection_counts": counter["projection_counts"],
            "count_scope": counter["projection_count_scope"],
            "activation_plans": [
                group["activation_retention"] for group in counter["replay_groups"]
            ],
        },
        "limitations": [
            "Representative 2048-row warmup is separate; new shape compilation can remain in complete-update times.",
            "Compiler wall time is not independently isolated. OS compiler caches are retained; warm component distributions describe steady-state primitives.",
            "Projection counters describe logical calls; compiler pruning/fusion and hardware instructions are separate.",
            "First ready means an in-memory Adam commit. Benchmark diagnostics and durable service-checkpoint I/O follow that boundary.",
            "Legacy gradient agreement is diagnostic. Independent derivatives and ten-update real-model learning define correctness.",
        ],
        "input_matrices": [
            str(path) for path in (args.ablations, args.final, args.batch)
        ],
        "counter_report": str(args.counters),
        "attention_report": str(args.attention),
    }


def _reduce(rows: list, expected: int) -> dict:
    assert len(rows) == expected
    assert len({row["initial_adapter_hash"] for row in rows}) == 1
    assert len({row["packed_sequences_sha256"] for row in rows}) == 1
    assert len({row["forward_metrics"]["sequences"] for row in rows}) == 1
    assert len({row["forward_metrics"]["total_tokens"] for row in rows}) == 1
    assert len({row["forward_metrics"]["supervised_tokens"] for row in rows}) == 1
    assert all(row["execution_receipt"]["status"] == "passed" for row in rows)
    assert all(row["forward_metrics"]["loss_kind"] == "cispo" for row in rows)
    phases = [row["profile"] for row in rows]
    forward = [
        sum(profile[name]["seconds"] for name in ("forward:gdn", "forward:attention"))
        for profile in phases
    ]
    backward = [
        sum(
            profile[name]["seconds"]
            for name in ("vjp_total:gdn", "vjp_total:attention")
        )
        for profile in phases
    ]
    times = [row["whole_update_seconds"] for row in rows]
    assert all(
        row["forward_metrics"]["completed_tokens"]
        == row["forward_metrics"]["total_tokens"]
        for row in rows
    )
    return {
        "trial_seconds": times,
        "median_update_seconds": statistics.median(times),
        "median_first_ready_seconds": statistics.median(
            row["first_update_ready_including_benchmark_preparation_seconds"]
            for row in rows
        ),
        "component_seconds": {
            name: statistics.median(profile[name]["seconds"] for profile in phases)
            for name in (
                "forward:gdn",
                "forward:attention",
                "vjp_total:gdn",
                "vjp_total:attention",
            )
        },
        "median_layer_forward_seconds": statistics.median(forward),
        "median_layer_backward_seconds": statistics.median(backward),
        "median_total_backward_seconds": statistics.median(
            layer_seconds
            + row["forward_metrics"]["phase_seconds"].get("head_backward", 0)
            for row, layer_seconds in zip(rows, backward, strict=True)
        ),
        "paths": rows[0]["forward_metrics"]["sequences"],
        "decoder_rows": rows[0]["forward_metrics"]["total_tokens"],
        "supervised_tokens": rows[0]["forward_metrics"]["supervised_tokens"],
        "median_head_backward_seconds": statistics.median(
            row["forward_metrics"]["phase_seconds"].get("head_backward", 0)
            for row in rows
        ),
        "median_representative_warmup_seconds": statistics.median(
            row["warmup"]["seconds_including_first_call_compilation"] for row in rows
        ),
        "maximum_mlx_bytes": max(
            row["update_peak_active_memory_bytes"] for row in rows
        ),
        "maximum_physical_footprint_bytes": max(
            row["process_memory_at_update_boundary"][
                "lifetime_max_physical_footprint_bytes"
            ]
            for row in rows
        ),
        "maximum_swap_growth_bytes": max(
            row["execution_metrics"]["swap_growth_bytes"] for row in rows
        ),
        "repeated_gradients_exact": len({row["gradient_hash"] for row in rows}) == 1,
        "repeated_updated_adapters_exact": len(
            {row["updated_adapter_hash"] for row in rows}
        )
        == 1,
        "kernel_build_ids": sorted(
            {row["runtime_identity"]["kernel_build_id"] for row in rows}
        ),
        "initial_adapter_hash": rows[0]["initial_adapter_hash"],
        "packed_sequences_sha256": rows[0]["packed_sequences_sha256"],
    }


def markdown(report: dict) -> str:
    """Render the measured results and every unmet numerical speed objective."""
    lines = [
        "# Matched BF16 backward qualification",
        "",
        "The frozen backbone uses 8-bit weights. BF16 tensors cover activations, carries, adapters, gradients, and stochastic-rounding Adam.",
        "The legacy profile remains selectable. No experiment or historical checkpoint is resumed.",
        "",
        "## Longest preserved path",
        "",
        "Each stage has three timed trials after a separate representative warmup. The path has 57,538 decoder rows and 21,675 supervised tokens.",
        "",
        "| Configuration | Update trials (s) | Median update (s) | First ready (s) | Layer forward / backward (s) | Peak MLX / physical (GiB) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in report["longest_path"].items():
        values = ", ".join(f"{value:.2f}" for value in row["trial_seconds"])
        lines.append(
            f"| {name} | {values} | {row['median_update_seconds']:.2f} | {row['median_first_ready_seconds']:.2f} | {row['median_layer_forward_seconds']:.2f} / {row['median_layer_backward_seconds']:.2f} | {row['maximum_mlx_bytes'] / GIB:.2f} / {row['maximum_physical_footprint_bytes'] / GIB:.2f} |"
        )
    lines += [
        "",
        "Layer timings are disjoint. Head and optimizer work remain in complete-update time. Nested VJP build/evaluation times are not added twice.",
        "",
        "## Complete preserved batch",
        "",
        "All 22 paths, 444,145 decoder rows, 156,208 supervised tokens, 329 captured responses, and 15 rollouts retain the original masks and CISPO weights. Each trial performs one accumulated Adam update.",
        "",
        "| Profile | Update trials (s) | Median update (s) | First ready (s) | Peak MLX / physical (GiB) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name, row in report["full_batch"].items():
        values = ", ".join(f"{value:.2f}" for value in row["trial_seconds"])
        lines.append(
            f"| {name} | {values} | {row['median_update_seconds']:.2f} | {row['median_first_ready_seconds']:.2f} | {row['maximum_mlx_bytes'] / GIB:.2f} / {row['maximum_physical_footprint_bytes'] / GIB:.2f} |"
        )
    lines += [
        "",
        f"Complete-update speedups are {report['longest_complete_update_speedup']:.3f}x longest-path and {report['full_batch_complete_update_speedup']:.3f}x full-batch. Total backward, including the vocabulary head, is {report['longest_total_backward_speedup']:.3f}x faster on the longest path and {report['full_batch_total_backward_speedup']:.3f}x on the full batch.",
        "",
        "Maximum reported net swap growth across the timed trials: "
        + str(
            max(
                [
                    row["maximum_swap_growth_bytes"]
                    for row in report["longest_path"].values()
                ]
                + [
                    row["maximum_swap_growth_bytes"]
                    for row in report["full_batch"].values()
                ]
            )
        )
        + " bytes.",
        "",
        "Representative warmup medians (including first-call setup): "
        + f"longest legacy {report['longest_path']['Legacy']['median_representative_warmup_seconds']:.2f} s, "
        + f"BF16 {report['longest_path']['Saved statistics (final)']['median_representative_warmup_seconds']:.2f} s; "
        + f"full batch legacy {report['full_batch']['Legacy']['median_representative_warmup_seconds']:.2f} s, "
        + f"BF16 {report['full_batch']['BF16 final']['median_representative_warmup_seconds']:.2f} s. "
        + "These are separate from update time and are included in first-ready latency.",
        "",
        "## Performance gates",
        "",
        "| Objective | Observed | Result |",
        "| --- | ---: | --- |",
    ]
    observed = {
        "longest_210_seconds_met": (
            "Longest update ≤210 s",
            f"{report['longest_path']['Saved statistics (final)']['median_update_seconds']:.2f} s",
        ),
        "full_batch_1200_seconds_met": (
            "Full batch update ≤20 min",
            f"{report['full_batch']['BF16 final']['median_update_seconds'] / 60:.3f} min",
        ),
        "longest_1_5x_update_met": (
            "Longest complete update ≥1.5x",
            f"{report['longest_complete_update_speedup']:.4f}x",
        ),
        "full_batch_1_5x_update_met": (
            "Full batch complete update ≥1.5x",
            f"{report['full_batch_complete_update_speedup']:.4f}x",
        ),
        "longest_2x_backward_met": (
            "Longest total backward ≥2x",
            f"{report['longest_total_backward_speedup']:.4f}x",
        ),
        "full_batch_2x_backward_met": (
            "Full batch total backward ≥2x",
            f"{report['full_batch_total_backward_speedup']:.4f}x",
        ),
        "all_attention_primitives_2x_backward_met": (
            "Every attention backward primitive ≥2x",
            ", ".join(
                f"{case['cached_backward_speedup']:.3f}x"
                for case in report["attention_primitives"]
            ),
        ),
    }
    for name, passed in report["targets"].items():
        label, value = observed[name]
        lines.append(f"| {label} | {value} | {'PASS' if passed else 'NOT MET'} |")
    lines += [
        "",
        "## Final component costs",
        "",
        "| Scope and component | Legacy forward / backward (s) | BF16 forward / backward (s) |",
        "| --- | ---: | ---: |",
    ]
    for scope, table, candidate_name in (
        ("Longest", report["longest_path"], "Saved statistics (final)"),
        ("Full batch", report["full_batch"], "BF16 final"),
    ):
        for kind in ("gdn", "attention"):
            baseline = table["Legacy"]["component_seconds"]
            candidate = table[candidate_name]["component_seconds"]
            lines.append(
                f"| {scope} {kind} | {baseline['forward:' + kind]:.2f} / {baseline['vjp_total:' + kind]:.2f} | {candidate['forward:' + kind]:.2f} / {candidate['vjp_total:' + kind]:.2f} |"
            )
    lines += [
        "",
        "## Attention primitive scope",
        "",
        "| Query / KV rows | Backward from saved primal | Complete VJP | 2x backward objective |",
        "| --- | ---: | ---: | --- |",
    ]
    for case in report["attention_primitives"]:
        lines.append(
            f"| {case['queries']} / {case['keys']} | {case['cached_backward_speedup']:.3f}x | {case['complete_vjp_speedup']:.3f}x | {'PASS' if case['twofold_backward_target_met'] else 'NOT MET'} |"
        )
    if "diagnostics" in report:
        lines += [
            "",
            "## Numerical diagnostics",
            "",
            "The gradient and Adam-update changes relative to legacy are large. These comparisons are diagnostics under the specified acceptance criteria; their completion status is not an agreement gate.",
            "",
            "| Scope | Cosine | Relative L2 difference |",
            "| --- | ---: | ---: |",
        ]
        for name, row in report["diagnostics"]["results"].items():
            lines.append(f"| {name} | {row['cosine']:.6f} | {row['relative_l2']:.6f} |")
        lines += [
            "",
            "Real-27B memorization and independent local derivatives define the requested correctness gate. Long-rollout reward improvement is not tested; the experiment remains stopped.",
        ]
    if "ablation_diagnostics" in report:
        lines += [
            "",
            "| Longest-path transition | Gradient cosine | Relative L2 difference |",
            "| --- | ---: | ---: |",
        ]
        for name, row in report["ablation_diagnostics"]["cases"].items():
            lines.append(f"| {name} | {row['cosine']:.6f} | {row['relative_l2']:.6f} |")
        lines += [
            "",
            "The explicit saved-activation VJP closely matches its preceding BF16 stage. Larger changes enter with numerical forward/kernel implementations, including the saved RMS realization.",
            "For the longest legacy/final comparison, layer 0 gradient cosine is 0.0206 and layer 63 is 0.9650. Differences are larger toward earlier layers; this observation alone does not establish their cause.",
        ]
    c = report["counter_trial"]
    lines += [
        "",
        "## Retention and remaining reconstruction",
        "",
        f"The release counter trial takes {c['update_seconds']:.2f} seconds with automatic admission and retains {c['optional_retained_activation_bytes_peak'] / GIB:.2f} GiB of optional outputs and {c['mandatory_saved_activation_bytes_peak'] / GIB:.2f} GiB of mandatory small residuals. Existing KV allocations and checkpointed hidden inputs are additional. Checkpoint-one backward consumes layer tapes.",
        "",
        c["count_scope"],
        "",
        (
            "The counter run uses the timed build."
            if c["build_matches_timed_trials"]
            else "The release counter run includes the qualified default and additional telemetry. It reproduces every timed gradient and Adam-result hash exactly; its build identity is recorded separately."
        ),
        "",
        "| Logical operation | Forward | Projection input-grad | Retained reads | Reconstructed |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    counts = c["projection_counts"]
    names = sorted({key.split(".", 1)[1] for key in counts})
    for name in names:
        values = [
            counts.get(prefix + "." + name, 0)
            for prefix in ("forward", "backward", "retained", "reconstructed")
        ]
        lines.append(f"| {name} | {' | '.join(str(value) for value in values)} |")
    lines += ["", "## Scope and limitations", ""]
    lines += ["- " + value for value in report["limitations"]]
    return "\n".join(lines) + "\n"


def _read(path: Path) -> dict:
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablations",
        type=Path,
        required=True,
        help="Passed three-trial ablation matrix.",
    )
    parser.add_argument(
        "--final",
        type=Path,
        required=True,
        help="Passed final saved-statistics matrix.",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        required=True,
        help="Passed two-trial legacy/final full-batch matrix.",
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        help="Completed CPU legacy/final gradient and update diagnostics; no agreement threshold.",
    )
    parser.add_argument(
        "--ablation-diagnostics",
        type=Path,
        help="Completed BF16 stage-transition diagnostic summaries.",
    )
    parser.add_argument(
        "--release-equivalence",
        type=Path,
        help="Passed final qualification proving a changed default/telemetry build reproduces the timed gradients and Adam result.",
    )
    parser.add_argument(
        "--attention",
        type=Path,
        required=True,
        help="Matched cached-backward and complete-VJP primitive evidence.",
    )
    parser.add_argument(
        "--counters",
        type=Path,
        required=True,
        help="Instrumented final longest-path report.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh report folder."
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = summarize(args)
    (args.output / "performance.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "performance.md").write_text(markdown(report))
