"""
Purpose: Account for useful model arithmetic and minimum logical byte traffic.
Input: Calibration, qualified inference, and prior complete-update evidence.
Output: A reproducible JSON cost report with explicit interpretation limits.
Note: Empirical GEMM rates do not prove a physical arithmetic maximum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any


def summarize(root: Path, training: Path) -> dict[str, Any]:
    """
    Purpose: Compare useful model work with measured hardware reference rates.
    Input: Authoritative raw evidence paths. Output: Costs and matched comparisons.
    Note: The report does not count omitted work as measured hardware idleness.
    """
    calibration = _read(root / "calibration/calibration.json")
    strict = _read(root / "strict-fp32/calibration.json")
    inference = _read(root / "inference-qualified/inference.json")
    old_training = _read(training)
    assert inference["status"] == "passed"
    assert inference["environment"]["MLX_ENABLE_TF32"] == "0"
    assert inference["frozen_base_before"] == inference["frozen_base_after"]
    bandwidth_bytes_s = max(
        row["useful_gb_s"] * 1e9
        for row in calibration["cases"]
        if row["name"].startswith(("copy-4294967296", "triad-4294967296"))
    )
    bf16_flops_s = statistics.mean(
        row["dense_equivalent_tflop_s"] * 1e12
        for report in (calibration, strict)
        for row in report["cases"]
        if row["name"] == "gemm-mlx.core.bfloat16-8192"
    )
    projections = inference["projection_inventory"]
    parameters = sum(row["dense_parameters"] for row in projections)
    projection_bytes = sum(row["stored_bytes"] for row in projections)
    config = inference["model_config"]
    full_layers = config["num_hidden_layers"] // config["full_attention_interval"]
    linear_layers = config["num_hidden_layers"] - full_layers
    head_parameters = config["vocab_size"] * config["hidden_size"]
    adapter_bytes = sum(
        2 * row["lora_rank"] * (row["width_in"] + row["width_out"])
        for row in projections
    )
    inference_rows = []
    for length, batch in ((2048, 1), (8192, 1), (32768, 1), (8192, 4)):
        rows = [
            row
            for row in inference["cases"]
            if row["prompt_tokens"] == length
            and row["batch"] == batch
            and not row["warmup"]
        ]
        assert len(rows) == 2
        prefill_s = statistics.mean(row["prefill_seconds"] for row in rows)
        decode_s = statistics.mean(row["decode_seconds"] for row in rows) / 63
        projection_flops = 2 * batch * length * parameters
        attention_flops = (
            4
            * batch
            * full_layers
            * config["num_attention_heads"]
            * config["head_dim"]
            * length
            * (length + 1)
            // 2
        )
        kv_bytes = (
            4
            * batch
            * full_layers
            * config["num_key_value_heads"]
            * config["head_dim"]
            * length
        )
        recurrent_bytes = (
            8
            * batch
            * linear_layers
            * config["linear_num_value_heads"]
            * config["linear_key_head_dim"]
            * config["linear_value_head_dim"]
        )
        logical_bytes = (
            projection_bytes
            + inference["head_weight_bytes"]
            + adapter_bytes
            + kv_bytes
            + recurrent_bytes
        )
        inference_rows.append(
            {
                "prompt_tokens": length,
                "batch": batch,
                "prefill_seconds": prefill_s,
                "decode_seconds_per_batch_step_including_token_selection": decode_s,
                "decode_aggregate_tokens_per_second": batch / decode_s,
                "prefill_projection_and_attention_flops": projection_flops
                + attention_flops,
                "prefill_large_gemm_reference_fraction": (
                    projection_flops + attention_flops
                )
                / prefill_s
                / bf16_flops_s,
                "estimated_minimum_logical_bytes_per_decode_batch_step": logical_bytes,
                "estimated_useful_gb_s": logical_bytes / decode_s / 1e9,
                "estimated_fraction_of_streaming_reference": logical_bytes
                / decode_s
                / bandwidth_bytes_s,
                "peak_active_memory_bytes": max(
                    row["peak_active_memory_bytes"] for row in rows
                ),
            }
        )
    tokens = old_training["path"]["decoder_tokens"]
    supervised = old_training["path"]["supervised_tokens"]
    required_projection_flops = 4 * tokens * parameters
    required_attention_flops = (
        12
        * full_layers
        * config["num_attention_heads"]
        * config["head_dim"]
        * tokens
        * (tokens + 1)
        // 2
    )
    required_head_flops = 4 * supervised * head_parameters
    required_adapter_flops = sum(
        6 * tokens * row["lora_rank"] * (row["width_in"] + row["width_out"])
        for row in projections
    )
    subtotal = (
        required_projection_flops
        + required_attention_flops
        + required_head_flops
        + required_adapter_flops
    )
    scheduling = _read(root / "inference-buffers-500-1024/inference.json")
    scheduling_rows = []
    for batch in (1, 4):
        reference = [
            row
            for row in inference["cases"]
            if row["batch"] == batch
            and row["prompt_tokens"] == 8192
            and not row["warmup"]
        ]
        candidate = [
            row
            for row in scheduling["cases"]
            if row["batch"] == batch and not row["warmup"]
        ]
        delta = max(
            abs(x - y)
            for a, b in zip(reference, candidate, strict=True)
            for c, d in zip(a["logprobs"], b["logprobs"], strict=True)
            for x, y in zip(c, d, strict=True)
        )
        scheduling_rows.append(
            {
                "batch": batch,
                "decode_speedup": statistics.mean(
                    row["decode_seconds"] for row in reference
                )
                / statistics.mean(row["decode_seconds"] for row in candidate),
                "extra_peak_active_memory_bytes": max(
                    row["peak_active_memory_bytes"] for row in candidate
                )
                - max(row["peak_active_memory_bytes"] for row in reference),
                "maximum_logprob_difference": delta,
                "decision": "reject: slower decode and larger memory peak",
            }
        )
    sources = [
        root / "calibration/calibration.json",
        root / "strict-fp32/calibration.json",
        root / "inference-qualified/inference.json",
        root / "inference-buffers-500-1024/inference.json",
        training,
    ]
    return {
        "status": "investigation_in_progress",
        "reference_bandwidth_bytes_s": bandwidth_bytes_s,
        "reference_bf16_flops_s": bf16_flops_s,
        "projection_parameters": parameters,
        "projection_stored_bytes": projection_bytes,
        "adapter_logical_bytes": adapter_bytes,
        "inference": inference_rows,
        "training": {
            "tokens": tokens,
            "supervised_positions": supervised,
            "necessary_projection_flops": required_projection_flops,
            "necessary_attention_gemm_flops": required_attention_flops,
            "necessary_head_flops": required_head_flops,
            "necessary_adapter_flops": required_adapter_flops,
            "necessary_flop_subtotal": subtotal,
            "optimistic_bf16_reference_seconds": subtotal / bf16_flops_s,
            "measured_checkpoint_1_update_seconds": old_training["settings"]["1"][
                "mean_update_seconds"
            ],
        },
        "scheduling_comparison": scheduling_rows,
        "interpretation_limits": [
            "These arithmetic rates are empirical references, not proven physical ceilings.",
            "The training subtotal excludes GDN recurrence, non-GEMM operations, and recomputation.",
            "The inference traffic estimate counts logical weights and one KV/state pass, not hardware DRAM counters.",
            "The prefill fraction excludes GDN, LoRA, norms, layout, and other operations.",
            "The inference timings use fixed-token in-process requests, not HTTP latency or free-running sampling.",
            "The complete-path checkpoint removal validation remains pending.",
        ],
        "evidence_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
    }


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="Downloaded hardware evidence folder."
    )
    parser.add_argument(
        "--training-summary",
        type=Path,
        required=True,
        help="Prior complete-update comparison summary.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output JSON report path."
    )
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(summarize(args.root, args.training_summary), indent=2) + "\n"
    )
