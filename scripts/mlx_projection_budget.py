"""
Purpose: Count executed frozen projections and measure their matched MLX costs.
Input: Typed replay graphs, an actual model inventory, and a qualified runtime.
Output: Reproducible operation recipes, measurements, and projection time budgets.
Note: This partial budget does not establish whole-model hardware utilization.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any

# B: batch, M: token rows, K: input features, N: output features.
# P: packed input features, G: quantization groups.


def plan(args: argparse.Namespace) -> None:
    """
    Purpose: Derive exact projection multiplicities from regular and terminal units.
    Input: Four graphs per unit, path facts, and model inventory.
    Output: Unique quantized operation recipes and executed arithmetic counts.
    """
    inventory = _read(args.inventory)
    profile = _read(args.profile)
    assert inventory["status"] == profile["status"] == "passed"
    assert inventory["environment"]["MLX_ENABLE_TF32"] == "0"
    assert profile["environment"]["MLX_ENABLE_TF32"] == "0"
    assert profile["checkpoint_layers"] == 1
    config = inventory["model_config"]
    layers = config["num_hidden_layers"]
    attention_layers = layers // config["full_attention_interval"]
    layer_counts = {"gdn": layers - attention_layers, "attention": attention_layers}
    tokens = profile["path"]["decoder_tokens"]
    full_units, tail = divmod(tokens, 2048)
    assert tail > 0
    recipes: dict[str, dict[str, Any]] = {}
    stages = []
    layout = _read(args.layout)
    assert layout["decoder_tokens"] == tokens
    unsupervised = layout["unsupervised_full_units"]
    sources = [args.inventory, args.profile, args.layout]
    for folder, repetitions, expected_tokens in (
        (args.regular, full_units, 2048),
        (args.terminal, 1, tail),
    ):
        for kind in ("gdn", "attention"):
            for stage in ("forward", "vjp"):
                path = folder / f"{kind}-{stage}.json"
                graph = _read(path)
                sources.append(path)
                assert graph["kind"] == kind and graph["stage"] == stage
                assert graph["tokens"] == expected_tokens
                if folder == args.terminal:
                    assert graph["terminal"]
                multiplicity = repetitions * layer_counts[kind]
                if kind == "attention" and stage == "vjp" and folder == args.regular:
                    multiplicity -= unsupervised
                counts: Counter[str] = Counter()
                flops = 0
                for matrix in graph["summary"]["matrices"]:
                    if matrix["name"] != "QuantizedMatmul":
                        continue
                    recipe = _recipe(matrix)
                    key = _recipe_key(recipe)
                    recipes.setdefault(key, {**recipe, "calls": 0, "uses": []})
                    recipes[key]["calls"] += multiplicity
                    counts[key] += 1
                    flops += matrix["flops"] * multiplicity
                for key, count in counts.items():
                    recipes[key]["uses"].append(
                        {
                            "kind": kind,
                            "stage": stage,
                            "unit_tokens": expected_tokens,
                            "calls": count * multiplicity,
                        }
                    )
                stages.append(
                    {
                        "kind": kind,
                        "stage": stage,
                        "unit_tokens": expected_tokens,
                        "unit_count": multiplicity,
                        "executed_projection_flops": flops,
                    }
                )
    carry_path = args.carry_only / "attention-vjp.json"
    carry_graph = _read(carry_path)
    sources.append(carry_path)
    assert carry_graph["carry_only"] and carry_graph["tokens"] == 2048
    carry_flops = 0
    counts = Counter()
    for matrix in carry_graph["summary"]["matrices"]:
        if matrix["name"] != "QuantizedMatmul":
            continue
        recipe = _recipe(matrix)
        key = _recipe_key(recipe)
        recipes.setdefault(key, {**recipe, "calls": 0, "uses": []})
        recipes[key]["calls"] += unsupervised
        counts[key] += 1
        carry_flops += matrix["flops"] * unsupervised
    for key, count in counts.items():
        recipes[key]["uses"].append(
            {
                "kind": "attention",
                "stage": "vjp",
                "unit_tokens": 2048,
                "calls": count * unsupervised,
                "carry_only": True,
            }
        )
    stages.append(
        {
            "kind": "attention",
            "stage": "vjp",
            "unit_tokens": 2048,
            "unit_count": unsupervised,
            "carry_only": True,
            "executed_projection_flops": carry_flops,
        }
    )
    parameters = sum(
        row["dense_parameters"] for row in inventory["projection_inventory"]
    )
    forward_flops = 2 * tokens * parameters
    executed_flops = sum(row["executed_projection_flops"] for row in stages)
    observed_forward = sum(
        row["executed_projection_flops"] for row in stages if row["stage"] == "forward"
    )
    assert observed_forward == forward_flops
    input_gradient_flops = sum(
        row["calls"] * row["flops"] for row in recipes.values() if not row["transpose"]
    )
    replay_flops = executed_flops - forward_flops - input_gradient_flops
    last_layer = layers - 1
    last = [
        r
        for r in inventory["projection_inventory"]
        if r["name"].startswith(f"layers.{last_layer}.")
    ]
    assert len(last) == 7
    sparse_parameters = sum(
        row["dense_parameters"]
        for row in last
        if row["name"].split(".")[-1] not in ("k_proj", "v_proj")
    )
    minimum_masked_flops = (
        4 * tokens * parameters
        - 4 * (tokens - layout["supervised_tokens"]) * sparse_parameters
    )
    measured = profile["measured"]["whole_update_seconds"]
    totals = profile["profile"]["totals"]
    disjoint = [
        f"{stage}:{kind}"
        for stage in ("forward", "vjp_total")
        for kind in ("gdn", "attention")
    ]
    attributed = sum(totals[key]["seconds"] for key in disjoint)
    result = {
        "status": "plan_ready",
        "path": profile["path"],
        "model_config": config,
        "checkpoint_layers": 1,
        "environment": {"MLX_ENABLE_TF32": "0"},
        "frozen_projection_parameters": parameters,
        "full_row_forward_and_input_gradient_reference_flops": 2 * forward_flops,
        "forward_and_input_gradient_graph_flops": forward_flops + input_gradient_flops,
        "minimum_projection_flops_with_sparse_last_layer": minimum_masked_flops,
        "input_gradient_projection_flops": input_gradient_flops,
        "executed_projection_flops": executed_flops,
        "recomputed_projection_flops": replay_flops,
        "unsupervised_final_layer_carry_only_units": unsupervised,
        "forward_equivalents_executed": executed_flops / forward_flops,
        "stages": stages,
        "activation_memory": {
            "saved_bf16_hidden_boundary_bytes": 2
            * tokens
            * config["hidden_size"]
            * layers,
            "two_bf16_mlp_intermediates_all_layers_bytes": 4
            * tokens
            * config["intermediate_size"]
            * layers,
            "physical_memory_bytes": inventory["device"]["memory_size"],
            "note": "The MLP estimate excludes the base, hidden boundaries, carries, optimizer, and scratch.",
        },
        "attention": _attention_work(config, tokens, layout),
        "recipes": [{"id": key, **row} for key, row in sorted(recipes.items())],
        "whole_update_seconds": measured,
        "disjoint_layer_seconds": {key: totals[key]["seconds"] for key in disjoint},
        "layer_attributed_fraction": attributed / measured,
        "other_update_seconds": measured - attributed,
        "limits": [
            "The graphs use CPU-default lazy construction and do not measure Metal dispatch.",
            "Quantized matrix shapes and dtypes are verified; CPU norm helper graphs are not timed.",
            "Executed FLOPs count graph-level dense equivalents, not physical quantization or padded instructions.",
            "Projection traffic counts logical packed weights, metadata, inputs and outputs.",
            "Warm isolated weight reuse can overstate the attainable rate in a complete model.",
            "GDN recurrence, attention, LoRA, norms, layout, accumulation and host overhead remain separate.",
        ],
        "evidence_sha256": {str(p): _digest(p) for p in sources},
    }
    _save(args.output, result)


def supervise(args: argparse.Namespace) -> None:
    """
    Purpose: Hold the exclusive lease and enforce one worker deadline.
    Input: A fresh output folder and bounded arguments. Output: Execution receipts.
    Note: The supervisor terminates only its own worker process group.
    """
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__)
    (output / "source.py").write_bytes(source.read_bytes())
    receipt = {
        "status": "running",
        "started_at": time.time(),
        "timeout_seconds": args.timeout_seconds,
        "supervisor_pid": os.getpid(),
        "source_sha256": _digest(source),
        "recipes_sha256": _digest(args.recipes),
        "swap_before": subprocess.check_output(
            ["sysctl", "vm.swapusage"], text=True
        ).strip(),
    }
    child = None
    with Path(args.lease).open("r") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            command = [sys.executable, "-u", str(source), *sys.argv[1:], "--worker"]
            with (output / "worker.log").open("w") as log:
                child = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                receipt["worker_pid"] = child.pid
                _save(output / "execution.json", receipt)
                child.wait(timeout=args.timeout_seconds)
                receipt.update(
                    status="passed" if child.returncode == 0 else "failed",
                    returncode=child.returncode,
                )
        except BaseException as error:
            receipt.update(
                status="failed", error_type=type(error).__name__, error=str(error)
            )
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
            raise
        finally:
            receipt.update(
                finished_at=time.time(),
                swap_after=subprocess.check_output(
                    ["sysctl", "vm.swapusage"], text=True
                ).strip(),
            )
            _save(output / "execution.json", receipt)


def calibrate(args: argparse.Namespace) -> None:
    """
    Purpose: Measure actual forward and input-gradient quantized matrix shapes.
    Input: Operation recipes. Output: Warm timing distributions and logical traffic.
    Note: The worker uses synthetic values and never loads model assets.
    """
    assert os.environ.get("MLX_ENABLE_TF32") == "0"
    import mlx.core as mx

    assert mx.default_device() == mx.gpu
    mx.set_cache_limit(2 << 30)
    mx.set_memory_limit(64 << 30)
    mx.random.seed(20261018)
    recipe_plan = _read(args.recipes)
    output = Path(args.output)
    report = {
        "status": "running",
        "started_at": time.time(),
        "device": mx.device_info(),
        "recipes_sha256": _digest(args.recipes),
        "environment": {"MLX_ENABLE_TF32": "0"},
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "scope": "Synthetic BF16 8-bit-affine/group64 operations, including input-gradient transpose=False and a dequantized BF16 GEMM reference.",
        "timing_scope": "Host graph construction plus synchronized evaluation; inputs materialized first.",
        "cases": [],
    }
    for recipe in recipe_plan["recipes"]:
        shape = recipe["weight_shape"]
        weight_NK = (mx.random.normal(shape) * 0.01).astype(mx.bfloat16)
        packed_NP, scales_NG, biases_NG = mx.quantize(
            weight_NK, group_size=64, bits=8, mode="affine"
        )
        x_BMK = mx.random.normal(recipe["input_shape"]).astype(mx.bfloat16)
        mx.eval(packed_NP, scales_NG, biases_NG, x_BMK)
        del weight_NK
        gc.collect()
        mx.clear_cache()
        weight_bytes = sum(x.nbytes for x in (packed_NP, scales_NG, biases_NG))

        operation = partial(
            mx.quantized_matmul,
            x_BMK,
            packed_NP,
            scales_NG,
            biases_NG,
            transpose=recipe["transpose"],
            group_size=64,
            bits=8,
            mode="affine",
        )

        timing = _measure_operation(mx, operation, args.warmups, args.repetitions)
        median_s = timing["median_seconds"]
        quantized_peak = mx.get_peak_memory()
        dequantized_NK = mx.dequantize(
            packed_NP, scales_NG, biases_NG, group_size=64, bits=8, mode="affine"
        )
        mx.eval(dequantized_NK)
        dense_weight_KN = (
            mx.swapaxes(dequantized_NK, -1, -2)
            if recipe["transpose"]
            else dequantized_NK
        )
        dense_operation = partial(mx.matmul, x_BMK, dense_weight_KN)
        dense_timing = _measure_operation(
            mx, dense_operation, args.warmups, args.repetitions
        )
        dense_peak = mx.get_peak_memory()
        native_BMN, dense_BMN = operation(), dense_operation()
        mx.eval(native_BMN, dense_BMN)
        assert list(native_BMN.shape) == list(dense_BMN.shape) == recipe["output_shape"]
        assert native_BMN.dtype == dense_BMN.dtype == mx.bfloat16
        assert bool(mx.isfinite(native_BMN).all().item())
        assert bool(mx.isfinite(dense_BMN).all().item())
        difference_BMN = dense_BMN.astype(mx.float32) - native_BMN.astype(mx.float32)
        error = {
            "bitwise_equal": bool(mx.array_equal(native_BMN, dense_BMN).item()),
            "maximum_absolute_error": float(mx.max(mx.abs(difference_BMN)).item()),
            "relative_l2_error": float(
                mx.sqrt(
                    mx.sum(difference_BMN * difference_BMN)
                    / mx.maximum(
                        mx.sum(mx.square(native_BMN.astype(mx.float32))), 1e-30
                    )
                ).item()
            ),
            "qualification": "Component diagnostic only; full-model gradients and Adam parity remain required.",
        }
        logical_bytes = (
            weight_bytes + x_BMK.nbytes + 2 * math.prod(recipe["output_shape"])
        )
        row = {
            "id": recipe["id"],
            "transpose": recipe["transpose"],
            "input_shape": recipe["input_shape"],
            "output_shape": recipe["output_shape"],
            "weight_shape": shape,
            "flops": recipe["flops"],
            "logical_bytes": logical_bytes,
            "stored_weight_bytes": weight_bytes,
            **timing,
            "dense_equivalent_tflops": recipe["flops"] / median_s / 1e12,
            "logical_gb_s": logical_bytes / median_s / 1e9,
            "peak_active_memory_bytes": quantized_peak,
            "dequantized_bf16_reference": {
                **dense_timing,
                "dense_tflops": recipe["flops"] / dense_timing["median_seconds"] / 1e12,
                "weight_cache_bytes": dequantized_NK.nbytes,
                "peak_active_memory_bytes": dense_peak,
                "speedup": median_s / dense_timing["median_seconds"],
                "output_comparison": error,
            },
        }
        report["cases"].append(row)
        _save(output / "projections.json", report)
        print(json.dumps(row), flush=True)
        del operation, dense_operation, dense_weight_KN, dequantized_NK
        del native_BMN, dense_BMN, difference_BMN
        del x_BMK, packed_NP, scales_NG, biases_NG
        gc.collect()
        mx.clear_cache()
    report.update(status="passed", finished_at=time.time())
    _save(output / "projections.json", report)


def summarize(args: argparse.Namespace) -> None:
    """
    Purpose: Sum matched isolated operation times using actual graph multiplicities.
    Input: A recipe plan and completed calibration. Output: A partial time budget.
    Note: A sum of isolated times is a reference, not an achieved update prediction.
    """
    recipe_plan, measurements = _read(args.recipes), _read(args.measurements)
    assert measurements["status"] == "passed"
    assert measurements["recipes_sha256"] == _digest(args.recipes)
    actual = {r["id"]: r for r in measurements["cases"]}
    stages: Counter[str] = Counter()
    total = dense_total = 0.0
    for recipe in recipe_plan["recipes"]:
        row = actual[recipe["id"]]
        total += recipe["calls"] * row["median_seconds"]
        dense_total += (
            recipe["calls"] * row["dequantized_bf16_reference"]["median_seconds"]
        )
        for use in recipe["uses"]:
            stages[f"{use['stage']}:{use['kind']}"] += (
                use["calls"] * row["median_seconds"]
            )
    _save(
        args.output,
        {
            "status": "partial_projection_budget",
            "projection_reference_seconds": total,
            "dequantized_projection_reference_seconds": dense_total,
            "hypothetical_dequantized_cache_bytes": 2
            * recipe_plan["frozen_projection_parameters"],
            "dequantized_maximum_component_relative_error": max(
                row["dequantized_bf16_reference"]["output_comparison"][
                    "relative_l2_error"
                ]
                for row in actual.values()
            ),
            "dequantized_cache_qualified": False,
            "stages_reference_seconds": dict(stages),
            "whole_update_seconds": recipe_plan["whole_update_seconds"],
            "executed_projection_flops": recipe_plan["executed_projection_flops"],
            "weighted_projection_tflops": recipe_plan["executed_projection_flops"]
            / total
            / 1e12,
            "limits": recipe_plan["limits"],
            "evidence_sha256": {
                str(p): _digest(p) for p in (args.recipes, args.measurements)
            },
        },
    )


def _measure_operation(
    mx: Any, operation: Any, warmups: int, repetitions: int
) -> dict[str, Any]:
    """
    Purpose: Measure materialized operations with separate graph and evaluation times.
    Input: One operation and repetition counts. Output: Raw times and their median.
    """
    for _ in range(warmups):
        result_BMN = operation()
        mx.eval(result_BMN)
    del result_BMN
    mx.reset_peak_memory()
    build_seconds, eval_seconds = [], []
    for _ in range(repetitions):
        started = time.perf_counter()
        result_BMN = operation()
        built = time.perf_counter()
        mx.eval(result_BMN)
        finished = time.perf_counter()
        build_seconds.append(built - started)
        eval_seconds.append(finished - built)
        del result_BMN
    durations = [a + b for a, b in zip(build_seconds, eval_seconds, strict=True)]
    return {
        "seconds": durations,
        "graph_build_seconds": build_seconds,
        "eval_seconds": eval_seconds,
        "median_seconds": statistics.median(durations),
    }


def _attention_work(
    config: dict[str, Any], tokens: int, layout: dict[str, Any]
) -> dict[str, Any]:
    """
    Purpose: Separate useful causal work from query-tile rectangle arithmetic.
    Input: Actual model dimensions and replay length. Output: FLOP accounting.
    Note: Fused forward kernels can skip causal cells and use padded hardware tiles.
    """
    heads, width = config["num_attention_heads"], config["head_dim"]
    layers = config["num_hidden_layers"] // config["full_attention_interval"]
    tiles = []
    for offset in range(0, tokens, 2048):
        length = min(2048, tokens - offset)
        span = offset + length
        tile = max(16, min(length, (2 << 30) // (heads * span * 4)))
        for start in range(0, length, tile):
            stop = min(length, start + tile)
            tiles.append(
                {
                    "segment": offset // 2048,
                    "queries": stop - start,
                    "keys": offset + stop,
                }
            )
    pairs = sum(row["queries"] * row["keys"] for row in tiles)
    skipped_units = {r["segment"] for r in layout["segments"] if r["head_rows"] == 0}
    skipped_pairs = sum(
        row["queries"] * row["keys"] for row in tiles if row["segment"] in skipped_units
    )
    checkpointed = (
        18 * layers * heads * width * pairs - 14 * heads * width * skipped_pairs
    )
    direct = 16 * layers * heads * width * pairs - 12 * heads * width * skipped_pairs
    return {
        "full_row_causal_forward_and_backward_reference_flops": 12
        * layers
        * heads
        * width
        * tokens
        * (tokens + 1)
        // 2,
        "minimum_masked_last_layer_attention_gemm_flops": 12
        * heads
        * width
        * (
            (layers - 1) * tokens * (tokens + 1) // 2
            + layout["supervised_causal_pairs"]
        ),
        "final_layer_carry_only_rectangle_pairs": skipped_pairs,
        "query_tiles_per_layer": len(tiles),
        "rectangle_to_causal_pair_ratio": pairs / (tokens * (tokens + 1) // 2),
        "fallback_equivalent_with_inner_checkpoint_flops": checkpointed,
        "fallback_equivalent_without_inner_checkpoint_flops": direct,
        "tiles": tiles,
        "limits": [
            "Fallback graph uses two forward and seven VJP matmuls per tile with the inner checkpoint.",
            "Removing the inner checkpoint reduces VJP from seven matmuls to six.",
            "Final-layer carry-only units omit attention VJP matrix products; the forward still executes.",
            "These totals exclude masks, softmax, layout, precision conversions, and quantization.",
            "Fused Metal forward work can differ from CPU fallback-equivalent arithmetic.",
        ],
    }


def _recipe(matrix: dict[str, Any]) -> dict[str, Any]:
    """
    Purpose: Validate one typed quantized projection and remove graph identities.
    Input: Matrix metadata. Output: An executable shape and precision recipe.
    """
    inputs, outputs = matrix["inputs"], matrix["outputs"]
    group, bits, mode, transpose = matrix["arguments"]
    assert (group, bits, mode) == (64, 8, 0)
    assert all(
        item[2] == "bfloat16" for item in [inputs[0], inputs[2], inputs[3], outputs[0]]
    )
    assert inputs[1][2] == "uint32"
    weight_shape = [inputs[1][1][0], inputs[1][1][1] * 4]
    width_out, width_in = weight_shape
    contracted = width_in if transpose else width_out
    produced = width_out if transpose else width_in
    assert inputs[0][1][-1] == contracted
    assert outputs[0][1] == inputs[0][1][:-1] + [produced]
    assert inputs[2][1] == inputs[3][1] == [width_out, width_in // group]
    assert matrix["flops"] == 2 * math.prod(outputs[0][1]) * contracted
    return {
        "input_shape": inputs[0][1],
        "output_shape": outputs[0][1],
        "weight_shape": weight_shape,
        "transpose": transpose,
        "flops": matrix["flops"],
    }


def _recipe_key(recipe: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    plan_parser = sub.add_parser(
        "plan", help="Count actual projection operations without MLX."
    )
    for name in (
        "regular",
        "terminal",
        "carry_only",
        "layout",
        "inventory",
        "profile",
        "output",
    ):
        plan_parser.add_argument(
            "--" + name.replace("_", "-"),
            type=Path,
            required=True,
            help=f"The {name} evidence path.",
        )
    measure_parser = sub.add_parser(
        "calibrate", help="Measure matched projections under a hard supervisor."
    )
    measure_parser.add_argument(
        "--recipes", type=Path, required=True, help="Exact operation recipe JSON."
    )
    measure_parser.add_argument(
        "--output", type=Path, required=True, help="Private measurement output folder."
    )
    measure_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=600,
        help="Hard job limit, at most one hour.",
    )
    measure_parser.add_argument(
        "--lease",
        default=str(Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"),
        help="Existing GPU lease file.",
    )
    measure_parser.add_argument(
        "--warmups", type=int, default=2, help="Materialized warmups per operation."
    )
    measure_parser.add_argument(
        "--repetitions", type=int, default=7, help="Measured repetitions per operation."
    )
    measure_parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    summary_parser = sub.add_parser(
        "summarize", help="Sum a partial projection time budget."
    )
    for name in ("recipes", "measurements", "output"):
        summary_parser.add_argument(
            "--" + name.replace("_", "-"),
            type=Path,
            required=True,
            help=f"The {name} evidence path.",
        )
    args = parser.parse_args()
    if args.mode == "plan":
        plan(args)
    elif args.mode == "summarize":
        summarize(args)
    else:
        assert (
            0 < args.timeout_seconds <= 3600
            and args.warmups > 0
            and args.repetitions >= 3
        )
        if args.worker:
            calibrate(args)
        else:
            supervise(args)
