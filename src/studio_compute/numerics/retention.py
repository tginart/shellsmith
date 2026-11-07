"""
Purpose: Measure projection costs and select residuals for one replay path.
Input: The current model, path layout, and admitted optional activation bytes.
Output: Stable retention choices, measured costs, and exact planned tensor bytes.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import mlx.core as mx

from ..activation_budget import RetentionGroup, select_residuals
from .saved import _projections

# B: batch, L: rows, D: projection input, F: projection output.


def mandatory_saved_bytes(layers: list, length: int) -> int:
    """Count compact LoRA residuals and saved attention outputs without KV duplication."""
    total = 0
    for layer in layers:
        total += (
            length * 8
        )  # Two BF16 statistic pairs for the input and post-mixer norms.
        total += sum(
            module.a.shape[0] * length * 2 for _, module in _projections(layer)
        )
        if not layer.is_linear:
            attention = layer.self_attn
            total += (
                length * attention.num_attention_heads * (attention.head_dim * 2 + 4)
            )
    return total


def plan_retention(
    layers: list,
    length: int,
    segment_tokens: int,
    budget_bytes: int,
    costs: dict,
    checkpoint_layers: int,
) -> tuple[dict, dict]:
    """
    Purpose: Admit optional residuals from measured recomputation costs per unique byte.
    Input: A longest live branch and a shared shape-cost cache.
    Output: Per-unit choices and a reproducible cost-model report.
    Note: Branches reuse the same budget after their last backward consumers finish.
    """
    started = time.perf_counter()
    row_counts = sorted(
        {
            min(segment_tokens, length - start)
            for start in range(0, length, segment_tokens)
        }
    )
    for count in row_counts:
        _calibrate(layers, count, costs)
    candidates = []
    for segment, start in enumerate(range(0, length, segment_tokens)):
        count = min(segment_tokens, length - start)
        for index, layer in enumerate(layers):
            prefix = f"{segment}.{index}."
            projections = dict(_projections(layer))
            for name, module in projections.items():
                if name.endswith((".out_proj", ".o_proj", ".down_proj", ".v_proj")):
                    # These primal outputs are unnecessary in checkpoint-one backward.
                    if checkpoint_layers == 1 or not name.endswith(".down_proj"):
                        continue
                width = module.base.weight.shape[0]
                key = _shape(module, count)
                cost = costs[key]
                if name.endswith(".down_proj"):
                    cost += sum(
                        costs[_shape(projections[n], count)]
                        for n in ("mlp.gate_proj", "mlp.up_proj")
                    )
                identity = prefix + name
                candidates.append(
                    RetentionGroup(identity, ((identity, count * width * 2),), cost)
                )
            mixer = "linear_attn." if layer.is_linear else "self_attn."
            output_name = mixer + ("out_proj" if layer.is_linear else "o_proj")
            cost = costs[_shape(projections[output_name], count)]
            if layer.is_linear:
                cost += costs[_inner_shape(layer, count)]
            identity = prefix + "middle"
            candidates.append(
                RetentionGroup(
                    identity,
                    ((identity, count * layer.input_layernorm.weight.size * 2),),
                    cost,
                )
            )
    plan = select_residuals(candidates, budget_bytes)
    choices = {}
    for identity in plan.selected:
        segment, index, name = identity.split(".", 2)
        choices.setdefault((int(segment), int(index)), set()).add(name)
    choices = {key: frozenset(value) for key, value in choices.items()}
    report = {
        "budget_bytes": budget_bytes,
        "retained_bytes": plan.used_bytes,
        "estimated_recomputation_ns_saved": plan.saved_ns,
        "calibration_and_plan_seconds": time.perf_counter() - started,
        "projection_costs_ns": {
            str(key): value for key, value in sorted(costs.items())
        },
        "selection": "Measured projection nanoseconds per unique BF16 output byte; stable identity breaks ties.",
        "middle_cost_scope": "Measured output projection plus the GDN inner forward; raw input projections remain needed for derivatives.",
        "selected_groups": len(plan.selected),
    }
    return choices, report


def _calibrate(layers: list, rows: int, costs: dict) -> None:
    """
    Purpose: Measure representative current frozen projections with exact input shapes.
    Input: Layers and a caller-owned shape cache. Output: Median warm nanoseconds.
    Note: The function does not update model tensors or random state.
    """
    for layer in layers:
        for _, module in _projections(layer):
            key = _shape(module, rows)
            if key in costs:
                continue
            x_BLD = mx.ones((1, rows, module.a.shape[1]), mx.bfloat16)
            mx.eval(x_BLD)
            times = []
            for iteration in range(5):
                started = time.perf_counter_ns()
                value_BLF = module.base(x_BLD)
                mx.eval(value_BLF)
                if iteration >= 2:
                    times.append(time.perf_counter_ns() - started)
                del value_BLF
            costs[key] = int(statistics.median(times))
            del x_BLD
        if layer.is_linear and _inner_shape(layer, rows) not in costs:
            from .bf16 import gdn_inner
            from .ops import initial_states

            linear = layer.linear_attn
            x_BLD = mx.ones((1, rows, linear.hidden_size), mx.bfloat16)
            raw = [
                getattr(linear, name).base(x_BLD)
                for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")
            ]
            states = [
                value
                for _, value in initial_states(
                    [layer], mx.bfloat16, recurrent_dtype=mx.bfloat16
                )[0]
            ]
            mx.eval(raw, states)
            times = []
            for iteration in range(5):
                started = time.perf_counter_ns()
                result = gdn_inner(linear, *raw, *states)
                mx.eval(result)
                if iteration >= 2:
                    times.append(time.perf_counter_ns() - started)
                del result
            costs[_inner_shape(layer, rows)] = int(statistics.median(times))
            del x_BLD, raw, states


def _inner_shape(layer: Any, rows: int) -> tuple:
    return rows, -1, layer.linear_attn.num_v_heads * layer.linear_attn.head_v_dim


def _shape(module: Any, rows: int) -> tuple:
    return rows, module.a.shape[1], module.base.weight.shape[0]
