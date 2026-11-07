"""
Purpose: Measure the error of the Flash training kernels against FP32 references,
    on real activations.
Input: A reconstructed update, one sequence index, the model, and a layer count.
Output: A JSON file. For each captured call, it gives the relative L2 error of
    the BF16 reference path and of the new training kernel, against FP32.
Method:
    1. Take the GPU lease and load the model as the replay engine does.
    2. Run the first layers of one 2048-token segment with the MLX reference
       paths. Keep the inputs of each hyper-connection and each GDN chunk scan.
    3. Hyper-connection: compare the upstream module and hyper_connection with
       an FP32 copy that uses dequantized dense weights. The check uses the
       outputs and the input gradient for random output cotangents.
    4. GDN: compare the substitution solve ("metal"), the explicit inverse
       ("metal_inverse"), and the fused chunk kernels ("metal_fused") with the
       FP32 reference backend. All paths start from the gate logits, so the
       check includes the gate math. It uses the outputs and the gradients of
       q, k, v, both logits, and the state.
Note: A gradient comparison of two BF16 sources shows only that they differ.
    This check shows which one is closer to the FP32 value.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path

LEASE = Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"
SEED = 20261026


def main() -> None:
    args = _parse_args()
    lease = LEASE.open("a")
    fcntl.flock(lease, fcntl.LOCK_EX)  # Wait in line for the shared lease.
    import mlx.core as mx

    lm = _load_language_model(args.model)
    ids = json.loads(args.sequences.read_text())["sequences"][args.index]
    ids = (ids["prompt_token_ids"] + ids["sampled_tokens"])[: args.tokens]
    hypers, scans = _capture(lm, mx.array([ids], mx.int32), args.layers)
    mx.random.seed(SEED)
    report = {
        "index": args.index, "tokens": len(ids), "layers": args.layers,
        "hyper": [_check_hyper(module, hyper_BLW) for module, hyper_BLW in hypers],
        "gdn": [_check_gdn(inputs) for inputs in scans],
    }
    report["summary"] = _summarize(report)
    args.output.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["summary"], indent=1))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, required=True,
                        help="Reconstructed update JSON.")
    parser.add_argument("--model", type=Path, required=True, help="Flash model path.")
    parser.add_argument("--output", type=Path, required=True, help="Report JSON.")
    parser.add_argument("--index", type=int, default=22, help="Sequence index.")
    parser.add_argument("--tokens", type=int, default=2048, help="Segment length.")
    parser.add_argument("--layers", type=int, default=6,
                        help="Run and check this many first layers.")
    return parser.parse_args()


def _load_language_model(path: Path):
    """Build the replay engine and return its Flash language model."""
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    from studio_compute.flash_model import QwenFlashModel
    from studio_compute.runtime import RuntimeEngine

    engine = RuntimeEngine(
        str(path), model_factory=QwenFlashModel, execution_profile="bf16-fast",
        headroom_gib=32.0, activation_budget_gib=16, inference_concurrency=8,
        prefill_tokens=1024, segment_tokens=2048, head_rows=128,
        replay_layout="segment_trie", training_moe_chunk=32, prefill_moe_chunk=32,
    )
    return engine.resident.lm


def _capture(lm, ids_BL, layers: int) -> tuple:
    """
    Purpose: Run the first layers with the reference paths and keep kernel inputs.
    Input: The language model, token IDs (1, L), and the layer count.
    Output: (hyper-connection calls, GDN chunk scan calls) as evaluated arrays.
    """
    import mlx.core as mx
    from studio_compute.numerics import bf16, flash_ops, ops

    hypers, scans, gates = [], [], []
    module_call, scan = flash_ops._module_call, ops.gated_delta_chunked
    inner = bf16.gdn_inner

    def keep_module_call(module, hidden_BLD):
        hypers.append((module, mx.stop_gradient(hidden_BLD)))
        return module_call(module, hidden_BLD)

    def keep_scan(*arguments, **options):
        scans.append([mx.stop_gradient(value) for value in arguments[:6]])
        return scan(*arguments, **options)

    def keep_inner(linear, qkv, z, a_BLH, b_BLH, *rest, **options):
        gates.append([a_BLH, b_BLH, linear.A_log, linear.dt_bias])
        return inner(linear, qkv, z, a_BLH, b_BLH, *rest, **options)

    flash_ops._module_call, ops.gated_delta_chunked = keep_module_call, keep_scan
    bf16.gdn_inner = keep_inner
    try:
        hidden_BLD = flash_ops.FlashOps.expand(
            lm, lm.model.embed_tokens(ids_BL).astype(mx.bfloat16))
        length = ids_BL.shape[1]
        positions = mx.broadcast_to(mx.arange(length)[None, None], (3, 1, length))
        context = (ids_BL, ids_BL[:, :0])
        initial = flash_ops.FlashOps.initial_states(lm.layers[:layers], mx.bfloat16)
        for layer, states in zip(lm.layers[:layers], initial):
            hidden_BLD, *_ = flash_ops.FlashOps.layer_math(
                layer, hidden_BLD, [value for _, value in states], positions, context)
            mx.eval(hidden_BLD, [value for _, value in hypers[-2:]],
                    scans[-1:] if layer.is_linear else [],
                    gates[-1:] if layer.is_linear else [])
    finally:
        flash_ops._module_call, ops.gated_delta_chunked = module_call, scan
        bf16.gdn_inner = inner
    calls = []
    for (q, k, v, _, _, state), (a, b, a_log, dt_bias) in zip(scans, gates,
                                                             strict=True):
        calls.append([q, k, v, a, b, a_log, dt_bias, state])
    return hypers, calls


def _check_hyper(module, hyper_BLW) -> dict:
    """Compare both BF16 hyper-connection paths with the FP32 dense copy."""
    import mlx.core as mx
    from studio_compute.numerics.hyper_metal import hyper_connection

    def outputs(function, value_BLW):
        mixed_BLD, _, injection_BLG = function(value_BLW)
        return [mixed_BLD, injection_BLG]

    shapes = [(*hyper_BLW.shape[:2], module.hidden_size),
              (*hyper_BLW.shape[:2], module.hc_count)]
    cotangents = [mx.random.normal(shape) for shape in shapes]
    reference = mx.vjp(lambda x: _reference_hyper(module, x),
                       [hyper_BLW.astype(mx.float32)], cotangents)
    paths = {
        "upstream": lambda x: outputs(module, x),
        "kernel": lambda x: outputs(lambda y: hyper_connection(module, y), x),
    }
    result = {}
    for name, function in paths.items():
        value = mx.vjp(function, [hyper_BLW],
                       [cotangent.astype(mx.bfloat16) for cotangent in cotangents])
        result[name] = _errors(value, reference, ["mixed", "injection"], ["dx"])
    return result


def _reference_hyper(module, hyper_BLW) -> list:
    """Return (mixed, injection weights) of one hyper-connection, all in FP32."""
    import mlx.core as mx
    import mlx.nn as nn

    count, width = module.hc_count, module.hidden_size
    norm = module.hc_norm
    groups_BLGD = hyper_BLW.reshape(*hyper_BLW.shape[:2], count, width)
    groups_BLGD = groups_BLGD * mx.rsqrt(
        mx.mean(groups_BLGD * groups_BLGD, axis=-1, keepdims=True) + norm.eps)
    groups_BLGD = groups_BLGD * (1.0 + norm.weight.astype(mx.float32).reshape(count, width))
    normed_BLW = groups_BLGD.reshape(hyper_BLW.shape)
    mix_BLR = nn.silu((normed_BLW @ _dense(module.input_mix_weight_down).T) / count)
    projected_BLW = mix_BLR @ _dense(module.input_mix_weight_up).T
    gates_BLGD = mx.sigmoid(projected_BLW).reshape(groups_BLGD.shape)
    mixed_BLD = mx.mean(gates_BLGD * groups_BLGD, axis=-2)
    injection_BLG = normed_BLW @ _dense(module.block_inject_weight).T
    return [mixed_BLD, 2 * mx.sigmoid(injection_BLG / count)]


def _dense(linear):
    """Return the FP32 dense weight of one quantized linear layer."""
    import mlx.core as mx

    return mx.dequantize(linear.weight, linear.scales, linear.biases,
                         group_size=linear.group_size, bits=linear.bits,
                         mode=linear.mode).astype(mx.float32)


def _check_gdn(inputs: list) -> dict:
    """
    Purpose: Compare the three BF16 GDN chunk paths with the FP32 reference.
    Input: q, k, v, the gate logits a and b, A_log, dt_bias, and the state.
    Output: The relative errors of each path, by output and gradient name.
    Note: The BF16 scan paths make the gates in BF16, as bf16.gdn_inner does.
        The fused path makes them in FP32 inside the kernel.
    """
    import mlx.core as mx
    from studio_compute.numerics.gdn_fused import gated_delta_chunks

    q, k, v, a, b, a_log, dt_bias, state = inputs
    length, heads, width = v.shape[1], v.shape[2], v.shape[3]
    cotangents = [mx.random.normal((1, length, heads, width)),
                  0.1 * mx.random.normal(state.shape)]
    primals = [0, 1, 2, 3, 4, 7]

    def run(function, values):
        def wrapped(*chosen):
            full = list(values)
            for index, value in zip(primals, chosen):
                full[index] = value
            return list(function(*full))

        seeds = [cotangent.astype(values[0].dtype) for cotangent in cotangents]
        return mx.vjp(wrapped, [values[index] for index in primals], seeds)

    fp32 = [value.astype(mx.float32) for value in inputs]
    reference = run(_reference_gdn, fp32)
    paths = {
        "metal": lambda *values: _bf16_gdn(*values, backend="metal"),
        "metal_inverse": lambda *values: _bf16_gdn(*values, backend="metal_inverse"),
        "metal_fused": gated_delta_chunks,
    }
    names = ["dq", "dk", "dv", "da", "db", "dstate"]
    return {name: _errors(run(function, inputs), reference, ["y", "state"], names)
            for name, function in paths.items()}


def _reference_gdn(q, k, v, a, b, a_log, dt_bias, state) -> tuple:
    """Return the FP32 gated delta rule with the gates made in FP32."""
    import mlx.core as mx
    import mlx.nn as nn
    from studio_compute.numerics.ops import gated_delta_chunked

    log_gate = mx.maximum(-mx.exp(a_log) * nn.softplus(a + dt_bias), math.log(1e-6))
    group = v.shape[2] // q.shape[2]
    return gated_delta_chunked(
        mx.repeat(q, group, -2), mx.repeat(k, group, -2), v, mx.exp(log_gate),
        mx.sigmoid(b), state, compute_dtype=mx.float32, backend="reference")


def _bf16_gdn(q, k, v, a, b, a_log, dt_bias, state, *, backend: str) -> tuple:
    """Return the BF16 scan path of bf16.gdn_inner for one chunk backend."""
    import mlx.core as mx
    import mlx.nn as nn
    from studio_compute.numerics.ops import gated_delta_chunked

    g = mx.exp(-mx.exp(a_log.astype(mx.bfloat16))
               * nn.softplus(a + dt_bias.astype(mx.bfloat16)))
    return gated_delta_chunked(q, k, v, g, mx.sigmoid(b), state,
                               compute_dtype=mx.bfloat16, backend=backend)


def _errors(value: tuple, reference: tuple, outputs: list, gradients: list) -> dict:
    """Return the relative L2 error of each output and gradient against FP32."""
    import mlx.core as mx

    errors = {}
    pairs = list(zip(outputs, value[0], reference[0]))
    pairs += list(zip(gradients, value[1], reference[1]))
    for name, actual, expected in pairs:
        actual, expected = actual.astype(mx.float32), expected.astype(mx.float32)
        difference = float(mx.sum((actual - expected) ** 2).item())
        norm = float(mx.sum(expected ** 2).item())
        errors[name] = math.sqrt(difference / norm) if norm else 0.0
    return errors


def _summarize(report: dict) -> dict:
    """Return the mean error per kernel family, path, and quantity."""
    summary = {}
    for family in ("hyper", "gdn"):
        totals = {}
        for call in report[family]:
            for path, errors in call.items():
                for name, error in errors.items():
                    totals.setdefault(f"{path}.{name}", []).append(error)
        summary[family] = {key: sum(values) / len(values)
                           for key, values in totals.items()}
    return summary


if __name__ == "__main__":
    main()
