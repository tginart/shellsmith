"""
Purpose: Audit actual-model BF16 intermediate dtypes and owned kernel dispatch.
Input: The approved 27B model, a pinned-runtime audit extension, and a private output.
Output: Graph facts, unexpected FP32 outputs, dispatch receipts, and frozen-base checks.
Note: The command does not commit an optimizer update or change model assets.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import sys
import time

from mlx_hardware_roofline import supervise


def run(args: argparse.Namespace) -> None:
    import mlx.core as mx

    assert importlib.metadata.version("mlx") == "0.32.3"
    sys.path.insert(0, str(args.extension_dir))
    import _shellsmith_graph_audit as extension
    from studio_compute.runtime import RuntimeEngine
    from studio_compute.numerics import attention_metal, gdn_metal, gdn_token
    from studio_compute.numerics.saved import (
        forward_saved,
        backward_saved,
        _projections,
    )
    from studio_compute.numerics.ops import initial_states
    from studio_compute.numerics.cispo import frozen_head_logprobs, cispo_loss
    from studio_compute.numerics.bf16_adam import update
    from studio_compute.numerics.sums import BalancedSums

    report = {
        "status": "running",
        "stages": {},
        "dispatches": [],
        "started_at": time.time(),
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def audit(name, values, *, allow_native_attention=False):
        nodes = extension.inspect(list(values))
        counts = Counter(value["dtype"] for node in nodes for value in node["outputs"])
        unexpected = [
            node
            for node in nodes
            if any(
                value["dtype"] in ("float32", "float64") for value in node["outputs"]
            )
        ]
        report["stages"][name] = {
            "primitives": len(nodes),
            "output_dtype_counts": dict(counts),
            "unexpected_fp32_outputs": unexpected,
        }
        (args.output / (name + ".json")).write_text(json.dumps(nodes, indent=2) + "\n")
        save()
        if unexpected:
            raise RuntimeError(
                f"{name} has explicit wider floating-point graph outputs: "
                + str([node["primitive"] for node in unexpected[:5]])
            )
        if not allow_native_attention and any(
            "ScaledDotProductAttention" in node["primitive"] for node in nodes
        ):
            raise RuntimeError(f"{name} unexpectedly used a native attention fallback.")

    # Prove that the auditor detects an explicit promotion, without evaluation.
    probe = mx.ones((8,), mx.bfloat16).astype(mx.float32)
    assert any(
        value["dtype"] == "float32"
        for node in extension.inspect([probe])
        for value in node["outputs"]
    )
    report["detects_explicit_fp32_promotion"] = True
    original_attention = attention_metal._kernel
    original_gdn = gdn_metal._nax_kernel
    original_token = gdn_token._kernel

    def instrument(kind, factory):
        def get(phase):
            kernel = factory(phase)

            def call(**kwargs):
                report["dispatches"].append(
                    {
                        "kind": kind,
                        "phase": phase,
                        "input_dtypes": [
                            str(value.dtype) for value in kwargs["inputs"]
                        ],
                        "output_dtypes": [
                            str(value) for value in kwargs["output_dtypes"]
                        ],
                    }
                )
                return kernel(**kwargs)

            return call

        return get

    attention_metal._kernel = instrument("attention", original_attention)
    gdn_metal._nax_kernel = instrument("gdn", original_gdn)
    token_factory = instrument("gdn_token", lambda _: original_token())
    gdn_token._kernel = lambda: token_factory("forward")
    engine = None
    try:
        engine = RuntimeEngine(
            args.model,
            prefix_cache_gib=0,
        )
        assert engine.execution_profile == "bf16-fast", "The qualified default changed."
        engine.create("dtype-audit", {"seed": 23})
        entry = engine.clients["dtype-audit"]
        factors = {
            name: value + 0.001 if name.endswith(".b") else value
            for name, value in entry["adapter"].state().items()
        }
        entry["adapter"].load(factors)
        engine.resident.select(entry["adapter"].state())
        engine.resident.mode(True)
        before = engine.resident.frozen_digest()
        lm = engine.lm
        ids = mx.array([list(range(17))], mx.int32)
        hidden = lm.model.embed_tokens(ids).astype(mx.bfloat16)
        mx.eval(hidden)
        positions = mx.broadcast_to(
            mx.arange(17, dtype=mx.int32)[None, None], (3, 1, 17)
        )
        for index in (0, 3):
            layer = lm.layers[index]
            assert layer.is_linear == (index == 0)
            incoming = [
                value
                for _, value in initial_states(
                    [layer], mx.bfloat16, recurrent_dtype=mx.bfloat16
                )[0]
            ]
            output, first, second, tape = forward_saved(
                layer,
                hidden,
                incoming,
                positions,
                retain=frozenset(name for name, _ in _projections(layer)),
            )
            values = [
                output,
                first,
                second,
                *tape.retained.values(),
                *tape.low_rank.values(),
                tape.input_statistics,
                tape.middle_statistics,
            ]
            if tape.attention is not None:
                values.extend(tape.attention)
            audit(f"layer_{index}_forward", values)
            mx.eval(values, tape.middle_BLD)
            _, back = backward_saved(
                tape, [mx.ones_like(output), mx.ones_like(first), mx.ones_like(second)]
            )
            audit(f"layer_{index}_backward", back)
            mx.eval(back)
            del output, first, second, tape, values, back, incoming
        target = mx.array(list(range(17)), mx.int32)

        def loss(value):
            current = frozen_head_logprobs(lm, value, target, dtype=mx.bfloat16)
            return cispo_loss(current, [-2.0] * 17, [0.1] * 17, dtype=mx.bfloat16)

        value, gradient = mx.value_and_grad(loss)(hidden[0])
        audit("head_loss_and_backward", [value, gradient])
        mx.eval(value, gradient)
        parameter = next(iter(entry["optimizer"].master.values()))
        zero = mx.zeros_like(parameter)
        gradient = mx.ones_like(parameter)
        optimizer = update(
            parameter,
            zero,
            zero,
            gradient,
            mx.array([0.9, 0.95, 0.001, 0.2, 1e-8], mx.bfloat16),
            mx.array([23, 1, 19], mx.uint32),
        )
        audit("optimizer", optimizer)
        mx.eval(optimizer)
        sums = BalancedSums()
        for i in range(5):
            sums.add("gradient", mx.ones_like(parameter) * (i + 1) * 0.001)
        total = sums.total("gradient")
        audit("balanced_accumulation", [total])
        mx.eval(total)
        engine.resident.mode(False)
        cache = lm.make_cache()
        inference = engine._decode_logits(
            mx.array([[1]], mx.int32),
            position_ids=mx.zeros((3, 1, 1), mx.int32),
            cache=cache,
        )
        state_tensors = [tensor for item in cache for tensor in item.state]
        audit(
            "inference_token",
            [inference.logits, *state_tensors],
            allow_native_attention=True,
        )
        mx.eval(inference.logits, state_tensors)
        assert any(row["kind"] == "gdn_token" for row in report["dispatches"])
        report["frozen_base_unchanged"] = before == engine.resident.frozen_digest()
        assert report["frozen_base_unchanged"]
        attention = {
            row["phase"] for row in report["dispatches"] if row["kind"] == "attention"
        }
        gdn = {row["phase"] for row in report["dispatches"] if row["kind"] == "gdn"}
        assert attention == {"forward", "query", "key_value"}
        assert gdn == {"forward", "backward"}
        report.update(
            status="passed",
            runtime_identity=engine.identity,
            peak_mlx_bytes=mx.get_peak_memory(),
        )
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        attention_metal._kernel = original_attention
        gdn_metal._nax_kernel = original_gdn
        gdn_token._kernel = original_token
        if engine is not None:
            engine.close()
        report["finished_at"] = time.time()
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Approved frozen 27B model path."
    )
    parser.add_argument(
        "--extension-dir",
        type=Path,
        required=True,
        help="Fresh build directory of the pinned read-only auditor.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh private audit evidence directory.",
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=600, help="Hard limit, at most one hour."
    )
    parser.add_argument(
        "--lease",
        default=str(Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"),
        help="Existing exclusive Studio GPU lease.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    assert 0 < args.timeout_seconds <= 3600
    if args.worker:
        run(args)
    else:
        supervise(args, script=Path(__file__))
