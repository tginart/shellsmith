"""
Purpose: Compare routed-expert implementations on real Flash replay inputs.
Input: Reconstructed update sequences, one segment and layer, and the Flash model.
Output: Forward and VJP timings plus errors against an FP32 dequantized reference.
Note: The script owns the GPU lease. It never opens a service state root.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import time
from pathlib import Path

LEASE = Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"
VARIANTS = ("unsorted", "sorted", "chunked16", "chunked32", "chunked64",
            "dequant32")


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    profiler = _load_profiler()
    sequence = json.loads(args.sequences.read_text())["sequences"][args.sequence]
    lease = LEASE.open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    profiler._write_identity(args.output / "process.json")
    import mlx.core as mx
    from studio_compute.numerics import flash_ops

    engine = profiler._load_engine(args.model)
    captured = profiler._capture_segment(engine, sequence, args.segment)
    report = {"segment": args.segment, "layers": {}}
    for index in args.layers:
        layer = captured["replay"].layers[index]
        hidden, _ = captured["inputs"][index]
        moe_input, _, _ = layer.mlp_hyper_connection(hidden)
        mx.eval(moe_input)
        reference = _reference(layer.mlp, moe_input)
        rows = {"routing": _routing_stats(layer.mlp, moe_input)}
        for variant in args.variants:
            function = _variant(flash_ops, variant)
            rows[variant] = _measure(layer.mlp, function, moe_input, reference,
                                     repeats=args.repeats)
            print(json.dumps({"layer": index, "variant": variant,
                              **{k: v for k, v in rows[variant].items()}}),
                  flush=True)
        report["layers"][index] = rows
        _write_json(args.output / "moe.json", report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, required=True,
                        help="Reconstructed update JSON.")
    parser.add_argument("--model", type=Path, required=True, help="Flash model path.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--sequence", type=int, default=0, help="Sequence index.")
    parser.add_argument("--segment", type=int, default=5, help="Segment index.")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 3, 40],
                        help="Decoder layers to measure.")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS),
                        choices=VARIANTS, help="Implementations to measure.")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repeats.")
    return parser.parse_args()


def _load_profiler():
    path = Path(__file__).with_name("profile_flash_replay.py")
    spec = importlib.util.spec_from_file_location("profile_flash_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _variant(flash_ops, variant: str):
    if variant == "unsorted":
        return lambda mixer, x: flash_ops._moe(mixer, x, sort_rows=False)
    if variant == "sorted":
        return lambda mixer, x: flash_ops._moe(mixer, x, sort_rows=True)
    if variant.startswith("chunked"):
        chunk = int(variant.removeprefix("chunked"))
        return lambda mixer, x: flash_ops._moe_chunked(mixer, x, chunk=chunk)
    chunk = int(variant.removeprefix("dequant"))
    return lambda mixer, x: flash_ops._moe_chunked(mixer, x, chunk=chunk,
                                                   dequantize=True)


def _routing_stats(mixer, hidden) -> dict:
    import mlx.core as mx

    gates = mx.softmax(mixer.gate(hidden), axis=-1, precise=True)
    indices = mx.argpartition(gates, kth=-mixer.top_k, axis=-1)[..., -mixer.top_k:]
    counts = mx.zeros((gates.shape[-1],), mx.int32).at[indices.reshape(-1)].add(1)
    values = sorted(counts.tolist())
    return {"rows": int(indices.size), "experts_used": sum(v > 0 for v in values),
            "max_rows": values[-1], "median_rows": values[len(values) // 2]}


def _reference(mixer, hidden) -> dict:
    """
    Purpose: Build an FP32 routed-expert reference with the production routes.
    Input: The MoE block and its real BF16 input rows.
    Output: Reference routed output, input gradient, and LoRA factor gradients.
    Method:
        1. Take expert IDs from the production BF16 router, as replay does.
        2. Compute scores, dequantized expert projections, and LoRA in FP32.
        3. Loop over used experts so no per-row weight copy exists.
        4. Add the shared-expert input gradient from the production BF16 path.
    """
    import mlx.core as mx

    expert = mixer.switch_mlp
    gates = mx.softmax(mixer.gate(hidden), axis=-1, precise=True)
    indices = mx.argpartition(gates, kth=-mixer.top_k, axis=-1)[..., -mixer.top_k:]
    flat = indices.reshape(-1)
    order = mx.argsort(flat)
    mx.eval(order, flat)
    experts = flat[order].tolist()
    runs, start = [], 0
    for position in range(1, len(experts) + 1):
        if position == len(experts) or experts[position] != experts[start]:
            runs.append((experts[start], mx.array(order.tolist()[start:position])))
            start = position
    weights = {}
    for name in ("up_proj", "gate_proj", "down_proj"):
        base = getattr(expert, name).base
        weights[name] = (base, getattr(expert, name).scale)

    def project(name, inputs, number, factors):
        base, scale = weights[name]
        weight = mx.dequantize(base.weight[number], base.scales[number],
                               base.biases[number], group_size=base.group_size,
                               bits=base.bits).astype(mx.float32)
        low = inputs @ factors[name + ".a"][number].T
        return inputs @ weight.T + scale * (low @ factors[name + ".b"][number].T)

    def run(x, factors):
        probabilities = mx.softmax(mixer.gate(x.astype(mx.bfloat16)).astype(mx.float32),
                                   axis=-1)
        scores = mx.take_along_axis(probabilities, indices, axis=-1)
        scores = (scores / mx.sum(scores, axis=-1, keepdims=True)).reshape(-1)
        rows = x.reshape(-1, x.shape[-1])
        outputs = []
        for number, positions in runs:
            values = rows[positions // mixer.top_k]
            up = project("up_proj", values, number, factors)
            gate = project("gate_proj", values, number, factors)
            down = project("down_proj", expert.activation(up, gate), number, factors)
            outputs.append(down * scores[positions][:, None])
        everything = mx.concatenate([p for _, p in runs])
        flat_out = mx.zeros((flat.size, rows.shape[-1]), mx.float32).at[everything].add(
            mx.concatenate(outputs))
        return flat_out.reshape(*indices.shape, -1).sum(-2)

    names = [f"{n}.{p}" for n in ("up_proj", "gate_proj", "down_proj") for p in "ab"]
    factors = {name: getattr(getattr(expert, name.split(".")[0]), name.split(".")[1])
               .astype(mx.float32) for name in names}
    x = hidden.astype(mx.float32)
    output = run(x, factors)
    cotangent = _cotangent(output.shape)
    _, grads = mx.vjp(lambda value, *parts: run(value, dict(zip(names, parts))),
                      [x, *[factors[n] for n in names]], [cotangent])
    shared = lambda value: mixer.shared_expert(value) * mixer._shared_expert_scale(value)
    shared_output = shared(hidden)
    _, (shared_grad,) = mx.vjp(shared, [hidden], [cotangent.astype(mx.bfloat16)])
    mx.eval(output, grads, shared_output, shared_grad)
    return {"routed": output, "shared": shared_output.astype(mx.float32),
            "cotangent": cotangent,
            "input_grad": grads[0] + shared_grad.astype(mx.float32),
            "factor_grads": dict(zip(names, grads[1:]))}


def _cotangent(shape):
    import mlx.core as mx

    return (mx.random.normal(shape, key=mx.random.key(7)) * 1e-3).astype(mx.float32)


def _measure(mixer, function, hidden, reference, *, repeats: int) -> dict:
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    expert = mixer.switch_mlp
    names = [f"{n}.{p}" for n in ("up_proj", "gate_proj", "down_proj") for p in "ab"]
    values = [getattr(getattr(expert, n.split(".")[0]), n.split(".")[1]) for n in names]

    def unit(x, *parts):
        for name, part in zip(names, parts):
            module, attribute = name.split(".")
            setattr(getattr(expert, module), attribute, part)
        return function(mixer, x)

    cotangent = reference["cotangent"].astype(mx.bfloat16)
    try:
        output = function(mixer, hidden)
        mx.eval(output)
        forward = _median(lambda: mx.eval(function(mixer, hidden)), repeats)
        vjp = _median(lambda: mx.eval(mx.vjp(unit, [hidden, *values], [cotangent])),
                      repeats)
        _, grads = mx.vjp(unit, [hidden, *values], [cotangent])
        mx.eval(grads)
    finally:
        for name, value in zip(names, values):
            module, attribute = name.split(".")
            setattr(getattr(expert, module), attribute, value)
    routed = output.astype(mx.float32) - reference["shared"]
    row = {"forward_seconds": forward, "vjp_seconds": vjp,
           "output_relative_error": _relative(routed, reference["routed"])}
    row["input_grad_relative_error"] = _relative(grads[0], reference["input_grad"])
    for name, gradient in zip(names, grads[1:]):
        row[name + "_grad_relative_error"] = _relative(
            gradient, reference["factor_grads"][name])
    return row


def _relative(value, reference) -> float:
    import mlx.core as mx

    value = value.astype(mx.float32)
    return float((mx.linalg.norm((value - reference).reshape(-1))
                  / mx.maximum(mx.linalg.norm(reference.reshape(-1)), 1e-30)).item())


def _median(function, repeats: int) -> float:
    function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return sorted(samples)[len(samples) // 2]


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


if __name__ == "__main__":
    main()
