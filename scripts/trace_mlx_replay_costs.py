"""
Purpose: Export typed replay operation graphs without model numerical execution.
Input: The actual decoder configuration and CPU-only lazy example arrays.
Output: Primitive names, tensor shapes, dtypes, and dense-equivalent FLOP counts.
Note: The script does not load model assets, evaluate model arrays, or execute GPU kernels.
"""

from __future__ import annotations

# B: batch, T: token rows, D: hidden width, P: position axes.
import argparse
import hashlib
import importlib.metadata
import json
import math
import resource
import signal
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def trace(args: argparse.Namespace) -> None:
    """
    Purpose: Trace representative GDN and attention replay units on CPU.
    Input: A qualified configuration and pinned source directory.
    Output: Typed forward and derivative graphs with no model array evaluation.
    """
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    assert mx.default_device() == mx.cpu
    from mlx import nn

    build = json.loads((Path(args.reader) / "build.json").read_text())
    assert (
        hashlib.sha256(Path(build["extension"]).read_bytes()).hexdigest()
        == build["extension_sha256"]
    )
    import _mlx_graph_reader as reader
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_vlm.models.qwen3_5.language import LanguageModel, TextConfig

    from studio_compute.numerics.adapter import Adapter
    from studio_compute.numerics.ops import initial_states, segment_layer
    from studio_compute.numerics.replay import Replay

    test_MK = mx.zeros((2, 3), mx.float32)
    test_KN = mx.zeros((3, 4), mx.float32)
    test_MN = test_MK @ test_KN
    test_graph = reader.read_graph([test_MN], [test_MK, test_KN])
    assert len(test_graph) == 1 and test_graph[0]["name"] == "Matmul"
    assert test_graph[0]["outputs"][0][1] == [2, 4]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.inventory).read_text())["model_config"]
    config["num_hidden_layers"] = 4
    lm = LanguageModel(TextConfig(**config))
    lm.set_dtype(mx.bfloat16)
    nn.quantize(lm, group_size=64, bits=8)
    Adapter(lm, ["linear_attention"] * 3 + ["full_attention"])
    lm.train()
    replay = Replay(SimpleNamespace(language_model=lm))
    report = {
        "status": "running",
        "started_at": time.time(),
        "default_device": "cpu",
        "model_array_evaluation": False,
        "model_assets_loaded": False,
        "scope": "Raw lazy graph with CPU default; device-specific helper graphs can differ. No model graph is evaluated.",
        "graph_reader_build": build,
        "tracer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256(
                (Path(args.source) / "studio_compute/numerics" / name).read_bytes()
            ).hexdigest()
            for name in ("adapter.py", "ops.py", "replay.py")
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mlx", "mlx-lm", "mlx-vlm")
        },
        "inventory_sha256": hashlib.sha256(
            Path(args.inventory).read_bytes()
        ).hexdigest(),
        "cases": [],
    }
    for index, kind in ((0, "gdn"), (3, "attention")):
        if args.carry_only and kind != "attention":
            continue
        layer = lm.layers[index]
        names, parameters = zip(*tree_flatten(layer.parameters()))
        tokens = args.tokens
        hidden_BTD = mx.zeros((1, tokens, 5120), mx.bfloat16)
        positions_PBT = mx.zeros((3, 1, tokens), mx.int32)
        sources = initial_states([layer], mx.bfloat16)[0]
        if kind == "attention":
            sources = []
            for chunk in range(args.key_prefix_chunks):
                for carry in range(2):
                    sources.append(
                        (
                            (chunk, index, carry),
                            mx.zeros((1, 4, 2048, 256), mx.bfloat16),
                        )
                    )
            carry_shapes = [(1, 4, tokens, 256)] * 2
        else:
            sources = [
                ((0, index, carry), value) for carry, (_, value) in enumerate(sources)
            ]
            carry_shapes = [value.shape for _, value in sources]
        values = [value for _, value in sources]
        cotangents = [mx.zeros(hidden_BTD.shape, mx.bfloat16)] + [
            mx.zeros(shape, mx.float32 if kind == "gdn" and carry == 1 else mx.bfloat16)
            for carry, shape in enumerate(carry_shapes)
        ]
        if args.terminal:
            cotangents = cotangents[:1]
        elif args.carry_only:
            cotangents = cotangents[1:]
        count = len(parameters)

        def forward(
            *arguments: Any,
            layer: Any = layer,
            names: tuple = names,
            count: int = count,
        ) -> tuple[Any, ...]:
            """Purpose: Trace one pure layer unit with explicit graph inputs."""
            layer.update(tree_unflatten(list(zip(names, arguments[:count]))))
            return segment_layer(
                layer,
                arguments[count],
                list(arguments[count + 2 :]),
                arguments[count + 1],
            )

        def backward(
            *arguments: Any,
            layer: Any = layer,
            names: tuple = names,
            count: int = count,
            sources: list = sources,
        ) -> list[Any]:
            """Purpose: Trace the existing replay VJP with actual carry edges."""
            layer.update(tree_unflatten(list(zip(names, arguments[:count]))))
            hidden_BTD = arguments[count]
            positions_PBT = arguments[count + 1]
            states = [
                (source, value)
                for (source, _), value in zip(
                    sources,
                    arguments[count + 2 : count + 2 + len(sources)],
                    strict=True,
                )
            ]
            incoming = list(arguments[count + 2 + len(sources) :])
            if args.terminal:
                incoming += [None, None]
            elif args.carry_only:
                incoming = [None, *incoming]
            _, back = replay._layer_vjp(
                layer, hidden_BTD, states, positions_PBT, incoming
            )
            return back

        arguments = [*parameters, hidden_BTD, positions_PBT, *values]
        for stage, function, inputs in (
            ("forward", forward, arguments),
            ("vjp", backward, arguments + cotangents),
        ):
            results = function(*inputs)
            if isinstance(results, mx.array):
                results = [results]
            events = [
                {"type": "inputs", "inputs": _plain(reader.metadata(inputs))},
                {"type": "outputs", "outputs": _plain(reader.metadata(results))},
            ]
            events.extend(_plain(reader.read_graph(results, inputs)))
            row = {
                "kind": kind,
                "stage": stage,
                "parameters": list(names),
                "tokens": tokens,
                "key_prefix_tokens": args.key_prefix_chunks * 2048
                if kind == "attention"
                else None,
                "terminal": args.terminal,
                "carry_only": args.carry_only,
                "events": events,
                "summary": summarize(events),
                "device_counts": dict(
                    Counter(
                        event["device_type"]
                        for event in events
                        if event["type"] == "primitive"
                    )
                ),
                "gpu_helper_nodes": [
                    event
                    for event in events
                    if event["type"] == "primitive" and event["device_type"] != 0
                ],
            }
            parameter_ids = {
                item[0]: name
                for item, name in zip(events[0]["inputs"][:count], names, strict=True)
            }
            for matrix in row["summary"]["matrices"]:
                matrix["parameter"] = parameter_ids.get(matrix["inputs"][1][0])
            (output / f"{kind}-{stage}.json").write_text(
                json.dumps(row, indent=2) + "\n"
            )
            report["cases"].append(
                {key: value for key, value in row.items() if key != "events"}
            )
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        assert report["peak_rss_bytes"] < 2 << 30
    report.update(status="passed", finished_at=time.time())
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


def summarize(events: list[dict]) -> dict[str, Any]:
    """
    Purpose: Count matrix arithmetic from typed primitive input and output shapes.
    Input: Export callback events. Output: Dtype-separated matrix FLOP totals.
    Note: Non-matrix operations and hardware-specific instruction costs are excluded.
    """
    counts = Counter()
    flops = Counter()
    matrices = []
    for event in events:
        if event["type"] != "primitive":
            continue
        name = event["name"]
        counts[name] += 1
        if name not in ("Matmul", "QuantizedMatmul"):
            continue
        inputs, outputs = event["inputs"], event["outputs"]
        operations = 2 * math.prod(outputs[0][1]) * inputs[0][1][-1]
        dtype = inputs[0][2]
        flops[f"{name}:{dtype}"] += operations
        matrices.append(
            {
                "name": name,
                "inputs": inputs,
                "outputs": outputs,
                "arguments": event["arguments"],
                "flops": operations,
            }
        )
    return {
        "primitive_counts": dict(counts),
        "matrix_flops": dict(flops),
        "matrices": matrices,
    }


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New private trace folder.")
    parser.add_argument(
        "--inventory", required=True, help="Prior qualified inference inventory JSON."
    )
    parser.add_argument(
        "--source", required=True, help="Pinned Shellsmith source directory."
    )
    parser.add_argument(
        "--reader", required=True, help="Isolated read-only graph extension folder."
    )
    parser.add_argument(
        "--tokens", type=int, default=2048, help="Actual replay unit length."
    )
    parser.add_argument(
        "--key-prefix-chunks",
        type=int,
        default=27,
        help="Prior 2048-token attention carries.",
    )
    parser.add_argument(
        "--terminal",
        action="store_true",
        help="Omit future carry cotangents at the terminal unit.",
    )
    parser.add_argument(
        "--carry-only",
        action="store_true",
        help="Trace final attention-layer units with no direct hidden cotangent.",
    )
    args = parser.parse_args()
    assert not (args.terminal and args.carry_only)
    assert 0 < args.tokens <= 2048 and args.key_prefix_chunks >= 0
    signal.alarm(60)
    sys.path.insert(0, args.source)
    sys.path.insert(0, args.reader)
    trace(args)
