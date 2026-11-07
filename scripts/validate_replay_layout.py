"""
Purpose: Validate a replay execution change against the original update replay.
Input: Reconstructed update sequences, a subset of indices, the model, and options.
Output: Logprob, loss, and gradient comparisons with the summation-order noise floor.
Method:
    0. With the initial adapter, replay the parity indices and compare the
       logprobs with the recorded production replay logprobs.
    1. Run the original prompt-group layout in the recorded sequence order.
    2. Run it again in reversed order. This change only reorders BF16 sums.
    3. Run each candidate configuration in the recorded order.
    4. Compare each run with the first run, tensor by tensor.
Note: The script owns the GPU lease. It never opens a service state root.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import math
import time
from pathlib import Path

LEASE = Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.sequences.read_text())
    subset = [payload["sequences"][index] for index in args.indices]
    lease = LEASE.open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    profiler = _load_profiler()
    profiler._write_identity(args.output / "process.json")
    engine = _load_engine(args.model, profiler)
    report = {"indices": args.indices, "runs": {}}
    report["production_parity"] = _production_parity(engine, payload, args.parity_indices)
    print(json.dumps({"production_parity": report["production_parity"]}), flush=True)
    _write_json(args.output / "validation.json", report)
    _nudge_adapter(engine)
    names = ["baseline"] + ([] if args.skip_reversed else ["baseline_reversed"])
    names += args.candidates
    results = {}
    for name in names:
        options = _candidate(name)
        result = _run(engine, subset, **options)
        results[name] = result
        report["runs"][name] = {
            **options, "seconds": result["seconds"], "loss": result["loss"],
            "peak_memory_bytes": result["peak_memory_bytes"],
            "stats": result["stats"],
            "versus_baseline": _compare(results["baseline"], result),
            "versus_sampler": _sampler_agreement(subset, result),
        }
        print(json.dumps({"run": name, "seconds": result["seconds"],
                          **report["runs"][name]["versus_baseline"]["summary"],
                          "sampler": report["runs"][name]["versus_sampler"]}),
              flush=True)
        _write_json(args.output / "validation.json", report)
    report["pairs"] = {f"{left}|{right}": _compare(results[left], results[right])["summary"]
                       for index, left in enumerate(names) for right in names[index + 1:]}
    _write_json(args.output / "validation.json", report)
    for pair, summary in report["pairs"].items():
        print(json.dumps({"pair": pair, "gradient_cosine": summary["gradient_cosine"],
                          "gradient_relative_l2": summary["gradient_relative_l2"],
                          "logprob_mean_abs": summary["logprob_mean_abs"]}), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, required=True,
                        help="Reconstructed update JSON.")
    parser.add_argument("--model", type=Path, required=True, help="Flash model path.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 11, 12],
                        help="Update sequence indices.")
    parser.add_argument("--parity-indices", type=int, nargs="+", default=[11],
                        help="Indices compared with the recorded production replay.")
    parser.add_argument("--candidates", nargs="+", default=["segment_trie"],
                        help="Candidates joined by +: segment_trie, sorted, chunkedN, "
                             "segmentsN.")
    parser.add_argument("--skip-reversed", action="store_true",
                        help="Skip the reversed-order baseline run.")
    return parser.parse_args()


def _candidate(name: str) -> dict:
    """Map a candidate name to replay options. Parts are joined by '+'."""
    options = {"layout": "prompt_groups", "moe": None, "reverse": name.endswith("_reversed"),
               "segment_tokens": None}
    for part in name.removesuffix("_reversed").split("+"):
        if part == "segment_trie":
            options["layout"] = part
        elif part == "sorted" or part.startswith("chunked"):
            options["moe"] = part
        elif part.startswith("segments"):
            options["segment_tokens"] = int(part.removeprefix("segments"))
        else:
            assert part == "baseline", part
    return options


def _load_profiler():
    path = Path(__file__).with_name("profile_flash_replay.py")
    spec = importlib.util.spec_from_file_location("profile_flash_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_engine(model: Path, profiler):
    engine = profiler._load_engine(model)
    engine.resident.mode(False)
    engine.create("validation", {
        "seed": 20261019, "lora_rank": 16, "lora_scale": 2.0, "learning_rate": 5e-5,
        "optimizer": {"beta1": 0.9, "beta2": 0.95, "epsilon": 1e-8},
    })
    return engine


def _nudge_adapter(engine) -> None:
    """A small nonzero B gives every A factor a nonzero gradient."""
    adapter = engine.clients["validation"]["adapter"]
    adapter.load({name: value + 1e-4 if name.endswith(".b") else value
                  for name, value in adapter.state().items()})


def _production_parity(engine, payload: dict, indices: list) -> dict:
    """Compare initial-adapter replay logprobs with the recorded production values."""
    sequences = [payload["sequences"][index] for index in indices]
    result = _run(engine, sequences, layout="prompt_groups", moe=None, reverse=False)
    differences = [abs(a - b) for index, values in zip(indices, result["logprobs"])
                   for a, b in zip(values, payload["replay_logprobs"][index],
                                   strict=True)]
    return {"indices": indices, "seconds": result["seconds"],
            "exact_fraction": sum(d == 0 for d in differences) / len(differences),
            "max_abs": max(differences)}


def _drain(work):
    while True:
        try:
            next(work)
        except StopIteration as stop:
            return stop.value


def _run(engine, sequences, *, layout, moe, reverse, segment_tokens=None) -> dict:
    """Run one complete forward_backward and keep the accumulated gradients."""
    import mlx.core as mx
    from studio_compute.numerics import flash_ops

    order = list(range(len(sequences)))
    if reverse:
        order.reverse()
    body = {"operation_id": f"validation-{layout}-{moe}-{reverse}-{time.time()}",
            "sequences": [sequences[index] for index in order], "loss": "cispo",
            "reduction": "sum", "clip_low": 0.0, "clip_high": 4.0}
    entry = engine.clients["validation"]
    entry["backward"] = None
    engine.replay_layout = layout
    default_segment_tokens = engine.segment_tokens
    engine.segment_tokens = segment_tokens or default_segment_tokens
    original = flash_ops._moe
    engine.replay_ops.training_moe_chunk = None
    if moe == "sorted":
        flash_ops._moe = lambda mixer, x, sort_rows=False: original(
            mixer, x, sort_rows=True)
    elif moe is not None:
        engine.replay_ops.training_moe_chunk = int(moe.removeprefix("chunked"))
    mx.reset_peak_memory()
    started = time.monotonic()
    try:
        result = _drain(engine.work("validation", "forward_backward", body))
    finally:
        flash_ops._moe = original
        engine.replay_ops.training_moe_chunk = None
        engine.segment_tokens = default_segment_tokens
    seconds = time.monotonic() - started
    gradients = dict(entry["optimizer"].accumulated)
    entry["optimizer"].zero()
    entry["backward"] = None
    logprobs = [None] * len(sequences)
    for position, index in enumerate(order):
        logprobs[index] = result["logprobs"][position]
    targets = [None] * len(sequences)
    for position, index in enumerate(order):
        targets[index] = result["logprob_target_indices"][position]
    return {"seconds": seconds, "loss": result["forward_metrics"]["loss"],
            "targets": targets,
            "logprobs": logprobs, "gradients": gradients,
            "peak_memory_bytes": mx.get_peak_memory(), "stats": result["groups"]}


def _compare(reference: dict, result: dict) -> dict:
    """Bitwise logprob agreement and gradient relative errors and cosine."""
    import mlx.core as mx

    differences = [abs(a - b) for left, right in zip(reference["logprobs"],
                                                     result["logprobs"])
                   for a, b in zip(left, right, strict=True)]
    dot = left_norm = right_norm = error = 0.0
    worst = (0.0, "")
    for name, value in reference["gradients"].items():
        left = value.astype(mx.float32)
        right = result["gradients"][name].astype(mx.float32)
        tensor_error = float(mx.sum((left - right) ** 2).item())
        tensor_norm = float(mx.sum(left * left).item())
        dot += float(mx.sum(left * right).item())
        left_norm += tensor_norm
        right_norm += float(mx.sum(right * right).item())
        error += tensor_error
        if tensor_norm > 0:
            worst = max(worst, (math.sqrt(tensor_error / tensor_norm), name))
    return {"summary": {
        "logprob_exact_fraction": sum(d == 0 for d in differences) / len(differences),
        "logprob_max_abs": max(differences),
        "logprob_mean_abs": sum(differences) / len(differences),
        "loss_difference": result["loss"] - reference["loss"],
        "gradient_relative_l2": math.sqrt(error / left_norm),
        "gradient_cosine": dot / math.sqrt(left_norm * right_norm),
        "gradient_norm_ratio": math.sqrt(right_norm / left_norm),
        "worst_tensor_relative_l2": worst[0],
    }, "worst_tensor": worst[1]}


def _sampler_agreement(sequences: list, result: dict) -> dict:
    """
    Purpose: Measure replay agreement with the sampler's behavior logprobs.
    Input: The sequences and one replay result with its target indices.
    Output: Mean, p99, and maximum absolute differences, and the mean signed
        difference (behavior minus replay) over loss tokens.
    """
    differences = []
    for sequence, values, targets in zip(sequences, result["logprobs"],
                                         result["targets"], strict=True):
        positions = targets if targets is not None else range(len(values))
        mask = sequence["loss_mask"]
        for value, position in zip(values, positions, strict=True):
            if mask[position]:
                differences.append(sequence["behavior_logprobs"][position] - value)
    magnitudes = sorted(abs(value) for value in differences)
    return {"tokens": len(differences),
            "mean_abs": sum(magnitudes) / len(magnitudes),
            "p99_abs": magnitudes[int(0.99 * (len(magnitudes) - 1))],
            "max_abs": magnitudes[-1],
            "mean_signed": sum(differences) / len(differences)}


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, default=str))
    temporary.replace(path)


if __name__ == "__main__":
    main()
