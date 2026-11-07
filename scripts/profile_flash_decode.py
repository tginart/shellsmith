"""
Purpose: Split Flash decode step time into CPU graph construction, GPU work, and sync.
Input: Reconstructed update sequences for real prompts, batch sizes, and the model.
Output: JSON timings per batch size for the production decode and sampling code.
Note: The script owns the GPU lease. It never opens a service state root.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

LEASE = Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sequences = json.loads(args.sequences.read_text())["sequences"]
    lease = LEASE.open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    _write_identity(args.output / "process.json")
    engine = _load_engine(args.model)
    report = {"batches": {}, "components": {}}
    for batch in args.component_batches:
        report["components"][batch] = _profile_components(
            engine, sequences, batch=batch, prompt_tokens=args.prompt_tokens)
        print(json.dumps({"components": batch, **report["components"][batch]}),
              flush=True)
        _write_json(args.output / "decode.json", report)
    for batch in args.batches:
        report["batches"][batch] = _profile_batch(
            engine, sequences, batch=batch, prompt_tokens=args.prompt_tokens,
            steps=args.steps)
        print(json.dumps({"batch": batch, **{
            k: v for k, v in report["batches"][batch].items() if k != "layers"}}),
            flush=True)
        _write_json(args.output / "decode.json", report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, required=True,
                        help="Reconstructed update JSON.")
    parser.add_argument("--model", type=Path, required=True, help="Flash model path.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8],
                        help="Decode batch sizes.")
    parser.add_argument("--prompt-tokens", type=int, default=12288,
                        help="Real prompt prefix length per request.")
    parser.add_argument("--steps", type=int, default=24, help="Timed decode steps.")
    parser.add_argument("--component-batches", type=int, nargs="*", default=[1, 8],
                        help="Batch sizes for the synchronized component breakdown.")
    return parser.parse_args()


def _write_identity(path: Path) -> None:
    pid = os.getpid()
    started = subprocess.check_output(["ps", "-o", "lstart=,command=", "-p", str(pid)],
                                      text=True).strip()
    _write_json(path, {"pid": pid, "ps_start_and_command": started,
                       "lease": str(LEASE)})


def _load_engine(model: Path):
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    from studio_compute.flash_model import QwenFlashModel
    from studio_compute.runtime import RuntimeEngine

    engine = RuntimeEngine(
        str(model), model_factory=QwenFlashModel, execution_profile="bf16-fast",
        activation_budget_gib=16, inference_concurrency=16, prefill_tokens=2048,
        segment_tokens=2048, head_rows=128,
    )
    factors = engine.resident.adapter.initial(20261019)
    engine.mx.eval(factors)
    engine.policies["profile"] = {"factors": factors, "revision": 0}
    return engine


class _TimedEval:
    """Record the time of every mx.eval call made by one module."""

    def __init__(self, mx):
        self._mx, self.calls = mx, []

    def __getattr__(self, name):
        return getattr(self._mx, name)

    def eval(self, *arrays):
        started = time.perf_counter()
        self._mx.eval(*arrays)
        self.calls.append((started, time.perf_counter()))


def _profile_batch(engine, sequences, *, batch, prompt_tokens, steps) -> dict:
    """Prefill real prompts, then time production decode and sampling steps."""
    from studio_compute import inference
    from studio_compute.numerics import flash_ops

    requests = []
    for row, sequence in enumerate(sequences[:batch]):
        tokens = sequence["prompt_token_ids"] + sequence["sampled_tokens"]
        body = {"prompt_token_ids": tokens[:prompt_tokens],
                "target_token_ids": tokens[prompt_tokens:prompt_tokens + steps + 4],
                "max_tokens": steps + 4, "seed": row, "revision": 0,
                "policy": "profile", "cache_precision": "8bit",
                "use_prefix_cache": False}
        request = inference.create_request(engine, f"profile-{row}", body)
        while request.logits is None:
            inference.prefill_request(engine, request)
        requests.append(request)
    inference.sample_requests(engine, requests)
    timed = _TimedEval(flash_ops.mx)
    flash_ops.mx = timed
    walls, builds, waits, samples = [], [], [], []
    try:
        for step in range(steps + 2):
            started = time.perf_counter()
            timed.calls.clear()
            inference.advance_requests(engine, requests)
            advanced = time.perf_counter()
            inference.sample_requests(engine, requests)
            stopped = time.perf_counter()
            if step < 2:
                continue
            eval_started, eval_stopped = timed.calls[-1]
            walls.append(stopped - started)
            builds.append(eval_started - started)
            waits.append(eval_stopped - eval_started)
            samples.append(stopped - advanced)
    finally:
        flash_ops.mx = timed._mx
        for request in requests:
            inference.release_request(engine, request)
    count = len(walls)
    return {
        "step_seconds": sum(walls) / count,
        "graph_build_seconds": sum(builds) / count,
        "eval_wait_seconds": sum(waits) / count,
        "sample_seconds": sum(samples) / count,
        "tokens_per_second": batch * count / sum(walls),
        "prompt_tokens": prompt_tokens,
    }


def _profile_components(engine, sequences, *, batch, prompt_tokens) -> dict:
    """
    Purpose: Attribute one decode step to components with a sync after each one.
    Input: The engine, real prompts, and a batch size.
    Output: Summed seconds per component over all decoder layers.
    Note: The extra synchronizations add latency. Use the shares, not the total.
    """
    import mlx.core as mx
    from mlx_vlm.models.cache import ArraysCache
    from studio_compute import inference
    from studio_compute.numerics import flash_ops
    from studio_compute.numerics.bf16 import gdn_forward

    requests = []
    for row, sequence in enumerate(sequences[:batch]):
        tokens = sequence["prompt_token_ids"] + sequence["sampled_tokens"]
        body = {"prompt_token_ids": tokens[:prompt_tokens],
                "target_token_ids": tokens[prompt_tokens:prompt_tokens + 8],
                "max_tokens": 8, "seed": row, "revision": 0, "policy": "profile",
                "cache_precision": "8bit", "use_prefix_cache": False}
        request = inference.create_request(engine, f"component-{row}", body)
        while request.logits is None:
            inference.prefill_request(engine, request)
        requests.append(request)
    inference.sample_requests(engine, requests)
    totals = {}

    def timed(name, function):
        mx.eval(mx.zeros(1))
        started = time.perf_counter()
        result = function()
        mx.eval(result)
        totals[name] = totals.get(name, 0.0) + time.perf_counter() - started
        return result

    try:
        for _ in range(3):
            totals.clear()
            _component_step(engine, requests, timed, flash_ops, gdn_forward,
                            ArraysCache)
            inference.sample_requests(engine, requests)
    finally:
        for request in requests:
            inference.release_request(engine, request)
    return totals


def _component_step(engine, requests, timed, flash_ops, gdn_forward, ArraysCache):
    """Mirror flash_ops.decode_rows with one timed region per component."""
    import mlx.core as mx

    lm = engine.lm
    engine.resident.select(requests[0].factors)
    engine.resident.mode(False)
    ids_BL = mx.array([[request.tokens[-1]] for request in requests], mx.int32)
    offsets = mx.array([len(r.body["prompt_token_ids"]) + len(r.tokens) - 1
                        for r in requests], mx.int32)
    positions = mx.broadcast_to(offsets[None, :, None], (3, len(requests), 1))
    hidden_BLD = timed("embed", lambda: flash_ops.FlashOps.expand(
        lm, lm.model.embed_tokens(ids_BL).astype(mx.bfloat16)))
    for index, layer in enumerate(lm.layers):
        caches = [request.cache[index] for request in requests]
        kind = "gdn" if layer.is_linear else "attention"
        if "ple" in layer:
            combined = ArraysCache(size=4)
            combined.cache = [mx.concatenate([c[slot] for c in caches], axis=0)
                              for slot in range(4)]
            hidden_BLD = timed("ple", lambda: hidden_BLD + layer.ple(
                hidden_BLD, ids_BL, combined, None))
        mixed_BLD, hyper_BLD, injection_BLH = timed(
            "hyper", lambda: layer.attn_hyper_connection(hidden_BLD))
        if layer.is_linear:
            states = [mx.concatenate([c[slot] for c in caches], axis=0)
                      for slot in range(2)]
            output_BLD, _, _ = timed(kind, lambda: gdn_forward(
                layer.linear_attn, mixed_BLD, states, decode_one=True,
                gate_activation=layer.linear_attn.norm.activation))
        elif len(requests) > 1:
            output_BLD = timed(kind, lambda: flash_ops._quantized_attention_rows(
                layer.self_attn, mixed_BLD, caches, positions))
        else:
            output_BLD = timed(kind, lambda: layer.self_attn(
                mixed_BLD, mask="causal", cache=caches[0], position_ids=positions))
        injected = output_BLD[..., None, :] * injection_BLH[..., None]
        hidden_BLD = hyper_BLD + injected.reshape(hyper_BLD.shape)
        mixed_BLD, hyper_BLD, injection_BLH = timed(
            "hyper", lambda: layer.mlp_hyper_connection(hidden_BLD))
        output_BLD = timed("moe", lambda: flash_ops._moe(layer.mlp, mixed_BLD,
                                                         sort_rows=True))
        injected = output_BLD[..., None, :] * injection_BLH[..., None]
        hidden_BLD = hyper_BLD + injected.reshape(hyper_BLD.shape)
    head = lm.lm_head
    timed("head", lambda: head(lm.model.hyper_connection_mixer(hidden_BLD)))


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


if __name__ == "__main__":
    main()
