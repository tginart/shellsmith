"""
Purpose: Measure Flash replay costs per component against a matrix FLOP floor.
Input: A recorded update batch, sequence and segment indices, and the Flash model.
Output: profile.json with per-layer component times and FLOPs, and a batch table
    that extrapolates the components over every trie segment of the batch.
Method:
    1. Advance one real sequence to each segment and keep the real layer inputs.
    2. Time the layer forward and the replay layer VJP, then each component alone:
       PLE, hyper-connections, mixer projections, mixer core, MoE, and the head.
    3. Count the matrix FLOPs of each component from the model shapes. The rule is
       the rule of scripts/trace_mlx_replay_costs.py: 2 * output size * inner width.
    4. Visit the batch trie. Sum the component times and FLOPs of each segment.
       Attention costs come from a linear fit over the profiled context lengths.
Note: The script owns the GPU lease. It never opens a service state root.
    The floor counts one forward, one recompute forward, and one backward. The
    backward of a frozen quantized projection is its input gradient only.
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
SEGMENT_TOKENS = 2048
HEAD_ROWS = 128
# FA2 backward: recompute scores, then dV, dP, dQ, and dK products.
ATTENTION_BACKWARD_RATIO = 2.5
COMPONENTS = ("ple", "attn_hc", "mixer_proj", "mixer_core", "mlp_hc", "moe_routed",
              "moe_shared", "other")

# B: batch, L: segment tokens, D: residual width (hc_count * hidden), F: features.


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sequences = _load_sequences(args.sequences)
    lease = LEASE.open("a")
    fcntl.flock(lease, fcntl.LOCK_EX)  # Wait in line for the shared lease.
    _write_identity(args.output / "process.json")
    import mlx.core as mx

    engine = _load_engine(args.model, args.moe_chunk)
    config = engine.lm.args
    report = {
        "sequence": args.sequence, "moe_chunk": args.moe_chunk, "segments": {},
        "args": {key: str(value) for key, value in vars(args).items()},
    }
    sequence = sequences[args.sequence]
    for segment in args.segments:
        captured = _capture_segment(engine, sequence, segment)
        context = segment * SEGMENT_TOKENS
        rows = {}
        for index in args.layers:
            layer = engine.lm.layers[index]
            times = _profile_layer(engine, captured, index, args.repeats)
            flops = layer_flops(config, layer, SEGMENT_TOKENS, context)
            rows[index] = {"times": times, "flops": flops}
            print(json.dumps({"segment": segment, "layer": index,
                              **{k: round(v * 1e3, 1) for k, v in times.items()}}),
                  flush=True)
        report["segments"][segment] = {
            "context_tokens": context, "layers": rows,
            "head": _profile_head(engine, captured, args.repeats),
            "group": _profile_group(engine, captured, args.group_layers, args.repeats),
            "peak_memory_bytes": mx.get_peak_memory(),
        }
        _write_json(args.output / "profile.json", report)
        del captured
        mx.clear_cache()
    report["batch"] = batch_table(report, sequences, config, engine.lm.layers,
                                  rate_tflops=args.rate_tflops)
    _write_json(args.output / "profile.json", report)
    print(json.dumps(report["batch"]["summary"], indent=1))


def layer_flops(config, layer, tokens: int, context: int) -> dict:
    """
    Purpose: Count the forward and floor-backward matrix FLOPs of one layer segment.
    Input: The text config, one decoder layer, segment tokens, and prior keys.
    Output: {component: {"forward", "backward", "dense_forward"}} in FLOPs.
    Note: "backward" is the floor: input gradients of frozen bases, input and
        factor gradients of LoRA, and the FA2 ratio for attention. The attention
        "dense_forward" is the masked dense product that the current code runs.
    """
    hidden, rank = config.hidden_size, 16
    width = hidden * config.hc_count
    result = {}

    def add(name, base, lora=0.0, backward=None):
        entry = result.setdefault(name, {"forward": 0.0, "backward": 0.0})
        entry["forward"] += base + lora
        entry["backward"] += base + 2 * lora if backward is None else backward

    def linear(name, inputs, outputs, *, adapted):
        add(name, 2 * tokens * inputs * outputs,
            2 * tokens * rank * (inputs + outputs) if adapted else 0.0)

    for name in ("attn_hc", "mlp_hc"):
        linear(name, width, config.hc_lowrank, adapted=False)
        linear(name, config.hc_lowrank, width, adapted=False)
        linear(name, width, config.hc_count, adapted=False)
    if "ple" in layer:
        linear("ple", config.ple_embed_dim, width, adapted=False)
        linear("ple", config.ple_embed_dim, hidden, adapted=False)
    if layer.is_linear:
        key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        for outputs in (2 * key_dim + value_dim, value_dim,
                        config.linear_num_value_heads, config.linear_num_value_heads):
            linear("mixer_proj", hidden, outputs, adapted=True)
        linear("mixer_proj", value_dim, hidden, adapted=True)
        # Chunked delta rule, chunk C=64: intra-chunk products and state updates.
        chunk, k_width = 64, config.linear_key_head_dim
        v_width = config.linear_value_head_dim
        per_head = 2 * chunk * (2 * k_width + v_width) + 6 * k_width * v_width
        core = tokens * config.linear_num_value_heads * per_head
        add("mixer_core", core, backward=2 * core)
    else:
        heads, head_dim = config.num_attention_heads, config.head_dim
        kv = config.num_key_value_heads * head_dim
        linear("mixer_proj", hidden, 2 * heads * head_dim, adapted=True)
        linear("mixer_proj", hidden, kv, adapted=True)
        linear("mixer_proj", hidden, kv, adapted=True)
        linear("mixer_proj", heads * head_dim, hidden, adapted=True)
        index_width = (config.indexer_n_heads + config.indexer_kv_heads)
        linear("mixer_proj", hidden, index_width * config.indexer_head_dim,
               adapted=False)
        sparse_keys = sum(_qsa_keys(config, context + row) for row in range(tokens))
        core = 4 * heads * head_dim * sparse_keys
        blocks = (context + tokens) // config.indexer_compress_ratio
        indexer = 2 * tokens * config.indexer_n_heads * config.indexer_head_dim * blocks
        add("mixer_core", core + indexer, backward=ATTENTION_BACKWARD_RATIO * core)
        dense = 4 * heads * head_dim * tokens * (context + tokens)
        result["mixer_core"]["dense_floor"] = (
            2 * (dense + indexer) + ATTENTION_BACKWARD_RATIO * dense)
    linear("moe_shared", hidden, config.num_experts, adapted=False)
    for inputs, outputs in ((hidden, config.shared_expert_intermediate_size),) * 2 + (
            (config.shared_expert_intermediate_size, hidden),):
        linear("moe_shared", inputs, outputs, adapted=True)
    routed = tokens * config.num_experts_per_tok
    for inputs, outputs in ((hidden, config.moe_intermediate_size),) * 2 + (
            (config.moe_intermediate_size, hidden),):
        add("moe_routed", 2 * routed * inputs * outputs,
            2 * routed * rank * (inputs + outputs))
    return result


def head_flops(config, rows: int) -> dict:
    """Return the forward and floor-backward FLOPs of one head tile."""
    width = config.hidden_size * config.hc_count
    mixer = 2 * rows * 2 * width * config.hc_lowrank
    vocabulary = 2 * rows * config.hidden_size * config.vocab_size
    return {"forward": mixer + vocabulary, "backward": mixer + vocabulary}


def batch_table(report: dict, sequences: list, config, layers, *, rate_tflops: float):
    """
    Purpose: Extrapolate the profiled components over every trie segment of a batch.
    Input: The profile report, the batch sequences, the config, and the floor rate.
    Output: Per-component measured seconds, floor seconds, and percent of floor.
    Method:
        1. Build the replay segment trie and list each forward segment context.
        2. Fit each attention component time as a + b * context over the profile.
        3. Use the mean GDN layer times for every GDN layer.
        4. A segment costs one checkpoint forward and one VJP per layer. The VJP
           runs its own forward, as the one VJP per checkpoint group does.
    """
    from studio_compute.numerics.replay import _segment_trie

    token_ids = [s["prompt_token_ids"] + s["sampled_tokens"][:-1] for s in sequences]
    contexts = _trie_contexts(_segment_trie(token_ids, SEGMENT_TOKENS))
    head_tiles = sum(_head_tiles(s) for s in sequences)
    profiles = {"gdn": [], "attention": []}
    for segment in report["segments"].values():
        for index, row in segment["layers"].items():
            kind = "gdn" if layers[int(index)].is_linear else "attention"
            profiles[kind].append((segment["context_tokens"], row["times"]))
    measured, floor, dense = {}, {}, {}
    for context in contexts:
        for layer in layers:
            kind = "gdn" if layer.is_linear else "attention"
            times = _fit(profiles[kind], context)
            for name, value in times.items():
                measured[name] = measured.get(name, 0.0) + value
            for name, entry in layer_flops(config, layer, SEGMENT_TOKENS,
                                           context).items():
                floor[name] = floor.get(name, 0.0) + 2 * entry["forward"] + entry[
                    "backward"]
                dense[name] = dense.get(name, 0.0) + entry.get(
                    "dense_floor", 2 * entry["forward"] + entry["backward"])
    head = list(report["segments"].values())[0]["head"]
    measured["head_forward"] = head["forward"] * head_tiles
    measured["head_vjp"] = head["vjp"] * head_tiles
    tile = head_flops(config, HEAD_ROWS)
    floor["head"] = head_tiles * (tile["forward"] + tile["backward"])
    rate = rate_tflops * 1e12
    tokens = sum(len(ids) for ids in token_ids)
    total_floor = sum(floor.values())
    total_measured = (measured["layer_forward"] + measured["layer_vjp"]
                      + measured["head_forward"] + measured["head_vjp"])
    rows = {}
    for name in COMPONENTS:
        forward = measured.get(name + "_forward", 0.0)
        vjp = measured.get(name + "_vjp", 0.0)
        seconds = forward + vjp
        rows[name] = {"measured_s": seconds, "forward_s": forward, "vjp_s": vjp,
                      "floor_s": floor.get(name, 0.0) / rate,
                      "tflop": floor.get(name, 0.0) / 1e12}
    rows["head"] = {"measured_s": measured["head_forward"] + measured["head_vjp"],
                    "floor_s": floor["head"] / rate, "tflop": floor["head"] / 1e12}
    return {
        "segments": len(contexts), "head_tiles": head_tiles, "tokens": tokens,
        "components": rows,
        "summary": {
            "forward_segments": len(contexts),
            "layer_forward_s": measured["layer_forward"],
            "layer_vjp_s": measured["layer_vjp"],
            "modelled_replay_s": total_measured,
            "floor_tflop": total_floor / 1e12,
            "floor_s": total_floor / rate,
            "percent_of_floor": 100 * total_floor / rate / total_measured,
            "gflop_per_token": total_floor / tokens / 1e9,
            "gflop_per_token_dense_attention": (total_floor - floor["mixer_core"]
                                                + dense["mixer_core"]) / tokens / 1e9,
            "rate_tflops": rate_tflops,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, required=True,
                        help="Recorded forward_backward body or reconstructed update JSON.")
    parser.add_argument("--model", type=Path, required=True, help="Flash model path.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--sequence", type=int, default=21,
                        help="Sequence index (21 is the 49k-token update 2 sequence).")
    parser.add_argument("--segments", type=int, nargs="+", default=[1, 8, 16, 23],
                        help="2048-token segment indices to profile.")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 3],
                        help="Decoder layers: 0 GDN, 1 GDN with PLE, 3 attention.")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repeats.")
    parser.add_argument("--moe-chunk", type=int, default=32,
                        help="training_moe_chunk; nonzero selects sorted expert segments.")
    parser.add_argument("--group-layers", type=int, default=10,
                        help="Checkpoint group size for the schedule comparison.")
    parser.add_argument("--rate-tflops", type=float, default=75.0,
                        help="Matrix rate of the floor; qmm8 measures 67-70 TFLOP/s.")
    return parser.parse_args()


def _load_sequences(path: Path) -> list:
    payload = json.loads(path.read_text())
    return payload["body"]["sequences"] if "body" in payload else payload["sequences"]


def _write_identity(path: Path) -> None:
    pid = os.getpid()
    started = subprocess.check_output(["ps", "-o", "lstart=,command=", "-p", str(pid)],
                                      text=True).strip()
    _write_json(path, {"pid": pid, "ps_start_and_command": started,
                       "lease": str(LEASE)})


def _load_engine(model: Path, moe_chunk: int):
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    from studio_compute.flash_model import QwenFlashModel
    from studio_compute.runtime import RuntimeEngine

    engine = RuntimeEngine(
        str(model), model_factory=QwenFlashModel, execution_profile="bf16-fast",
        activation_budget_gib=16, inference_concurrency=8, prefill_tokens=1024,
        segment_tokens=SEGMENT_TOKENS, head_rows=HEAD_ROWS,
        replay_layout="segment_trie", training_moe_chunk=moe_chunk,
    )
    factors = engine.resident.adapter.initial(20261019)
    # A small nonzero B keeps every LoRA product live in the gradient graph.
    factors = {k: (v + 1e-4 if k.endswith(".b") else v) for k, v in factors.items()}
    engine.mx.eval(factors)
    engine.resident.select(factors)
    engine.resident.mode(True)
    return engine


def _capture_segment(engine, sequence: dict, segment: int) -> dict:
    """Advance the real sequence to one segment and keep every layer's real inputs."""
    import mlx.core as mx
    from studio_compute.numerics.replay import Replay

    tokens = sequence["prompt_token_ids"] + sequence["sampled_tokens"][:-1]
    ids = mx.array([tokens], mx.int32)
    embeddings = engine.replay_ops.expand(
        engine.lm, engine.lm.model.embed_tokens(ids).astype(mx.bfloat16))
    positions = mx.broadcast_to(mx.arange(ids.shape[1], dtype=mx.int32)[None, None],
                                (3, 1, ids.shape[1]))
    replay = Replay(engine.resident.model, checkpoint_layers=2,
                    segment_tokens=engine.segment_tokens, head_rows=engine.head_rows,
                    execution_profile=engine.execution_profile,
                    model_ops=engine.replay_ops)
    branch = {"embeddings": embeddings, "positions": positions, "input_ids": ids,
              "prefix_length": len(sequence["prompt_token_ids"]),
              "tokens": sequence["sampled_tokens"], "target_indices": [0]}
    states = engine.replay_ops.initial_states(replay.layers, mx.bfloat16)
    work = replay._forward(branch, 0, segment, states, record=False, phase="warm")
    while True:
        try:
            next(work)
        except StopIteration as stop:
            states = stop.value[3]
            break
    start = segment * engine.segment_tokens
    stop = min(ids.shape[1], start + engine.segment_tokens)
    hidden = embeddings[:, start:stop]
    inputs = []
    for index, layer in enumerate(replay.layers):
        inputs.append((hidden, states[index]))
        hidden, *_ = replay._layer(layer, hidden, [value for _, value in states[index]],
                                   positions[:, :, start:stop],
                                   replay._context(branch, start, stop))
        mx.eval(hidden)
    return {"replay": replay, "branch": branch, "inputs": inputs, "output": hidden,
            "positions": positions[:, :, start:stop],
            "context": replay._context(branch, start, stop)}


def _profile_layer(engine, captured: dict, index: int, repeats: int) -> dict:
    """
    Purpose: Time one real layer segment and each of its components.
    Input: The engine, the captured segment, a layer index, and the repeat count.
    Output: Median seconds per "<component>_forward" and "<component>_vjp" key.
    Note: A VJP time includes graph build and evaluation. Each VJP differentiates
        the component inputs and the trainable LoRA factors in its module.
    """
    import mlx.core as mx

    replay, layer = captured["replay"], engine.lm.layers[index]
    hidden, sources = captured["inputs"][index]
    incoming = [value for _, value in sources]
    positions, context = captured["positions"], captured["context"]
    out = replay._layer(layer, hidden, incoming, positions, context)
    mx.eval(out)
    cotangents = [_noise(out[0])]
    for kind in range(engine.replay_ops.carry_count(layer)):
        live = kind + 1 < len(out) and engine.replay_ops.differentiable_carry(layer, kind)
        cotangents.append(_noise(out[kind + 1]) if live else None)
    times = {
        "layer_forward": _timed(lambda: replay._layer(
            layer, hidden, incoming, positions, context), repeats),
        "layer_vjp": _timed(lambda: replay._layer_vjp(
            layer, hidden, sources, positions, cotangents, None, context)[1], repeats),
    }
    parts = _layer_parts(engine, layer, hidden, incoming, positions, context)
    for name, (module, function, value) in parts.items():
        times[name + "_forward"] = _timed(lambda: function(value), repeats)
        times[name + "_vjp"] = _timed(
            lambda: _module_vjp(module, function, value), repeats)
    for stage in ("forward", "vjp"):
        times["mixer_core_" + stage] = max(
            0.0, times["mixer_full_" + stage] - times["mixer_proj_" + stage])
        times["moe_routed_" + stage] = max(
            0.0, times["moe_full_" + stage] - times["moe_shared_" + stage])
        known = sum(times.get(f"{name}_{stage}", 0.0)
                    for name in ("ple", "attn_hc", "mixer_full", "mlp_hc", "moe_full"))
        times["other_" + stage] = max(0.0, times["layer_" + stage] - known)
    return times


def _layer_parts(engine, layer, hidden, incoming, positions, context) -> dict:
    """Return {component: (module, function, real input)} for one layer."""
    import mlx.core as mx
    from studio_compute.numerics import flash_ops
    from studio_compute.numerics.bf16 import gdn_forward
    from studio_compute.numerics.hyper_metal import hyper_connection

    parts = {}
    if "ple" in layer:
        parts["ple"] = (layer.ple, lambda x: _ple(layer, x, incoming, context), hidden)
        hidden = hidden + _ple(layer, hidden, incoming, context)
    attn_hc = layer.attn_hyper_connection
    parts["attn_hc"] = (attn_hc, lambda x: list(hyper_connection(attn_hc, x)), hidden)
    mixed, hyper, injection = hyper_connection(attn_hc, hidden)
    if layer.is_linear:
        mixer = layer.linear_attn
        full = lambda x: list(gdn_forward(mixer, x, incoming[:2],
                                          gate_activation=mixer.norm.activation,
                                          backend="metal_fused"))
        names, out_name = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"), \
            "out_proj"
        out_width = mixer.out_proj.base.weight.shape[-1] * 32 // mixer.out_proj.base.bits
    else:
        mixer = layer.self_attn
        full = lambda x: [flash_ops._attention(mixer, x, incoming, positions)[0]]
        names, out_name = ("q_proj", "k_proj", "v_proj"), "o_proj"
        out_width = mixer.o_proj.base.weight.shape[-1] * 32 // mixer.o_proj.base.bits
    out_input = mx.random.normal((*mixed.shape[:2], out_width)).astype(mx.bfloat16)

    def projections(x):
        values = [getattr(mixer, name)(x) for name in names]
        if not layer.is_linear:
            values.append(mixer.indexer.index_qk_proj(x))
        values.append(getattr(mixer, out_name)(out_input))
        return values

    parts["mixer_proj"] = (mixer, projections, mixed)
    parts["mixer_full"] = (mixer, full, mixed)
    branch = full(mixed)[0]
    hidden = hyper + (branch[..., None, :] * injection[..., None]).reshape(hyper.shape)
    mlp_hc = layer.mlp_hyper_connection
    parts["mlp_hc"] = (mlp_hc, lambda x: list(hyper_connection(mlp_hc, x)), hidden)
    moe_input = hyper_connection(mlp_hc, hidden)[0]
    chunk = engine.replay_ops.training_moe_chunk
    shared = lambda x: layer.mlp.shared_expert(x) * layer.mlp._shared_expert_scale(x)
    if chunk:
        moe = lambda x: flash_ops._moe_sorted(layer.mlp, x)
    else:
        moe = lambda x: flash_ops._moe(layer.mlp, x)
    parts["moe_full"] = (layer.mlp, moe, moe_input)
    parts["moe_shared"] = (layer.mlp, shared, moe_input)
    mx.eval(hidden, moe_input, out_input)
    return parts


def _ple(layer, hidden, incoming, context):
    """Apply only the PLE residual term of FlashOps.layer_math."""
    import mlx.core as mx
    from mlx_vlm.models.cache import ArraysCache

    ids_BL, preceding_BL = context
    cache = ArraysCache(size=4)
    embedding = layer.ple.ple_embedding
    previous = mx.full((ids_BL.shape[0], embedding.context_len),
                       embedding.eos_token_id, mx.int64)
    if preceding_BL.shape[1]:
        previous[:, -preceding_BL.shape[1]:] = preceding_BL.astype(mx.int64)
    cache.cache = [*incoming[:2], incoming[2], previous]
    return layer.ple(hidden, ids_BL, cache, None)


def _profile_group(engine, captured: dict, layers: int, repeats: int) -> dict:
    """
    Purpose: Compare the two backward schedules of one checkpoint group.
    Input: The engine, the captured segment, and the group size from layer 0.
    Output: Seconds and peak transient bytes of each schedule.
    Note: "per_layer" recomputes the group inputs, then runs one VJP per layer.
        "group" runs one VJP over the whole group from its checkpoint.
    """
    import mlx.core as mx

    replay, ops = captured["replay"], engine.replay_ops
    indices = range(layers)
    positions, context = captured["positions"], captured["context"]
    sources = [captured["inputs"][index][1] for index in indices]
    hidden = captured["inputs"][0][0]
    carries = []
    for index in indices:
        layer = engine.lm.layers[index]
        out = replay._layer(layer, captured["inputs"][index][0],
                            [value for _, value in sources[index]], positions, context)
        carries.append([_noise(out[kind + 1]) if ops.differentiable_carry(layer, kind)
                        else None for kind in range(ops.carry_count(layer))])
    top = _noise(captured["inputs"][layers][0] if layers < len(engine.lm.layers)
                 else captured["output"])
    mx.eval(carries, top)

    def per_layer():
        inputs, value = [], hidden
        for index in indices:
            inputs.append(value)
            if index + 1 < layers:
                value = replay._layer(engine.lm.layers[index], value,
                                      [v for _, v in sources[index]], positions,
                                      context)[0]
                mx.eval(value)
        cotangent = top
        for index in reversed(indices):
            names, back = replay._layer_vjp(engine.lm.layers[index], inputs[index],
                                            sources[index], positions,
                                            [cotangent, *carries[index]], None, context)
            cotangent = back[len(names)]
            mx.eval(back)
        return cotangent

    def group():
        return replay._group_vjp(indices, hidden, sources, positions, top, carries,
                                 context)[1]

    result = {"layers": layers}
    for name, function in (("per_layer", per_layer), ("group", group)):
        mx.eval(function())
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        result[name + "_s"] = _timed(function, repeats)
        result[name + "_peak_transient_bytes"] = mx.get_peak_memory() - base
    print(json.dumps({"group": {k: round(v, 3) if isinstance(v, float) else v
                                for k, v in result.items()}}), flush=True)
    return result


def _profile_head(engine, captured: dict, repeats: int) -> dict:
    """Time one 128-row head tile forward and its VJP as trie replay runs them."""
    import mlx.core as mx

    replay = captured["replay"]
    rows_LD = captured["output"][0, :HEAD_ROWS]
    tokens_L = captured["branch"]["input_ids"][0, :HEAD_ROWS]
    cotangent_L = _noise(replay.head(rows_LD, tokens_L))
    return {
        "forward": _timed(lambda: replay.head(rows_LD, tokens_L), repeats),
        "vjp": _timed(lambda: mx.vjp(lambda x: replay.head(x, tokens_L),
                                     [rows_LD], [cotangent_L])[1], repeats),
    }


def _module_vjp(module, function, value):
    """VJP with respect to the input and the module's trainable LoRA factors."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    flat = tree_flatten(module.trainable_parameters())
    names, values = [name for name, _ in flat], [value for _, value in flat]
    outputs = function(value)
    outputs = outputs if isinstance(outputs, list) else [outputs]
    cotangents = [_noise(output) for output in outputs]

    def unit(*arguments):
        module.update(tree_unflatten(list(zip(names, arguments[:-1]))))
        result = function(arguments[-1])
        return result if isinstance(result, list) else [result]

    try:
        return mx.vjp(unit, [*values, value], cotangents)[1]
    finally:
        module.update(tree_unflatten(list(zip(names, values))))


def _noise(value):
    import mlx.core as mx

    return (mx.random.normal(value.shape) * 1e-3).astype(value.dtype)


def _fit(samples: list, context: int) -> dict:
    """Least-squares a + b * context per key; one sample gives a constant."""
    if len(samples) == 1:
        return dict(samples[0][1])
    xs = [x for x, _ in samples]
    mean_x = sum(xs) / len(xs)
    spread = sum((x - mean_x) ** 2 for x in xs)
    result = {}
    for key in samples[0][1]:
        ys = [times[key] for _, times in samples]
        mean_y = sum(ys) / len(ys)
        slope = 0.0 if spread == 0 else sum(
            (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / spread
        result[key] = max(0.0, mean_y + slope * (context - mean_x))
    return result


def _trie_contexts(nodes: list) -> list:
    """Return the prior-token count of every forward segment in a trie."""
    contexts = []
    for first, stop, _, children in nodes:
        contexts.extend(segment * SEGMENT_TOKENS for segment in range(first, stop))
        contexts.extend(_trie_contexts(children))
    return contexts


def _head_tiles(sequence: dict) -> int:
    """Count the 128-row head tiles of one sequence, per segment, as replay forms them."""
    prefix = len(sequence["prompt_token_ids"])
    mask = sequence.get("loss_mask") if sequence.get("source_spans") else None
    targets = [prefix - 1 + i for i in range(len(sequence["sampled_tokens"]))
               if mask is None or mask[i]]
    counts = {}
    for row in targets:
        counts[row // SEGMENT_TOKENS] = counts.get(row // SEGMENT_TOKENS, 0) + 1
    return sum(-(-count // HEAD_ROWS) for count in counts.values())


def _qsa_keys(config, query_end: int) -> int:
    """Return the keys one QSA query reads: dense below the budget, else top-k plus tail."""
    ratio = config.indexer_compress_ratio
    complete = (query_end + 1) // ratio
    if complete <= config.indexer_budget // ratio:
        return query_end + 1
    return config.indexer_budget + (query_end + 1) % ratio


def _timed(function, repeats: int) -> float:
    """Return the median wall seconds of build plus evaluation, after one warm run."""
    import mlx.core as mx

    mx.eval(function())
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        mx.eval(function())
        samples.append(time.perf_counter() - started)
    return sorted(samples)[len(samples) // 2]


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, default=str))
    temporary.replace(path)


if __name__ == "__main__":
    main()
