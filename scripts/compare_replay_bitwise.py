"""
Purpose: Compare the trie replay outputs of two source snapshots, bit for bit.
Input: run: reconstructed update sequences, indices, the model, and the trie
    sum levels. compare: two run directories and gradient tolerances.
Output: run: logprobs, loss, and gradients of one forward_backward.
    compare: a JSON verdict with bitwise and tolerance results.
Method:
    1. run takes the GPU lease and builds the engine with the production replay options.
    2. It nudges the LoRA B factors, so that every A factor gets a gradient.
    3. It runs one trie forward_backward and saves the accumulated gradients.
    4. compare checks logprobs, loss, and every gradient tensor for equal bits.
       It also gives the gradient relative L2 error and the cosine.
Note: Use the same script file with the PYTHONPATH of each snapshot. A fixed
    checkpoint interval and host headroom keep the replay plan the same.
    --fixed-routes replaces the MoE top-k routes and the QSA block top-k with
    fixed functions of the layer and the position. Then no BF16 rounding can flip
    a discrete choice, and two correct sources give close gradients.
    --reference-kernels puts named training kernels of this branch back to
    their MLX paths, to find which kernel moves the gradient.
    --fp32-reference runs the activations, carries, and gradients in FP32 with
    the MLX reference paths (python -O removes the BF16 asserts). With
    --fixed-routes, it gives the reference gradient for both BF16 sources.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import time
from pathlib import Path

LEASE = Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"
ROUTE_SHIFTS = 64


def main() -> None:
    args = _parse_args()
    if args.command == "run":
        _run(args)
    else:
        _compare(args)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run one trie forward_backward.")
    run.add_argument("--sequences", type=Path, required=True,
                     help="Reconstructed update JSON.")
    run.add_argument("--model", type=Path, required=True, help="Flash model path.")
    run.add_argument("--output", type=Path, required=True, help="New output directory.")
    run.add_argument("--indices", type=int, nargs="+", default=[0, 1, 11, 12],
                     help="Update sequence indices.")
    run.add_argument("--checkpoint-layers", type=int, default=2,
                     help="Fixed checkpoint interval (perf4 update 1 used 2).")
    run.add_argument("--headroom-gib", type=float, default=32.0,
                     help="Host headroom; the same value keeps the same budget.")
    run.add_argument("--sum-levels", default="auto",
                     help="auto: admission chooses; exact: only exact sums; N: cap N.")
    run.add_argument("--segment-tokens", type=int, default=2048,
                     help="Replay segment length (perf4 used 2048).")
    run.add_argument("--fixed-routes", action="store_true",
                     help="Fix the MoE routes and QSA blocks by layer and position.")
    run.add_argument("--reference-kernels", nargs="*", default=[],
                     choices=["hyper", "gdn", "experts", "qsa"],
                     help="Training kernels to replace with their MLX reference paths.")
    run.add_argument("--fp32-reference", action="store_true",
                     help="FP32 activations and gradients; needs python -O.")
    compare = commands.add_parser("compare", help="Compare two run directories.")
    compare.add_argument("reference", type=Path, help="Reference run directory.")
    compare.add_argument("candidate", type=Path, help="Candidate run directory.")
    compare.add_argument("--max-relative-l2", type=float, default=0.0,
                         help="Gradient tolerance; 0 requires equal bits.")
    compare.add_argument("--min-cosine", type=float, default=1.0,
                         help="Gradient cosine tolerance.")
    compare.add_argument("--output", type=Path, required=True, help="Verdict JSON.")
    return parser.parse_args()


def _run(args: argparse.Namespace) -> None:
    """Run one forward_backward with the GPU lease and save its outputs."""
    args.output.mkdir(parents=True)
    lease = LEASE.open("a")
    fcntl.flock(lease, fcntl.LOCK_EX)  # Wait in line for the shared lease.
    _write_identity(args.output / "process.json")
    payload = json.loads(args.sequences.read_text())
    sequences = [payload["sequences"][index] for index in args.indices]
    engine = _load_engine(args)
    import mlx.core as mx

    if args.fixed_routes:
        _fix_discrete_choices(engine.resident.lm)
    _use_reference_kernels(args.reference_kernels)
    if args.fp32_reference:
        _use_fp32_activations(engine)
    admission = _force_sum_levels(engine, args.sum_levels)
    adapter = engine.clients["validation"]["adapter"]
    adapter.load({name: value + 1e-4 if name.endswith(".b") else value
                  for name, value in adapter.state().items()})
    body = {"operation_id": f"bitwise-{time.time()}", "sequences": sequences,
            "loss": "cispo", "reduction": "sum", "clip_low": 0.0, "clip_high": 4.0}
    mx.reset_peak_memory()
    started = time.monotonic()
    result = _drain(engine.work("validation", "forward_backward", body))
    gradients = dict(engine.clients["validation"]["optimizer"].accumulated)
    mx.save_safetensors(str(args.output / "gradients.safetensors"), gradients)
    _write_json(args.output / "result.json", {
        "indices": args.indices, "seconds": time.monotonic() - started,
        "peak_memory_bytes": mx.get_peak_memory(),
        "loss": result["forward_metrics"]["loss"], "logprobs": result["logprobs"],
        "stats": [{key: value for key, value in group.items()
                   if not isinstance(value, (list, dict))}
                  for group in result["groups"]],
        "sum_levels": args.sum_levels, "admission": admission,
        "segment_tokens": args.segment_tokens, "fixed_routes": args.fixed_routes,
        "reference_kernels": args.reference_kernels,
        "fp32_reference": args.fp32_reference,
        "pythonpath": os.environ.get("PYTHONPATH"),
    })


def _force_sum_levels(engine, sum_levels: str) -> dict:
    """
    Purpose: Fix the trie sum levels and record what admission chooses.
    Input: The engine and auto, exact, or a level cap.
    Output: A dict that admission fills with the interval and the levels.
    Note: Source without trie gradient admission has exact sums only.
    """
    from studio_compute import engine as engine_module

    record = {}
    if not hasattr(engine, "_tree_admission"):
        assert sum_levels in ("auto", "exact"), "This source has exact sums only."
        record["levels"] = "exact (source has no level caps)"
        return record
    if sum_levels != "auto":
        engine_module.TREE_SUM_LEVELS = ((None,) if sum_levels == "exact"
                                         else (int(sum_levels),))
    original = engine._tree_admission

    def wrapped(sequences, future_gradient_bytes):
        interval, peak, budgets, levels = original(sequences, future_gradient_bytes)
        record.update(interval=interval, levels=levels, peak_bytes=peak)
        return interval, peak, budgets, levels

    engine._tree_admission = wrapped
    return record


def _load_engine(args: argparse.Namespace):
    """Build the engine with the perf4 replay options and one fresh client."""
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    from studio_compute.flash_model import QwenFlashModel
    from studio_compute.runtime import RuntimeEngine

    engine = RuntimeEngine(
        str(args.model), model_factory=QwenFlashModel, execution_profile="bf16-fast",
        headroom_gib=args.headroom_gib, activation_budget_gib=16,
        inference_concurrency=8, prefill_tokens=1024,
        segment_tokens=args.segment_tokens,
        head_rows=128, replay_layout="segment_trie", training_moe_chunk=32,
        prefill_moe_chunk=32, checkpoint_layers=args.checkpoint_layers,
    )
    engine.resident.mode(False)
    engine.create("validation", {
        "seed": 20261019, "lora_rank": 16, "lora_scale": 2.0, "learning_rate": 5e-5,
        "optimizer": {"beta1": 0.9, "beta2": 0.95, "epsilon": 1e-8},
    })
    return engine


def _fix_discrete_choices(lm) -> None:
    """
    Purpose: Make the MoE routes and the QSA block selection fixed functions.
    Input: The Flash language model of the engine.
    Output: None. The function patches flash_ops and the upstream indexer.
    Method:
        1. A layer_math wrapper keeps the positions of the current segment.
        2. MoE wrappers keep the layer index. In that scope, the flash_ops
           argpartition gives the experts (shift(p) + offset_k) mod E.
        3. The indexer select keeps its cache updates. Its block top-k uses a
           fixed hash score of the block, inside the causal limit of each query.
    Note: The route scores and the attention stay differentiable. Only the
        discrete IDs change, so both sources replay the same smooth function.
        Empty and large expert segments both occur, as with real routes.
    """
    from types import SimpleNamespace

    import mlx.core as mx
    from mlx_vlm.models.qwen4_exp import language as qwen4
    from studio_compute.numerics import flash_ops

    moe_layers = {id(layer.mlp): index for index, layer in enumerate(lm.layers)}
    indexer_layers = {id(layer.self_attn.indexer): index
                      for index, layer in enumerate(lm.layers)
                      if not layer.is_linear}
    scope = {"positions": None, "layer": None}

    layer_math = flash_ops.FlashOps.layer_math

    def fixed_layer_math(layer, hidden_BLD, states, positions, context, **options):
        scope["positions"] = positions
        return layer_math(layer, hidden_BLD, states, positions, context, **options)

    flash_ops.FlashOps.layer_math = staticmethod(fixed_layer_math)
    for name in ("_moe", "_moe_chunked", "_moe_sorted"):
        if hasattr(flash_ops, name):
            setattr(flash_ops, name, _in_layer_scope(getattr(flash_ops, name),
                                                     scope, moe_layers))

    class FixedRouteCore:
        def __getattr__(self, name):
            return getattr(mx, name)

        def argpartition(self, gates_BLE, kth, axis=-1):
            if scope["layer"] is None:
                return mx.argpartition(gates_BLE, kth=kth, axis=axis)
            return _fixed_experts(gates_BLE.shape, -kth, scope)

    flash_ops.mx = FixedRouteCore()
    select = qwen4.Qwen4ExpQSAIndexer.select

    def fixed_select(self, hidden_states, cache, position_ids):
        selection = select(self, hidden_states, cache, position_ids)
        if selection is None:
            return None
        blocks_BLS = _fixed_blocks(selection, indexer_layers[id(self)],
                                   self.block_topk)
        return SimpleNamespace(**{**vars(selection), "selected_blocks": blocks_BLS})

    qwen4.Qwen4ExpQSAIndexer.select = fixed_select


def _use_reference_kernels(names: list[str]) -> None:
    """
    Purpose: Replace named training kernels with the MLX paths that they replaced.
    Input: Names from hyper, gdn, experts, and qsa.
    Output: None. The function patches flash_ops and flash_adapter.
    Note: hyper uses the upstream module call. gdn uses the substitution solve.
        experts uses the MLX gather_qmm VJP. qsa uses dense masked attention
        over query tiles, as the source before the sparse kernels did.
    """
    if not names:
        return
    from studio_compute.numerics import flash_adapter, flash_ops

    if "hyper" in names:
        flash_ops.hyper_connection = flash_ops._module_call
    if "gdn" in names:
        gdn_forward = flash_ops.gdn_forward

        def substitution_gdn(*args, backend, **kwargs):
            return gdn_forward(*args, backend="metal", **kwargs)

        flash_ops.gdn_forward = substitution_gdn
    if "experts" in names:
        flash_adapter.sorted_expert_qmm = (
            lambda x, base, experts: base(x, experts, sorted_indices=True))
    if "qsa" in names:
        flash_ops.sparse_attention = _dense_selected_attention


def _use_fp32_activations(engine) -> None:
    """
    Purpose: Make the replay keep FP32 activations, carries, and gradients.
    Input: The engine. Output: None. The function patches the engine and replay.
    Note: The quantized weights stay the same. The GDN layer uses the FP32
        reference chunk scan. Attention uses the dense masked path of the source.
    """
    import mlx.core as mx
    from studio_compute.numerics import flash_ops
    from studio_compute.numerics.replay import Replay

    if __debug__:
        raise SystemExit("Run the FP32 reference with python -O.")
    embed = engine._embed
    engine._embed = lambda ids_BL: embed(ids_BL).astype(mx.float32)
    initialize = Replay.__init__

    def fp32_init(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        self.gradient_dtype = mx.float32

    Replay.__init__ = fp32_init
    flash_ops.gdn_forward = _fp32_gdn_forward
    if hasattr(flash_ops, "hyper_connection"):
        flash_ops.hyper_connection = flash_ops._module_call
    if hasattr(flash_ops, "sparse_attention"):
        flash_ops.sparse_attention = _dense_selected_attention


def _fp32_gdn_forward(linear, x_BLD, states, *, gate_activation="silu",
                      backend=None):
    """Apply the GDN layer of bf16.gdn_forward in FP32 with the reference scan."""
    import mlx.core as mx
    import mlx.nn as nn
    from studio_compute.numerics.ops import gated_delta_chunked

    del backend
    conv_BCD, state_BHVK = states
    qkv_BLD, z_BLD, a_BLH, beta_BLH = [
        getattr(linear, name)(x_BLD)
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")]
    batch, length, _ = qkv_BLD.shape
    z_BLHV = z_BLD.reshape(batch, length, linear.num_v_heads, linear.head_v_dim)
    beta_BLH = mx.sigmoid(beta_BLH)
    g_BLH = mx.exp(-mx.exp(linear.A_log.astype(mx.float32))
                   * nn.softplus(a_BLH + linear.dt_bias.astype(mx.float32)))
    convolution_BLD = mx.concatenate([conv_BCD, qkv_BLD], axis=1)
    convolved_BLD = nn.silu(linear.conv1d(convolution_BLD))
    q_BLHK, k_BLHK, v_BLHV = [
        value.reshape(batch, length, heads, width)
        for value, heads, width in zip(
            mx.split(convolved_BLD, [linear.key_dim, 2 * linear.key_dim], axis=-1),
            (linear.num_k_heads, linear.num_k_heads, linear.num_v_heads),
            (linear.head_k_dim, linear.head_k_dim, linear.head_v_dim))]
    q_BLHK, k_BLHK = linear._normalize_qk(q_BLHK, k_BLHK)
    y_BLHV, next_BHVK = gated_delta_chunked(
        q_BLHK, k_BLHK, v_BLHV, g_BLH, beta_BLH, state_BHVK,
        compute_dtype=mx.float32, backend="reference")
    normalized_BLHV = mx.fast.rms_norm(
        y_BLHV, linear.norm.weight.astype(mx.float32), linear.norm.eps)
    gate_BLHV = (mx.sigmoid(z_BLHV) if gate_activation == "sigmoid"
                 else nn.silu(z_BLHV))
    mixed_BLD = (normalized_BLHV * gate_BLHV).reshape(batch, length, -1)
    taps_BCD = convolution_BLD[:, -(linear.conv_kernel_size - 1):]
    return linear.out_proj(mixed_BLD), taps_BCD + mx.zeros_like(taps_BCD), next_BHVK


def _dense_selected_attention(q_BLHD, k_BJKD, v_BJKD, slots_LS, ends_L, scale):
    """Run masked dense attention over the slots of selection_slots, in query tiles."""
    import mlx.core as mx
    from studio_compute.numerics.flash_ops import ATTENTION_SCORE_BYTES

    _, length, heads, _ = q_BLHD.shape
    keys = k_BJKD.shape[2]
    blocks = -(-keys // 4)
    safe_LS = mx.where(slots_LS >= 0, slots_LS, blocks)
    chosen_LN = mx.put_along_axis(
        mx.zeros((length, blocks + 1), mx.bool_), safe_LS,
        mx.ones(safe_LS.shape, mx.bool_), axis=-1)
    token_K = mx.arange(keys)
    mask_LK = chosen_LN[:, token_K // 4] & (token_K[None] < ends_L[:, None])
    q_BHLD = q_BLHD.transpose(0, 2, 1, 3)
    tile_size = max(1, min(128, ATTENTION_SCORE_BYTES // (heads * keys * 4)))
    outputs = []
    for start in range(0, length, tile_size):
        stop = min(length, start + tile_size)

        def tile(q, k, v, mask=mask_LK[start:stop]):
            return mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=mask[None, None])

        outputs.append(mx.checkpoint(tile)(q_BHLD[:, :, start:stop], k_BJKD, v_BJKD))
    return mx.concatenate(outputs, axis=2).transpose(0, 2, 1, 3)


def _in_layer_scope(moe, scope: dict, moe_layers: dict):
    """Wrap one flash_ops MoE function so that argpartition knows its layer."""

    def wrapped(mixer, hidden_BLD, **options):
        scope["layer"] = moe_layers[id(mixer)]
        try:
            return moe(mixer, hidden_BLD, **options)
        finally:
            scope["layer"] = None

    return wrapped


def _fixed_experts(shape: tuple, top_k: int, scope: dict):
    """Return (B, L, K) expert IDs from the layer and the token positions."""
    import mlx.core as mx
    import numpy as np

    batch, length, experts = shape
    positions_L = np.array(scope["positions"][0, 0]).astype(np.int64)
    assert positions_L.shape == (length,), positions_L.shape
    layer = scope["layer"]
    offsets_K = np.random.default_rng(layer).choice(experts, top_k, replace=False)
    shift_L = (positions_L * positions_L * 31 + 17 * layer) % ROUTE_SHIFTS
    experts_LK = (shift_L[:, None] + offsets_K[None]) % experts
    return mx.broadcast_to(mx.array(experts_LK.astype(np.uint32))[None],
                           (batch, length, top_k))


def _fixed_blocks(selection, layer: int, block_topk: int):
    """Return (B, L, S) blocks with the top fixed hash scores below each limit."""
    import mlx.core as mx

    counts_BL = selection.complete_counts
    blocks = max(int(mx.max(counts_BL).item()), block_topk)
    block_N = mx.arange(blocks, dtype=mx.int32)
    # 40503 is a unit mod the prime 65521, so the scores have no ties.
    score_N = ((block_N * 40503 + 7919 * layer) % 65521).astype(mx.float32)
    valid_BLN = block_N[None, None] < counts_BL[..., None]
    score_BLN = mx.where(valid_BLN, score_N[None, None], -mx.inf)
    selected = mx.argpartition(score_BLN, kth=-block_topk, axis=-1)[..., -block_topk:]
    selected = selected.astype(mx.int32)
    assert selected.shape == selection.selected_blocks.shape
    return mx.where(selected < counts_BL[..., None], selected, -1)


def _compare(args: argparse.Namespace) -> None:
    """Compare logprobs, loss, and gradients of two runs."""
    import mlx.core as mx

    left = json.loads((args.reference / "result.json").read_text())
    right = json.loads((args.candidate / "result.json").read_text())
    left_gradients = mx.load(str(args.reference / "gradients.safetensors"))
    right_gradients = mx.load(str(args.candidate / "gradients.safetensors"))
    assert set(left_gradients) == set(right_gradients), "Gradient keys differ."
    unequal, dot, left_norm, right_norm, error = [], 0.0, 0.0, 0.0, 0.0
    for name in sorted(left_gradients):
        a, b = left_gradients[name], right_gradients[name]
        assert a.shape == b.shape, name
        if a.dtype != b.dtype or not bool(
                mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item()):
            unequal.append(name)
        a, b = a.astype(mx.float32), b.astype(mx.float32)
        dot += float(mx.sum(a * b).item())
        left_norm += float(mx.sum(a * a).item())
        right_norm += float(mx.sum(b * b).item())
        error += float(mx.sum((a - b) ** 2).item())
    relative_l2 = math.sqrt(error / left_norm)
    cosine = dot / math.sqrt(left_norm * right_norm)
    verdict = {
        "reference": str(args.reference), "candidate": str(args.candidate),
        "logprobs_bitwise": left["logprobs"] == right["logprobs"],
        "loss_bitwise": left["loss"] == right["loss"],
        "gradient_tensors": len(left_gradients),
        "gradient_tensors_unequal": len(unequal),
        "first_unequal": unequal[:5],
        "gradient_relative_l2": relative_l2, "gradient_cosine": cosine,
        "max_relative_l2": args.max_relative_l2, "min_cosine": args.min_cosine,
        "peak_memory_gib": [left["peak_memory_bytes"] / 2**30,
                            right["peak_memory_bytes"] / 2**30],
        "seconds": [left["seconds"], right["seconds"]],
    }
    if args.max_relative_l2 == 0:
        verdict["passed"] = (verdict["logprobs_bitwise"] and verdict["loss_bitwise"]
                             and not unequal)
    else:
        verdict["passed"] = (verdict["logprobs_bitwise"] and verdict["loss_bitwise"]
                             and relative_l2 <= args.max_relative_l2
                             and cosine >= args.min_cosine)
    _write_json(args.output, verdict)
    print(json.dumps(verdict))


def _drain(work):
    while True:
        try:
            next(work)
        except StopIteration as stop:
            return stop.value


def _write_identity(path: Path) -> None:
    pid = os.getpid()
    started = subprocess.check_output(["ps", "-o", "lstart=,command=", "-p", str(pid)],
                                      text=True).strip()
    _write_json(path, {"pid": pid, "ps_start_and_command": started,
                       "lease": str(LEASE)})


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


if __name__ == "__main__":
    main()
