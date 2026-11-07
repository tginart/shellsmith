"""
Purpose: Execute Qwen Sparse Attention for training replay with sparse Metal kernels.
Input: BF16 queries, keys, and values, and the indexer's block selection.
Output: BF16 attention values with a custom derivative for Q, K, and V.
Method:
    1. Each query reads only its selected key blocks and its incomplete tail block.
    2. The forward and backward kernels own one query row and one KV head. The
       query heads of the KV group form the rows of 16x16 NAX fragments.
    3. The backward runs two kernels. The dQ kernel owns one query row, as the
       forward does, and also writes P and dS in BF16. The dK and dV kernel owns
       one 16-key tile. It reads P and dS of the queries that selected its
       blocks. So no kernel needs atomics, and all three gradients are BF16.
Note: The key set of each query is the set that indexer.build_mask allows. No
    dense mask and no product against unselected keys exist, except the
    unselected blocks of a tile in the dK and dV kernel, which are masked.
    Register accumulators use FP32.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlx.core as mx

# L: query rows, H: query heads, J: KV heads, K: key rows, D: head width (256),
# S: selection slots per query.
SHADER_ROOT = Path(__file__).with_name("metal")
HEAD_WIDTH = 256
THREADS = 128
# Keys per dK and dV tile: four key blocks of the indexer.
TILE_KEYS = 16
BLOCK_KEYS = 4


def selection_slots(selection: Any) -> tuple:
    """
    Purpose: Convert one indexer selection to sorted block slots and query ends.
    Input: The SimpleNamespace that indexer.select returns, for batch one.
    Output: slots_LS (int32, -1 last) and ends_L (int32).
    Note: The slots hold the selected complete blocks and the incomplete tail
        block. A key t of block b is visible when t < the query end.
    """
    assert selection.selected_blocks.shape[0] == 1
    assert selection.zero_padding, "Sparse replay attention requires no left padding."
    blocks_LS = selection.selected_blocks[0].astype(mx.int32)
    tail_L1 = selection.complete_counts[0].astype(mx.int32)[:, None]
    slots_LS = mx.concatenate([blocks_LS, tail_L1], axis=1)
    pad = (-slots_LS.shape[1]) % 8
    if pad:
        slots_LS = mx.pad(slots_LS, [(0, 0), (0, pad)], constant_values=-1)
    # Ascending block order gives each chunk nearby rows; invalid slots go last.
    large = mx.array(2**31 - 1, mx.int32)
    ordered_LS = mx.sort(mx.where(slots_LS >= 0, slots_LS, large), axis=-1)
    slots_LS = mx.where(ordered_LS == large, -1, ordered_LS)
    ends_L = selection.query_ends[0].astype(mx.int32)
    return mx.stop_gradient(slots_LS), mx.stop_gradient(ends_L)


def sparse_attention(
    q_BLHD: Any, k_BJKD: Any, v_BJKD: Any, slots_LS: Any, ends_L: Any, scale: float
) -> Any:
    """
    Purpose: Apply QSA attention to every query row of one replay segment.
    Input: q (1, L, H, D), k and v (1, J, K, D), slots and ends from
        selection_slots, and the softmax scale.
    Output: o (1, L, H, D) BF16 with a custom VJP for q, k, and v.
    """
    batch, length, heads, width = q_BLHD.shape
    assert batch == 1 and width == HEAD_WIDTH and k_BJKD.shape == v_BJKD.shape
    assert k_BJKD.shape[0] == 1 and heads % k_BJKD.shape[1] == 0
    assert slots_LS.shape[0] == length == ends_L.shape[0]
    assert all(value.dtype == mx.bfloat16 for value in (q_BLHD, k_BJKD, v_BJKD))
    scale_1 = mx.array([scale], mx.float32)
    output_BLHD, _ = _sparse(q_BLHD, k_BJKD, v_BJKD, slots_LS, ends_L, scale_1)
    return output_BLHD


@mx.custom_function
def _sparse(q_BLHD, k_BJKD, v_BJKD, slots_LS, ends_L, scale_1):
    return _forward(q_BLHD, k_BJKD, v_BJKD, slots_LS, ends_L, scale_1)


@_sparse.vjp
def _sparse_vjp(primals: tuple, cotangents: tuple, outputs: tuple) -> tuple:
    q_BLHD, k_BJKD, v_BJKD, slots_LS, ends_L, scale_1 = primals
    output_BLHD, lse_LH = outputs
    dy_BLHD = cotangents[0]
    dq, dk, dv = _backward(
        q_BLHD, k_BJKD, v_BJKD, slots_LS, ends_L, scale_1, output_BLHD, lse_LH,
        dy_BLHD,
    )
    return (dq, dk, dv, mx.zeros_like(slots_LS), mx.zeros_like(ends_L),
            mx.zeros_like(scale_1))


def _forward(q_BLHD, k_BJKD, v_BJKD, slots_LS, ends_L, scale_1) -> tuple:
    """Run the forward kernel. Return o (1, L, H, D) and lse (L, H) in log2 units."""
    _, length, heads, _ = q_BLHD.shape
    kv_heads, keys = k_BJKD.shape[1:3]
    counts_L = mx.sum(slots_LS >= 0, axis=-1).astype(mx.int32)
    dims_3 = mx.array([length, keys, slots_LS.shape[1]], mx.int32)
    output_LHD, lse_LH = _kernel("forward")(
        inputs=[_padded(q_BLHD), _contiguous(k_BJKD), _contiguous(v_BJKD),
                mx.contiguous(slots_LS), counts_L, mx.contiguous(ends_L), dims_3,
                scale_1],
        template=_template(heads, kv_heads),
        grid=(length * THREADS, kv_heads, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(length, heads, HEAD_WIDTH), (length, heads)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )
    return output_LHD[None], lse_LH


def _backward(q_BLHD, k_BJKD, v_BJKD, slots_LS, ends_L, scale_1, output_BLHD,
              lse_LH, dy_BLHD) -> tuple:
    """
    Purpose: Calculate dQ with a query-major kernel, and dK and dV with a
        key-major kernel.
    Input: The forward primals and outputs, and the output cotangent.
    Output: dq (1, L, H, D), dk and dv (1, J, K, D), all BF16.
    """
    _, length, heads, _ = q_BLHD.shape
    kv_heads, keys = k_BJKD.shape[1:3]
    dy_BLHD = dy_BLHD.astype(mx.bfloat16)
    delta_LH = mx.sum(dy_BLHD[0] * output_BLHD[0], axis=-1).astype(mx.float32)
    counts_L = mx.sum(slots_LS >= 0, axis=-1).astype(mx.int32)
    shared = [_padded(q_BLHD), _contiguous(k_BJKD), _contiguous(v_BJKD),
              _padded(dy_BLHD), _padded_rows(lse_LH), _padded_rows(delta_LH)]
    width = slots_LS.shape[1]
    scores = (length * width * BLOCK_KEYS * heads,)
    dq_LHD, p_LSTH, ds_LSTH = _kernel("backward_dq")(
        inputs=[*shared, mx.contiguous(slots_LS), counts_L, mx.contiguous(ends_L),
                mx.array([length, keys, width], mx.int32), scale_1],
        template=_template(heads, kv_heads),
        grid=(length * THREADS, kv_heads, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(length, heads, HEAD_WIDTH), scores, scores],
        output_dtypes=[mx.bfloat16] * 3,
    )
    tiles = -(-keys // TILE_KEYS)
    entries_E, offsets_T, order_T = _tile_lists(slots_LS, tiles)
    dk_JKD, dv_JKD = _kernel("backward_dkv")(
        inputs=[shared[0], shared[3], p_LSTH, ds_LSTH, entries_E, offsets_T, order_T,
                mx.array([length, keys, tiles, width], mx.int32), scale_1],
        template=_template(heads, kv_heads),
        grid=(tiles * THREADS, kv_heads, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(kv_heads, keys, HEAD_WIDTH)] * 2,
        output_dtypes=[mx.bfloat16] * 2,
    )
    return dq_LHD[None], dk_JKD[None], dv_JKD[None]


def _tile_lists(slots_LS: Any, tiles: int) -> tuple:
    """
    Purpose: List, for each 16-key tile, the queries that selected its blocks.
    Input: Sorted block slots (L, S) with -1 last, and the tile count T.
    Output: entries (E,) int32, query << 14 | first slot << 4 | block mask,
        sorted by tile and query; offsets (T + 1,), so tile t owns entries [offsets[t], offsets[t + 1]);
        and the tiles by list length, longest first.
    Note: Sorted slots put the blocks of one tile side by side in a row, so the
        mask of a row and tile is the OR over the next three slots. The recent
        tiles have the longest lists. The kernel starts them first, so they do
        not run alone at the end.
    """
    length, width = slots_LS.shape
    per_tile = TILE_KEYS // BLOCK_KEYS
    valid_LS = slots_LS >= 0
    tile_LS = mx.where(valid_LS, slots_LS // per_tile, -1)
    bit_LS = mx.where(valid_LS, mx.left_shift(1, slots_LS % per_tile), 0)
    mask_LS = bit_LS
    for shift in range(1, per_tile):
        ahead_tile = mx.pad(tile_LS[:, shift:], [(0, 0), (0, shift)], constant_values=-2)
        ahead_bit = mx.pad(bit_LS[:, shift:], [(0, 0), (0, shift)])
        mask_LS = mask_LS | mx.where(ahead_tile == tile_LS, ahead_bit, 0)
    previous_LS = mx.pad(tile_LS[:, :-1], [(0, 0), (1, 0)], constant_values=-2)
    first_LS = valid_LS & (tile_LS != previous_LS)
    query_LS = mx.broadcast_to(mx.arange(length, dtype=mx.int32)[:, None], (length, width))
    order_key = mx.where(first_LS, tile_LS * length + query_LS, 2**31 - 1).reshape(-1)
    order = mx.argsort(order_key)
    assert length < 2**17 and width < 2**10
    slot_LS = mx.broadcast_to(mx.arange(width, dtype=mx.int32)[None], (length, width))
    packed_LS = mx.left_shift(query_LS, 14) | mx.left_shift(slot_LS, 4) | mask_LS
    entries_E = mx.take(packed_LS.reshape(-1), order).astype(mx.int32)
    counts_T = mx.zeros((tiles + 1,), mx.int32).at[
        mx.where(first_LS, tile_LS, tiles).reshape(-1)].add(1)
    offsets_T = mx.concatenate([mx.zeros((1,), mx.int32), mx.cumsum(counts_T[:tiles])])
    order_T = mx.argsort(-counts_T[:tiles]).astype(mx.int32)
    return tuple(mx.stop_gradient(value) for value in
                 (entries_E, offsets_T.astype(mx.int32), order_T))


def _template(heads: int, kv_heads: int) -> list:
    group = heads // kv_heads
    assert group <= 16, "One 16-row fragment holds the query heads of a KV group."
    return [("H", heads), ("G", group)]


def _padded(value_BLHD: Any) -> Any:
    """Flatten to (L * H + 16, D), so the 16 fragment rows of a group stay in bounds."""
    rows_RD = value_BLHD[0].reshape(-1, value_BLHD.shape[-1])
    return mx.contiguous(mx.pad(rows_RD, [(0, 16), (0, 0)]))


def _padded_rows(value_LH: Any) -> Any:
    return mx.contiguous(mx.pad(value_LH.reshape(-1), [(0, 16)]))


def _contiguous(value_BJKD: Any) -> Any:
    return mx.contiguous(value_BJKD[0])


@lru_cache(maxsize=4)
def _kernel(phase: str) -> Any:
    """Compile one repository-owned sparse QSA shader, keyed by its source hash."""
    header = (SHADER_ROOT / "gdn_nax_common.metal").read_text()
    header += (SHADER_ROOT / "qsa_sparse_common.metal").read_text()
    source = (SHADER_ROOT / f"qsa_sparse_{phase}.metal").read_text()
    names = {
        "forward": (["q", "k", "v", "slots", "counts", "ends", "dims", "scale"],
                    ["o", "lse"]),
        "backward_dq": (["q", "k", "v", "dy", "lse", "delta", "slots", "counts",
                         "ends", "dims", "scale"], ["dq", "p_store", "ds_store"]),
        "backward_dkv": (["q", "dy", "p_store", "ds_store", "entries", "offsets",
                          "order", "dims", "scale"], ["dk", "dv"]),
    }[phase]
    identity = hashlib.sha256((header + source).encode()).hexdigest()[:12]
    return mx.fast.metal_kernel(
        name=f"shellsmith_qsa_sparse_{phase}_{identity}",
        input_names=names[0],
        output_names=names[1],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )
