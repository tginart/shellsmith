"""
Purpose: Advance one inference token with BF16 recurrent state and no chunk padding.
Input: Normalized Q/K, values, gates, beta, and an owned BF16 state.
Output: One BF16 output and a new BF16 state. The function has no training VJP.
Note: SIMD reductions use float registers. No FP32 tensors exist.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import mlx.core as mx

# B: batch, H: value heads, J: key heads, K: key width, V: value width, L: one token.


def gated_delta_token(
    q_BLJK: Any, k_BLJK: Any, v_BLHV: Any, g_BLH: Any, beta_BLH: Any, state_BHVK: Any
) -> tuple:
    """
    Purpose: Apply the sequential recurrence for one autoregressive token.
    Input: Contiguous BF16 tensors with exactly one row. Output: Value and state tensors.
    """
    batch, length, key_heads, keys = q_BLJK.shape
    heads, values = v_BLHV.shape[-2:]
    assert length == 1 and keys % 32 == 0 and heads % key_heads == 0
    assert k_BLJK.shape == q_BLJK.shape and v_BLHV.shape[:2] == (batch, 1)
    assert g_BLH.shape == beta_BLH.shape == (batch, 1, heads)
    assert state_BHVK.shape == (batch, heads, values, keys)
    tensors = (q_BLJK, k_BLJK, v_BLHV, g_BLH, beta_BLH, state_BHVK)
    assert all(x.dtype == mx.bfloat16 for x in tensors)
    return tuple(
        _kernel()(
            inputs=[mx.contiguous(x) for x in tensors],
            template=[("DK", keys), ("DV", values), ("HJ", key_heads), ("HV", heads)],
            grid=(32, values, batch * heads),
            threadgroup=(32, min(4, values), 1),
            output_shapes=[v_BLHV.shape, state_BHVK.shape],
            output_dtypes=[mx.bfloat16] * 2,
        )
    )


@lru_cache(maxsize=1)
def _kernel() -> Any:
    """Compile the inference-only token recurrence with unique state ownership."""
    return mx.fast.metal_kernel(
        name="shellsmith_bf16_gdn_token_v1",
        input_names=["q", "k", "v", "gate", "beta", "state"],
        output_names=["output", "next_state"],
        source="""
            uint lane=thread_position_in_grid.x;
            uint value_row=thread_position_in_grid.y;
            uint bh=thread_position_in_grid.z;
            if(value_row>=DV)return;
            uint batch=bh/HV,head=bh%HV,key_head=head/(HV/HJ);
            uint q_base=(batch*HJ+key_head)*DK;
            uint state_base=(bh*DV+value_row)*DK;
            float decay=float(gate[bh]),prediction=0.0f;
            float decayed[DK/32];
            for(uint i=0;i<DK/32;++i) {
                uint d=lane*(DK/32)+i;
                decayed[i]=decay*float(state[state_base+d]);
                prediction+=decayed[i]*float(k[q_base+d]);
            }
            prediction=simd_sum(prediction);
            float delta=float(beta[bh])*(float(v[bh*DV+value_row])-prediction);
            float result=0.0f;
            for(uint i=0;i<DK/32;++i) {
                uint d=lane*(DK/32)+i;
                bfloat updated=bfloat(decayed[i]+delta*float(k[q_base+d]));
                next_state[state_base+d]=updated;
                result+=float(updated)*float(q[q_base+d]);
            }
            result=simd_sum(result);
            if(lane==0)output[bh*DV+value_row]=bfloat(result);
        """,
    )
