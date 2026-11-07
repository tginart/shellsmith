"""
Purpose: Attach rank-16 LoRA to Flash mixer, shared-expert, and routed-expert projections.
Input: One frozen quantized Flash decoder.
Output: Trainable BF16 factors that reference the original quantized base modules.
Note: Router, PLE, hyper-connection, embedding, and head weights remain frozen.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .adapter import ATTENTION_TARGETS, GDN_TARGETS, MLP_TARGETS, RANK, SCALE
from .adapter import Adapter, LoRALinear
from .expert_metal import sorted_expert_qmm

# E: experts, R: LoRA rank, D: input width, F: output width, A: other dimensions.


class SwitchLoRALinear(nn.Module):
    """
    Purpose: Apply one low-rank update per selected expert without a dense weight copy.
    Input: A quantized expert projection and routed input rows.
    Output: The base result plus selected BF16 low-rank updates.
    """

    def __init__(self, base: Any, rank: int = RANK, scale: float = SCALE):
        super().__init__()
        self.base = base
        self.scale = float(scale)
        experts, output_width, packed_width = base.weight.shape
        input_width = packed_width * 32 // base.bits
        self.a = mx.zeros((experts, rank, input_width), dtype=mx.bfloat16)
        self.b = mx.zeros((experts, output_width, rank), dtype=mx.bfloat16)

    def __call__(self, x_A: Any, indices: Any, sorted_indices: bool = False) -> Any:
        assert x_A.shape[-1] == self.a.shape[-1]
        # Training replay sends (R, 1, I) BF16 rows in expert order; its input
        # gradient uses the sorted-expert kernel.
        if sorted_indices and x_A.ndim == 3 and x_A.dtype == mx.bfloat16:
            value_A = sorted_expert_qmm(x_A, self.base, indices)
        else:
            value_A = self.base(x_A, indices, sorted_indices=sorted_indices)
        low_A = mx.gather_mm(
            x_A,
            mx.swapaxes(self.a.astype(x_A.dtype), -1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        update_A = mx.gather_mm(
            low_A,
            mx.swapaxes(self.b.astype(x_A.dtype), -1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        return value_A + (self.scale * update_A).astype(value_A.dtype)


class FlashAdapter(Adapter):
    """
    Purpose: Keep the dense adapter state contract for all Flash decoder layers.
    Input: The loaded Flash language model and its layer types.
    Output: Named attention, shared-expert, and per-expert BF16 factors.
    """

    def __init__(
        self,
        language_model: Any,
        layer_types: list[str],
        *,
        rank: int = RANK,
        scale: float = SCALE,
    ):
        if rank != RANK or not math.isfinite(scale) or scale != SCALE:
            raise ValueError("Flash qualification requires rank 16 and scale 2.")
        self.rank, self.scale = rank, scale
        self.language_model = language_model
        if len(language_model.layers) != len(layer_types):
            raise ValueError("Flash decoder layer inventory differs from its config.")
        language_model.freeze()
        self.modules = {}
        for index, (layer, kind) in enumerate(zip(language_model.layers, layer_types)):
            if layer.is_linear != (kind == "linear_attention"):
                raise ValueError(
                    "Flash mixer types differ from the declared layer types."
                )
            mixer_name = "linear_attn" if layer.is_linear else "self_attn"
            mixer = getattr(layer, mixer_name)
            targets = GDN_TARGETS if layer.is_linear else ATTENTION_TARGETS
            for name in targets:
                self._attach(mixer, name, f"layers.{index}.{mixer_name}.{name}")
            for expert_name in ("shared_expert", "switch_mlp"):
                expert = getattr(layer.mlp, expert_name)
                for name in MLP_TARGETS:
                    self._attach(
                        expert,
                        name,
                        f"layers.{index}.mlp.{expert_name}.{name}",
                    )
        language_model.freeze()
        for module in self.modules.values():
            module.unfreeze(keys=["a", "b"], recurse=False)

    def initial(self, seed: int) -> dict:
        key = mx.random.key(seed)
        state = {}
        for name, module in self.modules.items():
            key, subkey = mx.random.split(key)
            bound = module.a.shape[-1] ** -0.5
            state[name + ".a"] = mx.random.uniform(
                -bound,
                bound,
                module.a.shape,
                key=subkey,
            ).astype(mx.bfloat16)
            state[name + ".b"] = mx.zeros(module.b.shape, dtype=mx.bfloat16)
        return state

    def _attach(self, owner: Any, target: str, name: str) -> None:
        base = getattr(owner, target)
        if not hasattr(base, "bits") or base.mode != "affine":
            raise ValueError(f"Flash projection must retain its quantized base: {name}")
        is_expert = base.weight.ndim == 3
        expected_bits = 4 if is_expert else 8
        if base.bits != expected_bits or base.group_size not in (64, 128):
            raise ValueError(f"Flash projection has unexpected quantization: {name}")
        module = SwitchLoRALinear(base) if is_expert else LoRALinear(base)
        setattr(owner, target, module)
        self.modules[name] = module
