"""Canonical all-layer dense Qwen LoRA, adapted from High Command 10e830."""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

ATTENTION_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
GDN_TARGETS = ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj')
MLP_TARGETS = ('gate_proj', 'up_proj', 'down_proj')

RANK, SCALE = 16, 2.0


def canonical_targets(layer_types):
    for index, kind in enumerate(layer_types):
        if kind == "linear_attention":
            mixer = (("linear_attn", name) for name in GDN_TARGETS)
        elif kind == "full_attention":
            mixer = (("self_attn", name) for name in ATTENTION_TARGETS)
        else:
            raise ValueError(f"Unsupported dense adapter layer kind: {kind}")
        for parent, name in (*mixer, *(("mlp", name) for name in MLP_TARGETS)):
            yield f"layers.{index}.{parent}.{name}", (index, parent, name)


class LoRALinear(nn.Module):
    """Native projection + BF16 scale * B(A(x)); no custom or compiled kernel."""

    def __init__(self, base: Any, rank: int = RANK, scale: float = SCALE):
        super().__init__()
        self.base = base
        self.scale = float(scale)
        width_out = base.weight.shape[0]
        width_in = base.weight.shape[1] * 32 // base.bits if hasattr(base, "bits") else base.weight.shape[1]
        self.a = mx.zeros((rank, width_in), dtype=mx.bfloat16)
        self.b = mx.zeros((width_out, rank), dtype=mx.bfloat16)

    def __call__(self, x):
        value = self.base(x)
        low = (x @ self.a.T.astype(x.dtype)) @ self.b.T.astype(x.dtype)
        return value + (self.scale * low).astype(value.dtype)


class Adapter:
    def __init__(self, language_model, layer_types, *, rank: int = RANK, scale: float = SCALE):
        if rank != RANK or not math.isfinite(scale) or scale != SCALE:
            raise ValueError("MLX v1 requires canonical rank 16 / scale 2.0 factors.")
        self.rank, self.scale = rank, scale
        self.language_model = language_model
        layers = language_model.layers
        if len(layers) != len(layer_types) or any(
                layer.is_linear != (kind == "linear_attention") for layer, kind in zip(layers, layer_types)):
            raise ValueError("Loaded decoder disagrees with the admitted layer types.")
        language_model.freeze()
        self.modules = {}
        for name, (index, parent, target) in canonical_targets(layer_types):
            owner = getattr(layers[index], parent)
            base = getattr(owner, target)
            if not isinstance(base, (nn.Linear, nn.QuantizedLinear)):
                raise ValueError(f"Unexpected canonical projection type: {name}")
            if isinstance(base, nn.QuantizedLinear) and (base.bits != 8 or base.group_size != 64 or base.mode != "affine"):
                raise ValueError(f"Canonical projection has unexpected quantization: {name}")
            factor = LoRALinear(base, rank, scale)
            setattr(owner, target, factor)
            self.modules[name] = factor
        language_model.freeze()
        for factor in self.modules.values():
            factor.unfreeze(keys=["a", "b"], recurse=False)

    def state(self):
        return {f"{name}.{kind}": getattr(module, kind)
                for name, module in self.modules.items() for kind in ("a", "b")}

    def initial(self, seed: int):
        key = mx.random.key(seed)
        state = {}
        for name, module in self.modules.items():
            key, subkey = mx.random.split(key)
            bound = module.a.shape[1] ** -0.5
            state[name + ".a"] = mx.random.uniform(-bound, bound, module.a.shape, key=subkey).astype(mx.bfloat16)
            state[name + ".b"] = mx.zeros(module.b.shape, dtype=mx.bfloat16)
        return state

    def validate(self, state):
        expected = self.state()
        if set(state) != set(expected):
            raise ValueError("Adapter checkpoint does not cover canonical factors exactly.")
        for name, value in state.items():
            if value.shape != expected[name].shape or not bool(mx.isfinite(value).all().item()):
                raise ValueError(f"Invalid canonical adapter factor: {name}")

    def load(self, state):
        self.validate(state)
        published = {name: value.astype(mx.bfloat16) for name, value in state.items()}
        mx.eval(published)
        for name, value in published.items():
            if not bool(mx.isfinite(value).all().item()):
                raise FloatingPointError(f"Adapter factor cannot be represented in BF16: {name}")
        for name, module in self.modules.items():
            module.a, module.b = published[name + ".a"], published[name + ".b"]
