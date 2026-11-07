"""FP32 master Adam with summed gradients, no decay, no norm clipping."""

from __future__ import annotations

import math

import mlx.core as mx

BETA1, BETA2, EPSILON = 0.9, 0.95, 1e-8


class MasterAdam:
    """One Adam update on accumulated sum-reduced adapter gradients.

    BF16 published factors and FP32 master/moments are separate. A zero norm
    is a valid update (including momentum evolution); only nonfinite state
    fails. Publication happens after validating every prospective tensor.
    """

    def __init__(self, adapter, learning_rate: float):
        self.adapter = adapter
        self.learning_rate = self._rate(learning_rate)
        self.master = {name: value.astype(mx.float32) for name, value in adapter.state().items()}
        self.first = {name: mx.zeros_like(value) for name, value in self.master.items()}
        self.second = {name: mx.zeros_like(value) for name, value in self.master.items()}
        self.steps = 0
        self.accumulated = {}

    @staticmethod
    def _rate(value):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("Adam learning rate must be finite and positive.")
        return float(value)

    def zero(self):
        self.accumulated = {}

    def accumulate(self, gradients):
        if set(gradients) != set(self.master):
            raise ValueError("Gradient does not cover the canonical adapter exactly.")
        for name, value in gradients.items():
            if value.shape != self.master[name].shape:
                raise ValueError(f"Gradient shape differs from canonical factor: {name}")
            value = value.astype(mx.float32)
            self.accumulated[name] = value if name not in self.accumulated else self.accumulated[name] + value
        mx.eval(self.accumulated)

    def step(self, *, learning_rate: float | None = None):
        if not self.accumulated:
            raise ValueError("Adam requires at least one accumulated datum.")
        rate = self.learning_rate if learning_rate is None else self._rate(learning_rate)
        gradients = self.accumulated
        if not all(bool(mx.isfinite(value).all().item()) for value in gradients.values()):
            raise FloatingPointError("Nonfinite MLX gradient.")
        norm = math.sqrt(math.fsum(float(mx.sum(value * value).item()) for value in gradients.values()))
        if not math.isfinite(norm):
            raise FloatingPointError("Nonfinite accumulated gradient norm.")
        count = self.steps + 1
        first, second, master = {}, {}, {}
        size = rate / (1 - BETA1 ** count)
        correction = math.sqrt(1 - BETA2 ** count)
        for name, value in gradients.items():
            first[name] = BETA1 * self.first[name] + (1 - BETA1) * value
            second[name] = BETA2 * self.second[name] + (1 - BETA2) * value * value
            master[name] = self.master[name] - size * first[name] / (mx.sqrt(second[name]) / correction + EPSILON)
        mx.eval(master, first, second)
        if not all(bool(mx.isfinite(value).all().item())
                   for values in (master, first, second) for value in values.values()):
            raise FloatingPointError("Nonfinite prospective MLX optimizer state.")
        published = {name: value.astype(mx.bfloat16) for name, value in master.items()}
        self.adapter.load(published)
        self.master, self.first, self.second = master, first, second
        self.steps, self.learning_rate = count, rate
        self.accumulated = {}
        return norm

    def state(self):
        if self.accumulated:
            raise RuntimeError("Full MLX checkpoints require an optimizer update boundary.")
        tensors = {kind + "." + name: value for kind, values in
                   (("master", self.master), ("first", self.first), ("second", self.second))
                   for name, value in values.items()}
        return tensors, {"steps": self.steps, "learning_rate": self.learning_rate,
                         "beta1": BETA1, "beta2": BETA2, "eps": EPSILON, "weight_decay": 0.0}

    def validated_state(self, tensors, metadata):
        required = {kind + "." + name for kind in ("master", "first", "second") for name in self.master}
        if set(tensors) != required:
            raise ValueError("Checkpoint optimizer tensor inventory differs from canonical factors.")
        if (metadata.get("beta1"), metadata.get("beta2"), metadata.get("eps"), metadata.get("weight_decay")) != (
                BETA1, BETA2, EPSILON, 0.0):
            raise ValueError("Checkpoint optimizer semantics differ from MLX v1.")
        if type(metadata.get("steps")) is not int or metadata["steps"] < 0:
            raise ValueError("Checkpoint optimizer update count is invalid.")
        self._rate(metadata.get("learning_rate", float("nan")))
        state = {}
        for kind in ("master", "first", "second"):
            values = {name: tensors[kind + "." + name] for name in self.master}
            for name, value in values.items():
                if value.dtype != mx.float32 or value.shape != self.master[name].shape or not bool(mx.isfinite(value).all().item()):
                    raise ValueError(f"Invalid FP32 optimizer tensor {kind}.{name}")
                if kind == "second" and bool(mx.any(value < 0).item()):
                    raise ValueError(f"Negative second moment in checkpoint: {name}")
            state[kind] = values
        return state

    def load(self, tensors, metadata):
        state = self.validated_state(tensors, metadata)
        self.adapter.load({name: value.astype(mx.bfloat16) for name, value in state["master"].items()})
        self.master, self.first, self.second = state["master"], state["first"], state["second"]
        self.steps, self.learning_rate = metadata["steps"], float(metadata["learning_rate"])
        self.accumulated = {}
