"""Adam with explicit legacy FP32 or BF16 state, summed gradients, and no decay."""

from __future__ import annotations

import math
import hashlib

import mlx.core as mx

BETA1, BETA2, EPSILON = 0.9, 0.95, 1e-8


class MasterAdam:
    """One Adam update on accumulated sum-reduced adapter gradients.

    The legacy profile separates BF16 factors from FP32 master weights and moments.
    The BF16 profile uses BF16 tensors and counter-seeded stochastic parameter sums. A zero norm
    is a valid update (including momentum evolution); only nonfinite state
    fails. Publication happens after validating every prospective tensor.
    """

    def __init__(self, adapter, learning_rate: float, *, beta1=BETA1, beta2=BETA2, epsilon=EPSILON, execution_profile="legacy", seed=0):
        if execution_profile not in ("legacy", "bf16-fast"):
            raise ValueError("Unknown optimizer execution profile.")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Rounding seed must be uint32.")
        self.execution_profile = execution_profile
        self.rounding_seed = seed
        self.dtype = mx.bfloat16 if execution_profile == "bf16-fast" else mx.float32
        from .sums import BalancedSums
        self.sums = BalancedSums()
        self.adapter = adapter
        self.learning_rate = self._rate(learning_rate, allow_zero=True)
        if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x < 1 for x in (beta1, beta2)):
            raise ValueError("Adam betas must be finite and in [0, 1).")
        self.beta1, self.beta2, self.epsilon = float(beta1), float(beta2), self._rate(epsilon)
        self.moment_betas = (self.beta1, self.beta2)
        if execution_profile == "bf16-fast":
            self.moment_betas = tuple(
                mx.array(self.moment_betas, mx.bfloat16).tolist()
            )
            if any(beta >= 1 for beta in self.moment_betas):
                raise ValueError("BF16 Adam betas must remain below one.")
        self.master = {name: value.astype(self.dtype) for name, value in adapter.state().items()}
        self.first = {name: mx.zeros_like(value) for name, value in self.master.items()}
        self.second = {name: mx.zeros_like(value) for name, value in self.master.items()}
        self.steps = 0
        self.accumulated = {}

    @staticmethod
    def _rate(value, *, allow_zero=False):
        if isinstance(value, bool) or not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
            raise ValueError("Adam learning rate must be finite and positive.")
        return float(value)

    def zero(self):
        from .sums import BalancedSums
        self.accumulated = {}
        self.sums = BalancedSums()

    def accumulate(self, gradients):
        if set(gradients) != set(self.master):
            raise ValueError("Gradient does not cover the canonical adapter exactly.")
        for name, value in gradients.items():
            if value.shape != self.master[name].shape:
                raise ValueError(f"Gradient shape differs from canonical factor: {name}")
            value = value.astype(self.dtype)
            if self.execution_profile == "bf16-fast":
                self.sums.add(name, value)
                self.accumulated[name] = self.sums.total(name)
            else:
                self.accumulated[name] = value if name not in self.accumulated else self.accumulated[name] + value
        mx.eval(self.accumulated)

    def step(self, *, learning_rate: float | None = None):
        if not self.accumulated:
            raise ValueError("Adam requires at least one accumulated datum.")
        rate = self.learning_rate if learning_rate is None else self._rate(learning_rate, allow_zero=True)
        gradients = self.accumulated
        if not all(bool(mx.isfinite(value).all().item()) for value in gradients.values()):
            raise FloatingPointError("Nonfinite MLX gradient.")
        norm = math.sqrt(math.fsum(float(mx.sum(value * value).item()) for value in gradients.values()))
        if not math.isfinite(norm):
            raise FloatingPointError("Nonfinite accumulated gradient norm.")
        count = self.steps + 1
        first, second, master = {}, {}, {}
        # The kernel receives BF16 constants. Correct the moments with those same betas.
        beta1, beta2 = self.moment_betas
        size = rate / (1 - beta1 ** count)
        correction = math.sqrt(1 - beta2 ** count)
        if self.execution_profile == "bf16-fast":
            from .bf16_adam import update
            constants_C = mx.array(
                [beta1, beta2, size, correction, self.epsilon],
                mx.bfloat16,
            )
            for name, gradient_N in gradients.items():
                salt = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
                counter_R = mx.array([self.rounding_seed, count, salt], mx.uint32)
                master[name], first[name], second[name] = update(
                    self.master[name], self.first[name], self.second[name],
                    gradient_N, constants_C, counter_R,
                )
        for name, value in gradients.items():
            if self.execution_profile == "bf16-fast":
                continue
            first[name] = self.beta1 * self.first[name] + (1 - self.beta1) * value
            second[name] = self.beta2 * self.second[name] + (1 - self.beta2) * value * value
            master[name] = self.master[name] - size * first[name] / (mx.sqrt(second[name]) / correction + self.epsilon)
        mx.eval(master, first, second)
        if not all(bool(mx.isfinite(value).all().item())
                   for values in (master, first, second) for value in values.values()):
            raise FloatingPointError("Nonfinite prospective MLX optimizer state.")
        published = {name: value.astype(mx.bfloat16) for name, value in master.items()}
        self.adapter.load(published)
        self.master, self.first, self.second = master, first, second
        self.steps, self.learning_rate = count, rate
        self.zero()
        return norm

    def state(self):
        if self.accumulated:
            raise RuntimeError("Full MLX checkpoints require an optimizer update boundary.")
        tensors = {kind + "." + name: value for kind, values in
                   (("master", self.master), ("first", self.first), ("second", self.second))
                   for name, value in values.items()}
        metadata = {}
        if self.execution_profile == "bf16-fast":
            metadata = {"execution_profile": "bf16-fast", "rounding_seed": self.rounding_seed}
        return tensors, {**metadata, "steps": self.steps, "learning_rate": self.learning_rate,
                         "beta1": self.beta1, "beta2": self.beta2, "eps": self.epsilon, "weight_decay": 0.0}

    def validated_state(self, tensors, metadata):
        if metadata.get("execution_profile", "legacy") != self.execution_profile:
            raise ValueError("Checkpoint optimizer execution profile differs.")
        if self.execution_profile == "bf16-fast":
            seed = metadata.get("rounding_seed")
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError("Invalid stochastic-rounding checkpoint seed.")
        required = {kind + "." + name for kind in ("master", "first", "second") for name in self.master}
        if set(tensors) != required:
            raise ValueError("Checkpoint optimizer tensor inventory differs from canonical factors.")
        if (metadata.get("beta1"), metadata.get("beta2"), metadata.get("eps"), metadata.get("weight_decay")) != (
                self.beta1, self.beta2, self.epsilon, 0.0):
            raise ValueError("Checkpoint optimizer semantics differ from MLX v1.")
        if type(metadata.get("steps")) is not int or metadata["steps"] < 0:
            raise ValueError("Checkpoint optimizer update count is invalid.")
        self._rate(metadata.get("learning_rate", float("nan")), allow_zero=True)
        state = {}
        for kind in ("master", "first", "second"):
            values = {name: tensors[kind + "." + name] for name in self.master}
            for name, value in values.items():
                if value.dtype != self.dtype or value.shape != self.master[name].shape or not bool(mx.isfinite(value).all().item()):
                    raise ValueError(f"Invalid {self.dtype} optimizer tensor {kind}.{name}")
                if kind == "second" and bool(mx.any(value < 0).item()):
                    raise ValueError(f"Negative second moment in checkpoint: {name}")
            state[kind] = values
        return state

    def load(self, tensors, metadata):
        state = self.validated_state(tensors, metadata)
        self.adapter.load({name: value.astype(mx.bfloat16) for name, value in state["master"].items()})
        self.master, self.first, self.second = state["master"], state["first"], state["second"]
        self.steps, self.learning_rate = metadata["steps"], float(metadata["learning_rate"])
        self.rounding_seed = metadata.get("rounding_seed", self.rounding_seed)
        self.zero()
