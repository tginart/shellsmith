"""
Purpose: Compile a GDN unit without hidden policy factors or recurrent state.
Input: Current factors, hidden rows, two carries, positions, and live cotangents.
Output: The original three outputs or every factor and input gradient.
Note: A replay owns the function cache. Function closures do not retain that owner.
"""

from collections.abc import Callable
from typing import Any

# B: batch, L: token rows, D: hidden width.


class CompiledGDNUnit:
    """
    Purpose: Reuse compiled graphs for one frozen decoder layer.
    Input: The layer and its current explicit arguments. Output: Exact layer math.
    Note: Every mutable factor and carry enters the graph as an input.
    """

    def __init__(self, mx: Any, layer: Any, segment_layer: Callable) -> None:
        from mlx.utils import tree_flatten

        assert layer.is_linear
        self.mx, self.layer = mx, layer
        self.names = tuple(
            name for name, _ in tree_flatten(layer.trainable_parameters())
        )
        self._primal = _make_primal(layer, self.names, segment_layer)
        self._forward = mx.compile(self._primal)
        self._backward: dict[tuple[int, ...], Callable] = {}

    def forward(self, hidden_BLD: Any, states: list, positions: Any) -> list:
        self._validate_inputs(hidden_BLD, states, positions)
        return self._forward(*self._factors(), hidden_BLD, *states, positions)

    def backward(
        self, hidden_BLD: Any, states: list, positions: Any, cotangents: list
    ) -> tuple[tuple[str, ...], list]:
        """
        Purpose: Compile the full VJP in the original live-edge order.
        Input: Explicit unit arguments. Output: All original input gradients.
        """
        self._validate_inputs(hidden_BLD, states, positions)
        assert len(cotangents) == 3
        for value, shape in zip(
            cotangents,
            (hidden_BLD.shape, states[0].shape, states[1].shape),
            strict=True,
        ):
            assert value is None or value.shape == shape
        active = tuple(i for i in (1, 2, 0) if cotangents[i] is not None)
        assert active
        if active not in self._backward:
            self._backward[active] = self.mx.compile(self._derivative(active))
        arguments = [value.astype(self.mx.float32) for value in self._factors()]
        back = self._backward[active](
            *arguments,
            hidden_BLD,
            *states,
            positions,
            *[cotangents[i] for i in active],
        )
        return self.names, back

    def _validate_inputs(self, hidden_BLD: Any, states: list, positions: Any) -> None:
        assert hidden_BLD.ndim == 3 and len(states) == 2
        assert positions.shape == (3, hidden_BLD.shape[0], hidden_BLD.shape[1])
        assert states[0].ndim == 3 and states[1].ndim == 4
        assert states[0].shape[0] == states[1].shape[0] == hidden_BLD.shape[0]
        assert states[0].dtype == hidden_BLD.dtype
        assert states[1].dtype == self.mx.float32

    def _derivative(self, active: tuple[int, ...]) -> Callable:
        mx, primal, count = self.mx, self._primal, len(self.names) + 3

        def derivative(*arguments: Any) -> list:
            positions = arguments[count]

            def selected_primal(*inputs: Any) -> list:
                outputs = primal(*inputs, positions)
                return [outputs[i] for i in active]

            return mx.vjp(
                selected_primal,
                list(arguments[:count]),
                list(arguments[count + 1 :]),
            )[1]

        return derivative

    def _factors(self) -> list:
        from mlx.utils import tree_flatten

        names, values = zip(
            *tree_flatten(self.layer.trainable_parameters()), strict=True
        )
        assert names == self.names
        return list(values)


def _make_primal(
    layer: Any, names: tuple[str, ...], segment_layer: Callable
) -> Callable:
    """
    Purpose: Build a function that does not form an owner/function reference cycle.
    Input: A frozen layer and factor names. Output: An explicit-argument callable.
    Note: Each trace restores the layer's current parameter bindings.
    """

    def primal(*arguments: Any) -> list:
        from mlx.utils import tree_flatten, tree_unflatten

        count = len(names)
        original = list(tree_flatten(layer.trainable_parameters()))
        try:
            layer.update(
                tree_unflatten(list(zip(names, arguments[:count], strict=True)))
            )
            return list(
                segment_layer(
                    layer,
                    arguments[count],
                    list(arguments[count + 1 : count + 3]),
                    arguments[count + 3],
                )
            )
        finally:
            layer.update(tree_unflatten(original))

    return primal
