"""CPU-only ownership and exception checks. These tests do not simulate MLX math."""

from __future__ import annotations

import gc
import importlib.util
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def test_function_cache_does_not_retain_its_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Purpose: Detect owner/function cycles without assistance from cyclic collection.
    Input: A compiler that retains each callable. Output: Immediate owner release.
    """
    module = _load(monkeypatch)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        unit = module.CompiledGDNUnit(
            SimpleNamespace(compile=lambda function: function),
            FakeLayer(),
            lambda *_: None,
        )
        for active in ((0,), (1, 2, 0)):
            unit._backward[active] = unit.mx.compile(unit._derivative(active))
        reference = weakref.ref(unit)
        del unit
        assert reference() is None
    finally:
        if was_enabled:
            gc.enable()


def test_trace_failure_restores_current_parameter_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Purpose: Restore the active policy after an exception during a later trace.
    Input: Replaced factors after cache creation. Output: Those factors remain active.
    """
    module = _load(monkeypatch)
    layer = FakeLayer()
    supplied = {"a": object(), "b": object()}

    def fail(current: FakeLayer, *_: Any) -> None:
        assert current.trainable_parameters() == supplied
        raise RuntimeError("trace failure")

    unit = module.CompiledGDNUnit(
        SimpleNamespace(compile=lambda function: function), layer, fail
    )
    active = {"a": object(), "b": object()}
    layer.update(active)
    with pytest.raises(RuntimeError, match="trace failure"):
        unit._primal(supplied["a"], supplied["b"], None, None, None, None)
    assert layer.trainable_parameters() == active


class FakeLayer:
    """Parameter bindings only. Tensor calculations are outside this test's scope."""

    is_linear = True

    def __init__(self) -> None:
        self.parameters = {"a": object(), "b": object()}

    def trainable_parameters(self) -> dict:
        return dict(self.parameters)

    def update(self, values: dict) -> None:
        self.parameters = dict(values)


def _load(monkeypatch: pytest.MonkeyPatch) -> Any:
    """
    Purpose: Load only the ownership code without optional numerical dependencies.
    Input: A temporary import scope. Output: The real unit class with tree helpers.
    """
    package = ModuleType("mlx")
    utilities = ModuleType("mlx.utils")
    utilities.tree_flatten = lambda values: list(values.items())
    utilities.tree_unflatten = lambda values: dict(values)
    package.utils = utilities
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.utils", utilities)
    path = (
        Path(__file__).resolve().parents[1]
        / "src/studio_compute/numerics/compiled_gdn.py"
    )
    spec = importlib.util.spec_from_file_location("compiled_gdn_lifecycle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
