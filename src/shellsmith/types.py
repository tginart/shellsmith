"""Shared JSON type aliases and strict validators for Shellsmith contracts.

Why this exists:
    Credit and episode contracts load from JSON. These few aliases and
    validators keep them strict (no silent coercion) without a dependency.
    Pruned from the original monorepo's contract types to what Shellsmith uses.
"""

from __future__ import annotations

import math
from typing import Any, Literal


JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

AdvantageBaseline = Literal["zero", "group_mean"]
AdvantageNormalization = Literal["none", "capped_standardization"]
LossFnName = Literal["importance_sampling", "cispo"]


def require_dict(raw: Any, *, context: str) -> dict[str, Any]:
    """Return a JSON object or fail with a useful contract error.

    Main callers:
        ``from_dict`` constructors in the credit contracts.
    """
    if not isinstance(raw, dict):
        raise RuntimeError(f"{context} must be a JSON object.")
    return raw


def require_finite_float(value: Any, *, context: str) -> float:
    """Parse a finite float; booleans, non-numbers, NaN and infinity raise."""
    if isinstance(value, bool):
        raise RuntimeError(f"{context} must be numeric, got boolean {value!r}.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{context} must be numeric, got {value!r}.") from error
    if not math.isfinite(number):
        raise RuntimeError(f"{context} must be finite, got {value!r}.")
    return number
