"""
Purpose: Accumulate tensors in a deterministic balanced binary tree.
Input: A stream of same-shape tensors. Output: A sum in the original tensor dtype.
Note: A binary carry combines equal-size groups before the final sum. The sum
    keeps only the partials. A caller asks for the total when it consumes it.
"""

from __future__ import annotations

from typing import Any


class BalancedSums:
    """
    Purpose: Retain one partial tensor for each binary level and key.
    Input: Same-shape tensor contributions, and an optional cap on the partials.
    Output: A balanced sum in their dtype.
    Note: With max_levels None, the sum is exactly the binary carry order.
        A cap keeps at most max_levels partials for each key. When the cap
        applies, the two lowest partials merge. This changes only the BF16
        rounding order. Integer contributions give an exact memory model.
    """

    def __init__(self, max_levels: int | None = None) -> None:
        assert max_levels is None or max_levels >= 1
        self.max_levels = max_levels
        self.bins: dict[Any, list] = {}

    def __contains__(self, key: Any) -> bool:
        return key in self.bins

    def add(self, key: Any, value: Any) -> None:
        """
        Purpose: Combine equal-size partial groups before the final reduction.
        Input: A key and the next tensor. Output: None.
        """
        levels = self.bins.setdefault(key, [])
        level = 0
        while level < len(levels) and levels[level] is not None:
            value = levels[level] + value
            levels[level] = None
            level += 1
        if level == len(levels):
            levels.append(value)
        else:
            levels[level] = value
        if self.max_levels is not None:
            _cap_levels(levels, self.max_levels)

    def total(self, key: Any) -> Any:
        """
        Purpose: Add the partials of one key from the highest level to the lowest.
        Input: A key. Output: The current balanced sum. The partials stay.
        """
        parts = [part for part in reversed(self.bins[key]) if part is not None]
        total = parts[0]
        for part in parts[1:]:
            total = total + part
        return total

    def pop(self, key: Any) -> Any:
        """Return the total of one key and release its partials."""
        total = self.total(key)
        del self.bins[key]
        return total

    def keys(self) -> list:
        return list(self.bins)

    def arrays(self) -> list:
        """Return every retained partial, for evaluation."""
        return [part for levels in self.bins.values() for part in levels
                if part is not None]

    def partials(self, key: Any) -> list:
        return [part for part in self.bins[key] if part is not None]

    def partial_count(self, key: Any) -> int:
        return sum(part is not None for part in self.bins.get(key, []))


def _cap_levels(levels: list, max_levels: int) -> None:
    """Merge the lowest partial into the next one until max_levels remain."""
    while sum(part is not None for part in levels) > max_levels:
        occupied = [level for level, part in enumerate(levels) if part is not None]
        low, high = occupied[0], occupied[1]
        levels[high] = levels[high] + levels[low]
        levels[low] = None
