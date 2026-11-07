"""
Purpose: Admit saved activations and select optional residuals by measured benefit.
Input: Allocator facts, mandatory allocations, and measured recomputation costs.
Output: A deterministic selection whose unique allocations fit the byte budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Hashable, Iterable

GIB = 1 << 30
SAFETY_BYTES = 16 * GIB


@dataclass(frozen=True)
class RetentionGroup:
    """
    Purpose: Describe one optional residual group and its measured benefit.
    Input: Stable identity, allocation identities, and nanoseconds saved.
    Output: An immutable admission candidate.
    """

    identity: str
    allocations: tuple[tuple[Hashable, int], ...]
    saved_ns: int


@dataclass(frozen=True)
class RetentionPlan:
    """Store the selected stable identities and their unique allocation cost."""

    selected: frozenset[str]
    used_bytes: int
    budget_bytes: int
    saved_ns: int


def activation_budget(
    *,
    allocator_bytes: int,
    recommended_bytes: int,
    resident_bytes: int,
    checkpoint_bytes: int,
    scratch_bytes: int,
    requested_gib: float | None,
    safety_bytes: int = SAFETY_BYTES,
) -> int:
    """
    Purpose: Reserve mandatory work and a safety margin before optional retention.
    Input: Measured resident memory and conservative work estimates.
    Output: The admitted activation cap. Insufficient memory raises MemoryError.
    """
    values = (
        allocator_bytes,
        recommended_bytes,
        resident_bytes,
        checkpoint_bytes,
        scratch_bytes,
        safety_bytes,
    )
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError(
            "Activation admission requires nonnegative integer byte facts."
        )
    available = min(allocator_bytes, recommended_bytes) - sum(values[2:])
    if available < 0:
        raise MemoryError(
            "Mandatory checkpoints, scratch, and safety exceed the allocator limit."
        )
    requested = available if requested_gib is None else int(requested_gib * GIB)
    if requested < 0 or requested > available:
        raise MemoryError(
            f"The requested activation cap exceeds available {available / GIB:.2f} GiB."
        )
    return requested


def select_residuals(
    groups: Iterable[RetentionGroup],
    budget_bytes: int,
    *,
    resident_allocations: Iterable[tuple[Hashable, int]] = (),
) -> RetentionPlan:
    """
    Purpose: Select useful residuals by measured nanoseconds per unique byte.
    Input: Candidates, the byte cap, and already-resident alias allocations.
    Output: A stable selection with alias bytes counted once.
    Note: Candidates can share allocations only when all allocation sets match.
    """
    if type(budget_bytes) is not int or budget_bytes < 0:
        raise ValueError("The activation budget must be nonnegative integer bytes.")
    sizes = dict(resident_allocations)
    resident = set(sizes)
    grouped = {}
    identities = set()
    allocation_owners = {}
    for group in groups:
        if group.identity in identities or group.saved_ns < 0:
            raise ValueError(
                "Retention identities must be unique and costs nonnegative."
            )
        identities.add(group.identity)
        keys = frozenset(key for key, _ in group.allocations)
        for key, size_bytes in group.allocations:
            if type(size_bytes) is not int or size_bytes < 0:
                raise ValueError("Allocation sizes must be nonnegative integer bytes.")
            if key in sizes and sizes[key] != size_bytes:
                raise ValueError("An allocation identity has inconsistent byte sizes.")
            if (
                key not in resident
                and key in allocation_owners
                and allocation_owners[key] != keys
            ):
                raise ValueError(
                    "Overlapping residual groups must declare the same unique allocation set."
                )
            allocation_owners[key] = keys
            sizes[key] = size_bytes
        grouped.setdefault(keys, []).append(group)
    ranked = []
    for keys, aliases in grouped.items():
        size_bytes = sum(sizes[key] for key in keys - resident)
        saved_ns = sum(group.saved_ns for group in aliases)
        names = tuple(sorted(group.identity for group in aliases))
        ratio = Fraction(saved_ns, max(1, size_bytes))
        ranked.append((size_bytes == 0, ratio, names, size_bytes, saved_ns))
    ranked.sort(key=lambda value: (-value[0], -value[1], value[2]))
    selected, used, total_ns = set(), 0, 0
    for _, _, names, size_bytes, saved_ns in ranked:
        if used + size_bytes <= budget_bytes:
            selected.update(names)
            used += size_bytes
            total_ns += saved_ns
    return RetentionPlan(frozenset(selected), used, budget_bytes, total_ns)
