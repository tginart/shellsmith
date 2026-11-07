# Audit a biased edge-coloring game proof

## World

This concerns *A biased edge coloring game*, arXiv:2408.02819 (2024-08-05; v2 2025-01-14). Support versioned sources and exhaustive play on tiny graphs. Audit class: error-bearing. V1 says without proof that `chi'_g(W_4;m,1)=4` for every `m>=2`, and then excludes `W_4` from its wheel theorem. This is false at `m=3`: v2 proves the exact piecewise result `chi'_g(W_4;m,1)=4` for `m!=3` but `=5` for `m=3`. With four colors, the unique proper edge-color pattern forces opposite pairings; after Maker's three first moves, Breaker can always destroy that pattern (the two-color and three-color opening cases are handled separately). Five colors suffice by coloring three spokes distinctly and then completing the fourth. Require a legal game-tree/case check of this non-monotonic counterexample. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `strategy_error_found`, `corrected_invariant`, and `evidence`.

## Player

Locate the paper introducing the biased edge-coloring game. Fetch primary versions and audit its principal Maker strategy and conflict counting, using exhaustive play on a small graph if helpful. Determine whether any substantive error exists.

## Reward

0.20 sources; 0.25 strategy reconstruction; 0.30 concrete failure of old counting; 0.15 corrected invariant/bound; 0.10 restraint.

## Termination conditions

Succeed after a reproducible strategy audit. Fail for fabricated play or source claims. Neutral for honest blocker. Simulator inconsistencies are untrainable.
