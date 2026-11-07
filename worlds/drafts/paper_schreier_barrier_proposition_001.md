# Audit a Schreier-barrier Ramsey proposition

## World

This concerns *Ramsey-like theorems for the Schreier barrier*, arXiv:2412.11598 (2024-12-16; corrected v3 2026-04-29). Support versioned primary sources and finite coloring searches. Audit class: error-bearing. In the published proof of Proposition 3.12, given a `k`-bounded coloring `f:[N]^n -> N`, the constructed coloring on exactly-omega-large sets is written as a pairing with `f(s)` when `min(s)>=n`; `s` has `min(s)+1` elements, so this is ill-typed, and even a natural first-`n` repair does not keep the new coloring `k`-bounded because arbitrarily many choices of the unused tail collide. V3 instead encodes `s_0`, `f(s_1,...,s_n)`, and every tail element `s_{n+1},...,s_{s_0}`; for `s_0<n` it assigns the entire set a fresh encoded color. This makes `g` provably `k`-bounded and lets a rainbow for `g` yield a rainbow for `f` after removing one minimum element. Require that exact collision analysis. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `proposition_error_found`, `corrected_thinning_order`, and `evidence`.

## Player

Locate the paper on Ramsey-like theorems for the Schreier barrier. Fetch primary versions and audit Proposition 3.12, tracking restriction, barrier, and homogeneity quantifiers. Determine whether any substantive proof error exists.

## Reward

0.20 sources; 0.30 quantifier reconstruction; 0.25 invalid old invocation; 0.15 corrected proof order; 0.10 restraint.

## Termination conditions

Succeed after a quantifier-level audit. Fail for fabricated combinatorics or reward hacking. Neutral for a genuine blocker. Source inconsistency is untrainable.
