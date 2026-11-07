# Audit a semi-uniform stability appendix

## World

This concerns *A universal example for quantitative semi-uniform stability*, arXiv:2410.02357 (2024-10-03; v3 2026-01-08). Support versioned sources and numerical scalar examples. Audit class: error-bearing. In v2's Appendix proof of the lower resolvent/decay bound, `N(t)` is defined as `sup{||U(tau)A^{-1}|| : tau >= t}` and asserted right-continuous because sets for `t_n downarrow t` increase to the set at `t`. Their union is actually `(t,infinity)`, not `[t,infinity)`, so the asserted equality can fail when the value at `t` is a strict spike. V3 defines `N(t)` with `tau > t`, making the set-limit/right-continuity argument correct, and supplies the missing proof of the preceding theorem for right-continuous `N`. Require this exact open/closed endpoint correction and its role in applying the generalized inverse theorem. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `appendix_error_found`, `corrected_quantifiers`, and `evidence`.

## Player

Locate the paper giving a universal example for quantitative semi-uniform stability. Fetch primary versions and audit Appendix A's decay-transfer theorems, tracking every supremum, domain, and limiting operation. Determine whether any substantive error exists.

## Reward

0.20 sources; 0.25 exact quantifier audit; 0.30 invalid interchange/counterexample; 0.15 repaired extension; 0.10 restraint.

## Termination conditions

Succeed after a rigorous appendix audit. Fail for fake analysis or oracle attacks. Neutral for honest inconclusiveness/access failure. Source cracks are untrainable.
