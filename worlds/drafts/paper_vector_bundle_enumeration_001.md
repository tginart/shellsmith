# Audit a vector-bundle enumeration theorem

## World

This concerns *Enumerating complex rank n vector bundles on CP^{n+1}*, arXiv:2410.23520 (2024-10-31; v2 2026-03-01). Support primary versions and small characteristic-class calculations. Audit class: error-bearing. V1's proof of the extension theorem claims a pinch-map factorization `s o h ~= (h vee (p o h)) o s'` for the Hopf attaching map, based on a preimage-of-a-ball argument; that unstable homotopy factorization is not established. V2 replaces it with the standard action formula `(sigma V)o h - V o h = (n+1)(sigma o a) + [V o i_2,sigma]`. When `c_1(V)` is even, `V o i_2=2m`, so the Whitehead product is `[m,2sigma]=0`; for even `n`, the remaining term is the nonzero two-torsion class `sigma o a`. A new lemma also shows both extension obstructions lie in the relevant two-torsion subgroup, proving exactly one of `V` and `sigma V` extends. Require that exact replacement. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `lifting_error_found`, `corrected_obstruction_checked`, and `evidence`.

## Player

Find the recent paper enumerating complex rank-n vector bundles over CP^{n+1}. Fetch primary versions and audit Theorem 4.3's obstruction/lifting proof. Determine whether any substantive error exists and test the lowest nontrivial rank.

## Reward

0.20 sources; 0.30 obstruction sequence; 0.25 missing old step; 0.15 corrected proof/low-rank check; 0.10 restraint.

## Termination conditions

Succeed after a source-grounded obstruction audit. Fail for invented topology or oracle attacks. Neutral for honest blocker. Contradictory source simulation is untrainable.
