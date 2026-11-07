# Audit a qplex Bell-inequality optimization

## World

This audit concerns *Quantum correlations in QBism's reconstruction program*, arXiv:2606.07485, submitted 2026-06-05 and revised 2026-06-18. Support primary versions, symbolic trigonometric optimization, and numerical plotting.

Audit class: error-bearing. V1 maximizes its displayed `I_2233` upper-bound function at `theta=4pi/3` and claims the algebraic maximum 4. Direct optimization does not yield 4. V2 corrects the qplex-permitted upper bound to `2 + 2 sqrt(3)/3`, approximately `3.1547`, still above the quantum maximum near `2.8729`; thus the qualitative super-quantum conclusion survives. Full credit requires evaluating/optimizing the displayed expression under its stated geometric constraints, not merely repeating the revision comment. Keep the target and answer private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `optimization_error_found`, `correct_bound_verified`, and `evidence`.

## Player

Find the recent paper analyzing Bell-type correlations in the qplex/QBist reconstruction program. Use `webFetch` to obtain the primary paper and audit its optimization of the three-outcome CGLMP expression. Determine whether any substantive mathematical error exists and verify extrema analytically or numerically.

## Reward

- 0.15 source grounding.
- 0.30 faithful reconstruction and independent optimization of the displayed bound.
- 0.25 identification and explanation of any incorrect extremum.
- 0.20 derivation of the defensible value and consequence for the qualitative claim.
- 0.10 precise reporting. Deduct for confusing the qplex bound with the quantum maximum.

## Termination conditions

Succeed after a checked constrained optimization and sourced conclusion. Fail for fabricated equations, oracle extraction, or knowingly invalid numerics. Neutral for rigorous inconclusive work/access failure. A corrupted formula is untrainable.
