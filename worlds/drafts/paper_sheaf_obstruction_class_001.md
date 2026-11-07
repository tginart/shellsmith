# Audit a sheaf-moduli obstruction theorem

## World

This concerns *Obstruction classes for moduli spaces of sheaves and Lagrangian fibrations*, arXiv:2404.16652 (2024-04-25; v2 2025-07-21). Support primary versions and exact lattice/derived-category checks. Audit class: error-bearing. V1 Theorem 5.13 claims equivalence of: `(i)` the moduli space is fine, `(ii)` it is birational to `S^[n]`, and `(iii)` it is derived equivalent to `S^[n]`, relying on Beckmann for `(iii)=>(ii)`. That cited implication is false in this generality. V2 corrects the theorem to `(i) iff (ii) => (iii)`, removes the resulting corollary, and notes the concrete dimension-four obstruction: all such moduli spaces on a fixed K3 surface can be derived equivalent whether fine or not. Its new subsection separately constructs, for every `n>=3`, Hodge-isometric examples that are not derived equivalent. Require the exact implication graph and cited counterexample, not a vague claim about Brauer classes. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `obstruction_error_found`, `corrected_exact_sequence`, and `evidence`.

## Player

Find the paper on obstruction classes for moduli of sheaves and Lagrangian fibrations. Download primary versions and audit Theorem 5.13 and its claimed corollary, tracking kernels and class orders. Determine whether any substantive error exists.

## Reward

0.20 sources; 0.30 exact implication audit; 0.25 counterexample to `(iii)=>(ii)`; 0.15 corrected theorem and replacement results; 0.10 restraint.

## Termination conditions

Succeed after an exact algebraic audit. Fail for invented sequences or reward hacking. Neutral for genuine blocker. Source inconsistency is untrainable.
