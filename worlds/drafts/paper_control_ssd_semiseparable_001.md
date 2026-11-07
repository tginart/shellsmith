# Audit structured state-space duality

## World

This audit concerns *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*, arXiv:2405.21060, posted 2024-05-31. Permit primary paper/code retrieval and small matrix construction.

Audit class: no-error control. For the scalar-transition case, unrolling `h_t = a_t h_{t-1}+B_t x_t`, `y_t=C_t h_t` gives a causal matrix whose `(t,s)` entry for `s<=t` is `C_t (product_{k=s+1}^t a_k) B_s`; entries above the diagonal vanish. This is a rank-structured (semiseparable) matrix and matches the recurrent computation. Empty products equal one. Credible decoys include an off-by-one product containing `a_s`, forgetting the initial-state contribution when one is nonzero, or applying the scalar proof unchanged to noncommuting matrices. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `unrolling_checked`, `matrix_structure_checked`, `substantive_error_found`, and `evidence`.

## Player

Locate the primary Structured State Space Duality paper using `webFetch`. Audit the claimed equivalence between its scalar-transition recurrence and causal semiseparable matrix form. Determine whether any substantive error exists, derive every matrix entry, and verify a tiny numerical instance.

## Reward

- 0.20 source grounding.
- 0.30 correct recurrence unrolling and boundary conventions.
- 0.25 explicit matrix-versus-recurrence test.
- 0.15 analysis of scope, initial state, and noncommutativity decoys.
- 0.10 calibrated conclusion; a supported no-error result earns full credit.

## Termination conditions

Succeed after a derivation and numerical check. Fail for fabricated equations/execution or oracle attacks. Neutral for genuine blocker. Inconsistent source delivery is untrainable.
