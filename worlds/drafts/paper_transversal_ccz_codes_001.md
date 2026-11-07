# Audit a transversal-CCZ code construction

## World

This concerns *Good binary quantum codes with transversal CCZ gate*, arXiv:2408.10140 (2024-08-19; v2 2024-10-14). Permit primary versions and small binary-code computations. Audit class: error-bearing. In v1's qudit-to-qubit conversion, the claimed degree-three multiplication-friendly embedding uses the coordinatewise product `sigma(x) star sigma(y) star sigma(z)` and therefore concludes that physical CCZ gates act on the same coordinate `i` of each of three encoded registers. The displayed coefficient embedding does not align the monomial `x_i y_j z_k` that way. V2 adds two fixed coordinate permutations `pi_2,pi_3` so `psi(sigma(x) star pi_2(sigma(y)) star pi_3(sigma(z)))=xyz`; the physical CCZ triplets are consequently `(i,pi_2^{-1}(i),pi_3^{-1}(i))`. This still gives a transversal gate, but the missing permutations make the v1 equality false. Require a small coefficient-basis check. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `representative_error_found`, `corrected_ccz_condition`, and `evidence`.

## Player

Locate the paper constructing good binary quantum codes with transversal CCZ. Fetch primary versions and audit Section 4's passage from classical multiplication to a well-defined logical gate. Determine whether any substantive error exists, using a tiny code if useful.

## Reward

0.20 sources; 0.30 algebraic condition reconstruction; 0.25 representative-dependence witness; 0.15 corrected condition; 0.10 restraint.

## Termination conditions

Succeed after a code-level audit. Fail for fabricated code checks or source claims. Neutral for honest blocker. Contradictory tool output is untrainable.
