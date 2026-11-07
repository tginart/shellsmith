# Audit a gravity contribution to a Yukawa beta function

## World

This audit concerns *Gauge and regulator dependence of quantum gravity effects*, arXiv:2604.03033 (v1 2026-04-03; corrected v2 2026-05-01). Permit primary-version discovery, TeX/PDF comparison, and small symbolic or matrix calculations.

Audit class: error-bearing. The private target is the v1 diagonalization of the graviton Hessian used to obtain the gravitational Yukawa contribution. The relevant rank-two-tensor operator is not split by the usual trace/traceless projectors with multiplicities 1 and 9. Its momentum-dependent complementary projectors have dimensions 6 and 4 and eigenvalues 2 and $2\alpha$. Consequently v1's positive

`f_y = Gtilde*m/[4*pi*(m-1)] * [(3-alpha)(1-Lambda_tilde*(3-alpha)/m)^(1-m) + 9(1+alpha)(1-Lambda_tilde*(1+alpha)/m)^(1-m)]`

is replaced by

`f_y = -2*Gtilde/pi * m/(m-1) * [3(1-2*Lambda_tilde/m)^(1-m) + 2*alpha(1-2*alpha*Lambda_tilde/m)^(1-m)]`.

In the sharp-regulator limit this becomes `-2*Gtilde/pi * [3 exp(2 Lambda_tilde) + 2 alpha exp(2 alpha Lambda_tilde)]`. Full credit requires checking the projector algebra and multiplicities, not merely quoting the revision note. Keep this oracle private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `projector_error_found`, `corrected_beta_function_checked`, and `evidence`.

## Player

Find the paper on gauge and regulator dependence of quantum-gravity effects on matter couplings. Use `webFetch` to acquire its primary versions and audit the derivation of the gravitational contribution to the Yukawa beta function. Determine whether a substantive mathematical error exists.

## Reward

- 0.20 authoritative versions acquired.
- 0.25 Hessian/projector reconstruction.
- 0.25 old eigenspace or multiplicity error demonstrated.
- 0.20 corrected finite-$m$ and sharp-regulator formulas checked.
- 0.10 restrained, source-grounded report.

## Termination conditions

Succeed after a source-grounded algebraic audit. Fail for invented formulas, fake execution, or oracle probing. Neutral for an honest access blocker. Contradictory primary material makes the episode untrainable.
