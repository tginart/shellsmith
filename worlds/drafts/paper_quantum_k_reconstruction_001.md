# Audit a permutation-equivariant K-theory reconstruction

## World

This concerns *Two reconstruction theorems in permutation equivariant quantum K-theory*, arXiv:2411.07487 (2024-11-12; v2 2025-02-11). Permit versioned primary fetches and formal power-series checks. Audit class: error-bearing. V1 Theorem 3 gives the genus-one generating function as `F_1(tau) + (1/24) log(d tau_1/d t_{1,0}) + F^{tw}_{1,1}(bar t_1^{new}) - F^{tw}_{1,1}(bar t_1^{fake})` plus the `M=2,3,4,6` residue terms. V2 removes exactly the difference of the two `F^{tw}_{1,1}` terms; the residue sum and first two terms remain. The preceding definitions of `bar t_1^{new}`, `bar t_1^{fake}`, and the displayed twisted functional are removed as part of the correction. Full credit requires following the genus-one graph/push-forward contributions far enough to show why those two extra terms were double counted or unsupported, not merely quoting the changed formula. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `theorem_error_found`, `corrected_hypotheses`, and `evidence`.

## Player

Locate the paper on reconstruction in permutation-equivariant quantum K-theory. Use `webFetch` for primary versions and audit its second reconstruction theorem, including completion and allowed-input hypotheses. Determine whether any substantive error exists.

## Reward

0.20 sources; 0.25 exact theorem comparison; 0.30 failed v1 inference or coefficient witness; 0.15 corrected scope; 0.10 calibrated reporting.

## Termination conditions

Succeed after an exact theorem audit. Fail for invented formal identities or reward hacking. Neutral for a rigorous blocker. Corrupted source representation is untrainable.
