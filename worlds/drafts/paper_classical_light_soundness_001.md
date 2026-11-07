# Audit a classical-light verification soundness proof

## World

This audit concerns *Unconditional verification of quantum computation with classical light*, arXiv:2403.14142, submitted 2024-03-21 and revised 2024-03-29. Support versioned primary downloads and symbolic probability bounds.

Audit class: error-bearing. V1 splits malicious behavior using reported vacuum counts and obtains the invalid bound `p_acc <= max{exp(-m e^{-2 alpha^2} alpha^4/2), 1-b/2}`: its first case does not account for the probability that the *actual* vacuum-plus-one-photon count crosses the threshold in any of the `N` repetitions. V2 splits on that actual count, uses Hoeffding plus a union bound to get `p_(i) <= N exp(-m e^{-2 alpha^2} alpha^4/2)`, and therefore `p_acc <= 1-b/2 + N exp(...)`. With `alpha=1`, the required sample size changes exactly from `ceil(2 e^2 log(2 N^2 f(|x|)))` to `ceil(2 e^2 log(4 N^2 f(|x|)))`; the inverse-polynomial completeness--soundness gap then follows. Full credit requires reconstructing this event split rather than merely spotting the changed 2-to-4 constant. Keep all private guidance hidden.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `soundness_error_found`, `corrected_m_verified`, and `evidence`.

## Player

Find the recent paper proposing unconditional verification of quantum computation using classical light. Obtain primary versions with `webFetch` and audit the soundness proof quantitatively. Determine whether any substantive error exists; verify every probability budget and scaling parameter rather than trusting the theorem statement.

## Reward

- 0.15 source/version grounding.
- 0.30 reconstruction of the soundness events and probability bound.
- 0.25 demonstration that the original `m` does not establish the stated budget.
- 0.20 verification that the revised choice suffices and what results change.
- 0.10 clear calibrated report. Metadata-only claims receive little credit.

## Termination conditions

Succeed after a complete quantitative audit. Fail on fabricated equations, oracle extraction, or knowingly false probability claims. Neutral for honest bounded inconclusiveness or source failure. Contradictory source delivery is untrainable.
