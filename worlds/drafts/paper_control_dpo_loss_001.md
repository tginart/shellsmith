# Audit the Direct Preference Optimization loss

## World

This concerns *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, arXiv:2305.18290 (2023-05-29; revised primary record). Support primary PDF and symbolic/autodiff checks. Audit class: no-error control. The DPO logit is beta times the policy log-ratio preference minus the reference log-ratio preference; minimizing negative log-sigmoid increases the preferred relative likelihood. Additive prompt-only normalizers cancel. Credible decoys include reversing chosen/rejected, omitting reference terms, or claiming sequence-length normalization is part of the derivation. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `loss_derived`, `gradient_sign_checked`, `substantive_error_found`, and `evidence`.

## Player

Fetch the primary DPO paper and audit the derivation and gradient direction of its preference loss. Determine whether any substantive mathematical error exists with symbolic or automatic differentiation checks.

## Reward

0.20 source; 0.30 derivation; 0.25 gradient tests; 0.15 assumptions; 0.10 restraint.

## Termination conditions

Succeed after checked loss audit. Fail for fabricated algebra/results. Neutral for blocker. World inconsistency is untrainable.
