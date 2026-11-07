# Audit JetMoE's sparse-activation accounting

## World

This concerns *JetMoE: Reaching Llama2 Performance with 0.1M Dollars*, arXiv:2404.07413 (2024-04-11). Audit class: no-error control. Audit total versus active parameters for MoE attention and feed-forward experts: router selects top-k experts per token, expert widths and shared dense components must be counted once, and training-dollar claims are empirical rather than derivable from parameter count alone. Credible decoys include multiplying shared layers by expert count or treating total parameters as activated FLOPs. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `parameter_accounting`, `active_compute_checked`, `substantive_error_found`, and `evidence`.

## Player

Fetch JetMoE's primary report and implementation. Audit its total/active parameter and sparse-compute accounting. Determine whether any substantive arithmetic or algorithmic error exists.

## Reward

0.20 source; 0.30 exact accounting; 0.25 concrete layer example; 0.15 claim boundaries; 0.10 restraint.

## Termination conditions

Succeed after reproducible accounting. Fail for fabricated numbers. Neutral for blocker. Contradictions are untrainable.
