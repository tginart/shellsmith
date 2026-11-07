# Audit Mixture-of-Depths capacity routing

## World

This concerns *Mixture-of-Depths: Dynamically allocating compute in transformer-based language models*, arXiv:2404.02258 (2024-04-02). Audit class: no-error control. With capacity `C`, a block executes its expensive transform for the top-C router scores and passes other tokens through the residual path; static tensor shapes remain batchable. Expected compute follows selected tokens, but exact wall-clock savings depend on implementation. Credible decoys include dropping bypassed tokens, confusing top-C with independent thresholding, or claiming differentiability of hard selection without the training estimator. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `routing_semantics_checked`, `compute_accounting`, `substantive_error_found`, and `evidence`.

## Player

Locate the Mixture-of-Depths paper and code. Audit its fixed-capacity token-routing semantics, residual behavior, and compute accounting. Determine whether any substantive error exists with a toy batch.

## Reward

0.20 source; 0.30 routing reconstruction; 0.25 accounting/test; 0.15 estimator/scope; 0.10 restraint.

## Termination conditions

Succeed after executable routing audit. Fail for fake execution. Neutral for blocker. World cracks are untrainable.
