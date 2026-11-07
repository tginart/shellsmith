# Audit the Gated DeltaNet update

## World

This concerns *Gated Delta Networks: Improving Mamba2 with Delta Rule*, arXiv:2412.06464 (2024-12-09; v3 2025-03-06). Audit class: no-error control. The memory update first decays prior state, then applies a delta-rule correction proportional to the prediction residual for the current key. For normalized keys and admissible rates, the correction moves the retrieved value toward the target; it is not guaranteed to overwrite unrelated keys or converge under arbitrary correlations. Credible decoys include using target rather than residual, reversing matrix orientation, or claiming perfect associative recall. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `delta_update_checked`, `toy_retrieval_test`, `substantive_error_found`, and `evidence`.

## Player

Fetch the primary Gated Delta Networks paper and audit its gated delta-rule memory update. Determine whether any substantive mathematical error exists using a tiny retrieval simulation.

## Reward

0.20 source; 0.30 update derivation; 0.25 toy test; 0.15 scope/decoys; 0.10 calibrated report.

## Termination conditions

Succeed after a checked update. Fail for fabricated results. Neutral for blocker. Contradictions are untrainable.
