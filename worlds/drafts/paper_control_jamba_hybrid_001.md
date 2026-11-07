# Audit Jamba's hybrid-layer accounting

## World

This concerns *Jamba: A Hybrid Transformer-Mamba Language Model*, arXiv:2403.19887 (2024-03-28; v2 2024-07-03). Audit class: no-error control. Audit the repeated ratio of attention to Mamba layers and MoE placement: total parameters include all experts, active parameters include only routed experts plus shared/non-MoE components, and KV cache is contributed only by attention layers. Credible decoys include applying the attention fraction to embeddings, counting all experts active, or attributing recurrent state to KV cache. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `layer_schedule_checked`, `active_parameter_check`, `substantive_error_found`, and `evidence`.

## Player

Fetch the primary Jamba report and audit its hybrid architecture's layer, active-parameter, and cache accounting. Determine whether any substantive error exists using explicit formulas.

## Reward

0.20 source; 0.30 architecture reconstruction; 0.25 accounting; 0.15 edge cases; 0.10 calibrated conclusion.

## Termination conditions

Succeed after exact accounting. Fail for fabricated numbers. Neutral for blocker. Contradictions are untrainable.
