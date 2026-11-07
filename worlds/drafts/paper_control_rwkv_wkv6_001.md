# Audit RWKV-6's matrix-valued recurrence

## World

This concerns *Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence*, arXiv:2404.05892 (2024-04-08; v4 2024-09-26). Audit class: no-error control. Audit WKV-6's per-channel dynamic decay and matrix-valued state update by unrolling three steps. Elementwise decay acts on the stored matrix with specified broadcasting; order matters. Credible decoys include replacing dynamic decay with a scalar, transposing key/value outer products, or comparing a reset chunk to continuous state. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `recurrence_unrolled`, `shape_test`, `substantive_error_found`, and `evidence`.

## Player

Locate the RWKV Eagle/Finch paper and official code. Audit the WKV-6 matrix-state recurrence, tensor shapes, and chunk continuity. Determine whether any substantive algorithmic error exists with a toy implementation.

## Reward

0.20 source; 0.30 unrolling; 0.25 code/shape check; 0.15 boundary analysis; 0.10 restraint.

## Termination conditions

Succeed after reproducible recurrence audit. Fail for fake execution. Neutral for blocker. World cracks are untrainable.
