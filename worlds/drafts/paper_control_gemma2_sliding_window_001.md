# Audit Gemma 2's interleaved attention reach

## World

This concerns *Gemma 2: Improving Open Language Models at a Practical Size*, arXiv:2408.00118 (2024-07-31; v3 2024-10-02). Audit class: no-error control. Audit interleaved local sliding-window and global attention. A local layer alone has window-limited direct access, stacked locals expand receptive reach, and periodic global layers connect all prior causal positions. KV-cache memory depends on which layers retain full versus windowed keys. Credible decoys include claiming the whole model has only one-window context or assuming every layer stores full context. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `receptive_field_checked`, `cache_accounting`, `substantive_error_found`, and `evidence`.

## Player

Fetch the Gemma 2 report and audit the receptive field and cache implications of interleaved sliding-window/global attention. Determine whether any substantive algorithmic or arithmetic error exists with a small dependency graph.

## Reward

0.20 source; 0.30 dependency derivation; 0.25 cache accounting; 0.15 edge cases; 0.10 restraint.

## Termination conditions

Succeed after a checked graph/accounting audit. Fail for fabricated architecture. Neutral for blocker. World inconsistencies are untrainable.
