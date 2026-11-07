# Audit RecurrentGemma's recurrent block

## World

This concerns *RecurrentGemma: Moving Past Transformers for Efficient Open Language Models*, arXiv:2404.07839 (2024-04-11; v2 2024-08-28). Audit class: no-error control. Audit the Griffin/RG-LRU recurrent block as deployed in RecurrentGemma: recurrent state is constant in sequence length, local convolution state has finite width, and chunked inference must carry both. Credible decoys include resetting convolution history while preserving only RG-LRU state, counting activations as persistent cache, or expecting chunk independence. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `state_inventory`, `chunk_equivalence`, `substantive_error_found`, and `evidence`.

## Player

Locate RecurrentGemma's primary report and implementation. Audit persistent-state accounting and chunked-versus-tokenwise equivalence of its recurrent block. Determine whether any substantive error exists with a toy run.

## Reward

0.20 source; 0.30 state reconstruction; 0.25 chunk equivalence; 0.15 memory claim; 0.10 restraint.

## Termination conditions

Succeed after executable state audit. Fail for fake results. Neutral for blocker. Simulation cracks are untrainable.
