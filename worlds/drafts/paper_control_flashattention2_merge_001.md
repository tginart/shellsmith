# Audit FlashAttention-2's softmax merge

## World

This audit concerns *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, arXiv:2307.08691, posted 2023-07-17. Support primary-paper and official-code fetches plus CPU reference tests.

Audit class: no-error control. Audit the tiled online-softmax recurrence used to merge key/value blocks. The running row maximum and normalization sum must rescale the previous output when a later block has a larger maximum; with that rescaling the recurrence is algebraically identical to dense softmax attention. In causal mode, global row/column offsets determine the mask. Credible decoys are omitting the output rescale in a reimplementation, comparing mixed-precision results with zero tolerance, or attributing the number of non-matmul FLOPs to the mathematical identity. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `merge_identity_checked`, `reference_test_checked`, `substantive_error_found`, and `evidence`.

## Player

Find the primary FlashAttention-2 paper and obtain it with `webFetch`. Audit the online-softmax merge underlying its tiled attention computation, including a causal case. Determine whether any substantive mathematical or algorithmic error exists and compare against dense attention in a minimal program.

## Reward

- 0.20 source grounding.
- 0.30 exact stable merge derivation.
- 0.25 executable dense comparison with adversarial block maxima.
- 0.15 causal/mixed-precision edge-case analysis.
- 0.10 calibrated conclusion. Penalize faults caused only by the agent's incomplete reimplementation.

## Termination conditions

Succeed after a reproducible identity audit. Fail for fake execution, fabricated source claims, or reward hacking. Neutral for honest access/inconclusive ending. Simulation inconsistency is untrainable.
