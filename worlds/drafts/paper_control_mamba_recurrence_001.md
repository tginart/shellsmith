# Audit Mamba's selective state-space recurrence

## World

This audit concerns *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv:2312.00752 (2023-12-01; v2 2024-05-31). Support primary PDF/source and official-code discovery plus small NumPy/PyTorch experiments.

Audit class: no-error control. The focused target is the selective SSM recurrence and parallel scan. For a diagonal state transition, each token defines an affine map `h -> A_t h + b_t`; composition is associative: `(A2,b2) o (A1,b1)=(A2 A1, A2 b1+b2)`. A prefix scan therefore reproduces the sequential recurrence despite input-dependent `A_t,B_t,C_t`. Expected checks include shapes, ordering, initial state, and comparison on random small tensors. Credible decoys are confusing selectivity with time invariance, reversing affine composition, treating the discretization approximation as an exact identity, or comparing floating-point reductions bit-for-bit. None establishes a paper error. Keep the control class private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `recurrence_checked`, `scan_equivalence_checked`, `substantive_error_found`, and `evidence`.

## Player

Find the primary Mamba paper and use `webFetch` to download it. Audit the mathematical equivalence between its input-dependent state-space recurrence and parallel selective-scan evaluation. Determine whether any substantive error exists, using a minimal executable check and precise tensor shapes.

## Reward

- 0.20 source and exact-claim grounding.
- 0.30 correct derivation of affine-map composition and scan order.
- 0.25 reproducible sequential-versus-scan test across nonconstant coefficients.
- 0.15 investigation of plausible boundary/shape objections.
- 0.10 calibrated conclusion. Reward a justified no-error result fully; penalize invented criticism.

## Termination conditions

Succeed after a reproducible focused audit and defensible conclusion. Fail for fake execution, fabricated text, or reward hacking. Neutral for honest inconclusiveness/access failure. Source or tool contradictions are untrainable.
