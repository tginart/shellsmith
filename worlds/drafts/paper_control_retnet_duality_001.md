# Audit RetNet's parallel/recurrent duality

## World

This concerns *Retentive Network: A Successor to Transformer for Large Language Models*, arXiv:2307.08621 (2023-07-17; v4 2023-08-09). Audit class: no-error control. The focused retention kernel has multiplicative decay `gamma^(t-s)` and phase factors that factor into query/key rotations; a running state updated by `S_t=gamma S_{t-1}+K_t^T V_t` yields the same causal sum as the parallel form, with correct complex conjugation/real representation. Credible decoys are reversing the decay exponent, missing conjugation, or comparing chunk boundaries without carrying state. Support primary/code fetches and tests. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `duality_derived`, `numerical_check`, `substantive_error_found`, and `evidence`.

## Player

Download the primary RetNet paper via `webFetch` and audit the claimed equivalence of parallel and recurrent retention. Determine whether any substantive error exists and verify a small causal example.

## Reward

0.20 source; 0.30 derivation; 0.25 numerical/chunk test; 0.15 edge cases; 0.10 restraint.

## Termination conditions

Succeed after a checked equivalence. Fail for fabricated results. Neutral for blocker. Simulation cracks are untrainable.
