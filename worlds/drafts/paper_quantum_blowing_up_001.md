# Audit a distributed quantum-testing exponent proof

## World

This concerns *Distributed Quantum Hypothesis Testing under Zero-rate Communication Constraints*, arXiv:2410.08937 (2024-10-11; corrected revisions through 2026-04-28). Support all primary versions and finite-dimensional density-matrix experiments. Audit class: error-bearing. The concrete defect is the inequality on the fourth line at the top of page 31 of the earlier proof: the stated quantum blowing-up step is not valid for arbitrary noncommuting $\widetilde\rho_{AB}$ and $\rho_A\otimes\rho_B$. The corrected blowing-up lemma adds the missing commutation hypothesis $[\rho_A\otimes\rho_B,\widetilde\rho_{AB}]=0$, and the theorems that actually use that lemma inherit it. Separately, the important product-of-marginals single-letter result is recovered without that hypothesis by a new reverse-hypercontractivity and pinching argument. Require the agent to distinguish these two corrected proof routes and verify their hypotheses rather than claiming one universal replacement. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `blowing_up_error_found`, `replacement_proof_checked`, and `evidence`.

## Player

Find the paper on distributed quantum hypothesis testing with zero-rate communication. Download primary versions via `webFetch` and audit the proof of its single-letter Stein exponent, especially any blowing-up argument. Determine whether a substantive proof gap exists.

## Reward

0.15 sources; 0.30 reconstruction of the disputed bridge; 0.25 exact noncommutative gap; 0.20 verification of reverse-hypercontractive replacement; 0.10 restraint.

## Termination conditions

Succeed after a source-grounded exponent audit. Fail for fabricated inequalities or oracle probing. Neutral for honest blocker. Inconsistent primary evidence is untrainable.
