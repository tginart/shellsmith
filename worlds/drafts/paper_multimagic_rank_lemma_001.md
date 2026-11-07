# Audit a multimagic-square rank lemma

## World

This audit concerns *A circle method approach to K-multimagic squares*, arXiv:2406.08161, submitted 2024-06-12 and revised 2025-01-01. Support primary version downloads and small matrix-rank computations.

Audit class: error-bearing. V1 Lemma 4.1 claims for all `J subset [N]^2` a rank lower bound based on `ceil(2 sqrt(|J|))+1`, minus two plus `E_N(J)`. The greedy row/column-removal argument overlooks the exceptional size `|J|=(N-1)^2+1` and the interaction between irreducible components and the all-ones intersection. V2 replaces it with a piecewise bound: generally `min{ceil(2 sqrt(|J|))-1,2N-2}+E_N(J)`, but `2N-3+E_N(J)` at the exceptional size, and supplies the projection/irreducible decomposition. Full credit includes an explicit small-rank witness or verified enumeration and tracking the repaired lemma into the circle-method domination claim. Keep this private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `rank_lemma_error_found`, `piecewise_fix_verified`, and `evidence`.

## Player

Find the recent circle-method paper on multimagic squares. Obtain primary versions through `webFetch` and audit the combinatorial matrix-rank lemma underlying its main application. Determine whether any substantive error exists, using exact linear algebra or small exhaustive computations where useful.

## Reward

- 0.15 primary/version grounding.
- 0.25 independent reconstruction or computation of the rank quantity.
- 0.30 identification of the missed exceptional configuration.
- 0.20 verification of the corrected piecewise bound and downstream use.
- 0.10 clear restraint. Metadata-only answers cap at 0.30.

## Termination conditions

Succeed after a reproducible rank/proof audit. Fail for fake computations, oracle probing, or knowingly false claims. Neutral for well-supported inconclusiveness or access failure. Inconsistent simulated matrices or paper text are untrainable.
