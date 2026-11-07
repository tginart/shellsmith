# Audit the folded Reed--Solomon event lemma

## World

This audit concerns *Algorithmic Improvements to List Decoding of Folded Reed-Solomon Codes*, arXiv:2508.12548, submitted 2025-08-18 and revised 2025-08-20. Support primary arXiv discovery, versioned PDF/source download, text extraction, and small probability counterexamples.

Audit class: error-bearing. V1 Observation 2.15 assumes disjoint events with `Pr(union E_i) >= p` and each `Pr(E_i) <= p/2`, then asserts a two-way partition whose two union probabilities lie in `[p/4,3p/4]`. Its proof minimizes the larger part but silently treats the total mass as exactly `p`; the claim can fail when total mass exceeds `p`. V2 defines the actual union mass to equal `p`. In the downstream induction it introduces `q = Pr(union)` (called `p` there), partitions relative to `q`, and uses monotonicity of `q^2/16 - q p_r/32` for `q >= p_r` to recover the desired lower bound. Accept equivalent notation and valid counterexamples. Keep the oracle private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `event_lemma_error_found`, `downstream_fix_verified`, and `evidence`.

## Player

Locate the recent paper on near-linear-time list decoding of folded Reed--Solomon codes. Use `webFetch` to obtain the primary paper and audit its elementary probability observations and their use in the deterministic decoder. Determine whether any substantive error exists and justify the conclusion with a proof or counterexample.

## Reward

- 0.15 source and exact claim grounding.
- 0.30 independent proof attempt or concrete finite probability construction.
- 0.30 identification of the lower-bound-versus-equality defect and corrected parameterization.
- 0.15 verification of the repaired downstream induction rather than stopping at metadata.
- 0.10 clear, restrained reporting. Deduct for treating a renamed variable alone as an error.

## Termination conditions

Succeed after auditing the observation and its algorithmic use. Fail for fabricated text, hidden-oracle probing, or knowingly false probability arguments. Neutral endings require a serious bounded audit or genuine source blocker. Inconsistent source simulation is untrainable.
