# Audit a text-to-pattern Hamming-distance lemma

## World

This concerns *Faster Algorithms for Text-to-Pattern Hamming Distances*, arXiv:2310.13174 (2023-10-19; v3 2024-12-19). Support versioned source/code and random finite tests. Audit class: error-bearing. V2's auxiliary sparse-convolution lemma claims it can discover a small support superset `T` for `A star B-C` internally by folding the arrays to half the universe and recursing; that recursive support inference is not justified by the stated folded identity. V3 changes the lemma's contract to require `T` explicitly as input. The caller then constructs `T` recursively from a smaller pair-sum count problem: any coefficient at least `p` maps to one of two folded counts at least `p/2`, and the sum of folded counts bounds `|T|=O(n^2/p)`. Only after this proven support superset exists is the Chan--Lewenstein isolating-modulus lemma applied. Require this exact contract change, threshold/pigeonhole argument, and recurrence. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `hash_dependency_error_found`, `corrected_lemmas_checked`, and `evidence`.

## Player

Locate the recent paper on faster text-to-pattern Hamming distances. Fetch all primary versions and audit the proof of Lemma 5.3 together with its hashing auxiliary lemma. Determine whether any substantive probabilistic error exists.

## Reward

0.20 sources; 0.30 exact randomness/dependency reconstruction; 0.25 old invalid conditioning; 0.15 corrected lemma chain; 0.10 restraint.

## Termination conditions

Succeed after a reproducible proof audit. Fail for fabricated probability or code results. Neutral for honest blocker. Contradictory source simulation is untrainable.
