# Audit a spin-chain symmetry classification lemma

## World

This concerns *Classification of locality preserving symmetries on spin chains*, arXiv:2503.15088 (2025-03-19; v2 2025-05-09). Support versioned primary PDF/source and finite-chain algebra checks. Audit class: error-bearing. V1's Lemma 5.6/stabilization argument is used repeatedly to assert one stacked spin chain on which compatible covariant right restrictions exist at every site in an infinite lattice. The lemma only supplies a site-dependent auxiliary stabilization, so that simultaneous conclusion does not follow from the stated finite use. V2 repairs the construction by defining local degrees of freedom, allowing a countable family with uniformly bounded on-site dimension, and arranging the sitewise stabilizations on that single enlarged chain before decoupling. The anomaly classification survives. Require the exact old repeated-use step and the uniform boundedness argument, not a generic support complaint. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `lemma_error_found`, `corrected_support_argument`, and `evidence`.

## Player

Find the recent paper classifying locality-preserving symmetries on spin chains. Download primary versions with `webFetch` and audit the proof of the lemma that controls stable equivalence under composition. Determine whether any substantive error exists, checking all support neighborhoods explicitly.

## Reward

0.20 source grounding; 0.30 exact support/equivalence reconstruction; 0.30 localization and repair of the invalid v1 inference; 0.10 downstream scope; 0.10 restraint. Metadata-only answers cap at 0.30.

## Termination conditions

Succeed after a version-aware proof audit. Fail for fabricated text/algebra or oracle attacks. Neutral for honest blocker or bounded inconclusiveness. Contradictory source simulation is untrainable.
