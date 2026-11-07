# Audit a Brownian last-passage shock theorem

## World

This concerns *Shocks and instability in Brownian last-passage percolation*, arXiv:2407.07866 (2024-07-10; v3 2026-03-03). Support versioned sources and finite path simulations. Audit class: error-bearing. V2 Theorem 4.16 claims there are almost surely no non-degenerate bi-infinite geodesics, using a scale-invariance proof that does not establish the needed uniform control over arbitrary bi-infinite geodesics. V3 weakens the theorem to no non-degenerate bi-infinite geodesic that is locally rightmost or locally leftmost, explicitly noting that uniqueness of finite geodesics only supports this extremality assumption. Its replacement proof encodes crossings at integer levels and invokes a disjoint-geodesic probability estimate adapted from Hammond, plus scaling. The unrestricted statement is left expected rather than proved. Require the exact theorem-scope change and why the corrected proof needs local extremality. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `null_set_error_found`, `corrected_extension_checked`, and `evidence`.

## Player

Locate the paper on shocks and instability in Brownian last-passage percolation. Download the primary paper and audit its main theorem-level claims about instability points and shocks, with special attention to almost-sure quantifiers. Determine whether a proof error exists.

## Reward

0.20 sources; 0.30 exact probability quantifiers; 0.25 uncountable-union flaw; 0.15 corrected dense-set extension; 0.10 restraint.

## Termination conditions

Succeed after a measure-theoretic proof audit. Fail for fabricated probability arguments. Neutral for honest blocker. Source cracks are untrainable.
