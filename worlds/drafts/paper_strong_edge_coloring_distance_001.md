# Audit a strong edge-coloring minimal-counterexample proof

## World

This concerns *Strong edge-coloring of graphs with maximum edge weight seven*, arXiv:2505.20345 (2025-05-25; v2 2025-09-22). Support versioned primary source and exhaustive checks on small graphs. Audit class: error-bearing. V1 takes a minimal counterexample, deletes arbitrary edges, and treats a strong edge-coloring of the resulting subgraph `H'` as a valid partial strong coloring of `H`. That implication is false: a deleted edge can create a length-two relation in `H` between two retained edges whose distance is at least three in `H'`, so those retained edges may share a color legally in `H'` but conflict in `H`. V2 restricts the core extension lemma to `H'` obtained by deleting a vertex and all incident edges; it explicitly proves that distance at least three in `H'` then remains at least three in `H`, and rewrites the reducible configurations/recolorings around this safe deletion. Require a concrete counterexample to arbitrary-edge deletion and verification of the vertex-deletion invariant. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `distance_error_found`, `corrected_extension_lemma`, and `evidence`.

## Player

Find the recent paper on strong edge-coloring for graphs of maximum edge weight seven. Download primary versions with `webFetch` and audit the minimal-counterexample extension argument, especially whether a coloring of each chosen subgraph remains a valid partial coloring of the original graph. Determine whether any substantive error exists.

## Reward

0.20 sources; 0.30 exact distance/partial-coloring analysis; 0.25 counterexample to the old deletion step; 0.15 verification of the corrected vertex-deletion lemma; 0.10 restraint.

## Termination conditions

Succeed after a reproducible graph-distance audit. Fail for fabricated colorings or source claims. Neutral for genuine blocker. Contradictory simulated source or execution is untrainable.
