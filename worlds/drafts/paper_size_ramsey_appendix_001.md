# Audit a multicolor size-Ramsey appendix

## World

This audit concerns *A note on the multicolor size-Ramsey numbers of connected graphs*, arXiv:2404.05856, submitted 2024-04-08 and revised through v3 on 2026-04-08. Permit primary version discovery and source/PDF comparison.

Audit class: error-bearing. The private target is the appendix theorem, whose v1 argument overstates the general graph conclusion. The final revision explicitly says the appendix error significantly changed Theorem 7.4. The corrected theorem must be read from v3 and compared hypothesis-by-hypothesis with v1: the agent should identify the lost or strengthened graph-family/parameter condition and show where v1 invokes an estimate outside its established scope. Because numbering and notation move across revisions, judge semantic correspondence, not page-number matching. Do not invent a formula not present in the downloaded versions; Sol should use the live primary sources as the exact oracle and keep this classification private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `appendix_error_found`, `theorem_change_explained`, and `evidence`.

## Player

Locate the recent note on multicolor size-Ramsey numbers of connected graphs. Use `webFetch` to obtain its primary versions and audit the appendix theorem and proof. Determine whether any substantive error exists, stating the original and defensible theorem with exact hypotheses.

## Reward

- 0.20 correct primary/version acquisition.
- 0.25 exact comparison of theorem statements and hypotheses.
- 0.30 localization of the unsupported v1 proof step with a counterexample or scope argument.
- 0.15 accurate account of what survives in the final theorem.
- 0.10 restraint. Cap metadata-only reports at 0.30.

## Termination conditions

Succeed after a source-grounded theorem/proof audit. Fail for fabricated theorem text, oracle probing, or knowingly invalid reasoning. Neutral if careful work remains inconclusive or sources are genuinely inaccessible. An inconsistent rendered appendix is untrainable.
