# Audit a Boolean-lattice Ramsey lower bound

## World

This audit concerns *Erdős-Hajnal problems for posets*, arXiv:2310.02621. The primary record was submitted 2023-10-04 and revised 2025-03-29. Support versioned primary downloads and local combinatorial/numerical checks.

Audit class: error-bearing. V1 states `R(Q_n,Q_n) > 2.24n`, taking `c=0.24` in the alternating-chain construction. The density/union-bound inequalities needed to construct the parallel families are not justified at that constant. V2 explicitly reports the mistake and weakens the construction to `c=0.02`, hence `N=(2+c)n=2.02n`; the proof and corollary are rewritten consistently around `2.02`. The audit should locate the failed quantitative condition rather than awarding full credit for merely quoting the version comment. It should verify that the revised parameter lies within the proved asymptotic regime and track the constant through Lemma 4.6 and the Ramsey corollary. Keep this private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `constant_error_found`, `revised_bound_verified`, and `evidence`.

## Player

Find the paper on Erdős--Hajnal problems for colored posets and Boolean lattices. Use `webFetch` to obtain primary versions, then audit the quantitative construction behind its improved diagonal poset-Ramsey lower bound. Determine whether any substantive error exists and trace every claimed constant to the inequalities that support it.

## Reward

- 0.15 primary-source grounding.
- 0.25 reconstruction of the construction parameters and required estimates.
- 0.30 identification of why `0.24` is unsupported in the original proof.
- 0.20 verification of the revised `0.02`/`2.02n` claim and downstream corollary.
- 0.10 restraint and precise citations. Metadata-only answers cap at 0.35.

## Termination conditions

Succeed after a quantitative version-aware audit. Fail for fabricated inequalities, reward hacking, or knowingly false conclusions. Neutral endings cover honest inconclusive work or verified access failure. Source inconsistency is untrainable.
