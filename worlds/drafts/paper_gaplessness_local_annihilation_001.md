# Audit a gaplessness variational argument

## World

This audit concerns *Enforced Gaplessness from States with Exponentially Decaying Correlations*, arXiv:2503.01977, submitted 2025-03-03 and revised 2025-04-28. Enable primary PDF/source discovery and finite-dimensional operator checks.

Audit class: error-bearing. V1 Appendix C conflates a Hamiltonian being frustration-free (a decomposition into positive-semidefinite local terms each minimized by the state) with the weaker property actually used by parts of the variational calculation, that chosen local terms annihilate the state after harmless energy shifts. V2 introduces and defines `locally annihilating`, separates it from frustration freedom, adjusts Theorem 2's scope, and notes where positive semidefiniteness is unnecessary. The agent should track exactly which steps need annihilation, locality, positivity, and ground-state status; neither property implies every claimed conclusion without the other hypotheses. Keep oracle private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `assumption_error_found`, `corrected_scope_verified`, and `evidence`.

## Player

Locate the recent paper deriving gaplessness from exponentially decaying correlations. Use `webFetch` to obtain primary versions and audit the variational argument's Hamiltonian assumptions, especially its appendix. Determine whether any substantive logical error exists and state the weakest assumptions actually proved sufficient.

## Reward

- 0.15 source grounding.
- 0.25 precise definitions and implication analysis.
- 0.30 localization of the v1 conflation and its effect on theorem scope.
- 0.20 verification of the revised argument step by step.
- 0.10 careful communication. Deduct for claiming all frustration-free Hamiltonians are termwise zero without energy normalization.

## Termination conditions

Succeed after a source-grounded assumption audit. Fail for fabricated operator claims, hidden-oracle attacks, or intentional misstatement. Neutral after rigorous inconclusive work or true access failure. Simulation contradictions are untrainable.
