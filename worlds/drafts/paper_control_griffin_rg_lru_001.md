# Audit Griffin's gated linear recurrence

## World

This concerns *Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models*, arXiv:2402.19427 (2024-02-29). Audit class: no-error control. Focus on RG-LRU: the input-dependent decay is constrained to `[0,1]`, the state update combines decayed state and scaled input, and the square-root factor normalizes variance under the independence heuristic. It is not a convex interpolation unless coefficients sum accordingly, nor an exact stationarity proof. Credible decoys include dropping the square root, using an unconstrained decay, or treating heuristic variance assumptions as identities. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `recurrence_checked`, `variance_check`, `substantive_error_found`, and `evidence`.

## Player

Locate Griffin's primary paper and code with `webFetch`. Audit the RG-LRU update and its variance-normalization argument. Determine whether any substantive error exists and run a small stochastic check.

## Reward

0.20 source; 0.30 recurrence/shapes; 0.25 variance derivation/test; 0.15 assumptions; 0.10 restraint.

## Termination conditions

Succeed after a focused quantitative audit. Fail for fake tests. Neutral for blocker. Simulation inconsistency is untrainable.
