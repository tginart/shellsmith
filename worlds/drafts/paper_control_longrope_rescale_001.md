# Audit LongRoPE's nonuniform position interpolation

## World

This concerns *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens*, arXiv:2402.13753 (2024-02-21). Audit class: no-error control. Focus on nonuniform per-frequency rescaling and the retained unscaled prefix: the mapping must be monotone in position, paired dimensions share factors, and scale one reduces to original RoPE. The search procedure is empirical; it does not claim every factor vector preserves semantics. Credible decoys include discontinuity from an incorrectly joined prefix or applying factors to angles and positions simultaneously. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `position_map_checked`, `boundary_test`, `substantive_error_found`, and `evidence`.

## Player

Fetch LongRoPE's primary paper and implementation. Audit the nonuniform rotary-position rescaling and retained-prefix boundary. Determine whether any substantive algorithmic error exists with numerical boundary tests.

## Reward

0.20 source; 0.30 mapping derivation; 0.25 boundary/identity tests; 0.15 claim scope; 0.10 restraint.

## Termination conditions

Succeed after checked mapping. Fail for fabricated evidence. Neutral for blocker. Contradictions are untrainable.
