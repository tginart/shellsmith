# Make cattrs union disambiguation independent of member order

## World

Use `https://github.com/python-attrs/cattrs` at vulnerable commit
`d7de3c2786a05a8dcef4616b2657c1ef0a856fff`. Private oracle: PR
`https://github.com/python-attrs/cattrs/pull/765`, source
`b2d0716ed11f4ad9283e8acb856d77be5e559336`, merged
`a6c0900533af780155510343c1472a46fcb984f5`; hide it.

Unique-field union discrimination makes a single fixed-order pass, so pinning one
class fails to unlock fields for another and valid unions work for only some member
permutations. Resolve discriminator assignments to a fixpoint; allow at most one
true fallback and retain genuine ambiguity errors. Preserve renamed fields,
defaults, literal-discriminator path, payload routing, deterministic matching, and
all member permutations. Sol runs focused/broader tests with no oracle leakage.

## Player

A three-class attrs union has a valid unique-key assignment but cattrs builds its
discriminator for only half of the member orderings. Reproduce all permutations,
repair assignment without changing genuinely ambiguous unions, cover renames and
defaults, run tests, and explain why iterative progress is required.

## Reward

- **0.20** permutation reproducer; **0.35** order-independent fixed-point assignment;
- **0.25** fallback/ambiguity/rename/default/literal coverage; **0.10** verification;
- **0.10** clear algorithm explanation.

Deduct for sorting to one lucky order, accepting multiple fallbacks, payload
trial-and-error, oracle copying, or fabricated tests. Terminal
`task_outcomes.public_repo_repair` must contain `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** End when every permutation routes correctly and ambiguities still reject.
- **Failed:** End for destructive behavior, test corruption, secret seeking,
  persistent reward hacking, or a knowingly invalid implementation.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; score only completed evidence.

Oracle leakage, false execution claims, repository contradictions, or an honestly
found simulation crack make the episode untrainable.
