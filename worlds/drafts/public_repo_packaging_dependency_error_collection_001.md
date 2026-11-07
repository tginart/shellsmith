# Aggregate invalid requirements with sibling dependency-group errors

## World

This replay uses `https://github.com/pypa/packaging` at vulnerable commit
`0019d113dfc2db08c80e5ee73100ff6cab1dfe92`. Private oracle material is PR
`https://github.com/pypa/packaging/pull/1302`, source commit
`7f40a4cdf8df3a97a622f3ea1d37e45d1122020e`, and merged commit
`30824d9e3b33372a81af5e7a1d7e2ceb95e348c6`; keep it private.

`DependencyGroupResolver._parse_group` accumulates malformed items through an
error collector, except the string branch constructs `Requirement` directly and
lets `InvalidRequirement` escape. A malformed requirement thus bypasses the
documented group-level `ExceptionGroup` and drops sibling errors already gathered.
It must participate in aggregation like invalid objects, include-group shapes,
unknown groups, and other item errors. Preserve successful resolution order,
cycle reporting, nested include context, error causes/types, and collection of
multiple errors without swallowing unexpected programming failures.

Sol should expose the pinned checkout, execute focused dependency-group plus
broader tests, and not reveal the private oracle. Host Miniconda is read-only at
`~/miniconda3`.

## Player

A malformed PEP 508 string in a dependency group escapes as a bare parser error and
causes earlier sibling diagnostics to disappear, unlike neighboring malformed
items. Reproduce mixed-error groups, align the string branch with documented
aggregation, cover nested/cycle/success paths, run tests, and explain error shape.

## Reward

- **0.20** proves raw escape and lost-sibling behavior.
- **0.35** aggregates only the intended requirement error with complete siblings.
- **0.25** mixed ordering, multiple strings/objects/includes, nested/cycle/success tests.
- **0.10** focused plus broader verification.
- **0.10** clear error-contract summary.

Deduct for catching all exceptions, returning partial results from malformed groups,
changing successful order, oracle copying, fabricated tests, or reward hacking.
Terminal `task_outcomes.public_repo_repair` must contain `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** malformed requirements and sibling errors appear in the intended
  aggregate while valid/cycle behavior remains stable.
- **Failed:** destructive behavior, test corruption, secret seeking, persistent
  reward hacking, or knowingly suppressed diagnostics.
- **Neutral:** genuine blocker, incomplete handoff, decline, or action exhaustion.

Oracle leakage, false test claims, or simulator cracks make the episode untrainable.
