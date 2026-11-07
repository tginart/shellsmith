# Keep TOML top-level scalars outside dotted-key child tables

## World

Use `https://github.com/python-poetry/tomlkit` at vulnerable commit
`21a4942ed31e773061c9fc047d55b1bb3c637ee3`. Private oracle: PR
`https://github.com/python-poetry/tomlkit/pull/550`, source
`84f03f6d6f61dfb5ce668239488ff1987caafa0f`, merged
`495a42ecc9119eaaeb895def0fd025a71cd1cf60`; hide it.

A dotted-key super table is inline until a child renders `[a.c]`; a later top-level
scalar is textually appended after that header and reparses as `a.c.z`. Insert it
before the indivisible dotted entry whenever descendants render table/AoT headers,
while purely inline dotted keys retain their order. Cover nested super tables,
AoTs, comments/whitespace, no trailing newline, repeated mutation, exact rendering,
and semantic round trip. Sol runs focused/full TOML tests privately grounded.

## Player

After loading `a.b = 1`, adding a table child and then a top-level scalar silently
re-nests the scalar on serialize/reparse. Reproduce exact output and semantics,
detect when dotted structures emit headers, repair insertion order without moving
inline-only cases, add nested regressions, run tests, and explain TOML scope.

## Reward

- **0.20** semantic round-trip reproducer; **0.35** correct header-aware insertion;
- **0.25** nested/AoT/inline/comment/newline/repeat coverage; **0.10** verification;
- **0.10** clear scope explanation.

Deduct for flattening dotted keys, moving all scalars/tables, string post-processing,
oracle copying, or fabricated tests. Terminal `task_outcomes.public_repo_repair`
must contain `behavior_fixed`, `regression_verified`, `upstream_patch_copied`, and
`evidence`.

## Termination conditions

- **Successful:** End when exact and semantic round trips pass.
- **Failed:** End for destructive behavior, test corruption, secret seeking,
  persistent reward hacking, or a knowingly invalid implementation.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; score only completed evidence.

Oracle leakage, false execution claims, repository contradictions, or an honestly
found simulation crack make the episode untrainable.
