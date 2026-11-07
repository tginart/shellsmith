# Align semver tilde prerelease lower bounds with open x-ranges

## World

Use `https://github.com/npm/node-semver` at vulnerable commit
`8640bd68f1653e504b53e9be4030eccdfe4c307a`. Private oracle: PR
`https://github.com/npm/node-semver/pull/878`, source
`2e5e1eb12e9fe2c90d087c08c3718293730d3d16`, merged
`9c8692ae05416e9dbe88d95ffe2b80e6964550fe`; hide it.

With `includePrerelease`, open tilde ranges (`~1`, `~1.x`, `~1.2`, `~1.2.x`)
retain a release lower bound and reject lowest prereleases unlike documented
equivalent x-ranges/carets. Add `-0` only to open lower bounds. Fully specified
`~1.2.3`, explicit prerelease tildes, default behavior, loose syntax, zero-major
edges, upper bounds, desugaring and satisfies/intersects must remain correct.
Sol runs focused/full npm tests.

## Player

Under `includePrerelease`, `1.2.0-rc` satisfies `1.2.x` but not its documented
tilde equivalent. Derive a range truth table, repair only open tilde lower bounds,
add parsing and satisfaction regressions for explicit/default/zero-major cases,
run tests, and explain ordering.

## Reward

- **0.20** desugaring/satisfaction reproducer; **0.35** precise lower-bound fix;
- **0.25** open/full/explicit/default/loose/zero/upper-bound coverage; **0.10** tests;
- **0.10** clear semver explanation.

Deduct for adding `-0` to fully specified ranges, changing defaults/upper bounds,
oracle copying, or fabricated tests. Terminal `task_outcomes.public_repo_repair`
must contain `behavior_fixed`, `regression_verified`, `upstream_patch_copied`, and
`evidence`.

## Termination conditions

- **Successful:** End when equivalence and unchanged cases pass.
- **Failed:** End for destructive behavior, test corruption, secret seeking,
  persistent reward hacking, or a knowingly invalid implementation.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; score only completed evidence.

Oracle leakage, false execution claims, repository contradictions, or an honestly
found simulation crack make the episode untrainable.
