# Preserve native Jinja values when rendering block references

## World

Use `https://github.com/pallets/jinja` at vulnerable commit
`791dd3b0415b99faac229d42a81db1d7096b003c`. Private oracle: PR
`https://github.com/pallets/jinja/pull/1702`, source
`d3a0b1a4abac05a071f542aa83edea4551c88fa3`, merged
`a4abbfd7538e9ff216244266579232585033136c`; hide it.

`BlockReference` uses module-global string concatenation instead of the active
environment's concat policy. In `NativeEnvironment`, `self.block()` therefore
coerces native integer/list/dict results or combines them incorrectly. Sync and
async block calls must use the environment operation while preserving standard
string/Markup autoescape, inheritance, `super()`, empty/multi-node blocks, and
custom environments. Sol clones the base and executes focused/broader tests.

## Player

Calling a block through `self` under Jinja's native environment changes the result
type compared with ordinary native rendering. Reproduce sync and async values,
route block assembly through the environment contract, add normal/autoescape/
inheritance regressions, run tests, and explain the dispatch boundary.

## Reward

- **0.20** typed block reproducer; **0.35** environment-aware sync/async concat;
- **0.25** native/string/Markup/super/inheritance coverage; **0.10** verification;
- **0.10** concise explanation.

Deduct for special-casing NativeEnvironment, changing global concat, losing
autoescape, oracle copying, fabricated tests, or reward hacks. Terminal
`task_outcomes.public_repo_repair` must contain `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** End when block results follow environment semantics across modes.
- **Failed:** End for destructive behavior, test corruption, secret seeking,
  persistent reward hacking, or a knowingly invalid implementation.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; score only completed evidence.

Oracle leakage, false execution claims, repository contradictions, or an honestly
found simulation crack make the episode untrainable.
