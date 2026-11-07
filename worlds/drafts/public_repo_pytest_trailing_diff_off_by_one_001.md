# Skip long identical pytest diff suffixes from the actual end

## World

Use `https://github.com/pytest-dev/pytest` at vulnerable commit
`9db664649af9b24d496fb7ad482b66c0c341a28c`. Private oracle: PR
`https://github.com/pytest-dev/pytest/pull/14639`, source
`19935985bbc060392d043326ebe52c50f1fd6222`, merged
`3fa8d9b15b733aadb8a043cca3e98447804e1f28`; keep hidden.

The equal-length trailing scan starts at `i=0`; Python `s[-0]` is `s[0]`, so when
first characters differ it stops before examining a long identical suffix. Start
from the final character and preserve the 42-character threshold, unequal-length
logic, verbose mode, empty/one-character strings, Unicode, and exact diagnostic
shape. Sol clones the base and runs focused assertion plus broader tests.

## Player

Pytest expands a long shared suffix instead of reporting it as identical whenever
the strings differ at their first character. Build boundary cases around suffix
length and equal/unequal lengths, repair the scan, add regressions for diagnostics
and verbose behavior, run tests, and explain the indexing trap.

## Reward

- **0.20** exact output reproducer; **0.35** correct end scan and threshold;
- **0.25** boundary/length/empty/Unicode/verbose coverage; **0.10** verification;
- **0.10** clear explanation.

Deduct for changing thresholds, quadratic slicing, snapshot-only hard-coding,
oracle copying, fabricated tests, or reward hacks. Terminal
`task_outcomes.public_repo_repair` must contain `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** End when suffix elision is correct without output regressions.
- **Failed:** End for destructive behavior, test corruption, secret seeking,
  persistent reward hacking, or a knowingly invalid implementation.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; score only completed evidence.

Oracle leakage, false execution claims, repository contradictions, or an honestly
found simulation crack make the episode untrainable.
