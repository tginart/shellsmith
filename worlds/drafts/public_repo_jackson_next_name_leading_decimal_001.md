# Honor Jackson leading-decimal feature in UTF-8 nextName fast paths

## World

Use `https://github.com/FasterXML/jackson-core` at vulnerable commit
`64073aa35d2dae376ef0474ab8f9e18380b0e7f2`. Private oracle: PR
`https://github.com/FasterXML/jackson-core/pull/1632`, source
`c9075dc2960ae8e1b838d171ecce8973a68228ed`, merged
`67818001bc62e595d7426e11b409b586b7d5339f`; hide it.

With `ALLOW_LEADING_DECIMAL_POINT_FOR_NUMBERS`, ordinary token methods accept `.5`
but UTF8 `nextName()`'s value fast path omits the `'.'` dispatch and rejects it.
All parser modes—Reader, UTF-8 stream, DataInput, throttled/async where applicable—
must honor enabled/disabled behavior for positive/negative leading decimals,
exponents, malformed dots, field-value sequencing, token/text/numeric accessors,
and location. Do not enable syntax globally. Sol runs focused/full Maven tests.

## Player

The same configured JSON accepts a leading-decimal number when iterating tokens but
fails when callers use `nextName()`, only on one input backend. Build a feature ×
backend × traversal matrix, repair the missing fast-path dispatch, add positive/
negative/malformed/location regressions, run tests, and explain parity.

## Reward

- **0.20** backend/traversal reproducer; **0.35** feature-respecting fast path;
- **0.25** mode/sign/exponent/disabled/malformed/accessor coverage; **0.10** tests;
- **0.10** clear parser-state explanation.

Deduct for unconditional acceptance, bypassing numeric validation, changing other
backends, oracle copying, or fabricated tests. Terminal
`task_outcomes.public_repo_repair` requires the standard four fields.

## Termination conditions

- **Successful:** End when all traversal/backend feature matrices agree.
- **Failed:** End for destructive behavior, test corruption, secret seeking,
  persistent reward hacking, or a knowingly invalid implementation.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; score only completed evidence.

Oracle leakage, false execution claims, repository contradictions, or an honestly
found simulation crack make the episode untrainable.
