# Report null AtomicLongArray elements as Gson syntax errors

## World

Use `https://github.com/google/gson` at vulnerable commit
`004e7a4949e08b430e3c8996998ee5a17ff9423a`. Private oracle: PR
`https://github.com/google/gson/pull/3038`, source
`a872de0ca302e720d72cd4abe1ee1eb2ff845105`, merged
`e4f54f78c93dae0507fcac5be81253974bc9042d`; hide it.

The AtomicLongArray adapter unboxes a nullable number and leaks
`NullPointerException` for `[1,null,3]`. Invalid JSON data should produce the
library's structured `JsonSyntaxException` with useful context, consistently with
bad numeric values. Preserve valid numbers/strings under configured policies,
top-level null, empty arrays, overflow/coercion policy, reader position, nested
objects, AtomicLong behavior, and exception causes/messages. Sol runs focused/full
Java tests without oracle leakage.

## Player

Deserializing a null element inside `AtomicLongArray` leaks an implementation NPE
instead of a JSON data error. Reproduce null at several positions and neighboring
numeric failures, repair the adapter boundary, add valid/policy/nested regressions,
run tests, and explain exception consistency.

## Reward

- **0.20** faithful NPE reproducer; **0.35** structured syntax error;
- **0.25** positions/top-null/empty/numeric/policy/nested/AtomicLong coverage;
- **0.10** tests; **0.10** clear explanation.

Deduct for converting null to zero, catching all exceptions, changing numeric
policy, oracle copying, or fabricated tests. Terminal
`task_outcomes.public_repo_repair` requires the standard four fields.

## Termination conditions

- **Successful:** End when invalid nulls fail consistently and valid behavior passes.
- **Failed:** End for destructive behavior, test corruption, secret seeking,
  persistent reward hacking, or a knowingly invalid implementation.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; score only completed evidence.

Oracle leakage, false execution claims, repository contradictions, or an honestly
found simulation crack make the episode untrainable.
