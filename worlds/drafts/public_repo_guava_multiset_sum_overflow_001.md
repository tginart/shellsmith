# Saturate Guava multiset sum counts consistently

## World

Use `https://github.com/google/guava` at vulnerable commit
`117b5277123e3a2cb1cad35ec8e47c852a3a8858`. Private oracle: PR
`https://github.com/google/guava/pull/8509`, source and merged commit
`ee548013ec9aa94394c9bca2761edaca3be5e1ed`; keep hidden.

`Multisets.sum(...).size()` saturates at `Integer.MAX_VALUE`, but per-element
`count()` and `entrySet()` add with plain int overflow, producing negative counts
or exceptions. All public views must saturate consistently while preserving lazy
unmodifiable view behavior, iteration order, elements unique to either input,
zero/near/max counts, updates to backing multisets, equals/hash/contains, Android
source parity, and no allocation proportional to counts. Sol runs focused/full
Java tests.

## Player

Two large counts make `Multisets.sum().size()` saturate but `count()` and entry
iteration fail differently. Reproduce near-overflow boundaries, align every view
with the documented count domain, add lazy/order/update/Android regressions, run
tests, and explain the arithmetic policy.

## Reward

- **0.20** cross-view reproducer; **0.35** consistent saturation;
- **0.25** boundary/view/lazy/order/update/parity coverage; **0.10** tests;
- **0.10** clear numeric/API explanation.

Deduct for widening then narrowing unsafely, materializing occurrences, changing
ordering/laziness, oracle copying, or fabricated tests. Terminal
`task_outcomes.public_repo_repair` requires the standard four fields.

## Termination conditions

- **Successful:** End when all views saturate consistently and compatibility passes.
- **Failed:** End for destructive behavior, test corruption, secret seeking,
  persistent reward hacking, or a knowingly invalid implementation.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; score only completed evidence.

Oracle leakage, false execution claims, repository contradictions, or an honestly
found simulation crack make the episode untrainable.
