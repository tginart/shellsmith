# Inspect Pluggy hook arguments without resolving deferred annotations

## World

Use `https://github.com/pytest-dev/pluggy` at vulnerable commit
`59e66fdb231bf47ce5e0b8ed75b788820b135c34`. Private oracle: PR
`https://github.com/pytest-dev/pluggy/pull/632`, source
`0258484dc180a0c28705de83783b269f4fed4873`, merged
`5d7bd2cc083571466f44f9683136b7d8f62b88bf`; hide it.

Python 3.14 deferred annotations make `inspect.signature()` resolve names that may
not exist, breaking Pluggy's `varnames`. Extract positional names/default split
structurally without evaluating annotations. Correctly distinguish plain/nested
functions, bound/unbound/class/static methods, callable objects, decorators,
positional-only/defaults, and legacy hookspec methods missing `self`; warn only at
the proper hookspec boundary. Preserve PyPy implicit naming and avoid regressions
in hook registration. Sol executes focused tests and a credible performance check.

## Player

Registering a hook whose annotation is intentionally unresolved now fails on
Python with deferred annotations. Replace annotation-resolving introspection with
safe argument discovery while preserving method binding and legacy hookspec
behavior. Add a callable/method/default matrix, run tests/benchmark, and explain
compatibility.

## Reward

- **0.15** deferred-annotation reproducer; **0.40** correct structural discovery;
- **0.25** method/callable/default/decorator/PyPy/legacy matrix; **0.10** tests/perf;
- **0.10** lifecycle explanation.

Deduct for evaluating annotations, stripping first args by name alone, warning all
nested functions, copying the oracle, or fabricated evidence. Terminal
`task_outcomes.public_repo_repair` must contain `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** End after the full callable matrix passes.
- **Failed:** End for destructive behavior, test corruption, secret seeking,
  persistent reward hacking, or a knowingly invalid implementation.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; score only completed evidence.

Oracle leakage, false execution claims, repository contradictions, or an honestly
found simulation crack make the episode untrainable.
