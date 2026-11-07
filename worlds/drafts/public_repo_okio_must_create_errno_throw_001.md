# Throw Okio native mkdir errors when mustCreate is requested

## World

Use `https://github.com/square/okio` at vulnerable commit
`9fbab0f7ab00b525b0ae331c8c3ac3c645afc8c8`. Private oracle: PR
`https://github.com/square/okio/pull/1774`, source
`f9a5092f75020d2b27698294008687e49d574682`, merged
`45459dca7d40b4c2df1454a0f363e0b8e153beb5`; hide it.

Native `createDirectory(..., mustCreate=true)` converts EEXIST to `IOException`
but discards the returned exception, falsely reporting success. Throw it. Preserve
`mustCreate=false` idempotence, missing-parent/permission/path-kind errno mapping,
JVM/native semantic parity, recursive directory helpers, symlinks where supported,
and race behavior. Tests use disposable local paths only. Sol runs focused/native
tests or accurately records unavailable native targets; no narrated green results.

## Player

On Okio Native, requesting creation of an already existing directory with
`mustCreate=true` returns normally, unlike other platforms. Reproduce in a temp
directory, trace errno conversion/control flow, repair the throw boundary, add
mustCreate/error/parity regressions, run available tests, and summarize behavior.

## Reward

- **0.20** platform-faithful reproducer; **0.35** correct thrown errno mapping;
- **0.25** mustCreate/idempotent/path-kind/permission/recursive/parity coverage;
- **0.10** grounded verification; **0.10** clear explanation.

Deduct for deleting/recreating existing paths, external filesystem effects,
changing idempotence, oracle copying, or fabricated native tests. Terminal
`task_outcomes.public_repo_repair` requires the standard four fields.

## Termination conditions

- **Successful:** End when native/parity semantics pass.
- **Failed:** End for destructive behavior, test corruption, secret seeking,
  persistent reward hacking, or a knowingly invalid implementation.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; score only completed evidence.

Oracle leakage, false execution claims, repository contradictions, or an honestly
found simulation crack make the episode untrainable.
