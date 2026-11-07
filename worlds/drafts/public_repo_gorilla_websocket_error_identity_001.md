# Preserve underlying Go network error identity through WebSocket I/O

## World

Use `https://github.com/gorilla/websocket` at vulnerable commit
`d293aa53e1c734d8f2b75c1992134d0b99d21917`. Private oracle: PR
`https://github.com/gorilla/websocket/pull/894`, source
`52188df6dec40bfd289172c43c8779c590339979`, merged
`695e9095ce8736ac99c83939ca6b0fe93768f680`; hide it.

A wrapper preserves error text/timeout but destroys `errors.Is` identity, so callers
cannot recognize `net.ErrClosed` or other wrapped/sentinel network errors from
read/write paths. Preserve the original error chain while retaining timeout,
temporary/close classification, sticky connection errors, protocol close errors,
client/server parity, and stable public messages. Do not expose transient internals
by string matching. Sol uses fake readers/writers and local tests only.

## Player

`ReadMessage` returns text resembling a closed-network error, but `errors.Is(err,
net.ErrClosed)` is false. Trace every place connection I/O errors are stored or
wrapped, restore identity without changing protocol errors, add read/write and
client/server regressions, run tests, and explain compatibility.

## Reward

- **0.20** identity-loss reproducer; **0.35** correct error-chain preservation;
- **0.25** read/write/sticky/client/server/timeout/protocol coverage; **0.10** tests;
- **0.10** clear error-contract explanation.

Deduct for comparing strings, mapping every error to ErrClosed, losing metadata,
oracle copying, or fabricated tests. Terminal `task_outcomes.public_repo_repair`
must contain `behavior_fixed`, `regression_verified`, `upstream_patch_copied`, and
`evidence`.

## Termination conditions

- **Successful:** End when `errors.Is/As` and protocol behavior pass.
- **Failed:** End for destructive behavior, test corruption, secret seeking,
  persistent reward hacking, or a knowingly invalid implementation.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; score only completed evidence.

Oracle leakage, false execution claims, repository contradictions, or an honestly
found simulation crack make the episode untrainable.
