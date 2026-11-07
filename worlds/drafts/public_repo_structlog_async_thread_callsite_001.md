# Report the calling thread for structlog async log methods

## World

This replay uses `https://github.com/hynek/structlog` at vulnerable commit
`c6e7cad18ffe91febf2b97dac03a465134b1a5d1`. Private oracle material is PR
`https://github.com/hynek/structlog/pull/805`, source commit
`7d7708fdf39061b6f7515c13ed0404a9bc05fb33`, and merged commit
`ba04ac275291ea1754aba24643d17cb8e5638c66`; keep it private.

Async logging bridges synchronous work through an executor. Callsite stack context
is captured before dispatch, but `CallsiteParameterAdder` asks `threading` inside
the worker and records `asyncio_0`/worker identity instead of the caller's thread.
Capture caller thread id/name before dispatch and propagate it with context-local
state across native async, stdlib `BoundLogger`, legacy `AsyncBoundLogger`, generic
`alog`, and async exception paths. Reset tokens reliably on success/error and never
leak thread state to later synchronous logs, sibling tasks, or nested calls.

Sol should expose the pinned checkout, run focused async processor plus broader
tests, and keep oracle details hidden. Host Miniconda is read-only at
`~/miniconda3`.

## Player

Async structlog calls report the executor worker as `thread`/`thread_name`, while
other callsite fields identify the caller. Reproduce across logger wrappers,
preserve the pre-dispatch caller identity through the async bridge, test cleanup,
exceptions, concurrency, and synchronous behavior, run tests, and summarize the
context-propagation contract.

## Reward

- **0.15** reproduces caller/worker mismatch across relevant wrappers.
- **0.40** correct caller identity propagation with reliable token cleanup.
- **0.25** native/stdlib/legacy/alog/exception/concurrent/sync regressions.
- **0.10** focused and broader verification.
- **0.10** clear concurrency and lifecycle explanation.

Deduct for globally cached thread state, hard-coded MainThread, missing finally
cleanup, changing executor behavior, oracle copying, fabricated tests, or reward
hacking. Terminal `task_outcomes.public_repo_repair` must contain `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** all async bridges report caller thread identity without leakage
  and tests pass.
- **Failed:** destructive behavior, test corruption, secret seeking, persistent
  reward hacking, or knowingly unsafe global state.
- **Neutral:** genuine blocker, incomplete handoff, decline, or action exhaustion.

Oracle leaks, false concurrency/test claims, or simulator contradictions are untrainable.
