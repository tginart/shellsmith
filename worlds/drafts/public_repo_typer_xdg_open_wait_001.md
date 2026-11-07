# Respect Typer launch wait semantics on Linux

## World

This replay uses `https://github.com/fastapi/typer` at vulnerable commit
`2fe5d0c2b63db2243b3f3ad37d9f164791a390b1`. Private oracle material is PR
`https://github.com/fastapi/typer/pull/1820`, source commit
`7217cd9ffca56ede7c74d6911a3f7442ccda0cae`, and merged commit
`8e75bc3565d5dc270cd4ef3b09b5b2abf9f8491a`; never reveal it.

On Linux/BSD with `xdg-open`, `launch(..., wait=False)` still calls
`Popen(...).wait()`, blocking despite the API default. Start the process in both
cases, call `wait()` and return its code only when requested, and otherwise return
the established immediate success value. Preserve subprocess arguments and stream
redirection, URL/file distinctions, fallback browser behavior, and other platform
branches. Tests must mock process/platform boundaries and launch nothing real.

Sol should provide the pinned checkout and execute focused launch plus broader
tests; oracle details stay private. Host Miniconda is read-only at
`~/miniconda3`.

## Player

`typer.launch(url)` unexpectedly blocks on a Linux machine where `xdg-open` is
available even though `wait` defaults to false. Reproduce using mocks, repair true
and false wait modes without spawning real desktop applications, cover fallback
and compatibility paths, run tests, and summarize return semantics.

## Reward

- **0.15** deterministic reproducer proving the unwanted wait call.
- **0.40** correct wait/no-wait process and return behavior.
- **0.25** mocks both modes, process code, command/streams, platform/fallback paths.
- **0.10** focused and broader tests.
- **0.10** clear behavioral summary.

Deduct for real URL launches, always returning zero, removing wait support,
platform-wide behavior changes, oracle copying, fabricated tests, or reward hacks.
Terminal `task_outcomes.public_repo_repair` must contain `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** mocked no-wait never waits, requested wait returns process status,
  and relevant tests pass.
- **Failed:** destructive/external side effects, test corruption, secret seeking,
  persistent reward hacking, or knowingly broken platform behavior.
- **Neutral:** genuine blocker, incomplete handoff, decline, or budget exhaustion.

False process execution, oracle leakage, or simulator inconsistency is untrainable.
