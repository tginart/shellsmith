# Reduce p-limit queue overhead without changing async context

## World

Use `https://github.com/sindresorhus/p-limit` at vulnerable commit
`f53bdb5f464ae112b2859e834fdebedc0745199b`. Private oracle: PR
`https://github.com/sindresorhus/p-limit/pull/83`, source
`e3395efaf50efc25dc7abcf131b9fa40fd7893b7`, merged
`80273d742fa7d4d4845d7e598e19c2b430636519`; hide it.

Queueing via `AsyncResource.bind` allocates an async resource for every submitted
job and requires Node-only imports/stubs even though a promise continuation can
preserve the enqueue caller's async context. Queue a promise resolver so submission
is materially cheaper while concurrency, FIFO, active/pending counts, sync throws,
async rejection, clearQueue, and microtask scheduling remain correct. Cover
distinct `AsyncLocalStorage` values at concurrency 1 and greater, timing of
observable counts, and the browser-compatible import surface. Sol runs local Node
tests and repeatable queue-submission benchmarks.

## Player

Profiling shows substantial per-job queueing overhead from `AsyncResource.bind`,
and the implementation ships an async-hooks compatibility shim solely for it.
Establish a repeatable baseline, remove unnecessary resource binding using
platform-neutral promise mechanics, prove async context remains correct, verify
limiter accounting/FIFO/import behavior, and explain microtask timing tradeoffs.

## Reward

- **0.20** repeatable overhead baseline; **0.35** portable lower-overhead queueing;
- **0.25** concurrency/FIFO/count/error/clear/import matrix; **0.10** tests;
- **0.10** clear async lifecycle explanation.

Deduct for serializing all work, leaking Node-only imports, losing rejection/count
semantics, oracle copying, or fabricated tests. Terminal
`task_outcomes.public_repo_repair` must contain `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** End when submission overhead improves and context/limiter
  semantics pass.
- **Failed:** End for destructive behavior, test corruption, secret seeking,
  persistent reward hacking, or a knowingly invalid implementation.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; score only completed evidence.

Oracle leakage, false execution claims, repository contradictions, or an honestly
found simulation crack make the episode untrainable.
