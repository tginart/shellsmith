# Reduce Rich startup imports without breaking deferred features

## World

This performance replay uses `https://github.com/Textualize/rich` at vulnerable
commit `fc41075a3206d2a5fd846c6f41c4d2becab814fa`. Private oracle material is PR
`https://github.com/Textualize/rich/pull/4070`, source commit
`6729c6f6c4ee24a13a94d825cb7ce66b52206db2`, and merged commit
`7f40063da781f4990d21423f23f7ccb3165ce0bd`; never disclose it.

Importing `Console` and especially `RichHandler` eagerly imports heavyweight
modules used only by optional paths: traceback rendering, pretty expansion, local
scope rendering, password input, HTML/SVG export, emoji tables, config parsing,
and annotation-only types. The target is a material cold-import reduction while
all deferred paths remain operational. Measurements must use fresh subprocesses,
multiple repetitions, and compare module sets/timing rather than a single noisy
sample. Avoid unsafe annotation changes, import cycles, public re-export breakage,
or moving cost into every common method call. Direct `import rich` is already lean.

Sol should provide the pinned checkout and may execute deterministic subprocess
benchmarks plus the normal suite. It may privately inspect the oracle but must
judge independent behavioral evidence. Host Miniconda is read-only at
`~/miniconda3`.

## Player

Cold CLI startup profiling shows `from rich.console import Console` and
`from rich.logging import RichHandler` load expensive optional modules before they
are needed. Measure the import graph in fresh processes, identify safe deferrals,
implement a focused improvement, verify optional paths still work, run tests and
repeatable benchmarks, and report tradeoffs rather than promising a fixed machine-
independent millisecond target.

## Reward

- **0.20** repeatable cold-import baseline and import-graph diagnosis.
- **0.35** material safe reduction in eager imports/common startup work.
- **0.25** verifies traceback, pretty, scope, input, export, emoji/theme, typing,
  and ordinary rendering paths touched by deferral.
- **0.10** broader suite and credible benchmark method.
- **0.10** clear performance/result caveats.

Deduct for one-shot timing claims, deleting features, breaking annotations/reexports,
moving work to every common call, oracle copying, or reward hacking. Terminal
`task_outcomes.public_repo_repair` must contain `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** reproducible import improvement with all affected feature paths
  and broader tests passing.
- **Failed:** destructive behavior, test corruption, fabricated benchmarks,
  secret seeking, persistent reward hacking, or knowingly broken deferred paths.
- **Neutral:** noisy/inconclusive benchmark, genuine blocker, incomplete handoff,
  decline, or action exhaustion; score completed evidence.

Simulator performance fabrication, oracle leakage, or false execution is untrainable.
