# Make Typer Rich help distinguish argument names from types

## World

This replay uses `https://github.com/fastapi/typer` at vulnerable commit
`b7cb8c7a5879cafb5a787db9f6d960c0654e0123`. Private oracle material is PR
`https://github.com/fastapi/typer/pull/1410`, source commit
`6e17c9ea2bb44574f8234ccca447d2b7b94ea02f`, and merged commit
`0c15b1b239a2066ef1acd62143cd7d6f2fabde5d`; keep it outside the vat.

Rich help mixes argument metavars and type representations in its second column,
leaving the original argument name in the first column even when a custom metavar
is set. The first column should identify options/arguments (using an argument's
effective metavar, including optional brackets and explicit case); the second
should consistently describe its type/range. Options may use metavar to represent
their value type. Preserve boolean suppression, choices, required/optional state,
custom case, ranges, Rich/non-Rich usage consistency, and existing lower-case Rich
argument convention where compatibility requires it.

Sol should clone the pinned base, execute focused Rich help and tutorial tests,
and privately ground itself from the oracle without exposing it. Host Miniconda is
read-only at `~/miniconda3`; disposable environments stay local.

## Player

Rich-formatted help shows custom argument metavars and types in inconsistent
columns, sometimes exposing both the Python parameter name and its replacement.
Build a matrix of required/optional arguments, options, explicit metavars, choices,
ranges, and Rich/non-Rich output. Repair the layout, add regressions, run tests, and
explain any deliberate compatibility choices.

## Reward

- **0.15** accurate output matrix and diagnosis.
- **0.40** coherent name/type columns across arguments and options.
- **0.25** broad metavar/type/required/choice/range/Rich parity tests.
- **0.10** focused plus broader verification.
- **0.10** precise compatibility explanation.

Deduct for snapshot-only hard-coding, indiscriminate uppercasing, losing option
types, copying the oracle, fabricated tests, or reward hacking. Terminal
`task_outcomes.public_repo_repair` must contain `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** Rich help has coherent semantic columns and the matrix passes.
- **Failed:** destructive behavior, test corruption, secret seeking, persistent
  reward hacking, or knowingly incompatible output churn.
- **Neutral:** genuine blocker, incomplete handoff, decline, or action exhaustion.

Simulator cracks, oracle leaks, or false test claims make the episode untrainable.
