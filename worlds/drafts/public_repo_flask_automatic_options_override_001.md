# Honor explicit Flask automatic-OPTIONS enablement

## World

Use `https://github.com/pallets/flask` at vulnerable commit
`d3b78fd18a8d9e224cb9ef58a23cec9b1ffc9ce9`. Private oracle: PR
`https://github.com/pallets/flask/pull/5917`, source
`e82db2ca3a22c9614c1987392c9cfaa8c6ce99ad`, merged
`12e95c93b488725f80753f34b2e0d24838ca4646`; never expose it.

`provide_automatic_options=True` can only disable, not force-enable, OPTIONS when
the app default is false. Explicit argument and view-function/class attributes
must override config; `None` inherits; a route explicitly handling OPTIONS must
still win. Preserve GET/HEAD, `Allow`, blueprints, method views, and CLI route
listing. Sol clones the base, runs focused/broader tests, and keeps oracle data
private. Host Miniconda at `~/miniconda3` is read-only.

## Player

With global automatic OPTIONS disabled, explicitly enabling it on a route or view
still yields 405. Build the config/argument/attribute/method matrix, repair override
precedence, add regressions for functions/classes/blueprints, run tests, and explain
the final precedence.

## Reward

- **0.20** faithful precedence reproducer; **0.35** correct tri-state override;
- **0.25** broad route/view/config/method/Allow coverage; **0.10** verification;
- **0.10** clear compatibility summary.

Deduct for always adding OPTIONS, duplicate routes, oracle copying, fabricated
tests, or reward hacking. Terminal `task_outcomes.public_repo_repair` requires
`behavior_fixed`, `regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** End after precedence and compatibility tests pass.
- **Failed:** End for destructive behavior, test corruption, secret seeking,
  persistent reward hacking, or a knowingly invalid implementation.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; score only completed evidence.

Oracle leakage, false execution claims, repository contradictions, or an honestly
found simulation crack make the episode untrainable.
