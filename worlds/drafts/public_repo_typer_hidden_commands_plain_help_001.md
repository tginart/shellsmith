# Keep hidden Typer commands out of plain help

## World

This replay uses `https://github.com/fastapi/typer` at vulnerable commit
`c2d97d133a02bd358886d0006c6028c6789879f2`. Private oracle material is PR
`https://github.com/fastapi/typer/pull/1812`, source commit
`f992a163de36877381a3530612b0eb1eb0f19271`, and merged commit
`917248604affe32caf899cc8faddfa23f81cb1ef`; do not leak it.

After Click code was vendored, Typer's non-Rich `format_commands` stopped filtering
commands whose `hidden` flag is true. Rich output hides them, but disabling or
uninstalling Rich exposes hidden decorated commands, programmatically registered
commands, and hidden subgroups. Plain and Rich help must agree. Preserve visible
ordering/help, hidden options, unavailable-command handling, invocation by explicit
name where the framework permits it, and nested group behavior.

Sol should clone the pinned base and execute focused hidden-help tests in both
formatting modes plus broader tests. It may inspect the oracle only privately. Host
Miniconda is read-only at `~/miniconda3`.

## Player

Commands marked hidden appear in `--help` whenever Rich formatting is disabled.
Reproduce decorated, registered, and subgroup cases with both help renderers;
repair the shared semantics without hiding visible commands or changing invocation;
add tests, run relevant suites, and summarize the regression boundary.

## Reward

- **0.15** reproduces the Rich/plain divergence across hidden command forms.
- **0.40** filters hidden/unavailable commands correctly in plain help.
- **0.25** covers decorated, registered, subgroup, visible, nested, and hidden-option cases.
- **0.10** focused and broader verification.
- **0.10** concise compatibility explanation.

Deduct for disabling plain help, filtering by names/docs, changing invocation,
copying the oracle, fabricated tests, or reward hacking. Terminal
`task_outcomes.public_repo_repair` must contain `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** hidden state is respected by both renderers and tests pass.
- **Failed:** destructive behavior, test corruption, secret seeking, persistent
  reward hacking, or knowingly broad command removal.
- **Neutral:** genuine blocker, incomplete handoff, decline, or action exhaustion.

Simulator contradiction, oracle leakage, or false execution claims are untrainable.
