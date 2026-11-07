# Restore defaults for empty variadic Typer arguments

## World

This replay uses `https://github.com/fastapi/typer` at vulnerable commit
`7f888b24c867458ab9547702720b71dc920c8925`. Private oracle material is PR
`https://github.com/fastapi/typer/pull/1821`, source commit
`5bec961514de7c5da842c6e3170727022388414b`, and merged commit
`06fce7dc851de662792c840320abe990ddd3566e`; keep it private.

A variadic list argument with no CLI values arrives as `()`. The parser converts
that sentinel to `None` only if an envvar exists, so ordinary parameter/default-map
resolution is skipped and a declared list default is lost. Empty variadic input
must re-enter the normal precedence chain: explicit CLI values override everything;
an environment value is split when configured; a default-map value precedes the
parameter default; otherwise the declared default applies. List options already
work and must not regress. Preserve explicitly supplied empty-like values and
non-variadic parameters.

Sol should expose the pinned checkout, run focused parser/type tests and broader
checks, and not disclose oracle material. Host Miniconda is read-only at
`~/miniconda3`.

## Player

A command declaring a default list for a variadic positional argument receives no
default when invoked without values; an envvar-enabled variant behaves differently.
Map CLI, envvar, default-map, and declared-default precedence for list arguments and
options, repair the parser generally, add regressions, run tests, and explain why
the sentinel matters.

## Reward

- **0.20** accurate precedence reproducer.
- **0.35** restores ordinary resolution without disturbing explicit values.
- **0.25** covers argument/option, CLI/env/default-map/default, empty and multi-value.
- **0.10** focused plus broader verification.
- **0.10** clear parser-contract explanation.

Deduct for special-casing one command, bypassing env parsing, changing option
precedence, oracle copying, fabricated tests, or reward manipulation. Terminal
`task_outcomes.public_repo_repair` must contain `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** the precedence matrix is correct and relevant tests pass.
- **Failed:** destructive behavior, test corruption, secret seeking, persistent
  reward hacking, or knowingly incompatible parser changes.
- **Neutral:** genuine blocker, incomplete handoff, decline, or action exhaustion.

Oracle leakage, false test claims, or repository simulation cracks are
untrainable.
