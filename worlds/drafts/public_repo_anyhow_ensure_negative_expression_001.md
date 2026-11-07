# Parse non-literal negative expressions in anyhow ensure macros

## World

Use `https://github.com/dtolnay/anyhow` at vulnerable commit
`2737bbeb59f50651ff54ca3d879a3f5d659a98ab`. Private oracle: PR
`https://github.com/dtolnay/anyhow/pull/373`, source
`e0c74f26517944f4758da61079a16d88600d37c4`, merged
`0a1b405631348f38b780255ac78c86a87536feb6`; hide it.

`ensure!`'s token parser has a special negative-literal path whose macro matching
misclassifies a minus followed by a non-literal expression in comparison/generic-
looking token sequences, producing a compile error instead of falling back to Rust
expression parsing. Preserve negative numeric literals, unary expressions,
qualified paths, turbofish/comparisons, custom messages, precedence, and stable
compile-fail diagnostics. Use runtime and trybuild-style compile cases; do not
accept invalid Rust by overbroad token munching. Sol runs cargo tests/checks.

## Player

An otherwise valid `ensure!` condition containing unary minus before a computed
expression fails during macro parsing, while the literal form works. Minimize
compile cases, repair fallback boundaries without breaking generics/comparisons or
diagnostics, add pass/fail tests, run suites, and explain token ambiguity.

## Reward

- **0.20** minimal compile reproducer; **0.35** correct macro fallback;
- **0.25** literal/nonliteral/path/generic/message/pass-fail coverage; **0.10** tests;
- **0.10** clear token explanation.

Deduct for weakening syntax checks, changing macro evaluation count, snapshot-only
patches, oracle copying, or fabricated compile tests. Terminal
`task_outcomes.public_repo_repair` must contain `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** End when valid cases compile/evaluate and invalid cases remain rejected.
- **Failed:** End for destructive behavior, test corruption, secret seeking,
  persistent reward hacking, or a knowingly invalid implementation.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; score only completed evidence.

Oracle leakage, false execution claims, repository contradictions, or an honestly
found simulation crack make the episode untrainable.
