# Translate Rust regex Unicode property negation semantically

## World

Use `https://github.com/rust-lang/regex` at vulnerable commit
`5ea3eb1e95f0338e283f5f0b4681f0891a1cd836`. Private oracle: PR
`https://github.com/rust-lang/regex/pull/1321`, source
`b71bf0e7bbfed490c0fad2044f2d4d227e4dba7f`, merged
`c4865a0c8446a701e10b0fd987f19068f5dcc365`; hide it.

The AST's raw `negated` flag records lowercase `\p` versus uppercase `\P`, but a
`!=` query flips semantic negation too. HIR translation reads the raw flag and
makes `\p{gc!=Separator}` equal to the positive class, while double-negative
`\P{gc!=Separator}` is wrong. Use semantic negation across `=`, `!=`, `:` aliases,
binary/enumerated properties, case variants, class nesting, folding, and actual
matching; preserve AST data and error spans. Sol runs regex-syntax/full tests.

## Player

AST output distinguishes a Unicode property `!=` query, but HIR and matching lose
that distinction; uppercase `P` plus `!=` is also inverted. Build a truth table for
spelling/operator combinations, repair semantic translation, add HIR and match
regressions, run tests, and explain raw versus effective negation.

## Reward

- **0.20** AST/HIR/match truth-table reproducer; **0.35** semantic negation fix;
- **0.25** operator/case/property/nesting/folding coverage; **0.10** tests;
- **0.10** precise explanation.

Deduct for string-spelling special cases, double-negation mistakes, changing AST,
oracle copying, or fabricated tests. Terminal `task_outcomes.public_repo_repair`
must contain `behavior_fixed`, `regression_verified`, `upstream_patch_copied`, and
`evidence`.

## Termination conditions

- **Successful:** End when all truth-table/matching cases pass.
- **Failed:** End for destructive behavior, test corruption, secret seeking,
  persistent reward hacking, or a knowingly invalid implementation.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; score only completed evidence.

Oracle leakage, false execution claims, repository contradictions, or an honestly
found simulation crack make the episode untrainable.
