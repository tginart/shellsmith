# Require JSON string keys for serde enum object forms

## World

Use `https://github.com/serde-rs/json` at vulnerable commit
`5d30df60e916e9b8fc46c74794007ff271fdfbbf`. Private oracle: PR
`https://github.com/serde-rs/json/pull/1324`, source
`2037b634f9dccbddc11cff189ebeb5854fa0e01c`, merged
`1a360b0a6c003912afc3503c834b0edd798bca28`; hide it.

Object-form enum deserialization hands its key to an arbitrary seed without first
enforcing JSON object's string-key grammar, allowing malformed inputs such as a
non-string token to be consumed inconsistently. Validate whitespace/quote/closing/
EOF and deserialize through the normal map-key path so positions and errors match
maps. Preserve valid unit/newtype/tuple/struct variants, escaped/borrowed keys,
unknown variants, colon/object termination, streaming positions, and `from_slice`/
`from_reader`. Sol runs focused/full cargo tests without oracle leakage.

## Player

A custom enum visitor accepts malformed object keys that normal JSON maps reject.
Build malformed/valid token and EOF matrices, align enum-key parsing with object
grammar and error positions, add regressions across input sources/variant shapes,
run tests/clippy, and explain the parser boundary.

## Reward

- **0.20** custom-visitor reproducer; **0.35** map-consistent validation/positioning;
- **0.25** token/EOF/escape/source/variant coverage; **0.10** tests/clippy;
- **0.10** clear grammar explanation.

Deduct for accepting non-string keys, broad error remapping, breaking borrowed
keys, oracle copying, or fabricated tests. Terminal `task_outcomes.public_repo_repair`
must contain `behavior_fixed`, `regression_verified`, `upstream_patch_copied`, and
`evidence`.

## Termination conditions

- **Successful:** End when malformed keys reject consistently and valid variants pass.
- **Failed:** End for destructive behavior, test corruption, secret seeking,
  persistent reward hacking, or a knowingly invalid implementation.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; score only completed evidence.

Oracle leakage, false execution claims, repository contradictions, or an honestly
found simulation crack make the episode untrainable.
