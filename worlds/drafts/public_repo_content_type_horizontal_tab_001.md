# Distinguish HTTP horizontal tab from vertical tab in content-type values

## World

Use `https://github.com/jshttp/content-type` at vulnerable commit
`427eb1bad800ed74b4ae7f5700ab0105b94b2076`. Private oracle: PR
`https://github.com/jshttp/content-type/pull/64`, source
`e285de60ce6a4fbb6077b9fb378f86fe486be80d`, merged
`3817c1a331bc95aa00dcddbe824b24b1dbc6f9d4`; hide it.

The text-character regexp contains vertical tab U+000B where HTTP grammar permits
horizontal tab U+0009. Formatting therefore rejects/handles legal HTAB wrongly and
admits a control character that must reject. Preserve token versus quoted-string
selection, escaping, obs-text policy, parse/format round trip, other C0/DEL
rejection, Unicode boundaries, parameter ordering, and media type validation. Sol
runs focused/full TypeScript tests and keeps oracle private.

## Player

Formatting a parameter containing HTTP horizontal whitespace fails or behaves
incorrectly, while a different control whitespace slips through. Build a codepoint
boundary matrix, align accepted text with HTTP grammar, verify quoting and parse/
format round trips, add regressions, run tests, and explain the character distinction.

## Reward

- **0.20** codepoint reproducer; **0.35** exact grammar repair;
- **0.25** HTAB/VT/C0/DEL/obs-text/quote/round-trip coverage; **0.10** tests;
- **0.10** clear protocol explanation.

Deduct for allowing all whitespace/control chars, stripping values, changing token
grammar, oracle copying, or fabricated tests. Terminal
`task_outcomes.public_repo_repair` requires `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** End when grammar and round trips pass.
- **Failed:** End for destructive behavior, test corruption, secret seeking,
  persistent reward hacking, or a knowingly invalid implementation.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; score only completed evidence.

Oracle leakage, false execution claims, repository contradictions, or an honestly
found simulation crack make the episode untrainable.
