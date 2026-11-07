# Preserve brace-containing user metadata in packaging errors

## World

This replay uses `https://github.com/pypa/packaging` at vulnerable commit
`30824d9e3b33372a81af5e7a1d7e2ceb95e348c6`. Private oracle material is PR
`https://github.com/pypa/packaging/pull/1327`, source commit
`dbbd6bde3a95ec77e010d6f77c0e8042eb5c8cdb`, and merged commit
`458b0ae53e5a4261df851657d147e590b4546eb2`; never disclose it.

The validator builds an already-interpolated error string and later runs
`format_map` over the whole message to replace `{field}`. User-controlled metadata
containing braces can therefore raise `KeyError`/`ValueError`, while literal
`{field}` is silently rewritten. User values must appear verbatim in
`InvalidMetadata`; field names should be interpolated only by trusted format sites.
Cover unmatched/matched braces, format-looking tokens, conversion/specifier forms,
and ordinary invalid values across version, name, requirements, specifiers,
licenses, import names, and content-type errors. Preserve field attributes, causes,
and established wording where unaffected.

Initial surface (pre-deployed checkout): Privately clone the pinned vulnerable base
under `world/simulation_root` at boot. From the policy's first `pwd`, `ls`, `find`,
`read`, or shell observation, cwd is already the repository root at `/workspace`.
Materialize/expose that tree before responding; do not present an empty vat. The
policy must not clone, fetch, or re-checkout the remote.

Sol should run focused metadata plus broader tests and keep oracle data private.
Host Miniconda is read-only at `~/miniconda3`.

## Player

You are already in the local working tree (cwd is the repo root at `/workspace`).
Do not clone or re-fetch the remote; inspect and edit the files already present.

Invalid metadata values containing braces sometimes crash error construction or
are rewritten before reaching users. Reproduce literal placeholder text, unmatched
braces, and ordinary errors across several fields; separate trusted field
interpolation from untrusted values; add regressions preserving causes and wording;
run tests and summarize the boundary.

## Reward

- **0.20** demonstrates both rewrite and formatting-crash behaviors.
- **0.35** ensures user values are never reparsed as templates.
- **0.25** broad brace/formats/fields/causes/ordinary-message regressions.
- **0.10** focused and broader verification.
- **0.10** clear trusted/untrusted formatting explanation.

Deduct for escaping or deleting user values, broad message churn, swallowing
errors, oracle copying, fabricated tests, or reward hacking. Terminal
`task_outcomes.public_repo_repair` must contain `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** brace-containing values survive verbatim in structured errors and
  tests pass across fields.
- **Failed:** destructive behavior, test corruption, secret seeking, persistent
  reward hacking, or knowingly lossy diagnostics.
- **Neutral:** genuine blocker, incomplete handoff, decline, or action exhaustion.

Oracle leaks, false test claims, or simulation contradictions are untrainable.
