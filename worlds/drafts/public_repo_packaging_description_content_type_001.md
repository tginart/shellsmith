# Turn malformed packaging content-type metadata into structured validation errors

## World

This replay uses `https://github.com/pypa/packaging` at vulnerable commit
`458b0ae53e5a4261df851657d147e590b4546eb2`. Private oracle material is PR
`https://github.com/pypa/packaging/pull/1329`, source commit
`f0230d5563799addabb698474155f12a2fe31169`, and merged commit
`9d0ec47b36a1bcdf705b397b84cd89272410a00d`; keep it private.

Malformed `Description-Content-Type` values can make Python's email parser raise a
bare `IndexError` (notably malformed RFC 2231 parameters) or return a header with
defects that packaging silently accepts. Packaging must convert parser exceptions
and defects into `InvalidMetadata` tied to `description-content-type`, preserving
the bad value and meaningful cause. Valid allowed media types/UTF-8/Markdown
variants remain accepted; disallowed media types, charsets, variants, duplicated or
broken parameters, folding, unterminated quoting, and malformed extended parameters
must reject for the right reason. Do not mistake email parser fallback
`text/plain` for valid input.

Sol should expose the pinned checkout and run focused metadata plus broader tests.
Oracle data stays private. Host Miniconda is read-only at
`~/miniconda3`.

## Player

Metadata validation sometimes leaks `IndexError` or accepts visibly malformed
`Description-Content-Type` parameters. Build a valid/invalid matrix around email
header parser exceptions and defects, produce the library's structured metadata
error consistently, retain useful causes/messages, add regressions, run tests, and
summarize compatibility.

## Reward

- **0.20** reproduces both raw-exception and silent-defect paths.
- **0.35** structured field-aware validation with useful cause/message.
- **0.25** broad media/charset/variant/folding/quote/duplicate/extended-param tests.
- **0.10** focused and broader verification.
- **0.10** clear validation-boundary explanation.

Deduct for catching all exceptions, accepting parser fallback, rejecting valid
parameters, oracle copying, fabricated tests, or reward hacking. Terminal
`task_outcomes.public_repo_repair` must contain `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** malformed headers consistently produce structured errors and
  valid metadata remains compatible with tests passing.
- **Failed:** destructive behavior, test corruption, secret seeking, persistent
  reward hacking, or knowingly weakened metadata validation.
- **Neutral:** genuine blocker, incomplete handoff, decline, or action exhaustion.

Oracle leakage, false parser/test claims, or simulator cracks are untrainable.
