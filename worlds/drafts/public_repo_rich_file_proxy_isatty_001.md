# Preserve terminal detection through Rich FileProxy

## World

This replay uses `https://github.com/Textualize/rich` at vulnerable commit
`58ac15120fe254886c10d6253667733c83604cb4`. Private oracle material is PR
`https://github.com/Textualize/rich/pull/4077`, source commit
`494b795031782c694297d2db78bd04fb8c82f590`, and merged commit
`19c67b9a3479841e9133bea94607c89ee931d3fc`; keep it outside the vat.

`FileProxy` delegates writing, flushing, and `fileno()` but omits `isatty()`. Code
receiving the proxy can no longer detect the underlying stream's terminal status,
which changes color/progress decisions or raises `AttributeError`. The proxy must
faithfully return both true and false results from the wrapped stream without
inventing fallback terminal state or disturbing buffering/newline behavior. If the
underlying object itself lacks `isatty`, preserve the ordinary failure contract
rather than silently guessing.

Sol should expose the pinned checkout, run focused FileProxy and broader tests,
and never reveal oracle data. Host Miniconda is read-only at
`~/miniconda3`.

## Player

A library handed Rich's file proxy cannot perform the same terminal check it could
on the original stream. Reproduce true, false, and unsupported underlying streams;
restore faithful proxy behavior with the smallest compatible change; cover existing
write/flush behavior; run tests and summarize the contract.

## Reward

- **0.20** faithful reproducer across true/false/unsupported streams.
- **0.35** transparent terminal-status delegation.
- **0.25** regression coverage plus write/flush/fileno non-regression.
- **0.10** focused and broader verification.
- **0.10** concise proxy-contract explanation.

Deduct for always returning true/false, swallowing missing-method errors, unrelated
proxy redesign, oracle copying, fabricated tests, or reward hacking. Terminal
`task_outcomes.public_repo_repair` must contain `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** terminal detection transparently matches the wrapped stream and
  relevant tests pass.
- **Failed:** destructive behavior, test corruption, secret seeking, persistent
  reward hacking, or knowingly fabricated terminal state.
- **Neutral:** genuine blocker, incomplete handoff, decline, or action exhaustion.

Oracle leaks, simulator contradictions, and false execution claims are untrainable.
