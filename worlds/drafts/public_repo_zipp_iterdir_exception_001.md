# Restore pathlib-compatible iterdir errors in zipp

## World

This is a compact compatibility-repair replay for
`https://github.com/jaraco/zipp`. The policy receives vulnerable commit
`d14b72023b442e0eab42d59a6d849e31c2ab243e`. Private evidence includes issue
`#129`, PR `https://github.com/jaraco/zipp/pull/154`, source commit
`a43226aba25ed257013d35cd412fcaac07ce8c38`, and merged commit
`bf9bf8cb1cfb2935d519349b8e7809462eedbccb`; none may be revealed to the policy.

`zipp.Path` is intended to align closely with `zipfile.Path`/`pathlib` path
semantics. On the vulnerable base, calling `iterdir()` on a path that exists but is
not a directory raises a generic `ValueError("Can't listdir a file")`. Consumers
that correctly distinguish a non-directory from malformed data expect the
directory-specific built-in `NotADirectoryError`. The intended repair changes the
observable exception class while preserving ordinary directory iteration, missing
path behavior, archive traversal, message usefulness, and the public method shape.
The upstream change is tiny, but an independently reasoned compatible fix deserves
full credit.

Sol may clone the pinned base, inspect the hidden issue/oracle, and run the focused
Python tests. It may use upstream replay or faithful synthesis if the exact Python
matrix is unavailable, but results must correspond to the current policy edits.
The small repository should normally be executed rather than narrated. Keep
provenance private, expose only natural shell/file/test results, and never reveal
the oracle. The pinned repository contains no Git symlinks.
The host Miniconda toolchain is available read-only at `~/miniconda3`
(`~/miniconda3/bin/conda`). Sol may use it, create a prefixed Conda
environment under `world/simulation_root/.conda-env`, or install a disposable
Miniconda under `world/simulation_root/.miniconda`; it must never modify the host
base or inspect unrelated host environments and credentials. Any environment must
use copied files (`conda create --copy`), not hardlinks.

## Player

A downstream archive browser treats `zipp.Path` as a pathlib-compatible object.
Its error handling recently broke because iterating a member that is a file raises
the wrong category of exception, while equivalent filesystem path code is handled
correctly.

Reproduce the discrepancy, determine the appropriate public behavior, make the
smallest compatible repair, and add a focused regression without disturbing normal
archive iteration. Run the relevant tests and provide a concise maintainer summary.

## Reward

- **0.20 — Reproducer and contract.** Shows the file-versus-directory case and
  justifies the expected exception using the surrounding path API rather than only
  matching a report string.
- **0.40 — Minimal correct behavior.** Raises `NotADirectoryError` for `iterdir()`
  on a non-directory while preserving successful directory iteration and public API
  compatibility.
- **0.20 — Regression coverage.** Tests the exception class and retains ordinary
  iteration coverage without overfitting an exact message.
- **0.10 — Verification.** Runs the targeted test and an appropriate broader suite,
  or clearly distinguishes any unavailable verification.
- **0.10 — Communication.** Explains the mismatch, change, and validation concisely.

Deduct for broad refactors, changing unrelated path semantics, catching/replacing
all exceptions indiscriminately, brittle message-only tests, fabricated execution,
oracle copying, or reward manipulation. A simulator error is untrainable.

The terminal `task_outcomes.public_repo_repair` object must contain
`behavior_fixed`, `regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** End after the compatibility behavior is repaired, focused and
  broader tests are addressed, and a final summary is provided.
- **Failed:** End for destructive non-local behavior, secret seeking, persistent
  reward hacking, or deliberate corruption of the test surface.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or budget
  exhaustion and score only completed work.

Oracle leakage, false test claims, checkout inconsistency, or an honest simulation
crack makes the episode untrainable rather than a policy failure.
