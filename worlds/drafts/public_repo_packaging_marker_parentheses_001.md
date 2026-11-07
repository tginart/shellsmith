# Preserve marker meaning through packaging serialization

## World

This semantic-correctness replay uses `https://github.com/pypa/packaging` at
vulnerable commit `c4fb81ff6eba16fdd24d50e9d9d002fdbece6dd4`. Private oracle
material is issue `#1315`, PR `https://github.com/pypa/packaging/pull/1316`, source commit
`a9fc9fbc757d4bbe1ad6aa6253d55eb2b3b49d51`, and merged commit
`0a85b41e24c9b55b83f05ae696d73e8294a5b094`. Sol may inspect these privately;
the policy must not receive the oracle identifiers, diff, or exact solution.

The vulnerable marker formatter unwraps a redundant single-item nested structure
without preserving whether it is formatting the outermost expression. For a marker
equivalent to `A and (B or C)`, `str(Marker(...))` can drop the required inner
parentheses and emit `A and B or C`. Reparsing then follows different precedence
and can evaluate to the opposite result. One concrete case combines a Python
version constraint with a Linux-or-Darwin group; Python 3.12 on Darwin exposes the
flip.

The invariant is semantic round-trip preservation: parsing, serializing, and
reparsing must evaluate identically across relevant environments. The repair should
retain necessary nested parentheses without adding gratuitous outer parentheses or
breaking canonical strings, equality/hash behavior, chained associativity, marker
normalization, or other `and`/`or` shapes. Testing a single string literal is not
enough; cover both nesting directions, several truth assignments, and ordinary
markers. A structurally different formatter change earns full credit if it preserves
the same behavior.

Sol should clone the pinned base and normally execute focused Python tests. It may
privately use the issue, oracle, and upstream checks to ground unavailable
verification, but must evaluate the policy's current code rather than replaying a
green oracle result. Keep provenance private and output only the requested
repository/test surface. The pinned repository contains no Git symlinks.
The host Miniconda toolchain is available read-only at `~/miniconda3`
(`~/miniconda3/bin/conda`). Sol may use it, create a prefixed Conda
environment under `world/simulation_root/.conda-env`, or install a disposable
Miniconda under `world/simulation_root/.miniconda`; it must never modify the host
base or inspect unrelated host environments and credentials. Any environment must
use copied files (`conda create --copy`), not hardlinks.

## Player

A dependency resolver stores normalized environment markers by converting
`packaging.markers.Marker` objects to strings and reparsing them later. We found a
marker whose truth value changes after that round trip, causing a dependency to be
selected in an environment where the original marker was false.

Build a minimal reproducer, identify where serialization changes boolean grouping,
implement the smallest general repair, and add regression coverage across multiple
environments and nested `and`/`or` forms. Preserve existing canonical output where
parentheses are not semantically required. Run relevant tests and summarize the
cause, change, verification, and compatibility implications.

## Reward

- **0.20 — Semantic reproducer.** Demonstrates a real evaluation flip after
  parse/stringify/reparse and explains the precedence change.
- **0.35 — General formatter repair.** Preserves required grouping through nested
  serialization without gratuitous output churn or regressions to ordinary marker
  composition.
- **0.25 — Regression breadth.** Tests multiple truth assignments, nesting
  directions, repeated grouping, round-trip evaluation, and representative normal
  marker strings rather than one golden textual output.
- **0.10 — Verification.** Runs focused marker tests and a suitable broader suite or
  accurately records grounded unavailable checks.
- **0.10 — Communication.** Clearly explains affected semantics, implementation,
  validation, and downstream compatibility risk.

Deduct for hard-coding the reported marker, changing operator precedence globally,
wrapping every expression indiscriminately, asserting only exact text without
semantic evaluation, copying a leaked oracle, fabricated tests, broad unrelated
formatting churn, or reward manipulation. Simulator failure is untrainable.

The terminal `task_outcomes.public_repo_repair` object must contain
`behavior_fixed`, `regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** End after semantic round-trip preservation is implemented and
  verified across adversarial and ordinary marker cases with a final summary.
- **Failed:** End for destructive non-local behavior, secret seeking, deliberate
  test corruption, persistent reward hacking, or knowingly incorrect dependency
  semantics.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; score only completed evidence.

An oracle leak, inconsistent environment evaluation, false execution claim, or
honestly discovered simulation crack makes the episode untrainable.
