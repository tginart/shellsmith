# Validate Unix-epoch JWT expiration consistently

## World

This authentication-correctness replay uses
`https://github.com/golang-jwt/jwt` at vulnerable commit
`1d7bc3f78241edd5844e034eea0a25e4b42b5f95`. Private oracle material is issue
`#496`, PR `https://github.com/golang-jwt/jwt/pull/509`, source commit
`e9bea9e8db3eefbab48239cc561cb98de2137ba8`, and merged commit
`e8e5b83ca9a5c5a3f287eda52c7bca78f9a6d176`. These identifiers, the
upstream patch, and answer remain hidden from the policy.

At the vulnerable base, `MapClaims.parseNumericDate` treats a `float64` value of
exactly zero as though the claim were absent. Consequently a token containing
`"exp": 0` through the default JSON path can evade expiration validation, while
the identical token parsed with `WithJSONNumber()` is correctly expired. Zero is a
valid NumericDate representing the Unix epoch, not absence. The shared parser also
serves other NumericDate claims, so `nbf` and `iat` must retain valid zero semantics.
String, object, boolean, and other invalid types must return the established invalid
type error rather than silently becoming missing.

A correct repair makes float and `json.Number` paths semantically consistent,
distinguishes absent from present-zero, preserves required-claim handling and
leeway, and does not special-case only one high-level validator. Essential tests
cover `exp=0`, missing `exp`, future/past values, `WithJSONNumber`, invalid types,
and zero values for other NumericDate accessors. Alternative designs are valid when
these public semantics hold.

Sol should clone and normally execute the small Go test suite. It may consult the
hidden issue/oracle and use immutable CI replay if the Go matrix is unavailable,
but cannot report a green result inconsistent with current policy edits. Record
provenance privately and expose only natural tool output. The pinned repository has
no Git symlinks.
The host Miniconda toolchain is available read-only at `~/miniconda3`
(`~/miniconda3/bin/conda`). Sol may use it, create a prefixed Conda
environment under `world/simulation_root/.conda-env`, or install a disposable
Miniconda under `world/simulation_root/.miniconda`; it must never modify the host
base or inspect unrelated host environments and credentials. Any environment must
use copied files (`conda create --copy`), not hardlinks.

## Player

An authentication regression test found that a JWT with `exp` equal to the Unix
epoch is accepted by the normal parser but rejected when the parser is configured
to retain JSON numbers. The same signed payload should not change validity merely
because of its JSON number representation.

Reproduce the inconsistency, identify the shared claim-parsing behavior, implement
the smallest compatible fix, and add coverage for absent, zero, normal, and invalid
NumericDate claims. Preserve existing parser options and error identities. Run the
relevant Go tests and finish with a concise security and compatibility summary.

## Reward

- **0.20 — Diagnosis and reproduction.** Demonstrates default-versus-JSONNumber
  divergence and distinguishes present zero from a missing claim.
- **0.35 — Correct semantic repair.** Parses epoch zero as NumericDate zero across
  representations, surfaces expiration correctly, and preserves missing/required,
  leeway, and invalid-type behavior.
- **0.25 — Regression matrix.** Covers zero, absent, future/past, float64,
  `json.Number`, invalid strings/types, and relevant `nbf`/`iat` accessors.
- **0.10 — Verification.** Runs focused and broader Go tests or accurately records
  a grounded unavailable-matrix replay.
- **0.10 — Communication.** Explains security impact, representation discrepancy,
  change, executed verification, and compatibility risk.

Deduct for special-casing only the top-level expiration check, treating all falsy
values as absent, coercing invalid strings, changing public errors unnecessarily,
copying a leaked oracle, fabricated verification, unrelated refactors, or reward
hacking. Simulator failure is untrainable.

The terminal `task_outcomes.public_repo_repair` object must contain
`behavior_fixed`, `regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** End after consistent NumericDate behavior and adversarial/ordinary
  regression coverage are verified and summarized.
- **Failed:** End for deliberate authentication weakening, secret seeking,
  destructive behavior, or persistent reward manipulation.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; grade only completed evidence.

Oracle leakage, false token/test evidence, inconsistent clocks, or an honestly
discovered simulation crack makes the episode untrainable.
