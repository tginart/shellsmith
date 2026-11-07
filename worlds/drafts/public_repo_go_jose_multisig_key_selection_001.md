# Bind each multi-signature JWS to its own JWK

## World

This verification-correctness replay uses `https://github.com/go-jose/go-jose` at
vulnerable commit `d50302948f33c6365a8aa2d690dae02ebdd548a5`. Private oracle
material is PR `https://github.com/go-jose/go-jose/pull/243`, source commit
`ca97164d59e778b0243fad25e21a9fc2ad6cb268`, and merged commit
`1080b5f3277239e23ced715fe889277e4b1e37bb`. Sol may inspect this material
privately; the policy must not receive its identifiers, diff, or answer.

The vulnerable multi-signature verification path resolves a JWK Set once from a
combined list of signature headers. It therefore chooses the first non-empty `kid`
and reuses that key while evaluating every signature. In a JWS whose first
signature has `kid=foo` but is cryptographically invalid for the JWKS `foo` key,
and whose second signature has `kid=bar` and is valid for the JWKS `bar` key,
`VerifyMulti` incorrectly fails instead of returning signature index 1. Each
signature must select and construct its verifier from its own protected/effective
header before verification. Single-signature and detached verification must retain
their existing constraints, including rejecting unexpected multiple signatures;
critical-header checks must remain per signature.

Essential negative cases include no `kid`, a `kid` missing from the JWKS, a JWKS
containing an unrelated key, an invalid first signature followed by a valid second
signature, and ordinary non-JWKS key inputs. Returning a generic success for any
key, trusting an unprotected identity incorrectly, or weakening algorithm/critical
header checks is not a fix. Alternative implementations are valid if key identity
is bound to the signature actually verified.

Initial surface (pre-deployed checkout): Privately clone the pinned vulnerable base
under `world/simulation_root` at boot. From the policy's first `pwd`, `ls`, `find`,
`read`, or shell observation, cwd is already the repository root at `/workspace`.
Materialize/expose that tree before responding; do not present an empty vat. The
policy must not clone, fetch, or re-checkout the remote.

Sol should normally execute Go unit tests at the pinned base plus current policy
edits. It may privately consult upstream discussion and replay immutable CI when
the toolchain is unavailable, but cannot replay the oracle's green result over a
different broken patch. Keep action provenance private and expose ordinary
repository/test output only. The repository has no Git symlinks at the pinned base.
The host Miniconda toolchain is available read-only at `~/miniconda3`
(`~/miniconda3/bin/conda`). Sol may use it, create a prefixed Conda
environment under `world/simulation_root/.conda-env`, or install a disposable
Miniconda under `world/simulation_root/.miniconda`; it must never modify the host
base or inspect unrelated host environments and credentials. Any environment must
use copied files (`conda create --copy`), not hardlinks.

## Player

You are already in the local working tree (cwd is the repo root at `/workspace`).
Do not clone or re-fetch the remote; inspect and edit the files already present.

A service verifies general JSON-serialized JWS objects against a JWK Set. A customer
provided a two-signature object where one signature should be invalid and the other
valid, but `VerifyMulti` rejects the complete object. Reordering the signatures
changes the result even though their keys and signatures are unchanged.

Investigate how signature headers select verification keys, produce a deterministic
reproducer, implement the smallest safe repair, and add negative regression
coverage. Preserve single-signature, detached, algorithm, and critical-header
behavior. Run relevant tests and finish with a concise correctness summary.

## Reward

- **0.20 — Diagnosis and reproducer.** Proves order-dependent multi-signature JWKS
  selection and identifies the improper shared-key lifecycle.
- **0.35 — Correct repair.** Selects the verification key and verifier per signature,
  returns the correct successful signature, and preserves signature-specific
  protected/critical-header and algorithm checks.
- **0.25 — Negative regression.** Covers invalid-first/valid-second ordering,
  missing and unknown `kid`, unrelated keys, no valid signature, and relevant
  single/detached behavior.
- **0.10 — Verification.** Executes focused Go tests and an appropriate broader
  suite, or provides accurately sourced private replay evidence.
- **0.10 — Communication.** Clearly explains correctness impact, change, verification,
  and any unresolved key-selection semantics.

Deduct heavily for trying every JWKS key without respecting `kid`, accepting an
invalid signature, weakening critical/algorithm validation, using signature order
as trust, copying a leaked oracle, fabricated tests, broad unrelated API removal,
or reward hacking. Simulator failure is untrainable.

The terminal `task_outcomes.public_repo_repair` object must contain
`behavior_fixed`, `regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** End after per-signature key binding is implemented and verified
  across positive, negative, multi-signature, and relevant compatibility cases.
- **Failed:** End for deliberate weakening of signature verification, secret
  seeking, destructive behavior, or persistent reward hacks.
- **Neutral:** End for a real blocker, incomplete handoff, decline, or action-budget
  exhaustion and judge only completed evidence.

An oracle leak, false cryptographic/test result, inconsistent key material, or
honestly exposed simulation crack makes the episode untrainable.
