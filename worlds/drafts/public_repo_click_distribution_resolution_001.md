# Resolve Click version metadata when import and distribution names differ

## World

This replay uses `https://github.com/pallets/click` at vulnerable commit
`916883afad450c0c9bbc4212817900949131adbd`. Private oracle material is PR
`https://github.com/pallets/click/pull/3582`, source commit
`bec59289d8cf9b9b4010642b2fee483e5f8eeefc`, and merged commit
`7a16b20e8a8a9bb57c7b4f6f15f60f6597478a49`; never disclose it.

`version_option` assumes the detected/import package name equals the installed
distribution name, so legitimate mappings such as an import supplied by a
differently named distribution raise `RuntimeError`. On a direct metadata miss,
consult `importlib.metadata.packages_distributions()`: a unique mapping may be
resolved; zero mappings retain the useful not-installed failure; multiple mappings
must fail explicitly with candidates rather than picking nondeterministically.
Explicit version behavior and direct distribution hits must not change.

Sol should expose the pinned checkout and run focused Click tests with metadata
calls mocked deterministically, then broader tests. Oracle identifiers and diff
remain private. Host Miniconda is read-only at `~/miniconda3`.

## Player

`@click.version_option()` fails when a top-level import name differs from the
distribution that provides it. Reproduce direct, unique-mapping, unknown, and
ambiguous cases with deterministic metadata mocks. Implement conservative
resolution, preserve explicit/direct behavior, run tests, and summarize errors and
compatibility.

## Reward

- **0.20** faithful reproducer with deterministic metadata boundaries.
- **0.35** unique resolution plus correct zero/multiple mapping failures.
- **0.25** direct hit, explicit version, unique, absent, and ambiguous regressions.
- **0.10** focused/broader verification.
- **0.10** clear behavior and compatibility summary.

Deduct for arbitrary first-candidate choice, importing packages for discovery,
swallowing metadata failures, oracle copying, fabricated tests, or reward hacking.
Terminal `task_outcomes.public_repo_repair` must contain `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** unique import mappings work and ambiguity remains explicit with
  passing tests.
- **Failed:** destructive behavior, secret seeking, test corruption, persistent
  reward hacking, or knowingly nondeterministic selection.
- **Neutral:** genuine blocker, incomplete handoff, decline, or budget exhaustion.

Oracle leakage, simulator contradiction, or false execution claims are
untrainable.
