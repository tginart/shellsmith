# Honor chi response discard semantics on the ReaderFrom fast path

## World

Use `https://github.com/go-chi/chi` at vulnerable commit
`3b171578ca44dfd75ca3c5cbddc7b44c600a7b49`. Private oracle: PR
`https://github.com/go-chi/chi/pull/1110`, source
`8080605d27b5cc3743db5aaca80d6defd33a65a3`, merged
`2b9fca258f92830fc8232582e9e05a4d4a572a5d`; hide it.

`httpFancyWriter.ReadFrom` bypasses `basicWriter.Write` when no tee exists, so
`Discard()` fails for `io.Copy`/ServeContent-style writes even though normal Write
is suppressed. Route tee or discard through policy-aware writing while preserving
the pure pass-through fast path, accurate byte count, headers/status, errors, and
no double counting. Cover tee/discard combinations and ReaderFrom/no-ReaderFrom
underlyings without a live server. Sol runs focused/broader Go tests.

## Player

After middleware calls `Discard()`, ordinary writes vanish but `io.Copy` output
still reaches the underlying response. Reproduce with a fake ReaderFrom writer,
repair fast-path selection, test tee/discard/count/error combinations and normal
performance path, run tests, and explain the interface dispatch.

## Reward

- **0.20** faithful fast-path reproducer; **0.35** correct discard/tee routing;
- **0.25** matrix plus byte/header/error/no-double-count coverage; **0.10** tests;
- **0.10** clear explanation.

Deduct for removing ReaderFrom acceleration globally, dropping byte accounting,
network tests, oracle copying, or fabricated results. Terminal
`task_outcomes.public_repo_repair` must contain `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** End when all policy paths and accounting pass.
- **Failed:** End for destructive behavior, test corruption, secret seeking,
  persistent reward hacking, or a knowingly invalid implementation.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; score only completed evidence.

Oracle leakage, false execution claims, repository contradictions, or an honestly
found simulation crack make the episode untrainable.
