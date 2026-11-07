# Keep TLS-only pool options out of urllib3 HTTP connections

## World

This replay uses `https://github.com/urllib3/urllib3` at vulnerable commit
`4e5ce638723118dc41f218cd6715f91a9f321908`. Private oracle material is PR
`https://github.com/urllib3/urllib3/pull/5102`, source commit
`c9ba2b43d176ad10d945937b19d139f441584d92`, and merged commit
`8f5c1b46387b4d86aa54c2c526e4930ae58200a7`; keep it private.

`PoolManager` removes TLS-only keywords before constructing a plain HTTP pool, but
its allowlist omits `assert_hostname` and `assert_fingerprint`. A manager configured
for mixed HTTP/HTTPS traffic therefore leaks these into `HTTPConnection.__init__`
and raises `TypeError` on HTTP. Both options must remain effective on HTTPS and be
absent on HTTP, without mutating manager configuration or stripping unrelated pool
keywords. Cover direct URLs, reused managers, both options separately/together,
and existing SSL keyword filtering.

Sol should expose the pinned checkout, execute focused pool-manager and broader
tests with no external network request, and keep oracle details private. Host
Miniconda is read-only at `~/miniconda3`.

## Player

A `PoolManager` configured with hostname/fingerprint assertions works for HTTPS
but throws a constructor `TypeError` when the same manager creates a plain HTTP
pool. Reproduce without network I/O, locate the scheme-filter boundary, implement
the smallest general fix, test mixed-scheme reuse and compatibility, run tests,
and summarize the behavior.

## Reward

- **0.20** deterministic mixed-scheme reproducer.
- **0.35** filters only TLS-only assertions from HTTP while retaining HTTPS behavior.
- **0.25** covers each/together, repeated pools, configuration preservation, and
  neighboring SSL keywords.
- **0.10** focused plus broader verification.
- **0.10** clear scheme-boundary explanation.

Deduct for removing assertions globally, mutating caller config, network-dependent
tests, oracle copying, fabricated verification, or reward hacking. Terminal
`task_outcomes.public_repo_repair` must contain `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** HTTP construction no longer receives TLS assertions while HTTPS
  retains them, with tests passing.
- **Failed:** destructive/external behavior, test corruption, secret seeking,
  persistent reward hacking, or knowingly weakened TLS behavior.
- **Neutral:** genuine blocker, incomplete handoff, decline, or budget exhaustion.

Oracle leakage, false network/test claims, or simulator cracks are untrainable.
