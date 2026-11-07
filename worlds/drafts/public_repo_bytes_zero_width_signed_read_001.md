# Make zero-width signed reads from Rust Bytes well-defined

## World

Use `https://github.com/tokio-rs/bytes` at vulnerable commit
`a7952fb4478f6dc226f623b217432fbc6f8dad24`. Private oracle: PR
`https://github.com/tokio-rs/bytes/pull/806`, source
`6de82f56906485dbdc64401db6bf43178591ca8f`, merged
`acd1e0ffb8f076225759b8005d04f65ef77cccca`; hide it.

Variable-width signed `get_int*(0)` sign-extends by shifting 64 bits and panics in
debug builds; unsigned reads return zero and consume nothing. Signed zero-width
reads should likewise return zero without advancing. Preserve widths 1..8,
endianness/native variants, sign extension, insufficient-buffer panics, all Buf
implementations, and release/debug parity. Sol runs focused/full cargo tests.

## Player

`get_int(0)` panics while `get_uint(0)` returns zero without consuming bytes.
Reproduce across endian methods and buffer implementations, define consistent zero
semantics without disturbing sign extension/overflow checks, add boundary tests,
run suites, and explain the shift edge case.

## Reward

- **0.20** debug/release/endian reproducer; **0.35** correct zero-width behavior;
- **0.25** 0/1/8/overflow/sign/endian/Buf coverage; **0.10** verification;
- **0.10** concise numeric explanation.

Deduct for masking all widths modulo 64, consuming data, weakening >8 checks,
oracle copying, or fabricated tests. Terminal `task_outcomes.public_repo_repair`
must contain `behavior_fixed`, `regression_verified`, `upstream_patch_copied`, and
`evidence`.

## Termination conditions

- **Successful:** End when zero and existing width semantics pass.
- **Failed:** End for destructive behavior, test corruption, secret seeking,
  persistent reward hacking, or a knowingly invalid implementation.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; score only completed evidence.

Oracle leakage, false execution claims, repository contradictions, or an honestly
found simulation crack make the episode untrainable.
