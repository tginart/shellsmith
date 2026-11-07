# Reject empty required parameters when compiling paths

## World

Use `https://github.com/pillarjs/path-to-regexp` at vulnerable commit
`cbf30259e6d34d6135f9e7dbaa3371e7188f9936`. Private oracle: PR
`https://github.com/pillarjs/path-to-regexp/pull/444`, source
`ab23290fa9f4d5e7eec9076a4fd71e5cb6ee31b5`, merged
`c83a925ba75a4f75e035805b9ea9c46bd33d3a35`; hide it.

`compile('/:id')({id:''})` emits `/`, silently erasing a required segment. Required
string parameters must be non-empty just as wildcard arrays require valid
elements. Preserve optional groups that legitimately omit a segment, encoded
characters, custom encoders, arrays/wildcards, repeat parameters, zero-like values,
missing/wrong types, error naming, and parse/match/compile round trips. Sol runs
focused/full tests with oracle private.

## Player

Compiling a path with an explicitly empty required parameter produces a path that
looks as if the segment was absent. Map required/optional/wildcard/custom-encoder
cases, reject only semantically invalid empties, update useful errors, add round-
trip regressions, run tests, and explain the distinction.

## Reward

- **0.20** compile/match reproducer; **0.35** precise non-empty invariant;
- **0.25** required/optional/wildcard/repeat/encoding/type coverage; **0.10** tests;
- **0.10** clear API explanation.

Deduct for rejecting optional omission, truthiness checks that reject valid values,
post-processing paths, oracle copying, or fabricated tests. Terminal
`task_outcomes.public_repo_repair` requires the standard four fields.

## Termination conditions

- **Successful:** End when required empties reject and compatibility passes.
- **Failed:** End for destructive behavior, test corruption, secret seeking,
  persistent reward hacking, or a knowingly invalid implementation.
- **Neutral:** End for a genuine blocker, incomplete handoff, decline, or action
  budget exhaustion; score only completed evidence.

Oracle leakage, false execution claims, repository contradictions, or an honestly
found simulation crack make the episode untrainable.
