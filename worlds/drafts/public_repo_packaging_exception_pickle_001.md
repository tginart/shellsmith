# Make packaging's public structured exceptions survive process boundaries

## World

This replay uses `https://github.com/pypa/packaging` at vulnerable commit
`30824d9e3b33372a81af5e7a1d7e2ceb95e348c6`. Private oracle material is PR
`https://github.com/pypa/packaging/pull/1328`, source commit
`320b6b2a8eb657f967a54afa73a76532d9b75151`, and merged commit
`1a9c4cb6b6aac7fb6137fcdc4f440af912d8d87f`; keep it outside the vat.

`InvalidMetadata(field, message)` and
`CyclicDependencyGroup(requested_group, group, include_group)` call the base
exception with only their rendered message, so `BaseException.args` does not match
their constructor signatures. Default pickle reconstruction fails when these
cross multiprocessing/concurrent-futures boundaries. Round trips must preserve
class, `str()`, all public attributes, cause behavior where applicable, and work
across supported pickle protocols. Do not redesign constructors or alter rendered
messages. Internal exceptions that cannot escape public boundaries need not be
changed merely for symmetry.

Sol should provide the pinned checkout and execute direct pickle, process-boundary,
and broader tests while keeping oracle details private. Host Miniconda is read-only
at `~/miniconda3`.

## Player

Metadata validation and dependency-group errors fail to deserialize when returned
from worker processes. Inventory public exceptions whose constructor arguments do
not match their stored exception arguments, implement minimal stable reconstruction,
test direct and process-boundary round trips and attributes/messages, run suites,
and explain which exceptions are intentionally in scope.

## Reward

- **0.20** faithful pickle/process reproducer and bounded exception inventory.
- **0.35** stable reconstruction preserving types, attributes, and messages.
- **0.25** both exception classes, protocols/process path, causes, and ordinary errors.
- **0.10** focused and broader verification.
- **0.10** clear serialization-contract explanation.

Deduct for changing constructors/messages, generic monkeypatching of exceptions,
overbroad churn, oracle copying, fabricated process tests, or reward hacking.
Terminal `task_outcomes.public_repo_repair` must contain `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** scoped public exceptions round-trip intact locally and across a
  worker boundary with tests passing.
- **Failed:** destructive behavior, test corruption, secret seeking, persistent
  reward hacking, or knowingly unstable serialized state.
- **Neutral:** platform process blocker, incomplete handoff, decline, or exhaustion.

False process execution, oracle leakage, or simulator inconsistency is untrainable.
