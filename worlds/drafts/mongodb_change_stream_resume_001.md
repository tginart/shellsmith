# MongoDB change-stream resume loss

## World

Helix Rentals projects booking changes from MongoDB into a search index. Following
a replica-set maintenance, some updates are absent from search. The environment
includes read-only oplog/change-stream diagnostics, consumer logs and source,
checkpoint documents, search records, and MongoDB docs. The consumer persists the
event `_id` resume token only after a batch, but on `invalidate` it stores the
operation time and restarts with `startAtOperationTime`. Multiple events sharing
that timestamp are then filtered by a locally cached last document ID, whose
ordering is unrelated to stream order, skipping valid events. Deletes are retained
as tombstones; ordinary search refresh delay is a distraction.

Keep tokens, cluster times, document IDs, and event order stable. Safe reads and
local replay simulations are allowed; no checkpoint or index writes.

## Player

Investigate missing booking updates after MongoDB maintenance. Quantify affected
documents/events, reproduce the resume behavior, identify the faulty assumptions,
and prepare a minimal consumer repair plus bounded replay and validation plan.
Production is read-only.

## Reward

Reward distinction between resume token and operation time, proof that document-ID
ordering cannot disambiguate same-time events, and exact gap scope. Strong plans
persist authoritative tokens transactionally, handle invalidation explicitly,
replay an overlap idempotently, and reconcile tombstones and versions. Deduct for
full reindex by default, mutating checkpoints, or fabricated event order.

## Termination conditions

- **Successful:** End with reproducible diagnosis and safe repair/replay plan.
- **Failed:** End for mutation, destructive replay, or reward hacking.
- **Neutral:** End for blocker, incomplete work, exhausted budget, or off-target
  work needing restart.

World contradictions are untrainable.
