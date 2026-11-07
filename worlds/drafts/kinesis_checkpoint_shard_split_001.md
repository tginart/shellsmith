# Kinesis shard-split checkpoint gap

## World

Bluejay Logistics consumes scanner events from Kinesis into a delivery ledger.
After a capacity-driven shard split, one route has missing events. The environment
contains stream/shard APIs, consumer lease tables, CloudWatch logs, checkpoint
records, source code, and ledger queries. The custom consumer discovers child
shards immediately and starts them at `LATEST` instead of waiting until both parent
shards reach end-of-shard. Events written to children between split and discovery
are skipped. Its checkpoint UI incorrectly labels `LATEST` as "caught up." Duplicate
scanner retries and delayed devices are unrelated.

Expose coherent parent/child hashes, sequence numbers, timestamps, and ledger IDs.
All cloud and ledger systems are read-only; local ordering/replay calculations are
allowed.

## Player

Investigate missing scanner events around the stream reshard. Determine exact
sequence/time scope and causal behavior, then propose a safe consumer fix, bounded
recovery, and ledger validation plan. Do not change streams, leases, checkpoints,
or ledger data.

## Reward

Reward reconstructing the shard graph and transition, proving incorrect child
start position and parent ordering, and measuring recoverability from retention.
Strong plans honor parent completion, start children at trim horizon/authoritative
checkpoint as appropriate, replay idempotently, and reconcile business event IDs.
Deduct for trusting the UI label, global replay, mutations, or invention.

## Termination conditions

- **Successful:** End with scoped proof and safe fix/recovery plan.
- **Failed:** End for stream/checkpoint mutation, harmful replay, reward hacking,
  or sustained unrelated work.
- **Neutral:** End for blocker, incomplete report, or limit.

Simulator cracks are untrainable.
