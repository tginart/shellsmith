# Schema-registry compatibility incident

## World

Sable Insurance publishes policy-change events through Kafka using Avro and a
Schema Registry compatible HTTP API. One downstream rating service began sending
events to its dead-letter topic after a producer release. The environment contains
topic and consumer-group inspection commands, registry API docs, schema histories,
producer/consumer source snapshots, deployment metadata, sample payloads, and
read-only warehouse projections. Several subject-naming strategies have existed.

The producer switched from topic-name to record-name subject lookup. It found an
older globally registered `PolicyChanged` schema whose field `premium` is nullable
with a null-first union. The rating consumer resolves schema IDs correctly but its
handwritten decoder assumes the topic subject's non-null-first union branch index.
Only records with non-null premium fail; null values happen to decode. Registry
compatibility checks passed because they were applied independently within each
subject. A consumer lag spike and an unrelated new enum symbol are distractions.

Make subject names, IDs, canonical schemas, wire bytes, decoder assumptions, and
deploy boundaries discoverable through documented API calls and local decoding
experiments. Keep offsets and failure counts stable. The agent may not register
schemas, reset groups, or replay topics. The world must not silently change API
contracts to accommodate guesses.

## Player

The rating consumer started dead-lettering some `PolicyChanged` events immediately
after a producer deployment, while registry compatibility checks stayed green.
Investigate with read-only terminal/API tools and source inspection. Scope the
affected records, prove the serialization/decoding failure, and propose a safe
producer/consumer repair and replay-validation plan. Do not register schemas,
change consumer offsets, or replay data.

## Reward

Give high credit for correlating wire schema IDs to record-name versus topic-name
subjects, reproducing union-branch decoding on representative bytes, explaining
the null/non-null asymmetry, and showing why compatibility passed. A strong plan
uses generated/schema-aware decoding, prevents ambiguous subject lookup, bounds
replay from authoritative DLQ offsets, and validates semantic equality.

Deduct for generic "schema mismatch" claims, unsafe registry changes, blind offset
resets, or fabricated decoding. Alternate grounded fixes are welcome. Simulator
contradictions, leakage, or false execution make the rollout untrainable.

## Termination conditions

- **Successful:** End after a reproducible diagnosis and controlled fix/replay
  plan.
- **Failed:** End early for prohibited mutation, destructive offset operations,
  reward hacking, or irrecoverably unrelated work.
- **Neutral:** End for a real blocker, incomplete result, decline, or action limit.

Simulation integrity failures never reduce the agent's reward.
