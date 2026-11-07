# ClickHouse replacing-merge double count

## World

Nimble Ads stores mutable campaign facts in a ClickHouse
`ReplacingMergeTree(version)` table. A dashboard sometimes double-counts spend
immediately after corrections but converges hours later. The environment includes
read-only SQL, DDL, query logs, partition/part metadata, ingestion source, and BI
definitions. The BI query assumes physical deduplication and sums all rows without
`FINAL` or an equivalent argMax-by-version aggregation. Background merges explain
the apparent self-healing. One materialized view independently has late data but
does not feed this dashboard.

Keep row versions, parts, query IDs, and totals consistent; execute analytical SQL
when claimed. Production optimize/mutation commands are forbidden.

## Player

Investigate transient campaign-spend overstatement in ClickHouse. Quantify affected
queries and records, prove the storage/query interaction, and recommend a performant
durable fix plus validation. The cluster is read-only; do not run `OPTIMIZE`,
mutations, or change views.

## Reward

Reward comparing physical versions and logical latest rows, connecting convergence
to merges, measuring `FINAL` cost, and proposing an appropriate latest-version
query/materialization strategy with deterministic tie handling. Deduct for treating
merges as correctness guarantees, global optimize, unrelated late-data blame, or
invented results.

## Termination conditions

- **Successful:** End with measured cause and safe performant correction plan.
- **Failed:** End for mutation, destructive SQL, or reward hacking.
- **Neutral:** End for blocker, incomplete final, decline, limit, or off-target
  work requiring restart.

Integrity failures are untrainable.
