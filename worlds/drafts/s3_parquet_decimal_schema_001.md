# Object-lake decimal schema drift

## World

LumenPay lands daily processor extracts in an S3-compatible object lake and
exposes them through Trino and dbt. A finance aggregate suddenly understates gross
volume for one processor. The environment includes bucket listings and metadata,
Parquet inspection utilities, an external-table catalog, dbt artifacts, ingestion
logs, a schema registry, and processor documentation. Most access is through
ordinary shell commands and signed HTTP requests documented in READMEs.

The processor expanded `gross_amount` from DECIMAL(12,2) to DECIMAL(18,4). Its
new writer encoded the Parquet physical column as fixed-length byte array, while
the stale external-table definition still coerces it using the old precision.
Values above the former range become null under the connector's permissive mode;
small values remain correct. Files and checksums are complete, and no cents-versus-
dollars conversion occurred. A concurrent late partition and several harmless
statistics warnings are distractions.

The agent must discover the affected prefix and date boundary, compare schemas and
raw values, and trace the catalog definition to its deployment. Keep row counts,
amounts, object versions, and query results coherent. All production data is
read-only; local SQL and schema experiments are allowed. Do not narrate a query as
executed unless its result is actually simulated or computed.

Initial surface (pre-deployed data-platform workspace): Materialize the following
tree under the private simulation root and expose it from the policy's first
`pwd`, `ls`, `find`, `read`, or shell observation. cwd is `/workspace`. Do not
present an empty vat. The policy must not invent a second lake layout or redeploy
catalogs.

```text
/workspace/
  AGENTS.md
  README.md
  docs/
    object_lake.md
    processor_lumenpay.md
    decimal_conventions.md
  lake/
    README.md                 # signed listing / GET helpers
    listings/
      processor=acme/day=2026-07-17.txt
      processor=acme/day=2026-07-18.txt
    samples/
      processor=acme/day=2026-07-17/part-000.parquet
      processor=acme/day=2026-07-18/part-000.parquet
  tools/
    parquet_schema.py
    parquet_peek.py
    signed_get.sh
  catalog/
    external_tables/
      acme_gross.sql          # stale DECIMAL(12,2) definition
    deployments/
      acme_gross_2026-06-02.yaml
  dbt/
    models/
      finance_processor_gross.sql
    target/
      run_results_2026-07-19.json
    logs/
      dbt_2026-07-19.log
  schema_registry/
    acme_gross.json
    acme_gross.prev.json
  ingest_logs/
    acme_2026-07-18.log
    late_partition_note.md    # distraction
  sql/
    trino_examples.sql
```

## Player

You are already in the LumenPay object-lake investigation workspace at `/workspace`.
Do not clone or re-scaffold the lake tooling; use the docs, lake listings/samples,
Parquet tools, catalog definitions, dbt artifacts, schema registry, ingest logs,
and SQL notes already present.

Yesterday's processor-level gross-volume report is low, but object-arrival
monitoring is green. Investigate through the available terminal tools, internal
docs, object storage, catalog, and query engine. Quantify the affected partitions
and values, identify the cause with direct evidence, and provide the smallest safe
repair/backfill plan with validation queries. Do not alter production objects,
catalogs, or warehouse tables.

## Reward

Reward direct comparison of Parquet logical/physical schemas with catalog
definitions, a measured null/amount impact, correct isolation of the processor and
cutover, and a repair that preserves source objects and prevents mixed-schema
misreads. Strong validation reconciles raw decoded values, staged rows, and finance
aggregates before and after a bounded backfill.

Deduct for assuming missing files, proposing global recrawls, conflating scale and
currency units, or inventing query output. Normal late data and warning noise
should be separated from the cause. Simulator inconsistency, leakage, or false
execution makes the rollout untrainable.

## Termination conditions

- **Successful:** End after a quantified root-cause report and bounded repair plus
  validation plan.
- **Failed:** End for prohibited mutation, destruction of evidence, or reward
  manipulation.
- **Neutral:** End for a genuine blocker, incomplete final, decline, or action
  limit, including an irrecoverably off-target implementation after fair chance to
  recover; reward actual evidence.

Integrity failures are untrainable, never agent failures.
