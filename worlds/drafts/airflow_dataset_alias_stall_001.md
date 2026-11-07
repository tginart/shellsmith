# Airflow dataset-alias scheduling stall

## World

Redwood Commerce migrated an Airflow pipeline from cron scheduling to dataset-
driven scheduling. Downstream margin DAGs stopped starting for one source although
upstream tasks succeed. The environment contains DAG source, serialized DAG
metadata, task logs, dataset-event tables, scheduler logs, Airflow CLI output, and
versioned docs. The producer emits an event for URI `s3://curated/orders/` with a
trailing slash, while the consumer subscribes to `s3://curated/orders`. Airflow
treats dataset URIs as exact opaque identifiers. A stale alias with the no-slash
form makes the UI misleadingly show recent activity from a test DAG.

When materializing scheduler evidence, log timestamps that announce a DAG run's
successful transition precede or equal that run's recorded `end_date`; later log
flush or display timestamps must not be presented as the transition timestamp.

Keep DAG IDs, URIs, event producers, and schedule times consistent. Database and
scheduler are read-only; local DAG parsing is permitted.

## Player

Investigate why the dataset-scheduled margin DAG no longer starts despite green
upstream tasks. Establish scope and root cause across code, scheduler state, and
dataset events, and propose the smallest correction, catch-up, and validation plan.
Do not trigger DAGs or modify Airflow metadata.

## Reward

Reward exact producer-consumer URI comparison, identification of misleading test
events, and proof that no matching production event exists. Strong plans establish
a canonical constant/alias, safely deploy both sides, bound catch-up by source
partitions, and verify event-to-DAG linkage. Deduct for generic scheduler restarts,
manual triggers without scope, mutation, or invented output.

## Termination conditions

- **Successful:** End with evidence-backed diagnosis and controlled deploy/catch-up
  plan.
- **Failed:** End for metadata mutation, unsafe trigger, or reward manipulation.
- **Neutral:** End for blocker, incomplete final, action limit, or sustained
  commitment to an unrelated implementation that would require a restart.

Simulation failures are untrainable.
