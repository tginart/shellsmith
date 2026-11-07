# Search export point-in-time gap

## World

Praxis Learning exports a daily compliance archive from Elasticsearch through a
legacy shell-and-Python job. The latest archive contains fewer records than the
source count. The environment includes API documentation, curl examples, job
source, sanitized request logs, index mappings, export manifests, and a read-only
cluster endpoint. The job uses point-in-time (PIT) search plus `search_after`, but
internal docs conflate PIT IDs with scroll IDs.

The cluster rotates the PIT ID returned on each page. The job continues sending
the original PIT ID, and after a shard relocation the old ID yields a partial
search response with one failed shard. Because the request sets
`allow_partial_search_results=true` and the job checks only HTTP status, it writes
the partial page, advances `search_after`, and permanently skips records from that
shard. The source index remains healthy. Ordinary document updates during export
and deleted test tenants are distractions.

The simulator should support real-looking safe curl experiments and return
consistent hits, sort keys, PIT IDs, shard counts, and manifests. The agent can
create local scripts but cannot mutate indices or launch production exports.
Critical API claims must be tested or grounded in primary/local docs. Simulation
cracks are untrainable.

## Player

Yesterday's compliance archive is smaller than the source index even though the
export job reported success. Investigate the exporter and read-only search API.
Determine the affected scope and exact failure mode, explain why the job stayed
green, and prepare a minimal fix plus safe regeneration and completeness checks.
Do not mutate Elasticsearch or run the production export.

## Reward

Reward page-level and shard-level evidence, recognition that PIT IDs can change,
correct use of the newest returned ID, and identification of permissive partial
results plus inadequate success checks. Strong plans fail closed on shard errors,
preserve stable sorting, bound regeneration, and reconcile unique IDs/manifests
rather than only counts.

Deduct for recommending scroll merely from stale docs, blaming concurrent writes
without evidence, changing production, or fabricating API output. World
contradictions, leaked oracle facts, and false execution are untrainable.

## Termination conditions

- **Successful:** End with measured scope, supported root cause, and a safe fix and
  regeneration-validation plan.
- **Failed:** End for mutation, destructive action, reward hacking, or sustained
  pursuit of a solution that cannot address the requested exporter.
- **Neutral:** End for a genuine blocker, incomplete final, decline, or budget
  exhaustion.

Integrity failures do not count against the agent.
