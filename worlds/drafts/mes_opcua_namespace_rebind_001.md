# MES OPC-UA namespace rebinding

## World

Ironvale Packaging streams fill-weight and reject-counter tags from production
lines through OPC-UA into a manufacturing execution system (MES). After a planned
PLC gateway restart, one line's reject counts flatline while its fill weights
continue updating. The environment includes OPC-UA endpoint descriptions, namespace
arrays, browse/read/subscription tools, gateway restart logs, MES connector source,
historian queries, equipment topology, and read-only production dashboards.

The connector persists numeric NodeIds such as `ns=4;i=6021`. Namespace indexes are
session-local positions, not stable identities. After restart, a vendor diagnostics
namespace was inserted before the line-control namespace, moving its URI from
index 4 to 5. The old numeric NodeId now points to a valid but constant diagnostic
counter with the same data type, so subscription setup succeeds. Fill weights use
string NodeIds resolved by browse path and remain correct. A brief historian outage
and normal zero-reject intervals are distractions.

Keep namespace URIs/indexes, NodeIds, browse paths, data types, source timestamps,
and line topology coherent. Safe browse/read/subscription experiments against a
simulated mirror are allowed. No production writes, method calls, or subscription
changes are allowed.

## Player

Investigate the flatlined reject counts after the PLC gateway restart. Trace data
from OPC-UA discovery through the connector and MES/historian, measure affected
time and production scope, reproduce the binding behavior safely, and propose a
durable fix plus data-recovery and validation plan. Production controls are
read-only; do not write nodes or call equipment methods.

## Reward

Reward comparison of namespace arrays across sessions, URI-based resolution of the
intended node, proof that the stale NodeId validly targets the wrong counter, and
separation of fill-weight and historian behavior. Strong plans persist namespace
URI plus identifier/browse identity, resolve on every session, validate semantic
metadata, recover reject counts from PLC batch records where available, and alert
on suspiciously constant series. Deduct for PLC resets, hardcoding new index 5,
writing nodes, or fabricated telemetry.

## Termination conditions

- **Successful:** End with protocol-level diagnosis and a safe connector, recovery,
  and validation plan.
- **Failed:** End for equipment mutation, unsafe OPC-UA calls, or reward
  manipulation.
- **Neutral:** End for a genuine access blocker, incomplete result, or action
  exhaustion, including off-target implementation requiring restart.

Immersion or orchestration cracks make the episode untrainable.
