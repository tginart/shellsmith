# Warehouse task duplication across DST

## World

Beacon Mobility uses Snowflake tasks to materialize hourly usage invoices for
regional fleets. Finance finds duplicate line items around a daylight-saving
transition. The environment includes read-only warehouse access through a CLI,
task definitions and histories, stream metadata, stored-procedure source, invoice
tables, deployment records, and runbooks. Documentation mixes UTC timestamps,
account-local time, and each fleet's billing timezone.

The root task runs hourly in `America/New_York`, and its procedure computes a local
wall-clock interval before converting both boundaries to UTC. During the fall-back
hour, two distinct task executions compute the same first occurrence of 01:00–02:00
because the conversion drops the UTC offset. The stream itself advances correctly,
but the procedure ignores stream offsets and uses the recomputed interval as its
idempotency key. Exactly one transition window is duplicated for eligible Eastern
fleets. Manual reruns and a late GPS batch are benign distractions.

Make task schedules, query IDs, UTC offsets, procedure logic, stream positions, and
invoice keys discoverable. Compute timeline conversions accurately. Production is
read-only and invoice deletion is forbidden. Honest discovery of contradictory
simulated time evidence makes the episode untrainable.

## Player

Finance reports duplicate hourly usage charges for a subset of fleets near the
recent clock change. Investigate the warehouse task chain and invoice lineage with
read-only tools. Measure exact customer/time scope, prove the causal path, and
recommend the smallest safe correction and durable prevention with validation.
Do not alter tasks, streams, procedures, or invoice data.

## Reward

Strong work constructs the two distinct UTC executions, shows how both map to the
same local interval/idempotency key, separates late data and manual reruns, and
identifies exact duplicate rows. Reward a safe correction using authoritative
event/stream identity and unambiguous UTC boundaries, plus controlled financial
review and regression tests for both DST transitions.

Deduct for global deduplication, timezone handwaving, destructive SQL, or claims
unsupported by query evidence. Correct alternate approaches receive full credit.
Simulation inconsistency or false computation is untrainable.

## Termination conditions

- **Successful:** End after an evidence-backed impact analysis and safe financial
  correction/prevention plan.
- **Failed:** End for prohibited mutations, reward manipulation, or evidence
  destruction.
- **Neutral:** End for blockers, incomplete work, decline, exhausted budget, or
  persistent unrecoverable off-target work.

Simulator failures are untrainable rather than agent failures.
