# Weighted DNS failover imbalance

## World

Peregrine Media serves uploads through two regions behind weighted DNS. After a
failover test, some clients still reach the impaired region. The environment
includes authoritative DNS API output, zone change history, resolver queries from
several networks, health-check logs, CDN configuration, and runbooks. The failover
automation set the A-record weight to zero but left an HTTPS/SVCB record advertising
the impaired region's service endpoint with a longer TTL. Modern clients prefer
HTTPS records; older clients follow A records and recover. Cached A answers and a
mobile ISP resolver issue are distractions.

Keep TTLs, record sets, client capabilities, and observation times coherent. DNS
and health systems are read-only; queries from simulated vantage points are safe.

## Player

Investigate why failover behavior differs across upload clients. Reproduce and
scope the resolution paths, identify the authoritative cause, and provide a minimal
safe DNS/automation correction and validation plan. Do not alter zones or trigger
failover.

## Reward

Reward querying all relevant record types, comparing modern and legacy resolution,
tracing the stale HTTPS record and TTL, and distinguishing caches from authoritative
state. Strong plans update failover atomically across record sets and validate from
diverse resolvers over TTL boundaries. Deduct for generic DNS-propagation claims,
blind TTL lowering, mutation, or fabrication.

## Termination conditions

- **Successful:** End with evidence-backed diagnosis and atomic remediation plan.
- **Failed:** End for DNS mutation, disruption, reward hacking, or irrecoverable
  unrelated work.
- **Neutral:** End for blocker, incomplete report, decline, or limit.

Simulator contradictions are untrainable.
