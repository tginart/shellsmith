# GraphQL persisted-query rollout failure

## World

Mica Market's mobile checkout calls GraphQL through a CDN using automatic
persisted queries. After a backend release, older Android clients see intermittent
`PersistedQueryNotFound` errors. Available evidence includes CDN and origin logs,
GraphQL/APQ docs, deployment manifests, mobile release metadata, cache-key rules,
and safe curl access to staging and read-only production diagnostics.

The release changed the server's hash canonicalization from the raw query bytes to
a parsed-and-reprinted document, but old clients still hash raw bytes. Full-query
fallback requests succeed and register under the server-computed hash; subsequent
hash-only requests from old clients keep missing. The CDN caches the HTTP 200 error
body because the response lacks `Cache-Control: no-store`, amplifying failures.
New clients and non-persisted web traffic are healthy. A resolver latency increase
is unrelated.

Keep hashes, query variants, cache status, and client versions coherent and permit
safe reproduction with curl. No cache purge or deployment is allowed.

## Player

Investigate checkout failures affecting some mobile clients after the GraphQL
release. Use available docs, logs, terminal tools, and safe requests to scope and
reproduce the issue, identify the causal chain, and propose the smallest compatible
fix, rollout, and validation plan. Production is read-only.

## Reward

Reward byte-level hash comparison, old/new client scoping, proof of fallback
registration mismatch, and recognition of CDN amplification. Strong fixes restore
protocol-compatible client hashing, prevent caching APQ negotiation errors, and
stage validation across client versions. Deduct for resolver blame, blind cache
purge, unsupported claims, or mutation.

## Termination conditions

- **Successful:** End with reproducible diagnosis and controlled compatibility fix.
- **Failed:** End for mutation, reward hacking, or irrecoverably unrelated work.
- **Neutral:** End for blocker, incomplete result, decline, or action limit.

Simulation inconsistencies are untrainable.
