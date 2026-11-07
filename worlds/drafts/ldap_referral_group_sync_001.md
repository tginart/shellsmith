# LDAP referral group-sync omission

## World

An acquired subsidiary's employees are missing from an access-review export fed by
a legacy LDAP sync. The environment provides ldapsearch, directory topology docs,
sanitized bind configuration, sync source, logs, and read-only entitlement tables.
The root directory returns referrals for the subsidiary naming context. A library
upgrade changed referral chasing from enabled to disabled, while the sync logs the
initial search as successful and silently ignores referral entries. Direct queries
to the subsidiary server return the missing memberships. Disabled accounts and
nested-group cycles are valid distractions.

Expose coherent DNs, referral URLs, page cookies, group memberships, and release
history. Safe read-only LDAP queries are allowed; binds and directory state cannot
be changed.

## Player

Investigate why subsidiary employees disappeared from the access-review export.
Determine exact scope and causal path using documentation, source, LDAP queries,
and database evidence. Produce a minimal secure fix and backfill-validation plan.
Do not change directory or entitlement state.

## Reward

Reward inspecting referrals, proving direct naming-context results, correlating the
library default change, and separating disabled/nested cases. Strong remediation
explicitly configures trusted referral targets or queries contexts directly,
preserves TLS/auth boundaries, and reconciles identities and memberships. Deduct
for broadening trust to arbitrary referrals, manual grants, or fabrication.

## Termination conditions

- **Successful:** End with scoped proof and secure repair/backfill plan.
- **Failed:** End for directory writes, credential exposure, unsafe referral trust,
  or manipulation.
- **Neutral:** End for blockers, incomplete work, or exhausted actions.

Simulation cracks are untrainable.
