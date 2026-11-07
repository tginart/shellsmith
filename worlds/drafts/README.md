# Generative world files

This directory contains 150 campaign worlds and three simulator-operation
worlds promoted through [`../manifest.json`](../manifest.json). Its historical
`drafts/` name does not determine runtime membership. Static contract review
does not certify live environments or execution qualification. Each compact
world leaves concrete fixtures and peripheral evidence for the Game Master to
create coherently.

The generated [`CATALOG.md`](CATALOG.md) is the authoritative complete index for
the 150-world three-flavor campaign. Generic flavor discovery intentionally
excludes `meta_world_simulator_*`; core manifest or explicit ID selection can
select those three promoted worlds. The curated tables below document the
original seed set and its design intent.

Authoring templates:

- [`AGENTIC_WORLD_META_SPEC.md`](AGENTIC_WORLD_META_SPEC.md)
- [`PUBLIC_REPO_REPLAY_META_SPEC.md`](PUBLIC_REPO_REPLAY_META_SPEC.md)
- [`PAPER_AUDIT_META_SPEC.md`](PAPER_AUDIT_META_SPEC.md)

| Draft | Focus | Goal | Production posture |
| --- | --- | --- | --- |
| `warehouse_inventory_reconciliation_001.md` | Warehouse reconciliation | Explain and correct inventory drift | Read-only |
| `object_store_retention_incident_001.md` | Object-store retention | Investigate loss and plan recovery | Read-only |
| `legacy_settlement_batch_001.md` | Legacy system | Prepare a controlled settlement replay | Read-only |
| `access_review_exception_001.md` | Security/compliance | Assess a privileged-access exception | Read-only |
| `incident_bridge_handoff_001.md` | Multi-team coordination | Coordinate customer-impact mitigation | Limited operational actions |
| `webhook_signature_repair_001.md` | Software repair | Verify a webhook-authentication repair | Local edits allowed |
| `report_export_timezone_repair_001.md` | Software repair | Repair a date-boundary export defect | Local edits allowed |
| `partner_api_quota_outage_001.md` | API incident | Establish impact and recovery plan | Read-only |
| `analytics_lineage_break_001.md` | Data operations | Trace a metrics discrepancy | Read-only |
| `vendor_invoice_reconciliation_001.md` | Finance operations | Reconcile a billing variance | Read-only |
| `privacy_erasure_hold_001.md` | Privacy/compliance | Resolve deletion versus legal hold | Read-only |
| `feature_flag_audit_001.md` | Change safety | Assess an unsafe rollout | Read-only; staging allowed |

## Advanced drafts

These are deliberately closer to frontier terminal-agent work. Each requires a
long evidence chain across several imperfect interfaces plus an executable local
artifact or recovery proof. The intended difficulty is not obscurity: all critical
facts remain discoverable, multiple sound approaches remain valid, and the reward
cares about the resulting state and evidence rather than a hidden command sequence.

| Draft | Focus | Hard deliverable | Production posture |
| --- | --- | --- | --- |
| `cross_region_ledger_recovery_001.md` | Distributed recovery | Prove a gap-free, duplicate-free ledger reconstruction | Read-only; local recovery lab |
| `oidc_rotation_cache_repair_001.md` | Authentication/security | Repair a concurrent JWK cache and verify adversarial cases | Local edits; production read-only |
| `signed_artifact_provenance_001.md` | Supply-chain security | Identify the bad build edge and harden provenance verification | Local/CI sandbox only |
| `cdc_snapshot_boundary_repair_001.md` | Data systems | Repair snapshot/CDC handoff and prove reconciliation | Local replica and fixtures |
| `exactly_once_replay_planner_001.md` | Messaging/recovery | Generate and validate a safe selective replay plan | Production read-only |
| `legacy_mtls_chain_recovery_001.md` | PKI/legacy integration | Diagnose path-building failure and produce a tested bundle fix | Local TLS lab only |

The suite uses warehouses, object storage, legacy batch systems, incident response,
security, privacy, coordination, and focused repairs without prescribing golden
commands or exhaustive fixtures.

## Public-repository repair drafts

[`PUBLIC_REPO_REPLAY_META_SPEC.md`](PUBLIC_REPO_REPLAY_META_SPEC.md) defines the
shared authoring and simulation policy for replaying already-fixed public issues at
their vulnerable parent commit. These drafts are self-contained instances; the
meta-spec is not injected into a live episode.

| Draft | Repository | Repair focus |
| --- | --- | --- |
| `public_repo_hapi_content_duplicate_parameters_001.md` | `hapijs/content` | Reject ambiguous duplicate MIME/disposition parameters |
| `public_repo_zipp_iterdir_exception_001.md` | `jaraco/zipp` | Restore `pathlib`-compatible exception semantics |
| `public_repo_go_jose_multisig_key_selection_001.md` | `go-jose/go-jose` | Bind each JWS signature to its own verification key |
| `public_repo_node_tmp_path_escape_001.md` | `raszi/node-tmp` | Contain attacker-controlled temporary paths |
| `public_repo_golang_jwt_epoch_expiry_001.md` | `golang-jwt/jwt` | Validate Unix-epoch expiration consistently |
| `public_repo_packaging_marker_parentheses_001.md` | `pypa/packaging` | Preserve marker semantics through serialization |

## Research-audit drafts

| Draft | Primary source | Audit focus |
| --- | --- | --- |
| `lz_penalty_paper_audit_001.md` | arXiv:2504.20131 | Independently find and verify mathematical errors in the LZ penalty derivation |
