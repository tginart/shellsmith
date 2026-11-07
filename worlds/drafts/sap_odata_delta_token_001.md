# SAP OData delta-token reset

## World

Graystone Foods extracts material-master changes from SAP through an OData v2
service. After an SAP maintenance window, the warehouse stopped receiving updates
for one plant while the job remained green. The environment includes service
metadata, curl-accessible read-only endpoints, extractor source, saved checkpoints,
HTTP captures, SAP change logs, and warehouse tables. The maintenance invalidated
old delta tokens. SAP returns HTTP 410 with a new full-sync URL, but a compatibility
wrapper converts it to an empty HTTP 200 collection and preserves the stale token.
Other plants use newer tokens and remain healthy. A unit-of-measure cleanup is
unrelated.

Make token lineage, HTTP layers, entity keys, and counts coherent. Safe API reads
and local transformations are allowed; SAP and warehouse mutations are forbidden.

Initial surface (pre-deployed ops workspace): Materialize the following tree under
the private simulation root and expose it from the policy's first `pwd`, `ls`,
`find`, `read`, or shell observation. cwd is `/workspace`. Do not present an empty
vat. The policy must not invent a second checkout or redeploy the extractor.

```text
/workspace/
  AGENTS.md
  README.md
  docs/
    odata_v2_delta.md
    plant_scope.md
  metadata/
    MaterialMaster.$metadata.xml
    MaterialMaster.edmx
  extractor/
    README.md
    src/
      delta_client.py
      http_compat.py          # wrapper that remaps some SAP errors
      checkpoint_store.py
      warehouse_writer.py
    tests/
      test_delta_client.py
  checkpoints/
    plant_1000.json
    plant_2000.json           # healthy
    plant_3000.json           # healthy
  captures/
    2026-07-18/
      plant_1000_delta.http
      plant_1000_delta.body.json
      plant_1000_raw_410.http # private-ish evidence Sol may use when asked
  sap_change_logs/
    plant_1000_window.md
  warehouse/
    exports/
      material_master_plant_1000_sample.csv
      material_master_counts_by_plant.csv
  scripts/
    curl_examples.sh
```

## Player

You are already in the Graystone SAP extractor workspace at `/workspace`.
Do not clone or re-scaffold the job; use the docs, metadata, extractor source,
checkpoints, HTTP captures, change logs, warehouse exports, and scripts already
present.

Investigate missing material-master changes for the affected plant. Prove exact
scope and failure behavior using docs, requests, checkpoints, and data comparisons,
then prepare a bounded resynchronization and durable error-handling plan. Do not
change SAP, checkpoints, or warehouse data.

## Reward

Reward inspecting raw versus wrapped HTTP behavior, identifying invalid delta-token
handling, scoping by plant and checkpoint, and planning a controlled full snapshot
with overlap-safe key reconciliation before atomically replacing the token. Deduct
for accepting empty success, global reload, deleting checkpoints, or fabrication.

## Termination conditions

- **Successful:** End with supported diagnosis and safe resync/validation plan.
- **Failed:** End for mutation, checkpoint destruction, or reward hacking.
- **Neutral:** End for blocker, incomplete result, budget exhaustion, or
  irrecoverably off-target work.

Integrity failures are untrainable.
