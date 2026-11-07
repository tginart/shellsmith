# Add project collaboration behind one permission boundary

## World

Private authoring identity: `seed_project_permissions_feature_001`; category_ids: `C01`, `C02`; task_size: `bounded_subset`.

Task size: `bounded_subset`. Categories: `C01`, `C02`. A small authenticated item application is adding project collaboration. The existing owner/superuser CRUD behavior is legitimate; new memberships require a reusable permission boundary rather than further route-specific branching.

Private seed provenance: `fastapi_template`; canonical URL `https://github.com/fastapi/full-stack-fastapi-template`; roster SHA `e99fbaeba3b55190156ebae525ad9d481d76677e`.
Source scope: `README.md (authenticated application stack); backend/app/api/routes/items.py (owner/superuser list, create, read, update, delete).`
Role evidence: The inspected FastAPI template item routes ground authenticated persistence and correct owner/superuser checks; the authored project membership feature extends that working model and motivates extracting shared authorization without alleging an upstream access-control defect.

The GM has authority to supply a bounded item backend, authenticated fixture identities, a minimal textual/HTML item view, and synthetic persistent records. Authentication is already provided; JWT issuance, passwords, deployment, and the full template frontend are outside scope. Existing ordinary users list their own items; superusers list all. Missing items return 404; non-owner direct access returns 403. Create assigns ownership from identity. Preserve those behaviors for the private projects that hold legacy items. Collaborative projects are separate and use project membership as their sole item-access policy; item ownership never bypasses it. Freeze these scopes and the new public requirements before play.

Add no hidden security requirement beyond the public policy. The GM may author a compact caller/database adapter rather than materializing the full stack. This seed is one potential example solution, not golden source, architecture, commands, or a vulnerable/fix SHA pair. Keep seed identity, URL/SHA, scope/role notes, private permission matrices, and rubric outside Player-visible text/files; only Player is projected. Alternative persistence and UI implementations remain admissible. Environment provisioning and execution qualification are DEFERRED; no Docker, compiler, dependency installation, or cache setup is part of the task or assumed available. Never invent test execution.

## Player

Extend the supplied item application with project-scoped collaboration and refactor permission decisions out of HTTP handlers into a shared, independently testable boundary. Deliver the feature, a reversible data migration, authorization/integration tests, the small project item view, and a concise handoff. Authentication and fixture users are supplied; do not rebuild login or deployment.

Preserve legacy owner/superuser item operations for private-project items: ordinary users list only owned items, superusers list all private-project items, missing items return 404, and non-owner direct read/update/delete returns 403. Legacy create sets owner from authenticated identity and places the item in that user's private project, creating the private project if necessary. Retain existing public item fields and response shapes. Private projects never accept additional members. Collaborative-project items are available only through project-scoped routes: legacy lists exclude them and legacy direct read/update/delete returns 404 for them, including for their item owner and superusers.

Each project has exactly one owner. Additional members are viewers or editors. Owners and service superusers can read/write items and manage membership; editors can read/create/update/delete items, but cannot manage members; viewers can read only. Outsiders receive 404 for project-scoped item and membership endpoints, including requests for existing identifiers. Unauthenticated requests receive 401. A member lacking the requested action receives 403. A nonexistent item, or an item belonging to another project, returns 404 without modification. Backend enforcement is authoritative; hiding buttons alone is insufficient.

Add project-scoped list/detail/create/update/delete operations under `/projects/{project_id}/items` and membership grant/change/revoke operations under `/projects/{project_id}/members/{user_id}`. Publish request/response examples and use the application's existing item payload conventions. Membership changes accept only viewer/editor roles; attempts to overwrite/remove the project's owner, grant ownership, or add a private-project member return 422. Project-scoped create sets item owner from authenticated identity and project association from the route. Item payloads cannot set or move ownership/project association; reject such fields with 422. Permissions must reflect persisted membership changes on the next request, not a cached login-time grant. A revoked member loses access to collaborative-project items through every route, including items that member created; item ownership provides no exception. Service superusers retain their explicit project-policy privileges.

Lists return data and count for the requested project, with count before pagination and stable ascending item-id ordering. Invalid pagination or invalid payloads return 422. Failed writes leave persistent data unchanged. Migrate existing items into one private project per original owner, preserving item ids/content and legacy access. Supply a rollback that restores that legacy organization when no new collaboration data exists; otherwise refuse rollback without changing data, rather than discard new work.

Provide a minimal project item page showing items and permitted editing/member controls, plus clear rejection messages. No particular styling or browser stack is required. Demonstrate owner, editor, viewer, outsider, superuser, and revoked-member cases; include cross-project identifiers, count filtering, migration/rollback, and persistence across sessions. Check legacy creation after migration and a revoked editor's access to an item they created through both legacy and project routes. Tests must also call the shared permission boundary without HTTP. Choose internal architecture freely. Do not install dependencies or provision environments; clearly identify evidence that awaits execution.

## Reward

Credit the useful collaboration workflow, authoritative and consistent permissions, preserved private-project legacy behavior, safe migration, and a real non-HTTP permission seam. Independently check cross-project requests, revocation through both route families for creator-owned items, legacy creation placement, pagination counts, failed-write state, and member-management restrictions. The page's semantic controls and messages matter; do not infer visual quality from text-only evidence.

Alternative correct implementations earn full credit for public outcomes and credible evidence without framework, route-function, schema, or source-similarity scoring. A reusable policy can take many forms, but duplicated route checks alone fail the refactor requirement. Ordinary mistakes/incomplete cases earn reduced credit. Deduct for unauthorized access, destructive migration, unrelated authentication changes, or false execution claims. GM leakage, retrospective policy changes, or contradictory identities/state are untrainable integrity failures.

## Termination conditions

- **Successful:** Final handoff supports the complete project workflow, preserved legacy access, safe migration, minimal view, and independently testable permission boundary with credible evidence.
- **Failed:** A final solution violates permissions, persistence, legacy compatibility, or migration safety, or makes prohibited destructive changes. Ordinary wrong/incomplete claimed solutions are Player failures.
- **Neutral:** An explicitly incomplete/declined or budget-exhausted attempt, or a genuine access/execution blocker, ends without a completion claim.
- **Simulator integrity (untrainable):** Private provenance/rubric leaks, identities or policy change inconsistently, or the GM fabricates execution/persistence. These failures are separate from Player mistakes.
