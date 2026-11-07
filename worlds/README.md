# Promoted World catalog

`manifest.json` records the curated selection loaded by
`shellsmith.catalog.load_catalog_entries()`. That loader validates each path,
rejects paths outside this directory, and checks the world identifier.

The manifest is **not a security boundary** for the standalone campaign CLIs.
Explicit world-file arguments and draft-catalog selection can run valid worlds
that are not promoted in the manifest. Use manifest-based selection when an
experiment must contain only the curated worlds; review explicit and draft
selections before starting a campaign. The original Verifiers taskset and
High Command integrations are not included in Shellsmith.

The catalog contains **177 worlds**:

- `checkout_idempotency_001.md` in this directory;
- 150 campaign worlds stored under `drafts/`;
- 23 seed-grounded worlds stored under `seeds/`;
- three simulator-operation worlds stored under `drafts/meta_world_simulator_*`.

The five variants under `mlx_context/` remain specialized context tests outside
the core manifest. Core `manifest`, `random`, and explicit `ids` selection can
include the promoted simulator-operation worlds. Generic three-flavor draft
discovery intentionally omits them.

Promotion records static contract review and selection eligibility. It does not
certify episode solvability, performance, or completed execution qualification.

Format: [World-file contract](../docs/WORLD_FILE_SPEC.md)

Seed provenance: [`SEED_ROSTER.md`](../docs/SEED_ROSTER.md) and the
[seed collection](seeds/README.md). The roster remains research metadata; the
core manifest records runtime membership, and `seeds/coverage.json` pins each
seed world's reviewed content and public prompt.

Parent: [Shellsmith](../README.md)
