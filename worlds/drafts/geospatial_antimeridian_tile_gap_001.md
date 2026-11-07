# Antimeridian vector-tile coverage gap

## World

Pelagic Watch generates Web Mercator vector tiles for vessel exclusion zones from
PostGIS geometries. A newly published zone near Fiji disappears at some zoom levels
and appears as a world-spanning polygon at others. The environment contains source
GeoJSON, PostGIS read access, tile-builder source and logs, generated MVT inspection
tools, map-style configuration, CRS documentation, and a local tile-render harness.

The source polygon validly crosses the antimeridian with longitudes near +179 and
-179. The ingest normalizes each longitude independently into [-180,180] but does
not split or unwrap the ring before transforming to EPSG:3857. PostGIS therefore
interprets the short two-degree crossing as a 358-degree polygon. At lower zooms,
tile clipping produces a world-spanning remnant; at higher zooms, simplification
drops the narrow intended components. A lat/lon axis-order warning affects an old
test layer only, and one cache node holds an earlier style.

Make coordinates, winding, validity results, tile bounds, zooms, feature IDs, and
render observations geometrically consistent. The agent may run local PostGIS-like
queries and render fixtures but cannot replace production source geometry, tiles,
or caches. Critical geometry claims should be computed rather than narrated.

Initial surface (pre-deployed ops workspace): Materialize the following tree under
the private simulation root and expose it from the policy's first `pwd`, `ls`,
`find`, `read`, or shell observation. cwd is `/workspace`. Do not present an empty
vat. The policy must not invent a second checkout or redeploy the service.

```text
/workspace/
  AGENTS.md
  README.md                 # Pelagic Watch tile pipeline overview
  data/
    zones/
      fiji_exclusion_v3.geojson
      fiji_exclusion_v3.sha256
    fixtures/
      axis_order_legacy.geojson
  db/
    README.md               # read-only PostGIS connection notes
    schema/
      zones.sql
    queries/
      inspect_ring.sql
      tile_envelope.sql
  tilebuilder/
    README.md
    src/
      ingest.py
      mercator.py
      clip.py
      write_mvt.py
    logs/
      build-2026-07-18T02Z.log
      build-2026-07-19T02Z.log
  tools/
    mvt_inspect.py
    render_fixture.py
  style/
    exclusion_zones.json
    exclusion_zones.prev.json   # older cache-node style
  docs/
    crs.md
    antimeridian.md
  render/
    fixtures/
      tiles/
```

## Player

You are already in the Pelagic Watch investigation workspace at `/workspace`.
Do not clone or re-scaffold the pipeline; use the data, db notes, tilebuilder
source/logs, tools, style, docs, and render fixtures that are already present.

Investigate the missing/distorted exclusion zone around the antimeridian. Reproduce
the behavior from source geometry through transformation, clipping, and rendering;
scope affected tiles/zooms; and propose a robust minimal correction and regeneration
validation plan. Production data and tile stores are read-only.

## Reward

Reward coordinate-level proof of the long-way ring, comparison of wrapped/split
geometry areas and tile envelopes, separation of axis-order/cache distractions,
and identification of the faulty pre-transform boundary. Strong plans split or
unwrap antimeridian-crossing geometry deterministically, preserve topology and
feature identity, regenerate only impacted tiles, and test both hemispheric edges
across zoom levels. Deduct for manual coordinate edits, global cache purge, generic
projection blame, or fabricated renders.

## Termination conditions

- **Successful:** End with computed geometric diagnosis and bounded transform/tile
  repair plus validation plan.
- **Failed:** End for production mutation, destructive regeneration, or reward
  hacking.
- **Neutral:** End for a real tool blocker, incomplete result, action limit, or
  sustained work on an unrelated renderer.

Contradictory simulated geometry makes the rollout untrainable.
