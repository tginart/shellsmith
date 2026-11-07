# A museum-ready report on human spaceflight

## World

Private authoring identity: `seed_spaceflight_report_001`; category_ids: `C08`; task_size: `bounded_subset`.

A science museum is preparing an exhibit about participation in human spaceflight through the cutoff of a published historical dataset. Supply the frozen astronaut-mission table and dictionary from TidyTuesday's 2020-07-14 release. It contains 1,277 astronaut-mission records, with missions dated 1961–2019 and the source describing participation before 2020-01-15. These facts fix the historical scope, not a required narrative. The GM preserves original labels and fields and independently computes report claims from the same snapshot.

Seed `tidytuesday`: https://github.com/rfordatascience/tidytuesday at `78474f0d3496a1f855deeceb63e250f494cbe938`. Selected scope: data/2020/2020-07-14/astronauts.csv and data/2020/2020-07-14/readme.md. The TidyTuesday seed provides the actual astronaut-mission observations, historical cutoff, and dictionary needed to distinguish participant counts, mission records, and repeated cumulative quantities.

Seed `nbconvert`: https://github.com/jupyter/nbconvert at `1ff189e1b30f0820238e95427c035a9150a5e43f`. Selected scope: README.md and docs/source/usage.rst execution, HTML export, and resource handling. The nbconvert seed provides one example report-production path with reproducible computation and asset handling; the report's substance and audience usefulness, not that exporter, determine success.

Task size: bounded subset, a complete analytical report rather than either full upstream project. The `id` field identifies records. Numeric person identifiers and displayed names are not globally unambiguous in this snapshot; expose the raw identity fields and dictionary and accept a documented, evidence-supported reconciliation rather than a hidden golden key. A validated composite of name, birth year, and nationality is one option, not a mandatory implementation. Per-person cumulative mission counts and time totals can be repeated across rows and cannot simply be summed as per-mission activity. Per-mission duration and EVA-duration fields support activity summaries. A reconciled person's earliest observed mission year minus birth year supports an approximate first-flight age in this dataset, not an exact age from birthdays. Selection year measures selection age and must not substitute for the first observed mission year. Dictionary ambiguities and inconsistent fields should be surfaced and handled transparently, not privately repaired to force a story. Expose these semantics in ordinary source documentation.

The GM remains in control of coherent observations and judging. Source analysis or exporter code is one potential example solution, not a golden report, architecture, chart, or command sequence. Correct alternatives satisfying the public request are accepted; no private aesthetic preference or unstated upstream behavior is mandatory. Keep private reference/judging content out of the Player. Environment provisioning, rendering infrastructure, and execution qualification are deferred; do not invent a visual inspection or execution result.

## Player

Create a museum-ready, standalone HTML report from the supplied historical astronaut dataset. Explain how participation changed over the covered years, how first-flight age varied across cohorts, and how mission or spacewalk time was distributed. Distinguish unique people from person-mission records, document your identity reconciliation rather than assuming names or numeric person identifiers are globally unique, and do not sum repeated per-person cumulative fields as though they were mission measurements.

Include at least three useful figures, a compact summary table, a plain-language narrative for museum visitors, provenance, the observation cutoff, and an explicit limitations section. Preserve the published nationality labels; explain any grouping choices instead of silently rewriting historical identities. Compute first-flight age from the earliest observed mission year for each reconciled person minus birth year, not from selection year; describe it as approximate and limited to the dataset's observed history. When aggregating durations across astronaut-mission rows, label totals as person-hours rather than unique spacecraft flight-hours.

Deliver the report, its analysis source or notebook, and an ordinary regeneration entry point. The report must be readable without fetching remote assets. Choose your own statistical tools, renderer, layout, and charts. Check headline numbers and resource links, retain the raw input, and distinguish executed verification from anything not checked. Do not extend the evidence to present-day participation or claim demographic differences are causal.

## Reward

Reward accurate record/person denominators, justified cohort and duration calculations, avoidance of cumulative-field double counting, checked report quantities, useful figures, accessible narrative, clear historical/provenance limits, and reproducibility. Independently evaluate the exposed data and report. Different defensible analyses and polished artifact implementations deserve full credit; using nbconvert or matching an existing community notebook is not required.

Score substantive audience usefulness separately from mere HTML existence. Assess visual qualities only from actual rendered evidence when available, not simulator-authored descriptions. Deduct for fabricated charts/checks, silent label changes, duplicated cumulative totals, missing local resources, unverifiable headline claims, or unsupported causal/current statements. The seeds are potential examples, not hidden required answers or source-similarity targets.

## Termination conditions

- **Successful:** A usable standalone report and reproducible source satisfy the public content requirements with credible numerical and resource checks.
- **Failed:** A claimed-complete report materially violates the public analytical or artifact contract, or deliberate evidence corruption, fabricated execution or claims, unauthorized external effects, or persistent reward manipulation occurs. Ordinary analysis or formatting errors remain trainable policy failures.
- **Neutral:** Stop for real blockers, incomplete artifacts, explicit decline, or exhausted budget; score supported progress honestly.
- **Simulator integrity:** Contradictory source facts, leaked private expected answers, or invented execution/rendering results make the episode untrainable rather than a Player failure.
