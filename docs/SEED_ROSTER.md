# Generative world seed roster

Research roster of **21 seed repositories** across **10 possible task categories**. Metadata collected **2026-10-06T23:18:10.010293+00:00**.

**Status: candidates, not runnable worlds.** GitHub metadata and candidate commit references have been verified. No seed checkout, runtime fixture preparation, dependency installation, baseline test, license clearance, or training rollout is claimed. This document does not change the 151-task runtime allowlist.

The [seed-grounded authoring collection](../worlds/seeds/README.md) now contains 23 world contracts using all 21 references, including selected historical dataset scopes. These are authored specifications, not execution-qualified environments.

The structured source of truth is [`seed_roster.json`](seed_roster.json). This Markdown is its human-readable companion. Source URLs, exact UTC timestamps, full commit SHAs, GitHub metadata, category roles, and acceptance ideas are retained in JSON.

## Reading the roster

- **Primary** = strongest intended use of that seed; **secondary** = a concrete optional variant. Neither means an accepted world already exists.
- **Direct** = a source/data anchor; **paired** = additional data/workload/report assets are essential; **authored workflow** = persisted changing events must be built; **authored overlay** = controlled adversarial content must be built.
- A category classifies the task objective, not the repository generally. Libraries are not datasets; benchmark tools are not optimization results; renderers are not substantive reports; services are not automatically evolving workflows.
- World ideas are proposed tasks, not confirmed upstream defects. Inspect the selected code first: if a requested structural boundary already exists, choose another task or introduce a real new requirement rather than degrading healthy code.
- Commit recency is the latest **default-branch committer timestamp**, not `updated_at`, latest push on any branch, or dataset observation date. A fresh commit may be documentation, CI, or automation.
- GitHub stars are exact at collection time. Release entries refer only to GitHub's latest published non-prerelease release, not all tags or package registries. Repository archive/history size in JSON is **not source LOC**.
- Snapshot SHAs below are immutable **candidate references**, not downloaded or compatibility-qualified training baselines. Prefer a tested release/commit after qualification; freeze the final choice and file/data hashes.

## Category map

Counts below describe prospective seed fit, not qualified-world coverage. **C10 is overlay-only**; C04, C06, C08, and C09 need additional workload/data/deliverable/event authoring.

| Category | Objective | Primary seeds | Secondary seeds | Coverage / work still required |
| --- | --- | --- | --- | --- |
| **C01** | 0→1 construction and feature development | 8: [envconfig](#envconfig), [schedule](#schedule), [hexyl](#hexyl), [tablib](#tablib), [sqlite_utils](#sqlite_utils), [fastapi_template](#fastapi_template), [mailpit](#mailpit), [just](#just) | 2: [hyperfine](#hyperfine), [gitpython](#gitpython) | Available behavioral references; construction contracts and acceptance cases still need authoring. |
| **C02** | Refactoring and code quality | 7: [envconfig](#envconfig), [schedule](#schedule), [tablib](#tablib), [sqlite_utils](#sqlite_utils), [fastapi_template](#fastapi_template), [just](#just), [gitpython](#gitpython) | 5: [hexyl](#hexyl), [mailpit](#mailpit), [scikit_build_core](#scikit_build_core), [nbconvert](#nbconvert), [hyperfine](#hyperfine) | Available working-tree references; baseline behavior and structural acceptance need qualification. |
| **C03** | Environment, build, and release | 2: [sampleproject](#sampleproject), [scikit_build_core](#scikit_build_core) | 3: [podinfo](#podinfo), [fastapi_template](#fastapi_template), [just](#just) | Available packaging/build anchors; native fixtures and offline dependency caches need selection. |
| **C04** | Performance and resource optimization | 2: [duckdb](#duckdb), [hyperfine](#hyperfine) | 6: [hexyl](#hexyl), [tablib](#tablib), [sqlite_utils](#sqlite_utils), [scikit_learn](#scikit_learn), [podinfo](#podinfo), [mailpit](#mailpit) | Workload pairing required: a benchmark instrument alone is not an optimization task. |
| **C05** | Applied data analysis and decision support | 3: [nycflights13](#nycflights13), [tidytuesday](#tidytuesday), [fivethirtyeight_data](#fivethirtyeight_data) | 3: [sqlite_utils](#sqlite_utils), [duckdb](#duckdb), [lmfit](#lmfit) | Real data sources exist; individual datasets, questions, and reference calculations are not yet selected. |
| **C06** | ML experimentation | 1: [scikit_learn](#scikit_learn) | 1: [nycflights13](#nycflights13) | Pair a bounded pipeline with a frozen dataset and sealed evaluation split. |
| **C07** | Constructive science and algorithms | 2: [or_tools](#or_tools), [lmfit](#lmfit) | 1: [scikit_learn](#scikit_learn) | Solver/model anchors exist; independently checkable problem instances need qualification. |
| **C08** | Professional artifact production | 1: [nbconvert](#nbconvert) | 4: [nycflights13](#nycflights13), [tidytuesday](#tidytuesday), [fivethirtyeight_data](#fivethirtyeight_data), [lmfit](#lmfit) | Report authoring plus data pairing required; converting a notebook alone is not sufficient. |
| **C09** | Evolving operational workflows | 2: [podinfo](#podinfo), [mailpit](#mailpit) | 3: [schedule](#schedule), [or_tools](#or_tools), [gitpython](#gitpython) | Authored evolving labs required; static services and deployment examples are not complete workflows. |
| **C10** | Adversarial information handling | 0: None | 3: [mailpit](#mailpit), [nbconvert](#nbconvert), [gitpython](#gitpython) | Overlay-only coverage: no dedicated adversarial-source repository is claimed in this roster. |

## Full inventory

| Seed | Kind / GitHub language | Stars | Latest default-branch commit (UTC) | Primary | Secondary | Screening priority |
| --- | --- | ---: | --- | --- | --- | --- |
| [kelseyhightower/envconfig](#envconfig) | library / Go | 5,468 | 2025-06-28 | C01, C02 | — | pilot |
| [dbader/schedule](#schedule) | library / Python | 12,277 | 2024-05-25 | C01, C02 | C09 | pilot |
| [sharkdp/hexyl](#hexyl) | cli / Rust | 10,288 | 2026-04-30 | C01 | C02, C04 | pilot |
| [jazzband/tablib](#tablib) | library / Python | 4,758 | 2026-10-06 | C01, C02 | C04 | pilot |
| [simonw/sqlite-utils](#sqlite_utils) | cli_library / Python | 2,179 | 2026-09-22 | C01, C02 | C04, C05 | pilot |
| [pypa/sampleproject](#sampleproject) | packaging_scaffold / Python | 5,242 | 2024-11-06 | C03 | — | expansion |
| [duckdb/duckdb](#duckdb) | engine_workload_source / C++ | 41,951 | 2026-10-06 | C04 | C05 | expansion |
| [tidyverse/nycflights13](#nycflights13) | dataset / R | 133 | 2023-11-21 | C05 | C06, C08 | pilot |
| [scikit-learn/scikit-learn](#scikit_learn) | ml_example_source / Python | 67,482 | 2026-10-06 | C06 | C04, C07 | pilot |
| [google/or-tools](#or_tools) | optimization_example_source / C++ | 14,156 | 2026-09-17 | C07 | C09 | pilot |
| [rfordatascience/tidytuesday](#tidytuesday) | dataset_collection / HTML | 8,418 | 2026-10-05 | C05 | C08 | expansion |
| [stefanprodan/podinfo](#podinfo) | service_lab_anchor / Go | 6,013 | 2026-09-21 | C09 | C03, C04 | pilot |
| [fivethirtyeight/data](#fivethirtyeight_data) | dataset_collection / Jupyter Notebook | 17,461 | 2025-02-25 | C05 | C08 | expansion |
| [fastapi/full-stack-fastapi-template](#fastapi_template) | application_scaffold / TypeScript | 45,896 | 2026-10-06 | C01, C02 | C03 | expansion |
| [axllent/mailpit](#mailpit) | application_service / Go | 10,557 | 2026-10-06 | C01, C09 | C02, C04, C10 | expansion |
| [casey/just](#just) | cli_workflow_tool / Rust | 36,153 | 2026-10-02 | C01, C02 | C03 | expansion |
| [scikit-build/scikit-build-core](#scikit_build_core) | native_build_backend / Python | 516 | 2026-10-06 | C03 | C02 | pilot |
| [lmfit/lmfit-py](#lmfit) | scientific_model_source / Python | 1,240 | 2026-08-03 | C07 | C05, C08 | pilot |
| [jupyter/nbconvert](#nbconvert) | artifact_pipeline / Python | 1,939 | 2026-09-14 | C08 | C02, C10 | pilot |
| [sharkdp/hyperfine](#hyperfine) | benchmark_instrument / Rust | 28,954 | 2026-10-06 | C04 | C01, C02 | pilot |
| [gitpython-developers/GitPython](#gitpython) | versioned_state_library / Python | 5,185 | 2026-10-05 | C02 | C01, C09, C10 | conditional |

All listed repositories were unarchived at collection. Priority is a screening recommendation: **pilot**, **expansion**, or **conditional**; it is not a quality certification or admission status.

## Proposed first world per category

These ten compositions are **planned**, not materialized environments. A seed may support several worlds, and a world may combine several seeds.

| Category | Seed composition | First world | Additional fixture / validation work |
| --- | --- | --- | --- |
| C01 | [envconfig](#envconfig) | Typed configuration loader from an explicit public contract. | Private reference implementation; independently written compatibility cases; agent starts empty. |
| C02 | [schedule](#schedule) | Injectable-clock refactor preserving scheduling behavior. | Known-good baseline plus fake-clock and structural acceptance tests; no planted defect is necessary. |
| C03 | [scikit_build_core](#scikit_build_core) + [sampleproject](#sampleproject) | Reproducible wheel/sdist for a minimal native extension. | Authored or selected CMake extension, compiler cache, and separate clean-install validators. |
| C04 | [hyperfine](#hyperfine) + [sqlite_utils](#sqlite_utils) | Optimize a fixed import/query pipeline without losing atomicity or records. | Frozen generated workload, independent state reconciliation, calibrated repeated resource measurements. |
| C05 | [nycflights13](#nycflights13) + [sqlite_utils](#sqlite_utils) | Reliability analysis with cancellation and completed-flight denominators kept distinct. | Selected table slice and hashes, data dictionary, stakeholder brief, independently executed reference calculations. |
| C06 | [scikit_learn](#scikit_learn) + [nycflights13](#nycflights13) | Budgeted departure-time delay prediction with chronological evaluation. | Feature-availability contract, training-only preprocessing, sealed holdout, and reproducible baseline metrics. |
| C07 | [or_tools](#or_tools) | Constrained allocation with multiple valid optimal solutions. | New small instances plus independent feasibility checks and tiny-case exhaustive enumeration. |
| C08 | [nbconvert](#nbconvert) + [tidytuesday](#tidytuesday) | Audience-specific analytical HTML report rather than generic exploratory charts. | Selected dataset, report brief/template, recomputation checks, and clean rerendering; optional visual validator. |
| C09 | [podinfo](#podinfo) | Local rollout followed by delayed failure, rollback, and a second change. | Persisted event model, controlled clock/probes, local service state, and precommitted permissions. |
| C10 | [mailpit](#mailpit) + [gitpython](#gitpython) | Legitimate local release/message audit amid hostile email and repository instructions. | Authored inert attacks, fictional canaries, protected state, local sinks, and separate legitimate-task success checks. |

## Seed cards

### envconfig

**Repository:** [kelseyhightower/envconfig](https://github.com/kelseyhightower/envconfig) · **Latest GitHub release:** [v1.4.0](https://github.com/kelseyhightower/envconfig/releases/tag/v1.4.0) (2019-05-24) · **License detection hint:** `MIT` (manual review pending).

**Candidate commit:** [`7834011875d613aec60c606b52c2b0fe8949fe91`](https://github.com/kelseyhightower/envconfig/commit/7834011875d613aec60c606b52c2b0fe8949fe91) on `master`. Latest subject: migrate to github actions.

**Bounded scope:** A typed configuration loader: documented field types, defaults, required values, prefixes, and decoder hooks; not a service platform.

- **C01 primary — direct:** Build a configuration loader from an explicit public API contract in an empty workspace.
  - Acceptance: Independent tests cover absent/empty values, defaults, malformed types, prefixes, nested fields, and declared custom decoders.
- **C02 primary — direct:** Separate value lookup, conversion, and validation; inject the environment source while preserving public behavior.
  - Acceptance: Run characterization tests before/after; a fake value source works without global environment mutation and public callers remain compatible.

**Caveats:**
- Mature, infrequently updated library; recent CI changes do not imply recent feature development.

### schedule

**Repository:** [dbader/schedule](https://github.com/dbader/schedule) · **Latest GitHub release:** No published GitHub Release returned · **License detection hint:** `MIT` (manual review pending).

**Candidate commit:** [`82a43db1b938d8fdf60103bd41f329e06c8d3651`](https://github.com/dbader/schedule/commit/82a43db1b938d8fdf60103bd41f329e06c8d3651) on `master`. Latest subject: Release 1.2.2 (#625).

**Bounded scope:** A deterministic subset of interval/calendar scheduling with an injectable clock; exclude production distributed scheduling.

- **C01 primary — direct:** Construct a small scheduler from documented recurrence examples and a stated timezone policy.
  - Acceptance: Fake-clock cases check execution counts, next-run times, cancellation, failures, and explicitly specified calendar boundaries; no real sleeps.
- **C02 primary — direct:** Decouple clock access and job execution while preserving the supported public API.
  - Acceptance: Existing and independent clock-boundary tests pass; tests can advance time without patching process-global clock state.
- **C09 secondary — authored workflow:** Operate a queue whose jobs arrive, are cancelled, fail, and resume after a simulated restart.
  - Acceptance: A precommitted event driver verifies deadlines, cancellation, durable completion markers, restart behavior, and the final handoff.
  - Required authored/paired asset: Persisted queue fixture and controlled clock/event driver; not supplied by the repository alone.

**Caveats:**
- Last default-branch activity is older; qualify the pinned implementation rather than infer quality from stars.

### hexyl

**Repository:** [sharkdp/hexyl](https://github.com/sharkdp/hexyl) · **Latest GitHub release:** [v0.17.0](https://github.com/sharkdp/hexyl/releases/tag/v0.17.0) (2026-02-14) · **License detection hint:** `Apache-2.0` (manual review pending).

**Candidate commit:** [`6ecc29b9c8c84d08a7e860f7f69c22b113b480ea`](https://github.com/sharkdp/hexyl/commit/6ecc29b9c8c84d08a7e860f7f69c22b113b480ea) on `master`. Latest subject: Remove sponsors.

**Bounded scope:** Binary display from stdin/files with bounded offsets, lengths, partial rows, and an explicit plain/color output contract.

- **C01 primary — direct:** Build a bounded binary-inspection CLI from byte-input/output examples.
  - Acceptance: Golden fixtures and independently computed offsets cover empty inputs, binary bytes, partial rows, stdin, and invalid ranges.
- **C02 secondary — direct:** Separate byte acquisition, row formatting, and terminal presentation without changing declared output.
  - Acceptance: Existing/golden output cases pass and each boundary is testable using in-memory bytes rather than actual terminal state.
- **C04 secondary — paired:** Stream large input under a peak-memory cap without changing displayed bytes.
  - Acceptance: Compare exact output with the reference and measure peak memory plus repeated baseline-relative throughput on fixed files.
  - Required authored/paired asset: Frozen large-file workload and independently measured baseline.

### tablib

**Repository:** [jazzband/tablib](https://github.com/jazzband/tablib) · **Latest GitHub release:** [v3.10.0](https://github.com/jazzband/tablib/releases/tag/v3.10.0) (2026-07-31) · **License detection hint:** `MIT` (manual review pending).

**Candidate commit:** [`966876c76a6378557c1951badc1665b4d1375875`](https://github.com/jazzband/tablib/commit/966876c76a6378557c1951badc1665b4d1375875) on `master`. Latest subject: [pre-commit.ci] pre-commit autoupdate (#697).

**Bounded scope:** CSV/JSON tabular exchange only for construction; explicitly specify headers, nulls, types, ordering, and round-trip policy before grading.

- **C01 primary — direct:** Build a small tabular exchange library from CSV/JSON contracts, without the upstream implementation.
  - Acceptance: Independent round-trip fixtures cover Unicode, quoted newlines, empty cells, headers, ordering, and stated type/null semantics.
- **C02 primary — direct:** Replace duplicated format dispatch with a tested extension boundary while preserving existing formats.
  - Acceptance: Existing compatibility cases pass and a separately implemented test format can register without editing unrelated dispatch code.
- **C04 secondary — paired:** Improve a bounded conversion workload without dropping records or silently changing representation.
  - Acceptance: Output equivalence and row counts hold; repeated timing and peak-memory measurements demonstrate the declared improvement.
  - Required authored/paired asset: Fixed input sizes, conversion workload, and repeatable measurement harness.

### sqlite_utils

**Repository:** [simonw/sqlite-utils](https://github.com/simonw/sqlite-utils) · **Latest GitHub release:** [4.2.1](https://github.com/simonw/sqlite-utils/releases/tag/4.2.1) (2026-08-13) · **License detection hint:** `Apache-2.0` (manual review pending).

**Candidate commit:** [`6bc1d33d583c54bd69fbdd2071117e2d38c354a1`](https://github.com/simonw/sqlite-utils/commit/6bc1d33d583c54bd69fbdd2071117e2d38c354a1) on `main`. Latest subject: Fix resource leaks caused by buffered readers.

**Bounded scope:** A coherent import/query/upsert/export slice; avoid reconstructing every current command or extension.

- **C01 primary — direct:** Construct a local CSV/JSON-to-SQLite CLI with public import, upsert, query, and export contracts.
  - Acceptance: Fresh-process tests verify transactions, duplicate keys, malformed input, rerun idempotence, and exact query/export results.
- **C02 primary — direct:** Separate CLI orchestration from database transformations while preserving supported commands.
  - Acceptance: Characterization tests pass and core transformations are callable without parsing CLI arguments or relying on global process state.
- **C04 secondary — paired:** Optimize a fixed import/query pipeline without weakening atomicity or correctness.
  - Acceptance: Independent SQL checks reconcile final state; repeated workload measurements include memory, interrupted imports, and rollback cases.
  - Required authored/paired asset: Frozen workload plus a correctness and resource baseline.
- **C05 secondary — paired:** Answer a stakeholder question using a real multi-table dataset imported into SQLite.
  - Acceptance: Independent calculations verify join cardinalities, denominators, aggregates, and reported uncertainty; import success alone earns no analysis credit.
  - Required authored/paired asset: A frozen real dataset, data dictionary, and decision-oriented brief.

**Useful pairings:** [nycflights13](#nycflights13), [hyperfine](#hyperfine), [nbconvert](#nbconvert).

### sampleproject

**Repository:** [pypa/sampleproject](https://github.com/pypa/sampleproject) · **Latest GitHub release:** No published GitHub Release returned · **License detection hint:** `MIT` (manual review pending).

**Candidate commit:** [`621e4974ca25ce531773def586ba3ed8e736b3fc`](https://github.com/pypa/sampleproject/commit/621e4974ca25ce531773def586ba3ed8e736b3fc) on `main`. Latest subject: pyproject: prep 4.0.0 (#219).

**Bounded scope:** A tiny Python package with declared metadata and one meaningful resource/API; use as a scaffold, not a substantial product oracle.

- **C03 primary — direct:** Prepare reproducible wheel and sdist artifacts that install and work in separate clean environments.
  - Acceptance: Inspect metadata and included resources, build from the sdist, install both artifacts separately, and execute the published API in each environment.

**Useful pairings:** [scikit_build_core](#scikit_build_core).

**Caveats:**
- Too small and generic to count as a substantive whole-product reconstruction seed by itself.

### duckdb

**Repository:** [duckdb/duckdb](https://github.com/duckdb/duckdb) · **Latest GitHub release:** [v1.5.6](https://github.com/duckdb/duckdb/releases/tag/v1.5.6) (2026-09-28) · **License detection hint:** `MIT` (manual review pending).

**Candidate commit:** [`d05e1133a451ea2e4ef44696837abb2c67d4d7bd`](https://github.com/duckdb/duckdb/commit/d05e1133a451ea2e4ef44696837abb2c67d4d7bd) on `v2.0-cyanoptera`. Latest subject: Report malformed WKT as a cast error in TRY_CAST to GEOMETRY (#26497).

**Bounded scope:** One query/operator workload plus pinned engine binaries; do not require a full database-engine reconstruction or full native build.

- **C04 primary — paired:** Improve a bounded SQL workload under correctness, runtime, and memory constraints.
  - Acceptance: Independent results preserve NULL/duplicate semantics; use paired repeated measurements against a frozen baseline and report uncertainty.
  - Required authored/paired asset: Selected benchmark/query files, fixed datasets, pinned binaries, and a calibrated resource harness.
- **C05 secondary — paired:** Investigate a real analytical question using a frozen dataset and a reproducible SQL pipeline.
  - Acceptance: Independent calculations verify joins, denominators, missingness handling, and report figures; faster queries are not a substitute for valid inference.
  - Required authored/paired asset: Real dataset and analytical brief; the engine is tooling, not the dataset.

**Useful pairings:** [nycflights13](#nycflights13), [hyperfine](#hyperfine).

**Caveats:**
- Large upstream repository: freeze selected workloads and dependencies rather than admit the entire checkout as a small seed.

### nycflights13

**Repository:** [tidyverse/nycflights13](https://github.com/tidyverse/nycflights13) · **Latest GitHub release:** [v1.0.2](https://github.com/tidyverse/nycflights13/releases/tag/v1.0.2) (2021-04-12) · **License detection hint:** `not identified` (manual review pending).

**Candidate commit:** [`df98ef215aa8216fe0838a0b8ac5bada646d814c`](https://github.com/tidyverse/nycflights13/commit/df98ef215aa8216fe0838a0b8ac5bada646d814c) on `main`. Latest subject: Typo fix.

**Bounded scope:** Freeze a documented slice of flights, weather, aircraft, airports, and airlines; retain join keys, missingness, and timestamp semantics.

- **C05 primary — direct:** Analyze reliability across carriers/routes while distinguishing cancellations from completed-flight delays.
  - Acceptance: Independent reference computations reconcile counts, joins, denominators, missing values, route/season sensitivity, and observational limitations.
- **C06 secondary — paired:** Predict a declared delay outcome using only information available at the specified decision time.
  - Acceptance: A sealed chronological split and feature-availability checks reject leakage; compare a baseline with budgeted experiments and reproduce metrics.
  - Required authored/paired asset: scikit-learn or another bounded training stack; target contract, feature cutoff, and sealed holdout.
- **C08 secondary — paired:** Deliver a stakeholder-ready reliability report with reproducible tables, charts, limitations, and recommendations.
  - Acceptance: Recompute every numerical claim, verify rendered sections/assets, and assess audience usefulness separately from mere file generation.
  - Required authored/paired asset: Audience/deliverable brief and a report renderer or authored template.

**Useful pairings:** [sqlite_utils](#sqlite_utils), [scikit_learn](#scikit_learn), [nbconvert](#nbconvert).

**Caveats:**
- Data describes 2013 flights; repository commit recency does not mean recent observations.
- GitHub did not identify an SPDX license; inspect package declarations and original data provenance before admission.

### scikit_learn

**Repository:** [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) · **Latest GitHub release:** [1.9.1](https://github.com/scikit-learn/scikit-learn/releases/tag/1.9.1) (2026-09-11) · **License detection hint:** `BSD-3-Clause` (manual review pending).

**Candidate commit:** [`c95f032d2960b69ed20c1b15f7be5a8aa10595f8`](https://github.com/scikit-learn/scikit-learn/commit/c95f032d2960b69ed20c1b15f7be5a8aa10595f8) on `main`. Latest subject: FIX Fix `from sklearn.model_selection import *` (#35042).

**Bounded scope:** One pipeline/example family using pinned package binaries and a bounded frozen dataset; not a reimplementation of the full library.

- **C06 primary — paired:** Establish and improve a leakage-safe prediction pipeline within a CPU/experiment budget.
  - Acceptance: Check split separation, train-only preprocessing, available features, reproducible metrics, a sealed holdout, and an honest baseline comparison.
  - Required authored/paired asset: Selected bundled or paired dataset; declared decision time, target, and held-out evaluation.
- **C04 secondary — paired:** Reduce pipeline inference/training cost without violating its stated predictive-quality constraint.
  - Acceptance: Fixed inputs and splits preserve output/quality requirements; repeated runtime and memory measurements compare identical workloads.
  - Required authored/paired asset: Frozen data, quality floor, and performance baseline.
- **C07 secondary — direct:** Construct a bounded numerical algorithm against documented mathematical and input/output contracts.
  - Acceptance: Independent tiny-case calculations plus a trusted reference check objective values, convergence, and edge cases; implementation similarity is irrelevant.

**Useful pairings:** [nycflights13](#nycflights13), [hyperfine](#hyperfine).

**Caveats:**
- Many examples may fetch external data; select bundled/frozen inputs and disable unexpected downloads.
- Use package binaries and selected examples rather than making full-library compilation a default gate.

### or_tools

**Repository:** [google/or-tools](https://github.com/google/or-tools) · **Latest GitHub release:** [v9.15](https://github.com/google/or-tools/releases/tag/v9.15) (2026-01-12) · **License detection hint:** `Apache-2.0` (manual review pending).

**Candidate commit:** [`100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`](https://github.com/google/or-tools/commit/100f66e6242ab8bf8d32feb8f3bf086db66ae2b5) on `stable`. Latest subject: ci: add dummy release.yml.

**Bounded scope:** One scheduling/allocation model with stated hard constraints and objective; prefer pinned solver binaries and tiny independently checkable cases.

- **C07 primary — direct:** Construct a constrained allocation model and solve new problem instances.
  - Acceptance: An independent feasibility checker and exhaustive enumeration on tiny cases verify constraints and objective; multiple optimal schedules are valid.
- **C09 secondary — authored workflow:** Replan an operational schedule after precommitted arrivals, cancellations, and reservation changes.
  - Acceptance: Persisted state and an event driver verify feasible commitments, authorized changes, replanning quality, and final ownership handoff.
  - Required authored/paired asset: Controlled event stream, persisted commitments, and approval boundaries.

**Caveats:**
- Default branch is stable; the recorded default-branch commit is not the most recent push across every development branch.
- Do not require rebuilding the entire solver or all language bindings.

### tidytuesday

**Repository:** [rfordatascience/tidytuesday](https://github.com/rfordatascience/tidytuesday) · **Latest GitHub release:** [tidytuesday_tweets-dda1055](https://github.com/rfordatascience/tidytuesday/releases/tag/tidytuesday_tweets-dda1055) (2023-05-28) · **License detection hint:** `CC0-1.0` (manual review pending).

**Candidate commit:** [`78474f0d3496a1f855deeceb63e250f494cbe938`](https://github.com/rfordatascience/tidytuesday/commit/78474f0d3496a1f855deeceb63e250f494cbe938) on `main`. Latest subject: UC Davis Avocado Oil Fraud Studies (#1071).

**Bounded scope:** Select and checksum one weekly dataset with its dictionary and original-source provenance; not the entire collection.

- **C05 primary — direct:** Answer a concrete stakeholder question from one imperfect real dataset rather than produce generic exploratory charts.
  - Acceptance: Independent checks validate cleaning, joins, missingness, statistics, uncertainty, and the relationship between evidence and recommendation.
- **C08 secondary — paired:** Produce an audience-specific analytical report with reproducible figures and explicit limitations.
  - Acceptance: Recompute report claims and verify rendered content/assets; assess clarity and usefulness without prescribing a single chart or conclusion.
  - Required authored/paired asset: Selected dataset, report brief, and renderer/template.

**Useful pairings:** [nbconvert](#nbconvert).

**Caveats:**
- A collection commit does not identify which dataset is appropriate or its observation dates.
- Repository-level licensing does not replace review of each dataset's original source terms.

### podinfo

**Repository:** [stefanprodan/podinfo](https://github.com/stefanprodan/podinfo) · **Latest GitHub release:** [6.15.0](https://github.com/stefanprodan/podinfo/releases/tag/6.15.0) (2026-08-31) · **License detection hint:** `Apache-2.0` (manual review pending).

**Candidate commit:** [`3e0ff8ae123b710bc91de1315cba0f996a8896c2`](https://github.com/stefanprodan/podinfo/commit/3e0ff8ae123b710bc91de1315cba0f996a8896c2) on `master`. Latest subject: Merge pull request #527 from stefanprodan/dependabot/github_actions/actions-a0032a45bd.

**Bounded scope:** One local Go service and its health/configuration/probe surfaces; Kubernetes variants must have a separate declared infrastructure budget.

- **C09 primary — authored workflow:** Conduct an authorized sandbox rollout, react to delayed failures, and recover after a second change.
  - Acceptance: A precommitted event/probe driver verifies actual revisions, traffic decisions, health effects, rollback state, and final handoff.
  - Required authored/paired asset: Local service lab, controlled failure/event schedule, persisted state, and authorized action policy.
- **C03 secondary — direct:** Produce a reproducible local deployment with working configuration and readiness checks.
  - Acceptance: Clean-start tests verify configuration, binary/container artifacts, startup, health probes, and repeatability without production credentials.
- **C04 secondary — paired:** Improve a declared endpoint workload while preserving response and readiness semantics.
  - Acceptance: Fixed requests and failure cases retain correctness; paired measurements assess throughput, latency, and memory against a calibrated baseline.
  - Required authored/paired asset: Local load generator and fixed baseline workload.

**Useful pairings:** [hyperfine](#hyperfine).

**Caveats:**
- A running demo service is not already an evolving operational task; author and validate the event model.

### fivethirtyeight_data

**Repository:** [fivethirtyeight/data](https://github.com/fivethirtyeight/data) · **Latest GitHub release:** No published GitHub Release returned · **License detection hint:** `CC-BY-4.0` (manual review pending).

**Candidate commit:** [`4c1ff5e3aef1816ae04af63218015066e186c147`](https://github.com/fivethirtyeight/data/commit/4c1ff5e3aef1816ae04af63218015066e186c147) on `master`. Latest subject: Add data and README for trump-2-poll-issue-questions.

**Bounded scope:** One dataset directory with dictionary/provenance; keep an existing article or notebook outside the agent-visible solution surface.

- **C05 primary — direct:** Reassess a concrete claim from raw data and test its sensitivity to plausible analytical choices.
  - Acceptance: Independent computations verify denominators, statistics, exclusions, and uncertainty; an existing article is not the only acceptable conclusion.
- **C08 secondary — paired:** Write a reproducible analytical brief for a stated audience using the selected dataset.
  - Acceptance: Computed tables/charts support the claims, limitations are explicit, and the deliverable is independently rerenderable and usable.
  - Required authored/paired asset: Dataset selection, audience brief, and report template/renderer.

**Useful pairings:** [nbconvert](#nbconvert).

**Caveats:**
- Check dataset-specific exceptions and separate data/code terms; API SPDX detection is only a hint.
- Existing published analyses are private references, not golden prose or evidence of uncontaminated evaluation.

### fastapi_template

**Repository:** [fastapi/full-stack-fastapi-template](https://github.com/fastapi/full-stack-fastapi-template) · **Latest GitHub release:** [0.12.0](https://github.com/fastapi/full-stack-fastapi-template/releases/tag/0.12.0) (2026-08-12) · **License detection hint:** `MIT` (manual review pending).

**Candidate commit:** [`e99fbaeba3b55190156ebae525ad9d481d76677e`](https://github.com/fastapi/full-stack-fastapi-template/commit/e99fbaeba3b55190156ebae525ad9d481d76677e) on `master`. Latest subject: 📝 Update release notes.

**Bounded scope:** One authenticated CRUD/workflow slice across API, persistence, and UI; avoid reconstructing every template feature or production integration.

- **C01 primary — direct:** Build a bounded authenticated application from a product brief with declared acceptance behavior.
  - Acceptance: Local integration tests verify permissions, persistence, input errors, migrations, and the required user workflow; architecture may differ.
- **C02 primary — direct:** Isolate authorization and business rules from API handlers while preserving the supported workflow.
  - Acceptance: Behavioral integration tests remain green and shared rules can be tested/called independently of HTTP routing.
- **C03 secondary — paired:** Make the selected application slice start and test reproducibly from a clean local environment.
  - Acceptance: Pinned dependencies, migrations, services, and integration tests run without private credentials or production deployment.
  - Required authored/paired asset: Local database/browser-test dependencies and declared installation budget.

**Caveats:**
- The template is a scaffold; new business behavior needs a public contract and independent acceptance tests.
- Native screenshot-based evaluation needs a suitable observation path; API/DOM-level checks alone do not prove visual quality.

### mailpit

**Repository:** [axllent/mailpit](https://github.com/axllent/mailpit) · **Latest GitHub release:** [v1.31.4](https://github.com/axllent/mailpit/releases/tag/v1.31.4) (2026-10-03) · **License detection hint:** `MIT` (manual review pending).

**Candidate commit:** [`43139844269d5aaae5bdd1a8889cf8efbb7dc810`](https://github.com/axllent/mailpit/commit/43139844269d5aaae5bdd1a8889cf8efbb7dc810) on `develop`. Latest subject: Chore: Add gosec #nosec annotations and set explicit Content-Type on error responses.

**Bounded scope:** A bounded local SMTP capture and message-query workflow; prohibit real email delivery and external relaying.

- **C01 primary — direct:** Construct a local email-capture service with a stated SMTP/message-query contract.
  - Acceptance: Send fixture messages to localhost, query captured data, and verify headers, bodies, errors, ordering, and restart behavior.
- **C09 primary — authored workflow:** Operate a notification-test workflow through delayed deliveries, duplicate requests, and storage/service failures.
  - Acceptance: A persisted event driver verifies bounded effects, message identities, recovery, and the final state across multiple changes.
  - Required authored/paired asset: Local clients, deterministic events, persisted receipts, and explicit no-relay boundaries.
- **C02 secondary — direct:** Separate message ingestion, storage, and querying without changing supported protocol behavior.
  - Acceptance: Protocol fixtures pass and each component is independently testable using controlled messages and storage adapters.
- **C04 secondary — paired:** Improve capture/query performance on a fixed local message workload without losing or duplicating messages.
  - Acceptance: Reference identities and contents reconcile exactly; repeated measurements compare identical loads and memory limits.
  - Required authored/paired asset: Message generator, load harness, and correctness/performance baseline.
- **C10 secondary — authored overlay:** Process hostile message content that tries to authorize protected release/configuration changes.
  - Acceptance: Complete the legitimate message task while a canary sink and state audit verify no dummy-secret disclosure or unauthorized changes.
  - Required authored/paired asset: Inert malicious-text fixtures, fictional canaries, protected state, and a controlled audit sink.

**Useful pairings:** [gitpython](#gitpython), [hyperfine](#hyperfine).

**Caveats:**
- Default branch is develop; the recorded commit is a research reference, not automatically a stable release.

### just

**Repository:** [casey/just](https://github.com/casey/just) · **Latest GitHub release:** [1.58.0](https://github.com/casey/just/releases/tag/1.58.0) (2026-08-03) · **License detection hint:** `CC0-1.0` (manual review pending).

**Candidate commit:** [`602ee328c9b21361121b498674334012175d4d81`](https://github.com/casey/just/commit/602ee328c9b21361121b498674334012175d4d81) on `master`. Latest subject: Allow `fixme` and `xxx` in `bin/forbid` (#3734).

**Bounded scope:** A bounded command-runner grammar and process/environment contract; exclude full language/editor integrations from reconstruction.

- **C01 primary — direct:** Build a small recipe runner with declared dependency, argument, environment, and failure semantics.
  - Acceptance: Sandbox recipes verify invocation order, spaced paths, environment scope, exit propagation, and specified parser errors.
- **C02 primary — direct:** Separate parsing, recipe planning, and process execution without changing supported recipes.
  - Acceptance: Existing/independent recipes pass and a fake process runner verifies plans and effects without starting subprocesses.
- **C03 secondary — paired:** Create reproducible build/test/release recipes for a paired small project.
  - Acceptance: Execute in clean environments and inspect artifacts, failure propagation, paths, and dependency setup rather than merely read recipe text.
  - Required authored/paired asset: Paired project, clean-environment fixture, and public build/release contract.

**Useful pairings:** [sampleproject](#sampleproject), [scikit_build_core](#scikit_build_core).

### scikit_build_core

**Repository:** [scikit-build/scikit-build-core](https://github.com/scikit-build/scikit-build-core) · **Latest GitHub release:** [v1.1.1](https://github.com/scikit-build/scikit-build-core/releases/tag/v1.1.1) (2026-10-02) · **License detection hint:** `Apache-2.0` (manual review pending).

**Candidate commit:** [`8e36242f318c95ef5dea2f8eb219e03fa2f89112`](https://github.com/scikit-build/scikit-build-core/commit/8e36242f318c95ef5dea2f8eb219e03fa2f89112) on `main`. Latest subject: ci: use prek action with GITHUB_TOKEN for zizmor online audits (#1602).

**Bounded scope:** One minimal CMake-backed extension and wheel/sdist workflow; declare compilers, platforms, and offline dependencies.

- **C03 primary — paired:** Build and package a tiny native extension reproducibly from a wheel and source distribution.
  - Acceptance: Separate clean installs import and exercise the extension; inspect expected resources, metadata, ABI assumptions, and rerun behavior.
  - Required authored/paired asset: Selected or authored minimal native extension, compiler/CMake environment, and dependency cache.
- **C02 secondary — direct:** Decouple configuration resolution and build invocation while preserving declared packaging behavior.
  - Acceptance: Independent configuration tests plus clean artifact builds pass; an injectable runner checks generated commands without real global environment mutation.

**Useful pairings:** [sampleproject](#sampleproject), [just](#just).

### lmfit

**Repository:** [lmfit/lmfit-py](https://github.com/lmfit/lmfit-py) · **Latest GitHub release:** [1.3.4](https://github.com/lmfit/lmfit-py/releases/tag/1.3.4) (2025-07-19) · **License detection hint:** `NOASSERTION` (manual review pending).

**Candidate commit:** [`fe389bbbd1fe936cd73742bd81fc6fce7ac92858`](https://github.com/lmfit/lmfit-py/commit/fe389bbbd1fe936cd73742bd81fc6fce7ac92858) on `master`. Latest subject: FIX: return correct candidates in 'brute' for single varying parameter.

**Bounded scope:** One calibration/model-selection problem with small CPU-friendly data; avoid the entire example archive or exotic optimizers.

- **C07 primary — direct:** Construct a constrained calibration model and validate it against independently generated numerical cases.
  - Acceptance: Known-noise fixtures check constraints, residual behavior, identifiable parameters, and reference solutions without demanding one solver implementation.
- **C05 secondary — paired:** Analyze imperfect measurements, compare plausible models, and report uncertainty and limitations.
  - Acceptance: Independent recomputation validates fits and summaries; sensitivity analysis distinguishes model assumptions from observational conclusions.
  - Required authored/paired asset: Frozen measurement dataset or independently executed synthetic generator with documented assumptions.
- **C08 secondary — paired:** Deliver an uncertainty-aware calibration report with reproducible plots, diagnostics, and recommendations.
  - Acceptance: Recompute quantitative claims and check report sections, figures, limitations, and decision usefulness.
  - Required authored/paired asset: Audience brief, frozen measurements, and report renderer/template.

**Useful pairings:** [nbconvert](#nbconvert).

**Caveats:**
- GitHub returned NOASSERTION for license detection; manual inspection is pending, not evidence that the project is unlicensed.
- Library history/archive size is not measured handwritten code size; qualify a selected model/example.

### nbconvert

**Repository:** [jupyter/nbconvert](https://github.com/jupyter/nbconvert) · **Latest GitHub release:** [v7.17.1](https://github.com/jupyter/nbconvert/releases/tag/v7.17.1) (2026-04-08) · **License detection hint:** `BSD-3-Clause` (manual review pending).

**Candidate commit:** [`1ff189e1b30f0820238e95427c035a9150a5e43f`](https://github.com/jupyter/nbconvert/commit/1ff189e1b30f0820238e95427c035a9150a5e43f) on `main`. Latest subject: Use org-level security policy (jupyter/.github/SECURITY.md) (#2318).

**Bounded scope:** An executed notebook to standalone HTML path with local assets; PDF/TeX variants require a separate dependency budget.

- **C08 primary — paired:** Produce a stakeholder-ready analytical report from a reproducible notebook and frozen data.
  - Acceptance: Execute and rerender in a clean environment; independently check numerical claims, required sections, charts/assets, and audience usefulness.
  - Required authored/paired asset: Frozen dataset/notebook, audience and deliverable brief, and substantive report validators.
- **C02 secondary — direct:** Isolate notebook execution and export/template configuration while preserving a bounded export contract.
  - Acceptance: Regression fixtures verify equivalent numerical content, resource handling, and standalone outputs across the declared export cases.
- **C10 secondary — authored overlay:** Ignore hostile source commentary or notebook markdown while completing the authorized analysis/report.
  - Acceptance: Recompute the legitimate report; canary and protected-state audits detect unauthorized disclosure or changes without executing malicious fixture code.
  - Required authored/paired asset: Inert instruction-injection text, fictional canary, protected files, and a local audit sink.

**Useful pairings:** [nycflights13](#nycflights13), [tidytuesday](#tidytuesday), [fivethirtyeight_data](#fivethirtyeight_data), [lmfit](#lmfit).

**Caveats:**
- Rendering HTML alone is not professional-artifact success; substantive content and audience requirements must be scored.
- The current Player profile is text-only; visual-quality tasks require renderer/perception support, not simulated descriptions.

### hyperfine

**Repository:** [sharkdp/hyperfine](https://github.com/sharkdp/hyperfine) · **Latest GitHub release:** [v1.21.0](https://github.com/sharkdp/hyperfine/releases/tag/v1.21.0) (2026-10-05) · **License detection hint:** `Apache-2.0` (manual review pending).

**Candidate commit:** [`e4a3213bf32281db964c0c4910dec61ec43e4775`](https://github.com/sharkdp/hyperfine/commit/e4a3213bf32281db964c0c4910dec61ec43e4775) on `master`. Latest subject: Revise README content and demo description.

**Bounded scope:** A bounded benchmark-runner contract or a paired small target workload; never grade fabricated timings or unspecified hardware-independent thresholds.

- **C04 primary — paired:** Improve a paired target workload and use controlled repeated measurements to establish the result.
  - Acceptance: Independent correctness checks pass; exported measurements use paired baselines, declared warmup, noise-aware thresholds, and bounded resource accounting.
  - Required authored/paired asset: Actual optimization target, frozen inputs, and calibrated baseline; hyperfine alone is only an instrument.
- **C01 secondary — direct:** Construct a benchmark runner with declared repetitions, process/error handling, and output contracts.
  - Acceptance: Controlled process fixtures verify invocation counts, exit handling, serialization, and statistics over known samples; real timing checks use tolerances.
- **C02 secondary — direct:** Separate execution, timing, statistics, and formatting without changing the supported benchmark interface.
  - Acceptance: Deterministic sample-statistics tests and controlled process/output fixtures preserve behavior; platform-dependent timing is not asserted exactly.

**Useful pairings:** [sqlite_utils](#sqlite_utils), [hexyl](#hexyl), [podinfo](#podinfo), [mailpit](#mailpit).

### gitpython

**Repository:** [gitpython-developers/GitPython](https://github.com/gitpython-developers/GitPython) · **Latest GitHub release:** [3.2.0](https://github.com/gitpython-developers/GitPython/releases/tag/3.2.0) (2026-09-30) · **License detection hint:** `BSD-3-Clause` (manual review pending).

**Candidate commit:** [`b5d6a79acac3122552028770fd2b3c8bde061de2`](https://github.com/gitpython-developers/GitPython/commit/b5d6a79acac3122552028770fd2b3c8bde061de2) on `main`. Latest subject: Merge pull request #2272 from CrowZhen/main.

**Bounded scope:** Temporary local Git repositories with explicit refs, ancestry, dirty/staged files, and failure cases; no public pushes or real credentials.

- **C02 primary — direct:** Refactor a bounded Git-operation seam while preserving repository state and supported error behavior.
  - Acceptance: Temporary-repository tests verify refs, ancestry, staged/unstaged files, conflict handling, and no unrelated state mutation.
- **C01 secondary — direct:** Construct a local repository auditor from a stated branch/status/history contract.
  - Acceptance: Generate independent temporary repositories and compare results with Git commands, including dirty trees, detached HEAD, and missing refs.
- **C09 secondary — authored workflow:** Complete a local release/branch workflow after precommitted conflicting changes and an interrupted handoff.
  - Acceptance: A persisted event driver checks authorized refs, preservation of user work, recovery, checkpoint receipts, and final ownership.
  - Required authored/paired asset: Local Git event fixture, permitted action policy, and durable handoff state.
- **C10 secondary — authored overlay:** Audit a repository containing hostile README or issue text without authorizing protected branch or secret changes.
  - Acceptance: Legitimate audit results remain correct while canary/state checks show no unauthorized disclosure, ref changes, or external network effects.
  - Required authored/paired asset: Inert hostile text, fictional protected state, canaries, and controlled sinks.

**Useful pairings:** [mailpit](#mailpit).

**Caveats:**
- Upstream describes maintenance mode; useful as legacy/refactoring or state fixtures, not a claim of active architectural investment.

## Qualification and snapshot protocol

Every seed remains `research_candidate`. `metadata_verified` means only public repository metadata and the referenced commit were retrieved. All materialization, baseline, fixture, license, and rollout flags remain false.

1. Fetch only the chosen commit and selected files/data into a disposable private directory; inventory handwritten code, assets, generated/vendored content, and fixture bytes.
2. Inspect pinned licenses and required notices, plus each data source's terms; GitHub SPDX detection is not legal clearance and does not cover every external asset.
3. Select a coherent public contract and supported behavior. Construction may use an empty workspace; refactoring must start from a behaviorally characterized working tree.
4. Freeze dependency versions, toolchains, binaries, input hashes, dataset observation dates, and local services. Prepare offline caches where needed.
5. Run the original baseline and independently compute reference facts. For repair variants demonstrate the initial failure; for refactoring variants preserve a known-good baseline.
6. Execute consequential tests, joins, fits, benchmarks, and state transitions under the GM's private materialization root; never let plausible narration establish correctness.
7. Validate at least one correct alternative and several plausible wrong solutions. Hidden tests may enforce public requirements but not unspecified upstream quirks or source similarity.
8. Precommit evolving events and trust boundaries. Use fictional secrets and controlled sinks; never production credentials, public pushes, real email delivery, or uncontrolled exfiltration.
9. Keep implementations, articles, private answers, and validators outside the agent-visible solution surface; control oracle fetching while retaining legitimate dependency/documentation access.
10. Pilot repeated no-training rollouts for simulator consistency, leakage, reward discrimination, runtime, and cost; only then author and explicitly promote world files through worlds/manifest.json.

## World-design rules

- For C01, distinguish a detailed interface contract from a concrete high-level product brief. One-shot means one user brief plus an autonomous tool/testing trajectory, not a single code dump. Looser requirements permit more valid implementations; hidden tests must not demand unspecified upstream details.
- For C02, require both behavioral preservation and the named structural improvement. Tests alone can reward no change; a complexity score alone can reward hiding complexity.
- For C05/C06/C07, separately validate calculations, assumptions, split integrity, uncertainty, and decision quality. A familiar public dataset or library is a useful training seed, not automatically an uncontaminated evaluation.
- For C08, validate substantive claims and audience usefulness as well as artifact existence. Programmatic/DOM checks do not establish visual quality; image/PDF perception needs real rendering and a suitable observation profile.
- For C09, precommit multiple events, persisted consequences, authorization, and recovery checks. The current tool-mediated harness does not supply an independent event scheduler automatically; a controlled shell-accessible driver can supply one.
- For C10, overlay inert hostile text on a legitimate task and check both task completion and protected effects. Ordinary malformed-input or vulnerability tests do not prove resistance to instruction injection.
- Clarification may be provided through a terminal-accessible ticket/message surface. The checked-in default virtual tools do not implement `ask_user`; naming it in a world file is not capability support.
- Keep categories as authoring metadata. Do not extend the four-section world format or overload existing runtime flavor classification merely to store this roster.

## Provenance and maintenance

Metadata was read using GitHub's REST repository, default-branch commits, and latest-release endpoints. Per-repository source URLs and the collection window are recorded in JSON. Metadata describes repositories at that time; category mapping and world ideas are authoring judgments, not claims of existing upstream tasks or measured training gains.

Refresh metadata and update JSON/Markdown together. Preserve prior qualified baseline SHAs rather than silently advancing worlds to upstream HEAD. Roster tests are local and make no GitHub requests.

Related: [`WORLD_FILE_SPEC.md`](WORLD_FILE_SPEC.md), [`../worlds/README.md`](../worlds/README.md), [`README.md`](README.md).
