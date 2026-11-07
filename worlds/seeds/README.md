# Promoted seed-grounded worlds

This collection adds **23 world specifications** using **all 21 sourced seeds** across **all 10 task categories**. It includes **3 full-small-project reproductions** and **20 bounded subsets/workflows**.

**All 23 worlds are promoted in [`../manifest.json`](../manifest.json) after
static contract review. Environment provisioning and execution qualification
remain DEFERRED.** Promotion and coverage checks do not establish episode
solvability, performance results, or training success.

## Contract and judging

- Each Markdown contract has exactly one title and the four ordered sections `World`, `Player`, `Reward`, and `Termination conditions`.
- Only `Player` is projected into the agent prompt. Private `World` sections hold immutable source references, selected scope, and substantive seed roles.
- The GM stays in control. A seed is one potential example solution or evidence source—not a golden patch, architecture, command path, or hidden requirement. Public outcomes and credible evidence govern grading; correct alternatives and supported negative findings are valid.
- Full repro means the whole usable small product declared publicly, including artifacts, examples, tests, and docs. It does not secretly require undeclared upstream features or byte-for-byte source/CLI styling.
- Dataset seeds fix input evidence and its documented scope, not the analytical conclusion. Software references may be combined or replaced by independently correct implementations.
- Build/release remains a task category, but no environment provisioning, containers, dependencies, hardware integration, upstream builds, or GM episodes are implemented or certified here.

## World inventory

| World | Categories | Task size | Meaningful seed references |
| --- | --- | --- | --- |
| [seed_envconfig_rebuild_001](seed_envconfig_rebuild_001.md) — Build a complete typed configuration package | C01 | full_small_project | envconfig |
| [seed_schedule_rebuild_001](seed_schedule_rebuild_001.md) — Build a complete clock-testable scheduler | C01 | full_small_project | schedule |
| [seed_hexyl_rebuild_001](seed_hexyl_rebuild_001.md) — Build a complete binary-inspection command | C01 | full_small_project | hexyl |
| [seed_tablib_format_refactor_001](seed_tablib_format_refactor_001.md) — Isolate the tabular exchange extension boundary | C02 | bounded_subset | tablib |
| [seed_sqlite_import_tool_001](seed_sqlite_import_tool_001.md) — Deliver a transactional local ledger importer | C01 | bounded_subset | sqlite_utils |
| [seed_sampleproject_release_001](seed_sampleproject_release_001.md) — Prepare the field sample release artifacts | C03 | bounded_subset | sampleproject |
| [seed_duckdb_query_optimization_001](seed_duckdb_query_optimization_001.md) — Faster monthly order rollup, same answers | C04 | bounded_subset | duckdb, hyperfine |
| [seed_flights_reliability_analysis_001](seed_flights_reliability_analysis_001.md) — A reproducible flight-reliability investigation | C05 | bounded_subset | nycflights13, sqlite_utils |
| [seed_flights_prediction_pipeline_001](seed_flights_prediction_pipeline_001.md) — A delay prediction that could have been made before departure | C06 | bounded_subset | scikit_learn, nycflights13 |
| [seed_constraint_schedule_solver_001](seed_constraint_schedule_solver_001.md) — A small workshop schedule with checkable optimality | C07 | bounded_subset | or_tools |
| [seed_spaceflight_report_001](seed_spaceflight_report_001.md) — A museum-ready report on human spaceflight | C08 | bounded_subset | tidytuesday, nbconvert |
| [seed_service_rollout_recovery_001](seed_service_rollout_recovery_001.md) — Roll out, observe, and recover a sandbox service | C09 | bounded_subset | podinfo |
| [seed_airline_safety_analysis_001](seed_airline_safety_analysis_001.md) — What historical airline incident counts can actually support | C05 | bounded_subset | fivethirtyeight_data |
| [seed_project_permissions_feature_001](seed_project_permissions_feature_001.md) — Add project collaboration behind one permission boundary | C01, C02 | bounded_subset | fastapi_template |
| [seed_local_mailbox_service_001](seed_local_mailbox_service_001.md) — Build a bounded local test mailbox | C01 | bounded_subset | mailpit |
| [seed_recipe_runner_rebuild_001](seed_recipe_runner_rebuild_001.md) — Build a bounded executable recipe runner | C01 | bounded_subset | just |
| [seed_native_package_contract_001](seed_native_package_contract_001.md) — Package a native arithmetic module with a complete source release | C03 | bounded_subset | scikit_build_core, sampleproject |
| [seed_spectral_calibration_001](seed_spectral_calibration_001.md) — Two overlapping spectral lines and an honest calibration decision | C07, C05 | bounded_subset | lmfit |
| [seed_report_export_refactor_001](seed_report_export_refactor_001.md) — Separate report execution, rendering, and publication | C02, C08 | bounded_subset | nbconvert |
| [seed_benchmark_runner_rebuild_001](seed_benchmark_runner_rebuild_001.md) — Build an honest bounded benchmark runner | C01 | bounded_subset | hyperfine |
| [seed_git_state_refactor_001](seed_git_state_refactor_001.md) — Make repository status a read-only snapshot boundary | C02 | bounded_subset | gitpython |
| [seed_untrusted_release_messages_001](seed_untrusted_release_messages_001.md) — Finish a release despite untrusted messages | C10, C09 | bounded_subset | mailpit, gitpython |
| [seed_untrusted_report_sources_001](seed_untrusted_report_sources_001.md) — Produce an astronaut report from untrusted sources | C10, C08 | bounded_subset | nbconvert, tidytuesday |

## Category coverage

| Category | Objective | Corresponding worlds |
| --- | --- | --- |
| C01 | 0→1 construction and feature development | [seed_envconfig_rebuild_001](seed_envconfig_rebuild_001.md), [seed_schedule_rebuild_001](seed_schedule_rebuild_001.md), [seed_hexyl_rebuild_001](seed_hexyl_rebuild_001.md), [seed_sqlite_import_tool_001](seed_sqlite_import_tool_001.md), [seed_project_permissions_feature_001](seed_project_permissions_feature_001.md), [seed_local_mailbox_service_001](seed_local_mailbox_service_001.md), [seed_recipe_runner_rebuild_001](seed_recipe_runner_rebuild_001.md), [seed_benchmark_runner_rebuild_001](seed_benchmark_runner_rebuild_001.md) |
| C02 | Refactoring and code quality | [seed_tablib_format_refactor_001](seed_tablib_format_refactor_001.md), [seed_project_permissions_feature_001](seed_project_permissions_feature_001.md), [seed_report_export_refactor_001](seed_report_export_refactor_001.md), [seed_git_state_refactor_001](seed_git_state_refactor_001.md) |
| C03 | Environment, build, and release | [seed_sampleproject_release_001](seed_sampleproject_release_001.md), [seed_native_package_contract_001](seed_native_package_contract_001.md) |
| C04 | Performance and resource optimization | [seed_duckdb_query_optimization_001](seed_duckdb_query_optimization_001.md) |
| C05 | Applied data analysis and decision support | [seed_flights_reliability_analysis_001](seed_flights_reliability_analysis_001.md), [seed_airline_safety_analysis_001](seed_airline_safety_analysis_001.md), [seed_spectral_calibration_001](seed_spectral_calibration_001.md) |
| C06 | ML experimentation | [seed_flights_prediction_pipeline_001](seed_flights_prediction_pipeline_001.md) |
| C07 | Constructive science and algorithms | [seed_constraint_schedule_solver_001](seed_constraint_schedule_solver_001.md), [seed_spectral_calibration_001](seed_spectral_calibration_001.md) |
| C08 | Professional artifact production | [seed_spaceflight_report_001](seed_spaceflight_report_001.md), [seed_report_export_refactor_001](seed_report_export_refactor_001.md), [seed_untrusted_report_sources_001](seed_untrusted_report_sources_001.md) |
| C09 | Evolving operational workflows | [seed_service_rollout_recovery_001](seed_service_rollout_recovery_001.md), [seed_untrusted_release_messages_001](seed_untrusted_release_messages_001.md) |
| C10 | Adversarial information handling | [seed_untrusted_release_messages_001](seed_untrusted_release_messages_001.md), [seed_untrusted_report_sources_001](seed_untrusted_report_sources_001.md) |

## Every seed has an actual role

The index quotes each role and selected scope from the corresponding **private World section**; a repository mention alone does not count. Its source URL and exact roster SHA are also checked against that section.

| Seed / canonical repository | Corresponding worlds | First substantive role |
| --- | --- | --- |
| `envconfig` / `kelseyhightower/envconfig` | [seed_envconfig_rebuild_001](seed_envconfig_rebuild_001.md) | The envconfig seed supplies a substantive example of a complete typed configuration package whose naming rules, conversions, validation, and decoder extension points serve real application callers. |
| `schedule` / `dbader/schedule` | [seed_schedule_rebuild_001](seed_schedule_rebuild_001.md) | The schedule seed supplies a substantive example of a usable recurring-job library, informing the public registration, recurrence, cancellation, and inspection capabilities without imposing its internal timing choices. |
| `hexyl` / `sharkdp/hexyl` | [seed_hexyl_rebuild_001](seed_hexyl_rebuild_001.md) | The hexyl seed supplies a substantive example of an end-to-end binary inspection command, connecting real byte acquisition and range selection to a readable display rather than serving only as a formatting reference. |
| `tablib` / `jazzband/tablib` | [seed_tablib_format_refactor_001](seed_tablib_format_refactor_001.md) | Tablib's inspected Registry and import/export descriptors ground a real format-extension interface, while the task-specific caller supplies the new requirement for isolated registrations without prescribing that upstream architecture. |
| `sqlite_utils` / `simonw/sqlite-utils` | [seed_sqlite_import_tool_001](seed_sqlite_import_tool_001.md), [seed_flights_reliability_analysis_001](seed_flights_reliability_analysis_001.md) | The inspected sqlite-utils examples ground a practical local import/query/export interface and primary-key workflow; this task defines its own bounded transaction and type policies rather than pretending the README proves undocumented upstream behavior. |
| `sampleproject` / `pypa/sampleproject` | [seed_sampleproject_release_001](seed_sampleproject_release_001.md), [seed_native_package_contract_001](seed_native_package_contract_001.md) | The inspected sampleproject metadata, sample console entry point, and add_one function ground the small package's compatibility surface; the requested bundled defaults and artifact contract make this a substantive bounded release task rather than a whole-product reconstruction. |
| `duckdb` / `duckdb/duckdb` | [seed_duckdb_query_optimization_001](seed_duckdb_query_optimization_001.md) | DuckDB supplies the analytical SQL and frozen-table query setting in which the Player must improve a real rollup without changing its relational semantics. |
| `nycflights13` / `tidyverse/nycflights13` | [seed_flights_reliability_analysis_001](seed_flights_reliability_analysis_001.md), [seed_flights_prediction_pipeline_001](seed_flights_prediction_pipeline_001.md) | The nycflights13 seed supplies the historical observations and documented join/time semantics against which the agent's denominators, comparisons, and claims can be independently checked. |
| `scikit_learn` / `scikit-learn/scikit-learn` | [seed_flights_prediction_pipeline_001](seed_flights_prediction_pipeline_001.md) | Scikit-learn supplies a bounded mixed-type preprocessing and classification pattern that must be adapted to train-only fitting and chronological evaluation rather than copying the example's random split. |
| `or_tools` / `google/or-tools` | [seed_constraint_schedule_solver_001](seed_constraint_schedule_solver_001.md) | OR-Tools supplies a compact constructive job-shop modeling pattern whose intervals, precedence constraints, and makespan objective can be tested independently on bounded workshop instances rather than reproducing the solver suite. |
| `tidytuesday` / `rfordatascience/tidytuesday` | [seed_spaceflight_report_001](seed_spaceflight_report_001.md), [seed_untrusted_report_sources_001](seed_untrusted_report_sources_001.md) | The TidyTuesday seed provides the actual astronaut-mission observations, historical cutoff, and dictionary needed to distinguish participant counts, mission records, and repeated cumulative quantities. |
| `podinfo` / `stefanprodan/podinfo` | [seed_service_rollout_recovery_001](seed_service_rollout_recovery_001.md) | The documented separation of application responses from healthy probes anchors a recovery scene where routing decisions must follow real requests and survive a control-session restart. |
| `fivethirtyeight_data` / `fivethirtyeight/data` | [seed_airline_safety_analysis_001](seed_airline_safety_analysis_001.md) | The FiveThirtyEight seed supplies a real, small historical dataset and its measurement definitions; the published article is at most one potential analytical example, not the editor's required conclusion. |
| `fastapi_template` / `fastapi/full-stack-fastapi-template` | [seed_project_permissions_feature_001](seed_project_permissions_feature_001.md) | The inspected FastAPI template item routes ground authenticated persistence and correct owner/superuser checks; the authored project membership feature extends that working model and motivates extracting shared authorization without alleging an upstream access-control defect. |
| `mailpit` / `axllent/mailpit` | [seed_local_mailbox_service_001](seed_local_mailbox_service_001.md), [seed_untrusted_release_messages_001](seed_untrusted_release_messages_001.md) | Mailpit's capture-and-query product description anchors a small local developer mailbox with observable delivery receipts and durable message inspection rather than a reconstruction of the full upstream application. |
| `just` / `casey/just` | [seed_recipe_runner_rebuild_001](seed_recipe_runner_rebuild_001.md) | The just seed supplies a substantive example of a recipe tool that parses a task file, plans dependencies, and executes real commands with meaningful argument, environment, working-directory, and failure behavior. |
| `scikit_build_core` / `scikit-build/scikit-build-core` | [seed_native_package_contract_001](seed_native_package_contract_001.md) | The inspected scikit-build-core example grounds how a real native module is installed into a Python package and how wheel/sdist contents are declared; it supplies a possible build interface, not an assumed compiler environment or a mandated backend. |
| `lmfit` / `lmfit/lmfit-py` | [seed_spectral_calibration_001](seed_spectral_calibration_001.md) | Lmfit supplies named bounded parameters, noise-weighted residual fitting, and uncertainty exploration for a small calibration whose scientific recommendation depends on independently checked predictions and correlated parameter uncertainty. |
| `nbconvert` / `jupyter/nbconvert` | [seed_spaceflight_report_001](seed_spaceflight_report_001.md), [seed_report_export_refactor_001](seed_report_export_refactor_001.md), [seed_untrusted_report_sources_001](seed_untrusted_report_sources_001.md) | The nbconvert seed provides one example report-production path with reproducible computation and asset handling; the report's substance and audience usefulness, not that exporter, determine success. |
| `hyperfine` / `sharkdp/hyperfine` | [seed_duckdb_query_optimization_001](seed_duckdb_query_optimization_001.md), [seed_benchmark_runner_rebuild_001](seed_benchmark_runner_rebuild_001.md) | Hyperfine supplies repeated timing, explicit warmup and execution boundaries, and raw JSON samples; its command-order behavior motivates counterbalanced paired blocks rather than a ceremonial speed table. |
| `gitpython` / `gitpython-developers/GitPython` | [seed_git_state_refactor_001](seed_git_state_refactor_001.md), [seed_untrusted_release_messages_001](seed_untrusted_release_messages_001.md) | The inspected GitPython tutorial grounds attached and detached HEAD observations and separate staged, unstaged, and untracked state; the task-specific auditor caller supplies a new immutable snapshot seam without asserting a defect in upstream GitPython. |

## Inspect and validate promotion and provenance

The machine-readable index is [`coverage.json`](coverage.json). It records world identities, canonical content and Player-projection hashes, category/size mappings, selected source scopes, and literal private role evidence. Its index-relative core manifest reference connects each seed identity to the same promoted file path.

From the checkout:

```bash
PYTHONPATH=src python3 -m shellsmith.campaign.seed_world_catalog
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_seed_worlds.py" -v
```

The checker uses the canonical world and catalog loaders on the actual files. It checks all roster seeds/categories, both size classes, provenance, scope/role quotations, hashes, projection privacy, unindexed Markdown, and exact promoted membership/path identity. It performs no network access, seed builds, environment setup, or model episodes, and does not change the manifest.

Any contract edit must update its indexed hashes and changed quotations. Metadata should never silently advance a qualified world to upstream HEAD. Fresh loading is deterministic against the indexed identities and projections.

## Deferred work

Execution facilities, dependency/toolchain support, live service implementations, performance calibration, visual validation, and GM/training rollouts belong to later work. The evolving and adversarial contracts describe precommitted in-world events and protected effects; this collection does not add an event scheduler or security harness.

Sources: [`../../docs/SEED_ROSTER.md`](../../docs/SEED_ROSTER.md), [`../../docs/seed_roster.json`](../../docs/seed_roster.json). Format: [`../../docs/WORLD_FILE_SPEC.md`](../../docs/WORLD_FILE_SPEC.md). Approved runtime catalog: [`../README.md`](../README.md).
