# A reproducible flight-reliability investigation

## World

Private authoring identity: `seed_flights_reliability_analysis_001`; category_ids: `C05`; task_size: `bounded_subset`.

A regional travel-planning team has a frozen snapshot of 2013 departures from New York's three airports. They need defensible historical comparisons, not an assertion about today's carriers. Expose the flights, airlines, airports, aircraft, and hourly weather tables with their source dictionaries. Preserve the full supplied snapshot, missing values, published units, and row grain; the GM commits its inventory before the investigation and computes consequential joins and summaries rather than inventing them.

Seed `nycflights13`: https://github.com/tidyverse/nycflights13 at `df98ef215aa8216fe0838a0b8ac5bada646d814c`. Selected scope: the five 2013 package datasets and man/flights.Rd and man/weather.Rd. The nycflights13 seed supplies the historical observations and documented join/time semantics against which the agent's denominators, comparisons, and claims can be independently checked.

Seed `sqlite_utils`: https://github.com/simonw/sqlite-utils at `6bc1d33d583c54bd69fbdd2071117e2d38c354a1`. Selected scope: README.md import, in-memory query, and tabular export behavior. The sqlite-utils seed supplies an optional example of an auditable import/query/export pipeline; it is a substantive transformation reference, not a mandated tool or expected analysis.

Task size: bounded subset, covering an analytical workflow rather than reproducing either software package. Flight `time_hour` denotes the scheduled departure hour; weather keys include origin and observation hour. Missing actual departure is a no-departure proxy, not permission to classify every missing arrival delay as cancellation. Comparisons can be confounded by routes, season, and observation coverage. Make those facts discoverable in the public dictionaries and preserve unmatched records. No preferred ranking, chart, causal explanation, or SQL query is private ground truth.

The GM remains in control of the requester, coherent evidence surfaces, and judging. Each software reference is one potential example solution, not a golden architecture or command sequence; the dataset fixes evidence, not the answer. Substantively correct alternatives meeting the public brief receive equal credit. Private references and grading remain outside the Player prompt. Environment provisioning and execution qualification are deferred; this contract adds no setup or infrastructure requirements.

## Player

Our travel-planning team wants a reproducible investigation of the supplied 2013 New York flight data. Compare carrier and route reliability, but keep three things distinct: all scheduled records, records with no actual departure time, and arrival delays among records with an observed arrival delay. Label missing departure as a proxy rather than a certified cancellation count, and report missing-outcome coverage instead of dropping it silently.

Include at least one weather comparison and check whether important carrier/route conclusions change after accounting for route or season. Use the supplied dictionaries to justify joins and show that they do not inflate the flight inventory. State units, time interpretation, and the denominator behind each headline measure.

Deliver an executable analysis, machine-readable summary tables, and a concise report with reproducible figures, sensitivity checks, limitations, and proposals for follow-up investigation. Choose your own language, methods, and presentation. We need historical evidence, not unsupported causal claims or a ranking of current airlines. Preserve the source data, run credible checks, and clearly separate work you verified from work you could not verify.

## Reward

Reward correct inventory conservation, explicit missingness and denominators, defensible joins, useful carrier/route/weather comparisons, route/season sensitivity, reproducible outputs, and calibrated historical interpretation. Independently recompute reported quantities from the exposed snapshot; different valid statistical methods, pipelines, or conclusions are acceptable. Score only the requested outcomes and credible evidence, not use of SQLite, agreement with an unpublished ranking, or similarity to reference source.

Deduct for join multiplication, mixing no-departure and missing-arrival outcomes, unreported exclusions, incorrect units/time keys, causal or current-safety claims unsupported by the data, fabricated checks, and altered evidence. A careful negative or inconclusive finding can be a complete answer. The seed is one potential example solution, never a hidden additional contract.

## Termination conditions

- **Successful:** A reproducible analysis and report answer the public questions with checked quantities, explicit coverage, sensitivity, and limitations.
- **Failed:** A claimed-complete report materially violates the public analytical contract, or deliberate evidence destruction, fabricated execution or findings, unauthorized external effects, or persistent reward manipulation occurs. Ordinary analytical mistakes remain trainable policy failures, not simulator corruption.
- **Neutral:** Stop for a genuine blocker, incomplete analysis, explicit decline, or exhausted action budget; grade only supported work.
- **Simulator integrity:** Contradictory data surfaces, leaked private grading/reference answers, or false simulator execution make the episode untrainable, not a Player failure.
