# A delay prediction that could have been made before departure

## World

Private authoring identity: `seed_flights_prediction_pipeline_001`; category_ids: `C06`; task_size: `bounded_subset`.

An airport planning analyst wants to know whether a small mixed-feature classifier adds useful information beyond a constant risk estimate. This is C06, task_size `bounded_subset`: one pipeline experiment on a frozen, deterministic subset of 2013 NYC flights, at most 34,000 rows across all months. Select every tenth row after a stable chronological ordering with fixed row IDs, before inspecting targets. The scientific outcome may legitimately be no improvement.

Private seed provenance: `scikit_learn`, canonical URL https://github.com/scikit-learn/scikit-learn, exact roster commit `c95f032d2960b69ed20c1b15f7be5a8aa10595f8`. Selected source scope: `README.rst; examples/compose/plot_column_transformer_mixed_types.py: mixed-type preprocessing and estimator Pipeline`. Role evidence: Scikit-learn supplies a bounded mixed-type preprocessing and classification pattern that must be adapted to train-only fitting and chronological evaluation rather than copying the example's random split.

Private seed provenance: `nycflights13`, canonical URL https://github.com/tidyverse/nycflights13, exact roster commit `df98ef215aa8216fe0838a0b8ac5bada646d814c`. Selected source scope: `README.md; man/flights.Rd; man/weather.Rd`. Role evidence: Nycflights13 supplies the frozen historical rows and field meanings that distinguish scheduled information from realized delays, while its weather dictionary exposes why recording time alone cannot prove pre-departure availability.

Each seed is one potential example solution, not a golden architecture, source implementation, or command sequence. The GM commits extracted bytes, stable IDs, timezone conversion, eligibility rules, chronological split, and target labels before actions. Give the Player a public field dictionary and canonical scheduled timestamps, including explicit next-day handling for 2400. Treat schedule fields as an already-published schedule snapshot at the decision time, a declared retrospective assumption rather than verified historical feed latency. Weather is excluded because these sources do not establish release availability. No private predictive-score floor or guaranteed improvement.

Keep November-December rows and labels sealed until the final model, baseline, preprocessing, and evaluation plan are frozen. Actually train or execute submitted artifacts, generate held-out predictions, and independently compute metrics from the committed labels. Do not improvise plausible scores. A small reference pipeline can establish sanity checks but cannot become a hidden answer key. Public outcome reports may be shown after evaluation without permitting another model revision.

Environment provisioning and execution qualification are explicitly deferred; no training run, frozen extract, or usable runtime is claimed by this authoring spec.

## Player

Build and evaluate a small pipeline that predicts arrival delay of at least 15 minutes, using only information available 60 minutes before scheduled departure. Your materials describe a frozen sample of at most 34,000 historical 2013 departures from EWR, JFK, and LGA. Scheduled timestamps use the origin's America/New_York timezone and are supplied with an unambiguous UTC representation. The schedule is assumed already published at the decision time; document that limitation.

The target is `arr_delay >= 15` among flights with observed arrival delay. Exclude missing arr_delay from supervised scoring, report their counts by period, and explain that this is a completed-flight cohort, not a cancellation or all-bookings risk model. Do not silently label missing outcomes as on-time.

Allowed predictors are origin, destination, carrier, route distance, scheduled local departure hour/minute, weekday, and month. Do not use actual departure/arrival times, either realized delay, air_time, tail number, row IDs, target-derived aggregates, or weather. A recording timestamp is not proof that weather was available before the decision. Row IDs serve only alignment and split checks.

Use January-August 2013 for training, September-October for development, and a sealed November-December final holdout. Keep entire dates within one partition. Fit imputers, encoders, scaling, feature selection, and any calibration only on the fitting partition; within any development folds, learn them anew using earlier dates only. Handle previously unseen categories without consulting later rows. Compare a constant probability baseline equal to the fitting cohort's delay prevalence with at most four declared candidate configurations. Log candidates, selection criterion, fitted partitions, and random seeds; no unrestricted search.

Select the configuration using development data, then freeze the recipe and refit on January-October if desired. Freeze a correspondingly refitted prevalence baseline and submit both scorers before the single holdout evaluation. Do not obtain sealed rows or labels from another copy of the archive, inspect them early, or tune after seeing final metrics.

Report row accounting, split dates, Brier score as the primary comparison, precision-recall area with its definition, calibration diagnostics, and sample-supported origin/month breakdowns. Use a day-block bootstrap for a 95% interval on the paired Brier-score difference and explain its limitations. Provide reproducible predictions and a short recommendation that distinguishes demonstrated improvement from inconclusive or negative results. There is no required score or required win over the baseline.

## Reward

Reward decision-time validity, correct target/cohort construction, chronological separation, train-only transformations, bounded experiments, and reproducible held-out assessment. Independently recompute eligibility, row alignment, metrics, and uncertainty from the submitted probabilities. Substantively correct alternative pipeline designs or estimators earn equal credit; neither example similarity nor a favored library API is required. An honest, well-supported negative result can fully succeed. The recommendation must match the evidence, not an undisclosed accuracy target.

Ordinary Player failures include outcome leakage, holdout acquisition or repeated tuning, treating missing targets as negatives, and fabricated metrics. GM leakage of sealed labels or provenance, mutable labels/splits, falsely claimed training, contradictory scores, or inability to execute promised evaluation are simulator-integrity failures and untrainable.

## Termination conditions

- **Successful:** A final reproducible pipeline and report satisfy the public evaluation contract, including an honest no-improvement conclusion when warranted.
- **Failed:** The final artifact violates feature/split restrictions, misstates consequential evidence, or deliberate holdout compromise occurs.
- **Neutral:** The Player truthfully stops incomplete, declines, exhausts the experiment budget without a complete assessment, or encounters a genuine external block.
- **Simulator integrity (untrainable):** The simulator exposes private truth, changes committed data, fabricates consequential computation, or cannot honor the sealed single-evaluation boundary.
