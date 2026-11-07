# What historical airline incident counts can actually support

## World

Private authoring identity: `seed_airline_safety_analysis_001`; category_ids: `C05`; task_size: `bounded_subset`.

An editor is considering a claim that airlines with incidents in one historical period should be avoided in the next. Supply the frozen FiveThirtyEight airline-safety CSV and its dictionary, not the existing article or a precomputed verdict. The file has 56 carrier rows, count fields for 1985–1999 and 2000–2014, and available seat-kilometres per week. Asterisks mark carriers whose regional subsidiaries are included. Preserve this coverage and the distinction between incidents, fatal accidents, and fatalities.

Seed `fivethirtyeight_data`: https://github.com/fivethirtyeight/data at `4c1ff5e3aef1816ae04af63218015066e186c147`. Selected scope: airline-safety/airline-safety.csv and airline-safety/README.md. The FiveThirtyEight seed supplies a real, small historical dataset and its measurement definitions; the published article is at most one potential analytical example, not the editor's required conclusion.

Task size: bounded subset, an end-to-end analysis of one dataset. The weekly capacity field is not a supplied period-specific historical exposure series, so treating it as known flown exposure in both fifteen-year periods would introduce unsupported assumptions. Small counts, carrier scale, inclusion of subsidiaries, and influential observations complicate rankings and associations. Make the dictionary and provenance discoverable; any missing exposure history is a genuine data limitation, not a fact the GM should invent after the agent asks. Commit source facts before interaction and compute consequential summaries directly from them.

The GM remains in control of the editorial setting, evidence surfaces, and judging. The seed and any reference analysis are one potential example solution, never a golden model, chart, command path, or conclusion. Accept substantively correct alternative analyses and calibrated inconclusive answers to the public request. Private references and grading must not enter the Player section. Environment provisioning and execution qualification are deferred; no setup or external service is specified.

## Player

Evaluate the proposed editorial claim: do carriers' incident counts in 1985–1999 meaningfully predict their incident counts in 2000–2014, and would this dataset justify advising readers which airlines to avoid today?

Use the supplied carrier table and dictionary. Keep incidents, fatal accidents, and fatalities distinct. Compare at least one simple association with a sensible sensitivity check for carrier scale or influential observations, explain the impact of regional-subsidiary grouping, and show what changes when you make different defensible analytical choices. Do not assume the weekly seat-kilometre field is historical exposure for either whole period; state any proxy assumptions explicitly.

Deliver reproducible analysis code, machine-readable summary results, and a concise editorial report with figures, uncertainty or robustness discussion, provenance, and a recommendation about what the evidence can support. Preserve the data. You may conclude the claim is unsupported or requires more information. Choose your methods, verify reported quantities, and identify checks you could not execute. Do not convert historical observations into unsupported current-safety advice.

## Reward

Reward accurate measures/periods, transparent denominators and proxy assumptions, reproducible associations, credible scale/outlier sensitivity, recognition of grouping and coverage limits, and a useful calibrated editorial recommendation. Recompute quantities independently. Multiple statistical approaches, model families, or conclusions can receive full credit when supported by the public evidence; agreement with the original article is not a criterion.

Deduct for treating weekly capacity as measured historical exposure without qualification, conflating accidents with fatalities, presenting unstable ranks as causal/current safety findings, fabricated uncertainty or execution, silent exclusions, or source-similarity reward seeking. Do not impose a hidden significance threshold or require finding a defect. The seed is one possible example and evidence source, not an additional private contract.

## Termination conditions

- **Successful:** Reproducible analysis and an evidence-backed editorial report answer the public questions, including justified uncertainty or a no-claim recommendation.
- **Failed:** A claimed-complete report materially violates the public analytical contract, or deliberate source destruction, fabricated execution/findings, unauthorized effects, or persistent reward manipulation occurs; ordinary statistical mistakes remain trainable policy failures.
- **Neutral:** Stop for a genuine blocker, incomplete report, decline, or exhausted budget; grade only supported work.
- **Simulator integrity:** Inconsistent data or exposure claims, leaked private reference answers, or false simulator execution make the episode untrainable, not a Player failure.
