# Faster monthly order rollup, same answers

## World

Private authoring identity: `seed_duckdb_query_optimization_001`; category_ids: `C04`; task_size: `bounded_subset`.

A retail analytics team needs a faster monthly finance rollup, not a new database engine. This is C04, task_size `bounded_subset`: one query over a frozen snapshot of at most 100,000 orders and 600,000 line items. The existing executable baseline uses repeated per-order line aggregations; it is a plausible current implementation, not deliberately slowed or guaranteed to lose. Some completed orders have no lines, repeated lines are legitimate, amounts can be NULL or negative, and customers or regions can be missing.

Private seed provenance: `duckdb`, canonical URL https://github.com/duckdb/duckdb, exact roster commit `d05e1133a451ea2e4ef44696837abb2c67d4d7bd`. Selected source scope: `README.md: DuckDB overview and CSV/Parquet Data Import`. Role evidence: DuckDB supplies the analytical SQL and frozen-table query setting in which the Player must improve a real rollup without changing its relational semantics.

Private seed provenance: `hyperfine`, canonical URL https://github.com/sharkdp/hyperfine, exact roster commit `e4a3213bf32281db964c0c4910dec61ec43e4775`. Selected source scope: `README.md: Basic benchmarks; Warmup runs and preparation commands; Intermediate shell; Exporting results/JSON`. Role evidence: Hyperfine supplies repeated timing, explicit warmup and execution boundaries, and raw JSON samples; its command-order behavior motivates counterbalanced paired blocks rather than a ceremonial speed table.

Each seed is one potential example solution, not a golden architecture, source implementation, or command sequence. The GM may vary peripheral filenames but must commit the snapshot bytes, schema, baseline SQL, parameter values, resource settings, and independent correctness calculation before Player actions. Publish these task contracts, not these provenance paragraphs. Orders and customers have unique primary keys; line-item multiplicity must survive. No adaptive fixture changes or invented improvements. Actually execute baseline and candidate queries, compare their complete results with an independent aggregation, and obtain consequential timings from execution. An execution plan alone is not timing evidence.

Environment provisioning and execution qualification are explicitly deferred. This authoring spec neither provisions an environment nor claims any query or benchmark has already run. If qualification cannot support the declared measurement, do not improvise hardware thresholds or synthetic timings.

## Player

Optimize the July 2025 completed-order finance rollup while preserving its answers. Your starting materials are an immutable snapshot, schema, existing baseline query, parameterized query runner, and correctness examples. Limit your work to this rollup and its measurement; do not replace the database engine or modify source records.

For orders with status `completed` and ordered_at in the half-open interval July 1 through August 1, 2025, return each region's order count, actual line-item row count, and sum of signed integer amount_cents. Include orders without lines; count every duplicate line separately. NULL amounts contribute zero, and absent customers or NULL regions map to the literal label `Unknown`. Order and customer IDs are unique. Results are sorted by region, with exact integer totals and counts. The baseline's SQL and examples expose this contract; verify full results, not only a checksum or selected rows.

The public performance goal is at least a 10% reduction relative to the freshly measured baseline, with uncertainty excluding no improvement. Use a declared warm-cache protocol: three untimed warmups per implementation, then eight paired blocks of five measured runs per implementation, alternating baseline-first and candidate-first. Compute each block's candidate/baseline median-time ratio and the geometric mean across blocks; success requires a ratio at most 0.90 and a block-bootstrap 95% interval whose upper endpoint is below 1. Export every measured sample and identify warmups separately. Predeclare the bootstrap procedure and random seed. Do not discard inconvenient samples or keep testing until a favorable interval appears; document external interruptions and any whole-protocol restart.

Measure identical fixed parameters, engine settings, fresh-connection behavior, and full result consumption, including connection startup. Exclude snapshot creation equally. Any candidate-only materialization or setup belongs inside its measured run; persistent answer caching is not permitted. Report peak memory with measurement scope and limitations, without assuming an absolute hardware-dependent cap. Explain what changed and why. If the goal is not reached or the evidence is noisy, state that plainly and preserve the correct baseline rather than claiming a win.

## Reward

Reward exact rollup equivalence, an actual query improvement, and auditable baseline-relative evidence under the public protocol. Recompute counts and totals independently, including empty orders, duplicates, NULLs, missing regions, and negative amounts. Verify raw samples, included setup, the paired statistic, interval, and claims. Correct rewrites, different query organizations, and substantively equivalent measurement implementations are acceptable; neither upstream similarity nor a preferred command earns credit. Honest no-win results retain credit for correctness and measurement but do not meet the performance objective. No hidden speed, memory, or hardware thresholds apply.

Ordinary Player failures include changing data or semantics, hardcoding answers, omitting candidate-only costs, and falsely claiming improvement. GM leakage, invented execution, altered fixtures, contradictory timings, or an unavailable promised measurement mechanism make the episode untrainable, not a Player failure.

## Termination conditions

- **Successful:** A final handoff preserves all results and demonstrates the public relative-performance goal with credible raw evidence.
- **Failed:** The final deliverable is materially incorrect, forbidden manipulation occurs, or unsupported performance claims persist in the final response.
- **Neutral:** The Player reports no demonstrated improvement, inconclusive noise, an external block, or incomplete work without misrepresentation.
- **Simulator integrity (untrainable):** Private context leaks, frozen truth changes, execution is fabricated, or the simulator cannot maintain or compute the promised correctness and timing evidence.
