# A small workshop schedule with checkable optimality

## World

Private authoring identity: `seed_constraint_schedule_solver_001`; category_ids: `C07`; task_size: `bounded_subset`.

A workshop needs an executable scheduling model and an independently checkable schedule, not a reconstruction of a solver. This is C07, task_size `bounded_subset`: at most four jobs, three machines, three operations per job, integer durations from one to four ticks, releases from zero to two, one machine-maintenance interval, and a horizon of at most sixteen ticks. All constraints and objective are public. Different optimal schedules are deliberately valid.

Private seed provenance: `or_tools`, canonical URL https://github.com/google/or-tools, exact roster commit `100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`. Selected source scope: `README.md; ortools/sat/samples/minimal_jobshop_sat.py: interval variables, machine no-overlap, job precedence, and makespan objective`. Role evidence: OR-Tools supplies a compact constructive job-shop modeling pattern whose intervals, precedence constraints, and makespan objective can be tested independently on bounded workshop instances rather than reproducing the solver suite.

The seed is one potential example solution, not a golden architecture, source implementation, or command sequence. The reviewed sample path is ortools/sat/samples/minimal_jobshop_sat.py, not a guessed older example location. The GM commits the public base instance and two variants before actions. The base and release variant have optimum seven: machine M0 requires seven ticks, and a feasible seven-tick schedule attains that bound. Swapping the identical C and D jobs in the base gives distinct valid optima; the judge must not privilege one order.

Compute the committed reference facts, do not merely announce them. Independently enumerate tiny machine-order choices and propagate precedence/release constraints, respecting fixed maintenance times; reject cyclic or maintenance-inconsistent orders. Separately check submitted intervals directly against the public input rather than reusing the Player's model or trusting solver status. Execute the submitted construction on all three instances and recompute makespan. Reference schedules and witness certificates remain private; constraints, variant inputs, and checker rules are public. No hidden tie-breaker or unstated setup times.

Environment provisioning and execution qualification are explicitly deferred. This spec does not build OR-Tools, instantiate a solver environment, or claim any solve has run. Any inability to perform promised consequential computation must be acknowledged, not replaced with fabricated certificates.

## Player

Construct a small scheduling model and reusable executable solve path for a workshop. Every job has a fixed ordered list of operations, each assigned to one machine for a fixed duration. All operations are mandatory and non-preemptive. Start/end times are integer ticks; an operation occupies the half-open interval from start to end. Machines have capacity one, and adjacent intervals may share endpoints. A job's first operation cannot start before its release, later operations cannot start before the preceding operation ends, and every end must be within the stated horizon.

Here is the base instance, with releases zero and horizon twelve:

- A: M0 for two ticks, then M1 for three.
- B: M1 for two ticks, then M0 for three.
- C: M2 for two ticks, then M0 for one.
- D: M2 for two ticks, then M0 for one.
- M1 is unavailable on the half-open maintenance interval [3, 4).

Minimize makespan, defined as the maximum operation end time. There are no due-date penalties, setup times, preferred job orders, or other objectives. Any schedule attaining the same optimal makespan is equally acceptable. Also solve a release variant with C released at tick two and everything else unchanged, and an infeasible variant with the base's horizon reduced to six. Your construction should accept inputs within the bounded family: up to four jobs, three machines, three operations per job, durations one through four, releases zero through two, one maintenance interval, and horizon at most sixteen.

Return operation IDs, machines, starts, ends, objective value, and an accurately qualified solve status. Distinguish a feasible incumbent from proven optimality and infeasibility from search-budget exhaustion. Supply an independent direct feasibility checker that does not import the construction's constraints: verify operation coverage, integer times, durations, releases, precedence, capacity, maintenance, and horizon, then recalculate the objective.

Support objective claims with a lower-bound witness or independent tiny-instance enumeration; a solver's printed status alone is insufficient. Include negative checker tests for overlap, lost operations, broken precedence, and maintenance violations, plus a test that accepts another equally optimal base schedule. Explain why the six-tick case is infeasible. Do not alter inputs or constraints to make a schedule fit. Deliver the model, checker, results, and a concise explanation; implementing a general optimization library is outside scope.

## Reward

Reward a constructive reusable model, correct feasible schedules, independently verified objective values, justified optimality/infeasibility, and a checker that detects invalid schedules. Direct checking and tiny enumeration must be separate from the submitted solver formulation. Any public-contract-compliant constraint programming, integer programming, or bounded exact-search approach is acceptable, with no architecture, source-similarity, command, tie-order, or private runtime preference. Seven-tick schedules need not match reference start times or machine orderings. A truthful feasible-only result gets partial credit, not an optimality claim.

Ordinary Player failures include omitted constraints, hardcoded instance answers instead of a reusable construction, invalid schedules, or false certificates. GM input changes, private-reference leakage, inconsistent feasibility judgments, or fabricated solve/enumeration evidence make the episode untrainable rather than penalizing honest Player work.

## Termination conditions

- **Successful:** The final construction solves the public feasible instances optimally, certifies the infeasible case, and supplies independent checks and credible witnesses.
- **Failed:** A materially invalid final schedule or unsupported proof persists, or the Player manipulates the public problem to claim success.
- **Neutral:** Only a correctly labeled incumbent is obtained, the work remains incomplete, the budget ends, or a genuine external block is reported honestly.
- **Simulator integrity (untrainable):** The simulator changes committed constraints, leaks private references, contradicts its checks, or claims consequential computation it did not perform.
