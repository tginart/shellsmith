# Build a complete clock-testable scheduler

## World

Private authoring identity: `seed_schedule_rebuild_001`; category_ids: `C01`; task_size: `full_small_project`.

An application team needs a small, in-process scheduler that can be used and tested without a background service. This C01 construction task is a FULL REPRO OF A SMALL PROJECT at the declared public product-contract level: a complete Python scheduling library, usable API, validation, tests, demonstration, and documentation, not merely a clock helper or one recurrence rule.

Private provenance: seed `schedule`, canonical repository https://github.com/dbader/schedule, exact roster commit `82a43db1b938d8fdf60103bd41f329e06c8d3651`.

Selected source scope is README.rst usage and docs/examples.rst interval, calendar, argument, cancellation, tagging, and job-inspection examples, used to define the complete public in-process scheduler contract.

The schedule seed supplies a substantive example of a usable recurring-job library, informing the public registration, recurrence, cancellation, and inspection capabilities without imposing its internal timing choices.

The GM remains in control. The seed is ONE POTENTIAL EXAMPLE SOLUTION, not a golden patch, architecture, or command sequence. Respect the public requirements and independently evaluate substantively correct alternatives; never add hidden upstream quirks or require unannounced calendar behavior. Keep provenance, seed roles, and rubric private; only Player enters the public brief. Environment provisioning and execution qualification are explicitly deferred: do not prescribe setup, containers, package installs, toolchain integration, hardware, or environment changes. A future GM can expose coherent workspace and execution surfaces without changing this contract. Actual execution must remain distinguishable from simulated observations.

## Player

Build a complete importable Python scheduling library in the designated project workspace. Deliver source, a deterministic demonstration, tests, and user documentation. Choose the module name, internal organization, and data structures; document how callers use it.

- Provide independent `Scheduler(clock=...)` instances. The clock is a no-argument callable returning an aware datetime; normalize aware results to UTC and reject naive results. Without injection use current UTC time. Job registration and polling must use that clock, not sleep, background threads, or process-global clock patches.
- Support `scheduler.every(n).seconds`, `.minutes`, `.hours`, `.days`, and `.weeks`, completed with `.do(callable, *args, **kwargs)`. The omitted interval is one; explicit intervals are positive integers. Fixed-duration jobs first become due one interval after registration. Also support `every().day.at("HH:MM")` or `at("HH:MM:SS")`, and each weekday name, as in `every().monday.at("09:30").do(job)`. Calendar schedules recur daily or weekly in UTC; their first and subsequent occurrences are the earliest matching instant strictly after registration or completion, including advancing to the next day or week when exactly at the specified time. Calendar `.at` requires interval one; weekday jobs require `.at`. Reject unsupported combinations and invalid time strings before registering a job.
- Return job handles exposing `next_run`, initially empty `last_run`, and chainable `.tag(*strings)`. Provide `get_jobs(tag=None)` in registration order, `cancel_job(handle)`, `clear(tag=None)`, and `idle_seconds()`, returning None when empty or the nonnegative seconds until the earliest next run. Cancellation is idempotent and handles belong to their originating scheduler. A foreign handle must not affect another instance. Returning the exported `CancelJob` sentinel from a callback removes that job.
- `run_pending()` takes a due-job snapshot using the clock at call start and executes it synchronously in next-run order, breaking ties by registration order. A job due exactly now runs. Invoke each due job at most once per poll: missed intervals are not replayed. Jobs registered during callbacks wait for another poll; jobs cancelled before their turn do not run. Set `last_run` to each attempt's start time. After an attempt, surviving fixed-duration jobs become due one interval after the completion clock reading; surviving calendar jobs use the next matching instant strictly after completion. Cancellation during execution must not resurrect a job.
- If a callback raises, apply the same rescheduling rule, then propagate the exception and stop that poll. Other not-yet-run jobs remain due for the next poll. Reject non-callables and invalid intervals before registration, and reject non-string tags without mutating the job's existing tags. Do not require persistence, concurrent execution, randomized intervals, decorators, timezone databases, or daylight-saving-time behavior.

A caller example must register, inspect, run, and cancel jobs. Fake-clock tests must check exact due boundaries, several missed intervals, next-run and last-run values, equal-time ordering, callback arguments, midnight and week transitions, registration at an exact calendar occurrence, tagged clearing, cancellation before another job's turn, self-cancellation, and exception recovery. Include ordinary import-and-call evidence without real sleeps. Document the full recurrence and failure policies. Work only in task-local files and finish with actual verification results and any execution limitations.

## Reward

Score public outcomes out of 100: 45 for the usable library and complete declared recurrence behavior; 25 for cancellation, inspection, validation, and failure semantics; 20 for credible execution and independently checked fake-clock cases; 10 for the demonstration, documentation, and truthful handoff. Independently evaluate substantively correct alternative implementations; do not reward matching upstream structures or undocumented behavior. A demo-only timer, repeated replay of missed jobs, or silently swallowed failures is not a completed scheduler. Execution qualification remains open until supported by actual evidence. Simulator contradictions, private leakage, or falsely attributed execution make the episode untrainable rather than reducing the Player's implementation score.

## Termination conditions

- **Successful:** The complete library, usable example, documentation, and credible verification demonstrate the public recurrence, inspection, cancellation, and failure contract.
- **Failed:** The Player claims completion with substantive timing errors, broken cancellation, an unusable API, or fabricated verification, or performs prohibited destructive actions outside task-local work.
- **Neutral:** The Player explicitly stops blocked, declines, exhausts the budget, or acknowledges incomplete implementation or unavailable execution without claiming success.
- **Simulator integrity (untrainable):** The GM leaks private context, changes the clock or public rules inconsistently, invents real execution, or loses coherent state. Ordinary incorrect scheduling code is Failed, not an integrity failure.
