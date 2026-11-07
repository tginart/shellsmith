# Studio run monitoring and GM timeouts

Optional telemetry must never become training authority. The micro workflow
runs `shellsmith.jobs.studio_monitor` as a separate disposable process. SQLite
polls explicitly close their cursors and connections; monitor startup failures,
monitor death, telemetry write failures, and failed failure notifications leave
the trainer running. The owner retains the child monitor’s timing observations
and avoids appending completed-operation metrics twice when taking its final
snapshot. Required compute identity checks still fail closed.

The independent supervisor owns the absolute deadline. It tracks owned process
identities using kernel birth information and a stable owner-session enclosure.
Workflow children use separate process groups within that session, so a child
reparented after an abrupt owner exit remains discoverable. Owner startup and
launch-receipt publication are inside the cleanup-protected region. The
supervisor checks cleanup before committing
`owner-limit.json`. It repairs stale `instance.json` status after owner failure.
The offline report also reconciles that terminal receipt over stale running
snapshots, and displays stopped pending operations as interrupted while keeping
their original journal status. Missing owner verification is never promoted to
successful training.

Stage controls only for a fresh run enclosure:

```sh
python scripts/stage_studio_micro_controls.py --run-root /path/to/new-run
```

Staging copies `workflow.py`, `supervisor.py`, and `native_preflight.py`; it does
not launch anything. It rejects symlinked destinations, dispatched runs and
historical runs. The existing absolute two-hour run deadline is never reset.
The staging receipt rejects symlinks and is atomically replaced without
following a link target. The selected GM account remains owned by its environment
and account guard;
no account fingerprint or credential is built into the templates.

GM timeouts are separate settings in `[gm]`:

```toml
reasoning_effort = "medium"
timeout_seconds = 900
api_idle_timeout_seconds = 180
```

The 900-second ceiling covers an entire Codex invocation, including local tool
execution. The 180-second inactivity limit applies to an upstream API exchange:
it covers absent response headers and stalled response bodies, including SSE.
Incoming bytes, including keepalives, reset inactivity. Local tool execution
between API exchanges has no API inactivity timer. Provider retries remain in
effect inside the invocation ceiling. Authentication probes use the same
settings. The Player response limit remains the agreed 32,768 tokens.

The pinned Codex CLI's native SSE timeout does not cover missing HTTP headers.
A private loopback relay therefore applies upstream socket timeouts and disables
WebSocket transport so provider requests use this protected HTTP path. Upstream
HTTPS uses normal certificate verification. Authentication commands and account
headers remain CLI-owned and are forwarded unchanged; the relay never logs
credentials or payloads. `logs/*.api-transport.json` records timeout phases and
limits. Cancellation closes active relay sockets and the existing Codex process
cleanup terminates the owned process group.

Render a snapshot report with no model runtime or network access:

```sh
python scripts/render_studio_run_report.py --run-root /path/to/run --output /path/to/report.html
```

Tests use temporary SQLite databases, a low descriptor limit with GC disabled,
real dummy process failure/deadline injection, and local HTTP fixtures. Native
Codex fixture tests run only when the CLI is installed, with empty temporary
Codex homes, disabled plugins, local mock endpoints and synthetic credentials.
No real model rollout or Studio training run is part of this verification.

A malformed, unreadable or symlinked primary operation snapshot is an error,
not a reason to substitute a stale legacy snapshot. When the training client is
known, report token totals, training numerics and publication receipts include
only that client. Preflight/control, other-client and unassigned operations
remain visible separately and do not count as rollout/training work.
