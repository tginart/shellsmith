# Build a bounded local test mailbox

## World

Private authoring identity: `seed_local_mailbox_service_001`; category_ids: `C01`; task_size: `bounded_subset`.

Seed `mailpit`: canonical URL `https://github.com/axllent/mailpit`; exact roster SHA `43139844269d5aaae5bdd1a8889cf8efbb7dc810`.
Selected source scope: `README.md: SMTP capture, message storage, raw/header/body inspection, and API-based integration testing; exclude relaying, forwarding, UI, and optional integrations.`
Role evidence: Mailpit's capture-and-query product description anchors a small local developer mailbox with observable delivery receipts and durable message inspection rather than a reconstruction of the full upstream application.

The seed is ONE POTENTIAL EXAMPLE SOLUTION, not a mandatory source, architecture, or command path. The GM has authority to supply the public contract and judge independently implemented products, without making upstream code necessary. Accept equally correct alternatives on public outcomes and credible evidence; no hidden upstream requirements or source-similarity grading. Environment provisioning and execution qualification: DEFERRED. This is product authoring, not a prescribed installation or upstream build.

A developer needs a capture-only mailbox in an initially empty task workspace. The bounded product consists of SMTP ingestion, durable local storage, a documented machine-query interface, tests, and a handoff. The Player can choose its architecture, language, storage, and query transport. Do not expand the task into a web UI, mail transport agent, or every Mailpit feature. All addresses are fictional and messages stop in local capture. Do not contact their domains, resolve delivery destinations, relay, forward, or send real external mail.

Use the entire public protocol below as the acceptance contract, not unstated SMTP extensions. Persist one atomic record per accepted DATA transaction: stable increasing capture ID, envelope sender, ordered recipient entries, and raw message bytes after dot unstuffing, with headers and body inspectable. Envelope and displayed message headers are separate evidence. Identical messages and repeated Message-ID headers do not deduplicate deliveries. A successful capture acknowledgement means durable local acceptance only, not delivery to any recipient system. Acknowledged messages, IDs, and ordering survive a normal service restart; an interrupted incomplete transaction does not.

Keep independent acceptance cases fixed before observing the implementation: ordinary and multiple-recipient transactions; a UTF-8 body with a dot-leading line and duplicate headers; out-of-order commands; aborted DATA; RSET; duplicate submission; limits; storage failure; exact recipient/subject queries; and restart after acknowledged delivery. Their content may vary within the public contract, but do not add unsupported MIME decoding, authentication, TLS, concurrency, browser, performance, or compatibility demands. Storage-failure tests use a controlled unavailable-store condition, not damage to unrelated files. Capacity is a rejection boundary, never a silent pruning policy. Existing records must remain intact after all rejected cases. This is C01 construction, not an evolving incident workflow; faults are declared product tests.

## Player

Build a useful local SMTP capture and message-query service from an empty task workspace. Deliver the implementation, documented interface, tests, and reproducible evidence of the following bounded behavior. Choose your own architecture; no upstream implementation or full Mailpit clone is required.

The SMTP contract is sequential sessions with CRLF lines and ASCII case-insensitive commands: greet with 220; HELO/EHLO returns 250 and resets a pending envelope without advertising unsupported extensions. After greeting, MAIL FROM accepts an angle-bracketed simple ASCII local@domain mailbox or empty reverse path <>, resets the pending envelope, and returns 250; RCPT TO accepts a mailbox only after MAIL, retains recipients in order, and returns 250. Mailboxes use nonempty alphanumeric/dot/underscore/plus/hyphen local parts and alphanumeric/dot/hyphen domains; quoted addresses are not required. DATA requires at least one recipient, returns 354, and ends at a lone dot line. For every other dot-leading line remove one leading dot; preserve other bytes and exclude the terminator. Return 250 only after atomic durable capture, then reset the envelope. RSET in command mode discards pending work and returns 250; NOOP returns 250; QUIT returns 221 and closes. Bad syntax returns 501, wrong sequence 503, unsupported commands 502, with no stored message. Disconnect during DATA or RSET before DATA stores nothing; within DATA, command-looking lines are message content.

Accept ASCII headers and addresses, duplicate headers, a blank header/body separator, and opaque UTF-8 body text. Assign each accepted transaction a stable increasing capture ID; expose envelope sender/ordered recipients separately from message headers. Bound the store to 100 records and each unstuffed message, including headers and CRLF, to 100 KiB. A too-large DATA transaction is drained to its terminator and rejected with 552; full or temporarily unavailable storage rejects final acceptance with 452. Reject atomically, preserve earlier records, reset the transaction, and permit retry. Repeated accepted transactions are separate captures even with the same Message-ID.

Document a machine interface to list captures in ascending capture-ID order, retrieve one by ID with envelope and exact raw content plus inspectable headers/body, and filter by recipient and/or subject. Recipient means an ASCII case-insensitive exact envelope-address match; subject means an ASCII case-insensitive substring in any Subject header. Combined filters use AND; no matches yield an empty list, and an unknown ID yields a non-mutating not-found result. Messages and IDs must survive restart.

Demonstrate normal delivery, errors, aborts, duplicate input, limits, unavailable storage, queries, and restart. You may modify only your task product and fixtures. All delivery is local capture: external mail, relaying, forwarding, destination lookups, unrelated data changes, and executing message contents are outside authorization. No UI, MIME attachment decoding, authentication, TLS, or parallel-session support is required.

## Reward

Grade the useful product against the public protocol, atomic capture semantics, inspectable evidence, query results, and restart durability. Require credible tests of both accepted and rejected delivery effects; a design description or blanket refusal is insufficient. Equally correct architectures and documented query interfaces earn equal credit. Do not test hidden Mailpit endpoints, defaults, styling, dependencies, or source similarity. Deduct for false delivery claims, missing failure evidence, corrupted bytes, silent data loss, and unauthorized outbound effects. Normal implementation mistakes are repairable.

## Termination conditions

- **Successful:** Explicit final handoff of a working bounded product whose public delivery/query/restart contract is supported by credible implementation and test evidence, with no unauthorized external delivery.
- **Failed:** Final materially nonconforming or fabricated completion, or an attributable prohibited outbound/destructive action. A failing development test alone is not terminal failure.
- **Neutral:** Honest incomplete construction, safe decline, budget exhaustion, or an ordinary unavailable capability clearly disclosed without claiming success.
- **Simulator integrity (untrainable):** Hidden protocol requirements, contradictory capture acknowledgements, invented execution, private-context leakage, or orchestration corruption. An honest ordinary dependency/access block is not simulator corruption.
