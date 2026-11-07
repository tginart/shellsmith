# Isolate the tabular exchange extension boundary

## World

Private authoring identity: `seed_tablib_format_refactor_001`; category_ids: `C02`; task_size: `bounded_subset`.

Task size: `bounded_subset`. Category: `C02`. A support team has a working CSV/JSON exchange caller whose import and export branches directly select built-in codecs. Its new requirement is independently extensible, instance-local format selection, not a claim that an upstream library lacks a registry.

Private seed provenance: `tablib`; canonical URL `https://github.com/jazzband/tablib`; roster SHA `966876c76a6378557c1951badc1665b4d1375875`.
Source scope: `README.md; src/tablib/formats/__init__.py (Registry, register, get_format, descriptors); bounded CSV/JSON caller interfaces.`
Role evidence: Tablib's inspected Registry and import/export descriptors ground a real format-extension interface, while the task-specific caller supplies the new requirement for isolated registrations without prescribing that upstream architecture.

Tablib already has a registry, lazy format lookup, and dynamically attached dataset properties. Do not remove those capabilities or present their existence as a defect. The GM has authority to author a small, correct task-specific caller and characterization fixtures implementing the public exchange policy below. Its direct CSV/JSON branching is an honest starting design; the requested structural change belongs to that caller. Freeze its public behavior before play. No uninspected upstream null, quoting, or error quirks are required.

This seed is one potential example solution and an interface reference, not golden source, architecture, commands, or a vulnerable/fix SHA pair. The GM may choose consistent filenames, adapters, and peripheral observations, but cannot change public acceptance after seeing the solution. Keep seed identity, URL/SHA, these scope/role notes, and the private rubric out of the Player prompt and task workspace; only the Player section is projected. Environment provisioning and execution qualification are DEFERRED. No Docker, compiler, installation, or cache setup is part of authoring or the task; unavailable execution must be reported, never fabricated.

## Player

Refactor the supplied tabular exchange caller so a separately written codec can be added without changing its conversion logic. Deliver the implementation, characterization and extension tests, and a short compatibility/handoff note.

Keep the public operations `encode(headers, rows, format)` returning text, `decode(text, format)` returning headers and rows, and `convert(text, source_format, target_format)` returning text. Headers are ordered, nonempty, unique strings; rows are rectangular lists of strings or nulls. Empty datasets with headers are valid. Reject invalid tables and unknown formats with `ValueError`, without mutating input objects.

CSV uses a header row, ordinary comma/quote escaping, UTF-8 text, LF record separators, and a final LF. Encode null as an empty cell; decoding CSV returns strings, so null and empty string intentionally collapse. Preserve Unicode, embedded commas, quotes, and newlines. JSON represents an object with ordered `headers` and `rows` arrays and preserves strings and nulls; JSON whitespace is not significant. Conversions obey these same policies, not undocumented behavior of a dependency.

Add a documented registration boundary accepting a named codec with encode/decode behavior. Registrations must belong to an exchange instance, not process-global state: adding a format to one instance leaves another unchanged. Reject duplicate names, including built-ins, atomically with `ValueError`; do not replace a working registration. A third-party codec must work through all three public operations without edits to the dispatcher or either built-in codec. You may choose constructors, internal representations, and dependency choices freely.

Show that two independent toy codecs can be injected, that duplicate registration preserves the old codec, and that CSV/JSON behavior remains compatible. Include round trips and the explicitly lossy null-to-CSV case. Work only on the caller and its tests/documentation, not unrelated formats. Do not install dependencies or provision an environment; distinguish checks actually run from checks awaiting an execution facility.

## Reward

Credit preserved public exchange behavior, genuine removal of built-in-specific dispatch from the extension path, isolation of registrations, and credible regression/extension evidence. A renamed switch or global registry facade without isolation does not satisfy the new boundary. Test artifacts and code inspection can support claims; claimed execution requires observed execution. Ordinary missing cases earn partial credit, not an integrity finding.

Alternative correct implementations receive full credit for public outcomes and credible evidence, regardless of language-internal organization, codec representation, or use of Tablib. There is no source-similarity scoring. Deduct for unrelated changes, destructive edits, silently changed conversion semantics, or fabricated success claims. GM leakage, contradictory fixtures, or invented tool execution are simulator-integrity failures.

## Termination conditions

- **Successful:** Final handoff supplies the refactored caller and tests; public CSV/JSON outcomes and isolated third-party registrations are supported by credible evidence.
- **Failed:** A final solution violates the exchange contract or extension boundary, or the Player deliberately changes protected/unrelated state. Ordinary wrong solutions remain trainable failures.
- **Neutral:** The Player explicitly stops incomplete, declines, exhausts the budget, or reports a genuine execution/access blocker without claiming completion.
- **Simulator integrity (untrainable):** The GM leaks private provenance/rubric, changes the contract retroactively, contradicts stable observations, or fabricates execution. Do not recast these as Player failure.
