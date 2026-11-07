# Deliver a transactional local ledger importer

## World

Private authoring identity: `seed_sqlite_import_tool_001`; category_ids: `C01`; task_size: `bounded_subset`.

Task size: `bounded_subset`. Category: `C01`. A community office needs a small local CSV/JSON-to-SQLite tool, not an entire database utility suite. It has no existing application implementation. Atomic imports, repeatable upserts, and independently readable exports are new product requirements stated publicly below.

Private seed provenance: `sqlite_utils`; canonical URL `https://github.com/simonw/sqlite-utils`; roster SHA `6bc1d33d583c54bd69fbdd2071117e2d38c354a1`.
Source scope: `README.md (CLI CSV/JSON insert, primary-key option, SQL query/export examples, and Database.insert_all example).`
Role evidence: The inspected sqlite-utils examples ground a practical local import/query/export interface and primary-key workflow; this task defines its own bounded transaction and type policies rather than pretending the README proves undocumented upstream behavior.

The GM has authority to supply small synthetic ledger inputs and independently inspect their database results, including late malformed rows, Unicode, duplicate keys, and reopening between commands. Fix fixture contents before play and retain their original bytes. The README supplies interface examples, not an implementation or an asserted defect; rollback and upsert rules here are authored requirements. Do not introduce migrations, plugins, full-text search, external services, or an undisclosed schema-inference oracle.

This seed is one potential example solution, not golden source, architecture, commands, or a vulnerable/fix SHA pair. Independent use of SQLite is valid. The GM may choose realistic filenames and observations but cannot alter the product policy to favor the reference. Keep seed identity, URL/SHA, scope/role notes, private expected rows, and the rubric out of the Player prompt/workspace; project only Player. Environment provisioning and execution qualification are DEFERRED; no Docker, compiler, dependency installation, or cache setup is required or assumed. Report unavailable execution honestly.

## Player

Build a local Python module CLI named `ledger_import`. Deliver its implementation, integration tests, sample inputs, and a concise usage/error-policy note. Internal architecture is your choice.

Support `import DB TABLE INPUT --format csv|json --key KEY --mode insert|upsert`, `query DB SQL`, and `export DB TABLE --format csv|json` through `python -m ledger_import`. INPUT is a file or `-` for stdin. Table and column names must match `[A-Za-z_][A-Za-z0-9_]*`; reject unsupported names safely. Imports create a missing table with TEXT columns and KEY as primary key, or require the existing table to have the same ordered columns and primary key. Do not silently migrate schemas or alter unrelated tables.

CSV has a header and at least one data row; headers are unique, cells are strings including empty strings, and row widths must agree. JSON is a nonempty array of objects sharing the same key set; column order follows the first object. Values are strings or nulls only. Reject other types, malformed input, duplicate headers, or missing/empty/null key values. JSON null becomes SQL NULL; CSV empty cells remain empty strings. Preserve Unicode, quoting, embedded newlines, and input column order.

Each import is one transaction. In insert mode, any duplicate key within the input or existing table fails the whole import. In upsert mode, replace all non-key column values for matching keys; the last row for a repeated key wins. Repeating an identical upsert leaves the same table contents. A failure, including a bad final row, leaves prior rows and schema unchanged; a failed first import must not leave a partially created table. A harmless empty database file may remain.

Query accepts one read-only SELECT and emits a JSON array of rows with SQL result column names, retaining result order and nulls; reject writes and multiple statements. Export sorts by the declared primary key using binary text ordering. JSON retains nulls and column order; CSV includes headers and intentionally represents null as an empty cell. JSON whitespace and CSV line-ending style are not acceptance criteria. Successful data output goes to stdout; failures give a useful stderr message and nonzero exit, not a success payload.

Test from separate invocations, reopen the database, and inspect final state independently. Cover both input formats, duplicate keys, reruns, rollback, schema mismatch, and exports. Do not install dependencies or set up an environment; identify unavailable execution checks in the handoff.

## Reward

Credit useful end-to-end construction: durable import correctness, all-or-nothing failure, specified upserts, read-only queries, compatible exports, and honest integration evidence. Independent SQL inspection outweighs a self-reported row count. A toy command that handles only one happy input is incomplete; extra unrequested utility features earn no substitute credit.

Alternative correct implementations earn full credit for public outcomes and credible evidence, whether they use sqlite-utils, the standard library, or another declared adapter. No source-similarity scoring or hidden upstream CLI quirks apply. Deduct for data corruption, unauthorized writes, unsafe name handling, or fabricated test claims. GM contradictions, private leakage, and nonexistent claimed execution make the episode untrainable, not the implementation wrong.

## Termination conditions

- **Successful:** Final handoff includes the usable CLI and tests, with credible evidence for durable imports, rollback, repeatable upserts, queries, and exports.
- **Failed:** A final solution produces wrong or partial database state, violates the public contract, or performs prohibited destructive actions. Ordinary missing/wrong functionality is Player failure.
- **Neutral:** An explicitly incomplete/declined attempt, budget exhaustion, or genuine access/execution blocker ends without a completion claim.
- **Simulator integrity (untrainable):** Fixture truth changes, private answers/provenance leak, or the GM fabricates database execution or contradicts stable state; these are not Player failures.
