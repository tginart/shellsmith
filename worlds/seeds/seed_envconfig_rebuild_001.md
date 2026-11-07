# Build a complete typed configuration package

## World

Private authoring identity: `seed_envconfig_rebuild_001`; category_ids: `C01`; task_size: `full_small_project`.

A small service team needs a reusable configuration package rather than another application-specific environment helper. This C01 construction task is a FULL REPRO OF A SMALL PROJECT at the declared public product-contract level: the complete usable Go package, validation, tests, example, and documentation. It is not an upstream compatibility exercise or a one-tag feature.

Private provenance: seed `envconfig`, canonical repository https://github.com/kelseyhightower/envconfig, exact roster commit `7834011875d613aec60c606b52c2b0fe8949fe91`.

Selected source scope is README.md usage, struct-tag, supported-type, and custom-decoder sections plus envconfig.go lookup, conversion, and validation behavior, used to define the complete public configuration-package contract.

The envconfig seed supplies a substantive example of a complete typed configuration package whose naming rules, conversions, validation, and decoder extension points serve real application callers.

The GM remains in control. The seed is ONE POTENTIAL EXAMPLE SOLUTION, not a golden patch, architecture, or command sequence. Respect the public requirements and independently evaluate substantively correct alternatives; never add hidden upstream quirks, including undocumented naming cases, fallback lookups, or error wording. Keep provenance, seed roles, and rubric private; only Player enters the public brief. Environment provisioning and execution qualification are explicitly deferred: do not prescribe setup, containers, package installs, toolchain integration, hardware, or environment changes. A future GM can expose coherent workspace and execution surfaces without changing this contract. Actual execution must remain distinguishable from simulated observations.

## Player

Build a complete Go configuration package for service developers in the designated project workspace. Deliver source, tests, a small caller example, and concise API and usage documentation. Choose the internal design and project layout. The package must work for arbitrary caller-defined structs, not just the example.

- Expose `Process(prefix string, target any) error`, which reads the process environment, and `ProcessWithLookup(prefix string, target any, lookup func(string) (string, bool)) error`, which uses the supplied lookup exclusively. Neither changes the environment. Accept a non-nil pointer to a struct; reject invalid targets and a nil lookup without panicking.
- For exported fields, uppercase the prefix and field name and join them with an underscore; an empty prefix adds no underscore. The `envconfig` tag replaces the field-name segment, still under the prefix. There is no unprefixed fallback. `split_words:"true"` inserts word boundaries before capitals following lowercase letters or digits and before the last capital of an acronym followed by lowercase letters: `HTTPPort` becomes `HTTP_PORT`, `Port2` stays `PORT2`. An explicit name overrides splitting. Named nested value structs add their own name segment; anonymous embedded value structs flatten into the containing prefix. Ignore unexported fields and `ignored:"true"` fields completely. Reject conflicting effective keys.
- Distinguish absence from a present empty string. A `default` tag, including an empty default, applies only when the key is absent. `required:"true"` rejects absence without a default; present empty values satisfy presence and then undergo normal conversion. Optional absent fields without defaults keep their existing values.
- Support strings, booleans accepting case-insensitive `true` or `false`, range-checked decimal signed and unsigned integers, finite decimal floating-point values, and Go duration strings for `time.Duration`. Support comma-separated slices of these scalar types and comma-separated maps with string keys and scalar values, using the first colon to separate each key and value. Trim surrounding whitespace on collection items, keys, and values; preserve standalone strings. Empty collection input means an empty collection. Reject missing map separators, empty map keys, duplicate map keys, overflow, and malformed scalar values. Collection quoting and escaping are outside scope.
- For a field implementing `Decode(string) error`, then `Set(string) error`, then `encoding.TextUnmarshaler`, use the first available hook in that order before built-in conversion or nested traversal. Support pointer-receiver hooks on addressable fields. Pass the original value unchanged and propagate hook failures. Ordinary pointer fields, recursive struct types, nested collections, and binary-decoder hooks are outside scope; other unsupported exported field types must produce an error rather than silently succeed.
- Package-generated conversion diagnostics identify the effective key, field path, and destination type without embedding raw configuration values. Make underlying conversion or hook causes inspectable. Document whether failure can partially update the target; either policy is acceptable if accurately implemented.

Tests must cover custom callers, prefixes, nested and embedded fields, all declared tags and conversions, missing versus empty values, malformed inputs, invalid targets, collisions, and decoder precedence and failure. Use injected lookups for deterministic cases and include an environment-reading API check. Document the entire supported contract and exclusions. Work only in task-local files. Finish with the artifact entry points, verification actually performed, and any unverified behavior; do not invent passing executions.

## Reward

Score public outcomes out of 100: 45 for the complete callable package and naming/presence contract; 25 for conversions, validation, and hooks; 20 for credible executed tests and independently checked public cases; 10 for a usable example, documentation, and accurate handoff. Evaluate alternate architectures and the publicly permitted partial-update policies on their merits, not resemblance to the seed. A hardcoded example, ignored errors, or fabricated passing results cannot earn completion credit. Do not infer runtime qualification from this authored spec. Simulator contradictions, private leakage, or falsely attributed execution make the episode untrainable, not a Player implementation defect.

## Termination conditions

- **Successful:** The Player hands off the complete package, caller example, documentation, and credible verification satisfying the public contract.
- **Failed:** The Player declares completion with substantive contract violations, an unusable package, or fabricated verification, or performs prohibited destructive actions outside task-local work.
- **Neutral:** The Player explicitly stops blocked, declines, exhausts the budget, or reports incomplete work without claiming success; unavailable execution is an acknowledged qualification gap.
- **Simulator integrity (untrainable):** The GM leaks private context, changes the contract after the fact, presents invented execution as real, or loses coherent state. An ordinary incorrect solution remains Failed, never an integrity failure.
