# Build a complete binary-inspection command

## World

Private authoring identity: `seed_hexyl_rebuild_001`; category_ids: `C01`; task_size: `full_small_project`.

A support team needs a portable, plain-text binary viewer for inspecting files and piped data. This C01 construction task is a FULL REPRO OF A SMALL PROJECT at the declared public product-contract level: a complete usable command, input handling, range validation, row rendering, help, tests, and documentation. It is not just a formatter helper or one offset flag.

Private provenance: seed `hexyl`, canonical repository https://github.com/sharkdp/hexyl, exact roster commit `6ecc29b9c8c84d08a7e860f7f69c22b113b480ea`.

Selected source scope is the README.md binary-viewer product description plus src/main.rs file-or-stdin acquisition, skip and length options, presentation selection, and error reporting, used to define the complete public plain-text viewer contract.

The hexyl seed supplies a substantive example of an end-to-end binary inspection command, connecting real byte acquisition and range selection to a readable display rather than serving only as a formatting reference.

The GM remains in control. The seed is ONE POTENTIAL EXAMPLE SOLUTION, not a golden patch, architecture, or command sequence. Respect the public requirements and independently evaluate substantively correct alternatives; never add hidden upstream quirks, ANSI styling, negative offsets, unit suffixes, or terminal-width heuristics. Keep provenance, seed roles, and rubric private; only Player enters the public brief. Environment provisioning and execution qualification are explicitly deferred: do not prescribe setup, containers, package installs, toolchain integration, hardware, or environment changes. A future GM can expose coherent workspace and execution surfaces without changing this contract. Actual execution must remain distinguishable from simulated observations.

## Player

Build a complete binary-inspection CLI, called `hexview`, in the designated project workspace. Deliver the implementation and a documented executable entry point, tests, help, and user documentation. Choose the implementation language and internal design. The command must handle actual files and standard-input bytes, not only embedded examples.

- Support `hexview [--skip N] [--length N] [--width N] [--plain] [FILE|-]` and `--help`. An omitted file or `-` reads stdin; otherwise read the one supplied path in binary mode. Support `--` before the positional path so a filename may start with a dash. Do not modify input files, decode input as text, or translate newlines.
- `--skip` defaults to zero and discards that many bytes from the start. `--length` limits bytes displayed after skipping; omission means through EOF. Accept nonnegative decimal counts and `0x`-prefixed hexadecimal counts through the unsigned 64-bit range; reject negative, malformed, and overflowing counts. Zero length, empty input, and skipping at or beyond EOF succeed with empty stdout. A length extending beyond EOF displays only available bytes. Skipping must work for non-seekable stdin as well as files.
- `--width` is a decimal integer from 1 through 64, default 16. Each output row contains at most that many consecutive selected bytes, starting a fresh row at the selected offset rather than rounding the skip down. Display each row's original zero-based input offset in lowercase hexadecimal, zero-padded to at least eight digits, followed by two spaces. Then render exactly WIDTH two-character byte slots separated by one space: present bytes are lowercase hex, missing final-row slots are two spaces. Follow the hex area with two spaces, a vertical bar, the character panel for present bytes only, another vertical bar, and a newline. Bytes from `0x20` through `0x7e` appear as their ASCII characters; all other bytes appear as a dot. Do not escape printable vertical bars or backslashes.
- For example, a width-four row containing bytes `00 41 7f ff` at offset zero is exactly `00000000  00 41 7f ff  |.A..|` followed by a newline. Rows continue at original offsets and the final partial row uses the padding rule above. Never squeeze repeated rows or add headers, borders, an EOF offset, or a trailing blank line. Default output and `--plain` both follow this contract, independent of terminal width or color settings; color and decorative styling are not required.
- Process bytes incrementally so memory need not grow with total input length or skipped-prefix length. No throughput target or hardware-specific limit is required. Help describes all supported options and exits successfully without reading input. Reject extra paths, unknown options, invalid arguments, directories, and unreadable paths with a concise stderr diagnostic and nonzero exit status, without a successful dump. Validate arguments before producing dump output. Mid-stream read or write failures report failure; already written rows need not be retracted. Success exits zero.

Test the actual command on named files and piped stdin, byte values spanning all 256 possibilities, invalid UTF-8, empty and partial rows, repeated rows, varied widths, range boundaries, a path containing spaces, and argument and I/O errors. Include independently computed expected text, not only self-generated snapshots. Document exact layout, input/range behavior, entry point, and exclusions. Work only in task-local files. Finish with the artifact and actual verification evidence, explicitly identifying anything not executed.

## Reward

Score public outcomes out of 100: 40 for a working end-to-end CLI and byte/range handling; 30 for exact declared rendering and validation; 20 for credible execution and independently checked file/stdin cases; 10 for help, documentation, and truthful handoff. Judge alternative languages, buffering approaches, and architectures by public outcomes, not source similarity. Plain output is sufficient; upstream colors and cosmetic fidelity earn no required credit. A library-only renderer or a text-mode substitute cannot earn completion credit. No performance claims may be invented. Simulator contradictions, private leakage, or falsely attributed execution make the episode untrainable, not a Player implementation failure.

## Termination conditions

- **Successful:** The usable CLI, documentation, and credible verification satisfy the complete declared file/stdin, range, rendering, and error contract.
- **Failed:** The Player claims completion with substantive byte corruption, incorrect offsets or layout, an unusable command, or fabricated evidence, or performs prohibited destructive actions outside task-local work.
- **Neutral:** The Player explicitly stops blocked, declines, exhausts the budget, or acknowledges incomplete work or unavailable execution without claiming success.
- **Simulator integrity (untrainable):** The GM leaks private context, changes byte fixtures or public output rules inconsistently, invents real execution, or loses coherent state. Ordinary wrong output remains Failed, never an integrity failure.
