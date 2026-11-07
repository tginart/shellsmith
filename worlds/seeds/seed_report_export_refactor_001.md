# Separate report execution, rendering, and publication

## World

Private authoring identity: `seed_report_export_refactor_001`; category_ids: `C02`, `C08`; task_size: `bounded_subset`.

Task size: `bounded_subset`. Categories: `C02`, `C08`. An operations team has a correct but tightly coupled notebook-to-HTML report caller. It now needs replaceable execution/rendering/writing boundaries and a useful decision brief, not generic cleanup or conversion alone.

Private seed provenance: `nbconvert`; canonical URL `https://github.com/jupyter/nbconvert`; roster SHA `1ff189e1b30f0820238e95427c035a9150a5e43f`.
Source scope: `README.md; docs/source/usage.rst (HTML export, embedded images, notebook execution/error policy, output and supporting-file handling).`
Role evidence: The inspected nbconvert usage grounds notebook execution, static HTML conversion, and resource handling as distinct behaviors; a task-specific operations report caller supplies the requested injectable boundaries and analytical deliverable without alleging an upstream coupling defect.

The GM has authority to author a small healthy caller, a trusted notebook, and synthetic weekly data with columns week/team/planned/completed/late. Use at least two teams and four weeks, including a zero-denominator row and unequal group sizes. Freeze data before play; independently derive aggregate truth from CSV, not cached notebook output. The caller may initially combine execution, rendering, and writing in one function while correctly handling the public policy. No upstream refactor target or vulnerable/fix commit is invented. The data is authored simulation data, not an additional public dataset seed.

This seed is one potential example solution and export-interface reference, not golden source, architecture, commands, or pixel output. The GM may choose fixture paths and adapters but must preserve the Player contract. Keep seed identity, URL/SHA, scope/role notes, private totals/recommendation, and rubric outside Player-visible prompt/files; only Player is projected. Environment provisioning and execution qualification are DEFERRED. Do not assume a kernel, exporter, browser, or image tool exists. No Docker, compiler, dependency installation, or cache setup is part of the task; unavailable execution is a blocker, never fictional evidence.

## Player

Refactor the supplied reporting caller and deliver its implementation/tests, a reproducible analysis notebook, standalone `report.html`, any local assets, and a short handoff explaining the recommendation and checks. Preserve `export_report(notebook_path, output_dir)` and its `report.html` destination. You may change internals and choose another compatible exporter.

Separate notebook execution, report rendering, and output publication behind documented replaceable boundaries. Tests must inject fake execution/rendering/writing implementations without starting a kernel or mutating global settings. The real path must use freshly computed notebook results, not trust cached outputs. Leave the supplied input notebook and CSV unchanged; save the reproducible notebook as a separate deliverable. A notebook execution error must fail the export without marking partial output as a completed report or replacing an existing complete report. Publish HTML and its assets as a coherent set.

The supplied CSV is synthetic weekly delivery data: week and team identify each unique row; planned, completed, and late are nonnegative integers with `late <= completed <= planned`. Reject malformed or duplicate rows. Analyze only observed weeks; do not invent missing activity. Completion rate is total completed divided by total planned. On-time rate is total completed minus total late, divided by total completed. Aggregate counts first; do not average row percentages. A zero denominator is unavailable, not zero percent.

Produce an executive summary, overall totals/rates, a team comparison table, a weekly trend chart with a readable text/table equivalent, a methods section, and a staffing follow-up recommendation. Recommend investigation of the team with the lowest aggregate on-time rate among teams with nonzero completed totals; break exact ties by team identifier. If none qualifies, say evidence is insufficient. Explain the selection using counts and denominators, and explicitly note the small synthetic sample cannot establish causation or a staffing effect. Display percentages to one decimal place while retaining unrounded calculations in the notebook.

HTML must be readable without network access: embed assets or bundle local relative assets, including the chart, with no required CDN/fonts/scripts. Include units/labels and meaningful report headings. Styling, DOM structure, and pixel appearance are not prescribed; content and usable local resource references are. Show that repeat exports preserve analytical content and that replacing a renderer does not reimplement execution or publication logic. Cover cell failure, missing assets, invalid input, and zero denominators.

Do not provision kernels, browsers, environments, or dependencies. If a qualified execution facility is unavailable, deliver reviewable source/tests and explicitly identify which fresh-execution or final-report checks remain pending; never claim a cached or fabricated result proves execution.

## Reward

Credit both causal refactoring and professional usefulness: independently replaceable stages, preserved entrypoint/publication semantics, verified fresh numerical content, a justified recommendation, honest limitations, and standalone resources. A nicely converted notebook with wrong denominators or no decision brief fails the artifact requirement; renamed stages that still share global execution state fail the boundary requirement. Check semantic content and local links, not inferred visual quality.

Alternative correct exporters, templates, pipelines, and charts receive full credit for public outcomes and credible evidence. No source-similarity, nbconvert markup, or golden pixel scoring applies. Ordinary analytical/code mistakes earn reduced credit. Deduct for changing source data, falsely claiming execution, silently publishing failed results, or dropping limitations. Private leakage, inconsistent frozen data, and fabricated GM execution are untrainable integrity failures.

## Termination conditions

- **Successful:** Final handoff includes the refactored pipeline and credible fresh-analysis evidence, plus the standalone substantive report, recommendation, reproducible notebook, and tests.
- **Failed:** A claimed completed solution has wrong analytics, incomplete/broken report resources, failed publication semantics, or no real replaceable boundary. Ordinary wrong solutions are Player failures.
- **Neutral:** An explicitly incomplete/declined or budget-exhausted attempt, or a genuine qualified-execution/access blocker, ends without claiming report completion.
- **Simulator integrity (untrainable):** The GM changes data truth, leaks private provenance/answers/rubric, contradicts artifact observations, or fabricates notebook/render execution. These are separate from Player failure.
