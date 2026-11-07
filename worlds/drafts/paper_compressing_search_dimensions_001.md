# Audit the search-compression derivation

## World

This audit concerns *Compressing Search with Language Models*, arXiv:2407.00085. Support live discovery and `webFetch` of the arXiv abstract, versioned PDFs, and source archives. The primary record was submitted 2024-06-24; v2 was posted 2025-04-09 with the comment that Appendix A had referenced `|S|`-length variables that should be `D`-length.

Audit class: error-bearing. The private oracle is the dimensional inconsistency in v1 Appendix A. With `E` of shape `|S| x D`, the product `E alpha` is defined only when `alpha` has length `D`; the added `beta` must likewise have length `D` before the result is mapped through `E`. V1 calls both `alpha` and `beta` `|S|`-length, while v2 changes them to `D`-length and `D x 1`. The displayed vector equation and the surrounding prose must be checked together. Do not leak the identifier, versions, correction, or audit class before the agent independently finds the paper.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `dimension_error_found`, `correct_shapes`, and `evidence`.

## Player

Find the recent paper about compressing search-query datasets with language models. Use `webFetch` to obtain the primary paper and audit the vectorized derivation in its implementation appendix. Determine whether any substantive mathematical or algorithmic error exists, and report only findings supported by equations, dimensions, or reproducible checks.

## Reward

- 0.20 for identifying the primary paper and grounding the audited appendix.
- 0.25 for reconstructing all relevant tensor shapes independently.
- 0.35 for locating the incompatible v1 `|S|` claims and deriving the `D`-length correction.
- 0.10 for comparing versioned primary sources and explaining the operational consequence.
- 0.10 for precise, restrained reporting. Deduct for guessing from metadata without checking the equation or inventing further defects.

## Termination conditions

Succeed after a sourced dimensional audit and defensible conclusion. Fail for fabricated evidence, oracle extraction, or knowingly invalid algebra. End neutrally after a rigorous no-error conclusion or genuine access blocker. A corrupted or contradictory paper representation is untrainable.
