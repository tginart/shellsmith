# Audit BitNet b1.58's ternary-bit claim

## World

This audit concerns *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*, arXiv:2402.17764, posted 2024-02-27. Support primary-paper retrieval, official-code inspection, and simple quantization experiments.

Audit class: no-error control. The name `1.58-bit` is `log2(3)=1.58496...`, the information needed per independently coded ternary symbol under an ideal uniform alphabet; it is not a claim that ordinary hardware stores a weight in a fractional fixed-width cell. The forward quantizer maps normalized weights to `{-1,0,1}` with a scale, while training retains higher-precision state and uses a surrogate gradient. Credible decoys include comparing to a two-bit packed implementation, forgetting scale/metadata, interpreting average entropy as physical cell width, or demanding the straight-through gradient equal the quantizer's true derivative. These caveats do not falsify the stated alphabet calculation. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `ternary_entropy_checked`, `quantizer_checked`, `substantive_error_found`, and `evidence`.

## Player

Locate and download the primary BitNet b1.58 paper with `webFetch`. Audit the mathematical and storage interpretation of “1.58 bits” and its ternary forward quantizer. Determine whether any substantive error exists, separating information-theoretic, packed-storage, and training-state claims.

## Reward

- 0.20 source grounding.
- 0.25 exact ternary alphabet/entropy calculation.
- 0.25 faithful quantizer and scale analysis.
- 0.20 nuanced storage/training interpretation and executable toy check.
- 0.10 calibrated conclusion; implementation caveats are not automatically paper errors.

## Termination conditions

Succeed after a sourced quantitative audit. Fail for fabricated claims/results, oracle probing, or knowingly false conclusions. Neutral for honest blocker. Source/tool contradictions are untrainable.
