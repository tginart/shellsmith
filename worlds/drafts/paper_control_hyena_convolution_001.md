# Audit Hyena's implicit long convolution

## World

This concerns *Hyena Hierarchy: Towards Larger Convolutional Language Models*, arXiv:2302.10866 (2023-02-21; v3 2023-04-19). Audit class: no-error control. Audit that the implicit filter is evaluated by position, windowed, and used in a causal long convolution with gating; FFT padding must be at least the linear-convolution length to avoid circular aliasing. The advertised subquadratic sequence mixing follows from FFT convolution, not from filter generation being free. Credible decoys include insufficient padding, correlation-vs-convolution reversal in a toy implementation, or counting projection cost twice. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `causal_convolution_checked`, `complexity_checked`, `substantive_error_found`, and `evidence`.

## Player

Fetch the primary Hyena paper and official implementation. Audit its implicit-filter convolution, causal boundary, and complexity claim. Determine whether any substantive algorithmic error exists with a direct-versus-FFT test.

## Reward

0.20 source; 0.30 convolution derivation; 0.25 executable equivalence; 0.15 complexity; 0.10 calibrated report.

## Termination conditions

Succeed after a reproducible audit. Fail for fake execution. Neutral for access blocker. Contradictions are untrainable.
