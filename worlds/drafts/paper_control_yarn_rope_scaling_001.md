# Audit YaRN's RoPE rescaling

## World

This concerns *YaRN: Efficient Context Window Extension of Large Language Models*, arXiv:2309.00071 (2023-08-31; v3 2026-02-06). Audit class: no-error control. Audit the per-dimension wavelength/ramp interpolation between extrapolation and interpolation and the attention-temperature correction. Endpoints must recover the stated regimes, paired rotary dimensions share frequency, and monotonic ramp bounds hold. Credible decoys include interpolating positions twice, mixing dimension index with pair index, or treating the empirically chosen temperature as a theorem. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `frequency_ramp_checked`, `endpoint_test`, `substantive_error_found`, and `evidence`.

## Player

Locate YaRN's primary paper and code. Audit its rotary-frequency interpolation and temperature scaling for context extension. Determine whether any substantive error exists and test boundary dimensions/scale one.

## Reward

0.20 source; 0.30 formula/endpoints; 0.25 numerical check; 0.15 assumptions; 0.10 restraint.

## Termination conditions

Succeed after a reproducible formula audit. Fail for fake results. Neutral for access blocker. Source inconsistency is untrainable.
