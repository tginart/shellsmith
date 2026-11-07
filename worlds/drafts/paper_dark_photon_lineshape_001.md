# Audit a dark-photon lineshape calculation

## World

This audit concerns *First search for dark photon dark matter with a MADMAX prototype*, arXiv:2408.02368, submitted 2024-08-05 and revised 2025-03-07. Support primary version download and numerical evaluation of the stated halo lineshape.

Audit class: error-bearing. V1 uses `sigma_v=218.0 km/s` in the expected signal lineshape where the cited convention requires the one-dimensional dispersion `154.1 km/s` (the former is larger by `sqrt(2)`). It consequently says the signal spans about seven bins. V2 uses `154.1 km/s`, about five bins, updates the matched filtering, changes the SG efficiency from `0.87` to `0.91`, and improves the exclusion from `3.0e-12` to `2.7e-12` with peak `1.2e-13` to `1.1e-13`. The audit must distinguish this derivation error from unrelated presentation changes and reproduce the width scaling. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `versions_compared`, `lineshape_error_found`, `limit_consequence_verified`, and `evidence`.

## Player

Locate the MADMAX prototype paper reporting its first dark-photon dark-matter search. Use `webFetch` to obtain primary versions and audit the derivation and discretization of the expected signal lineshape used in matched filtering. Determine whether any substantive error exists and quantify its downstream effect.

## Reward

- 0.15 source/version grounding.
- 0.25 reconstruction of velocity convention, frequency width, and binning.
- 0.30 identification of the `sqrt(2)` dispersion mismatch.
- 0.20 verification of the revised filter/limit consequences.
- 0.10 calibrated reporting. Deduct for treating changed experimental numbers as proof by themselves.

## Termination conditions

Succeed after a reproducible lineshape audit. Fail for fabricated data, oracle probing, or knowingly invalid physics. Neutral for careful inconclusiveness or source failure. Contradictory source/data delivery is untrainable.
