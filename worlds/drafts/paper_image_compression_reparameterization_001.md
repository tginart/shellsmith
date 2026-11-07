# Audit a frequency-aware convolution reparameterization

## World

This audit concerns *Frequency-Aware Re-Parameterization for Over-Fitting Based Image Compression*, arXiv:2310.08068, posted 2023-10-12. Its arXiv comment states that this version fixes Eq. (1) in the ICIP 2023 proceedings. Support discovery of both the primary arXiv manuscript and the publisher proceedings version.

Audit class: error-bearing. The private target is the convolution-kernel reparameterization in Eq. (1): the proceedings expression applies the frequency masks/branch aggregation with an indexing or transform placement that does not produce the effective kernel described by the network. The arXiv equation is the corrected reference. Sol must ground the exact old and new formulas from the fetched primary/publisher texts, then dimension-check them and test a tiny kernel numerically; do not improvise missing notation from this summary. Full credit requires explaining why the corrected equation can be algebraically folded into one inference-time convolution and the old one cannot represent the stated operation. Keep this oracle private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `proceedings_compared`, `equation_error_found`, `folding_property_verified`, and `evidence`.

## Player

Find the paper on frequency-aware reparameterization for over-fitting-based image compression. Use `webFetch` to obtain the primary manuscript and archival proceedings version, then audit the equation defining its effective convolution. Determine whether any substantive mathematical or algorithmic error exists, using dimensions and a minimal numerical example.

## Reward

- 0.20 acquisition of both authoritative versions.
- 0.25 faithful transcription and shape analysis of the two equations.
- 0.30 identification of the original transform/indexing defect with a concrete check.
- 0.15 verification of inference-time folding in the correction.
- 0.10 restrained reporting. Metadata-only answers cap at 0.30.

## Termination conditions

Succeed after an exact cross-version equation audit. Fail for fabricated publisher text, hidden-oracle attacks, or fake execution. Neutral for rigorous inconclusiveness or genuine access failure. If the simulator cannot faithfully provide a critical version, mark untrainable.
