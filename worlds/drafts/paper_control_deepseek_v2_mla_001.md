# Audit latent-attention cache accounting

## World

This audit concerns *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, arXiv:2405.04434 (2024-05-07; v5 2024-06-19). Allow primary and official-code web fetches and symbolic shape accounting.

Audit class: no-error control. Focus on Multi-head Latent Attention. The unrotated key/value content is generated from a cached low-rank latent; the small decoupled RoPE key must also be cached because position rotation prevents simply absorbing that path into a fixed projection. Cache accounting must therefore include the latent plus RoPE component, not full per-head K and V. Matrix absorption/reassociation preserves the content dot products and output projection in exact arithmetic. Credible decoys include dropping the RoPE cache, counting training intermediates as inference cache, or applying absorption through a position-dependent rotation. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `mla_shapes_checked`, `cache_accounting_checked`, `substantive_error_found`, and `evidence`.

## Player

Locate the DeepSeek-V2 primary report and download it using `webFetch`. Audit the derivation and inference-cache accounting of Multi-head Latent Attention, including its decoupled positional component. Determine whether any substantive mathematical or algorithmic error exists.

## Reward

- 0.20 source grounding.
- 0.30 faithful matrix/shape derivation.
- 0.25 exact cache accounting and comparison with MHA.
- 0.15 investigation of RoPE/absorption edge cases.
- 0.10 calibrated conclusion; rigorous validation can earn full credit.

## Termination conditions

Succeed after a complete focused accounting audit. Fail for invented formulas, oracle probing, or false execution claims. Neutral for bounded inconclusiveness/access failure. Inconsistent evidence simulation is untrainable.
