# Audit QLoRA's NF4 construction

## World

This concerns *QLoRA: Efficient Finetuning of Quantized LLMs*, arXiv:2305.14314 (2023-05-23). Support primary paper/code and numerical quantizer tests. Audit class: no-error control. Audit the NormalFloat construction: equiprobable bins under a zero-mean normal prior, asymmetric treatment needed to include an exact zero, blockwise scale normalization, and double quantization of scales. `NF4 is information-theoretically optimal` is conditional on the normal-prior/equal-probability scalar model, not a universal distortion theorem. Credible decoys are expecting uniform spacing, counting optimizer/LoRA state as 4-bit base weights, or forgetting metadata. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `nf4_levels_checked`, `memory_accounting_checked`, `substantive_error_found`, and `evidence`.

## Player

Find QLoRA's primary paper through ordinary web search, download the primary
source, and audit the NF4 quantizer construction and memory accounting. Determine
whether any substantive mathematical or algorithmic error exists; reproduce
levels or quantization behavior numerically.

## Reward

0.20 source; 0.30 quantizer derivation; 0.25 memory accounting; 0.15 decoy/scope analysis; 0.10 calibrated conclusion.

## Termination conditions

Succeed after a reproducible audit. Fail for fake code/evidence. Neutral for honest blocker. World contradictions are untrainable.
