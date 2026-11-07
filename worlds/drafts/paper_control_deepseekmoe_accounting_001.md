# Audit DeepSeekMoE expert accounting

## World

This audit concerns *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*, arXiv:2401.06066, posted 2024-01-11. Support primary report and official repository retrieval plus symbolic parameter/FLOP calculations.

Audit class: no-error control. Focus on fine-grained expert segmentation and shared-expert isolation. Splitting a conventional expert of intermediate width `d_ff` into `m` experts of width `d_ff/m` preserves aggregate expert parameters; activating `mK` fine-grained experts matches the activated width of `K` conventional experts, before separately counting always-active shared experts. Total parameters and activated parameters are different quantities. Credible decoys include holding top-k fixed after segmentation, omitting shared experts, counting router parameters as expert MLP width, or assuming equal load is guaranteed by arithmetic. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `segmentation_accounting_checked`, `activated_cost_checked`, `substantive_error_found`, and `evidence`.

## Player

Find and download the primary DeepSeekMoE report with `webFetch`. Audit the parameter and activated-computation accounting behind fine-grained expert segmentation and shared-expert isolation. Determine whether any substantive mathematical or algorithmic error exists, with a concrete symbolic or numeric example.

## Reward

- 0.20 source grounding.
- 0.30 correct total/active parameter derivation.
- 0.20 explicit example that includes shared experts and routing.
- 0.20 investigation of plausible accounting pitfalls.
- 0.10 calibrated conclusion. Penalize comparisons that silently change activated width.

## Termination conditions

Succeed after reproducible accounting and a defensible conclusion. Fail for fake evidence, oracle extraction, or intentional miscalculation. Neutral for access blocker/inconclusiveness. Simulation cracks are untrainable.
