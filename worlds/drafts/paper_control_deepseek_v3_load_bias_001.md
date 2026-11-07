# Audit auxiliary-loss-free expert balancing

## World

This audit concerns *DeepSeek-V3 Technical Report*, arXiv:2412.19437 (2024-12-27; v2 2025-02-18). Support primary/official-code fetches and small routing simulations.

Audit class: no-error control. The focused rule maintains per-expert routing biases used only for top-k selection: after a batch, decrease the bias of overloaded experts and increase the bias of underloaded experts by a small step. Selection bias must not be multiplied into the final gating weight, preserving the learned score semantics. This negative-feedback sign is correct locally; it is not a proof of global convergence under arbitrary nonstationarity. Credible decoys include reversing the overload sign, updating model logits directly, expecting exact balance each batch, or conflating the sequence-level auxiliary loss retained for another purpose. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `bias_rule_checked`, `routing_simulation_checked`, `substantive_error_found`, and `evidence`.

## Player

Locate and download the DeepSeek-V3 technical report with `webFetch`. Audit its auxiliary-loss-free load-balancing update as a feedback algorithm: check signs, where bias enters routing, and behavior in a toy expert system. Determine whether any substantive error exists.

## Reward

- 0.20 source grounding.
- 0.25 exact update/routing reconstruction.
- 0.25 toy dynamics that test overload and underload cases.
- 0.20 distinction between local negative feedback and stronger convergence claims.
- 0.10 calibrated conclusion; reward rigorous confirmation.

## Termination conditions

Succeed after a reproducible feedback audit. Fail for fake simulations, fabricated text, or oracle probing. Neutral for bounded inconclusiveness/access failure. Contradictory world output is untrainable.
