# Audit streaming attention with sink tokens

## World

This concerns *Efficient Streaming Language Models with Attention Sinks*, arXiv:2309.17453 (2023-09-29; revised primary record). Support paper/code fetches and tiny attention tests. Audit class: no-error control. Audit cache eviction with fixed initial sink tokens plus a rolling recent window: absolute/rotary position handling must follow the implementation, evicted middle tokens cannot be attended, and sinks stabilize softmax mass without recreating lost content. Credible decoys include reindexing RoPE inconsistently, comparing against full attention as an exact identity, or claiming sinks store summary information. Keep private.

The terminal `task_outcomes.paper_audit` object must contain `primary_source_found`, `cache_policy_checked`, `position_test`, `substantive_error_found`, and `evidence`.

## Player

Find the primary attention-sinks/StreamingLLM paper and code. Audit its cache policy and positional treatment with a small streaming example. Determine whether any substantive algorithmic error exists.

## Reward

0.20 source; 0.30 cache semantics; 0.25 position test; 0.15 claim scope; 0.10 restraint.

## Termination conditions

Succeed after reproducible streaming audit. Fail for fake execution. Neutral for blocker. Contradictions are untrainable.
