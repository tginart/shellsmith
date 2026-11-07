# Matched BF16 backward qualification

These measurements qualify the original source snapshot. Subsequent [PR review corrections](review-corrections.json) pass fresh numerical, real-model learning, and dtype checks, but do not repeat the matched timings or legacy-update diagnostics.

The frozen backbone uses 8-bit weights. BF16 tensors cover activations, carries, adapters, gradients, and stochastic-rounding Adam.
The legacy profile remains selectable. No experiment or historical checkpoint is resumed.

## Longest preserved path

Each stage has three timed trials after a separate representative warmup. The path has 57,538 decoder rows and 21,675 supervised tokens.

| Configuration | Update trials (s) | Median update (s) | First ready (s) | Layer forward / backward (s) | Peak MLX / physical (GiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| Legacy | 324.47, 322.57, 322.51 | 322.57 | 349.97 | 81.93 / 233.97 | 93.43 / 93.27 |
| BF16 reference | 306.12, 306.29, 305.97 | 306.12 | 332.92 | 78.76 / 220.88 | 94.99 / 95.30 |
| BF16 + GDN | 291.68, 292.36, 292.11 | 292.11 | 318.69 | 76.32 / 208.93 | 94.99 / 95.27 |
| BF16 + fused attention | 211.09, 211.65, 211.70 | 211.65 | 238.04 | 57.25 / 146.82 | 80.28 / 82.03 |
| Saved projections | 198.65, 196.49, 198.13 | 198.13 | 223.83 | 57.55 / 132.38 | 193.28 / 195.84 |
| Saved statistics (final) | 197.54, 195.76, 196.49 | 196.49 | 222.31 | 57.80 / 130.80 | 193.28 / 195.85 |

Layer timings are disjoint. Head and optimizer work remain in complete-update time. Nested VJP build/evaluation times are not added twice.

## Complete preserved batch

All 22 paths, 444,145 decoder rows, 156,208 supervised tokens, 329 captured responses, and 15 rollouts retain the original masks and CISPO weights. Each trial performs one accumulated Adam update.

| Profile | Update trials (s) | Median update (s) | First ready (s) | Peak MLX / physical (GiB) |
| --- | --- | ---: | ---: | ---: |
| Legacy | 1818.33, 1822.13 | 1820.23 | 1848.10 | 93.86 / 93.69 |
| BF16 final | 1214.70, 1213.73 | 1214.22 | 1240.36 | 199.94 / 202.46 |

Complete-update speedups are 1.642x longest-path and 1.499x full-batch. Total backward, including the vocabulary head, is 1.776x faster on the longest path and 1.728x on the full batch.

Maximum reported net swap growth across the timed trials: 0 bytes.

Representative warmup medians (including first-call setup): longest legacy 6.32 s, BF16 4.66 s; full batch legacy 6.32 s, BF16 4.65 s. These are separate from update time and are included in first-ready latency.

## Performance gates

| Objective | Observed | Result |
| --- | ---: | --- |
| Longest update ≤210 s | 196.49 s | PASS |
| Full batch update ≤20 min | 20.237 min | NOT MET |
| Longest complete update ≥1.5x | 1.6416x | PASS |
| Full batch complete update ≥1.5x | 1.4991x | NOT MET |
| Longest total backward ≥2x | 1.7756x | NOT MET |
| Full batch total backward ≥2x | 1.7276x | NOT MET |
| Every attention backward primitive ≥2x | 1.808x, 2.561x | NOT MET |

## Final component costs

| Scope and component | Legacy forward / backward (s) | BF16 forward / backward (s) |
| --- | ---: | ---: |
| Longest gdn | 42.35 / 103.91 | 34.97 / 74.10 |
| Longest attention | 39.53 / 130.21 | 22.86 / 56.69 |
| Full batch gdn | 304.08 / 810.64 | 276.97 / 496.13 |
| Full batch attention | 158.69 / 505.58 | 126.24 / 258.86 |

## Attention primitive scope

| Query / KV rows | Backward from saved primal | Complete VJP | 2x backward objective |
| --- | ---: | ---: | --- |
| 2048 / 57344 | 1.808x | 2.178x | NOT MET |
| 2048 / 262144 | 2.561x | 2.565x | PASS |

## Numerical diagnostics

The gradient and Adam-update changes relative to legacy are large. These comparisons are diagnostics under the specified acceptance criteria; their completion status is not an agreement gate.

| Scope | Cosine | Relative L2 difference |
| --- | ---: | ---: |
| longest-gradients | 0.306889 | 1.138858 |
| longest-update-delta | 0.414508 | 1.086097 |
| batch-gradients | 0.199716 | 1.214052 |
| batch-update-delta | 0.408232 | 1.091862 |

Real-27B memorization and independent local derivatives define the requested correctness gate. Long-rollout reward improvement is not tested; the experiment remains stopped.

| Longest-path transition | Gradient cosine | Relative L2 difference |
| --- | ---: | ---: |
| legacy-reference | 0.234862 | 1.379395 |
| reference-gdn | 0.349871 | 1.086200 |
| gdn-attention | 0.463611 | 0.959846 |
| attention-saved | 0.999783 | 0.020822 |
| saved-statistics | 0.551835 | 0.958805 |

The explicit saved-activation VJP closely matches its preceding BF16 stage. Larger changes enter with numerical forward/kernel implementations, including the saved RMS realization.
For the longest legacy/final comparison, layer 0 gradient cosine is 0.0206 and layer 63 is 0.9650. Differences are larger toward earlier layers; this observation alone does not establish their cause.

## Retention and remaining reconstruction

The release counter trial takes 199.13 seconds with automatic admission and retains 108.43 GiB of optional outputs and 11.50 GiB of mandatory small residuals. Existing KV allocations and checkpointed hidden inputs are additional. Checkpoint-one backward consumes layer tapes.

Logical frozen-projection and nonlinear reconstruction calls; compiler pruning and fusion are not hardware counters.

The release counter run includes the qualified default and additional telemetry. It reproduces every timed gradient and Adam-result hash exactly; its build identity is recorded separately.

| Logical operation | Forward | Projection input-grad | Retained reads | Reconstructed |
| --- | ---: | ---: | ---: | ---: |
| attention_gate | 464 | 0 | 0 | 0 |
| attention_gate_vjp_primal | 0 | 0 | 0 | 450 |
| attention_prepare | 464 | 0 | 0 | 464 |
| attention_prepare_vjp_primal | 0 | 0 | 0 | 464 |
| gdn_inner | 1392 | 0 | 0 | 0 |
| gdn_inner_vjp_primal | 0 | 0 | 0 | 1392 |
| head_norm_statistics | 177 | 0 | 0 | 177 |
| head_softmax | 177 | 0 | 0 | 177 |
| input_norm_output | 0 | 0 | 0 | 1856 |
| input_norm_statistics | 1856 | 0 | 1856 | 0 |
| linear_attn.in_proj_a | 1392 | 1392 | 1392 | 0 |
| linear_attn.in_proj_b | 1392 | 1392 | 1392 | 0 |
| linear_attn.in_proj_qkv | 1392 | 1392 | 1031 | 361 |
| linear_attn.in_proj_z | 1392 | 1392 | 1392 | 0 |
| linear_attn.out_proj | 1392 | 1392 | 0 | 0 |
| middle_norm_output | 0 | 0 | 0 | 1842 |
| middle_norm_statistics | 1856 | 0 | 1842 | 0 |
| mlp.down_proj | 1856 | 1842 | 0 | 0 |
| mlp.gate_proj | 1856 | 1842 | 64 | 1778 |
| mlp.up_proj | 1856 | 1842 | 64 | 1778 |
| mlp_swiglu | 1856 | 0 | 0 | 1842 |
| self_attn.k_proj | 464 | 464 | 464 | 0 |
| self_attn.o_proj | 464 | 450 | 0 | 0 |
| self_attn.q_proj | 464 | 450 | 16 | 434 |
| self_attn.v_proj | 464 | 464 | 0 | 0 |
| vocabulary_head | 177 | 177 | 0 | 177 |

## Scope and limitations

- Representative 2048-row warmup is separate; new shape compilation can remain in complete-update times.
- Compiler wall time is not independently isolated. OS compiler caches are retained; warm component distributions describe steady-state primitives.
- Projection counters describe logical calls; compiler pruning/fusion and hardware instructions are separate.
- First ready means an in-memory Adam commit. Benchmark diagnostics and durable service-checkpoint I/O follow that boundary.
- Legacy gradient agreement is diagnostic. Independent derivatives and ten-update real-model learning define correctness.
