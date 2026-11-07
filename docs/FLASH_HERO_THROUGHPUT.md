# Flash hero epoch: end-to-end throughput (2026-10-22)

Run before: `studio-flash-hero177-20261021T080529Z-r3` (R, left intact).
Run after: `studio-flash-hero177-20261022T213210Z-perf2` (clean restart; stopped
during update 2 to add tiled prefill).
Tiled-prefill run: `studio-flash-hero177-20261023T010937Z-perf3`. It halted in
update 1 on GM provider errors (see below).
Live run: `studio-flash-hero177-20261023T031605Z-perf4` (clean restart of perf3).
Source snapshots: perf2 `c2badbcae71c` (manifest `062881b9…`), perf3 `2f50b7e5bb7b`
(manifest `1d39cd22…`). Benchmarks and validation:
`user@studio-host:~/.local/state/shellsmith/flash-perf-20261022/bench`.

## Verdict

One update (2 worlds) took 12,440 s in r3, 7,198 s in perf2, and 6,400 s in
perf4. Committed throughput rose from 0.58 to 1.13 worlds per hour (1.94x).
The full epoch projects to about 6.6 days instead of 12.8 days. Attempts, budgets, LoRA,
precision, GM settings, and the loss are unchanged. Most of the gain comes
from replay. Tiled prefill made player inference faster per sample (prefill 4x,
decode 1.8x), but rollout wall time moved only 5%. GM turns were slower
(49 s vs 42 s mean), and the slowest episode sets the collection time.

## Lifecycle, update 1 (same two worlds)

| Phase | r3 | perf2 | perf4 | Ceiling or note |
|---|---|---|---|---|
| Rollouts (collection) | 4,600 s | 4,217 s | 4,002 s | Bound by the slowest of 16 episodes; GM is ~45% of episode time |
| Player prefill per sample | — | 4.73 s | 0.97 s | Tiled prefill |
| Player decode per request | — | 7.0 tok/s | 12.9 tok/s | Batched decode is 94 ms/step at 8 rows |
| GM turn mean | — | 41.7 s | 49.3 s | External provider |
| Replay forward+backward | 7,815 s (536k tok, 24 seq) | 2,952 s (573k tok, 33 seq) | 2,364 s (517k tok, 24 seq) | ~260 s at 75 TFLOPS (36 GFLOP/token) |
| Replay per token | 14.6 ms | 5.2 ms | 4.6 ms | |
| Optimizer | 15 s | 2.7 s | 2.6 s | |
| Checkpoint + publish + rest | ~60 s | ~26 s | ~30 s | |
| Whole update | 12,440 s | 7,198 s | 6,400 s | |
| Failed episodes | 0 / 16 | 0 / 16 | 0 / 16 | |
| Prefix-cache hit | 94.1% | 94.3% | 94.9% | |
| Peak allocator | 242 GiB | 248.4 GiB | 240.3 GiB | Budget 240 GiB, physical 256 GiB, no swap growth |

perf2 and perf4 run the same worlds, but trajectories differ (sampling is
stochastic). So per-sample and per-token rates are the fair comparison. Memory
headroom did not improve: replay still peaks at the budget. See the limits below.

## Code changes (commits after the baseline import `8845f2b`)

- `60c449c`, `4f4f5a7`: segment-trie replay (`replay_layout = "segment_trie"`).
  One forward/backward per shared token-prefix segment across all sequences.
  Update 1 ran 202 of 298 segments.
- `732bcb3`: grouped routed-expert training tiles (`training_moe_chunk = 32`).
  Static-capacity expert tiles with `gather_qmm`, no host sync. MoE layer
  forward is 4.9x faster and the VJP 4.7–7x faster.
- `43ce70d`: one joined device table for the PLE n-gram shards. Bitwise equal.
  Lookup is 63 ms → 0.3 ms at 2048 rows, and decode PLE is 32 ms → 0.6 ms at 8 rows.
- `c2badbc`: bf16-fast restore admission. The old bound assumed FP32 optimizer
  maps (16x the adapter bytes, 261 GB vs a 239 GB budget). Every Flash
  restore failed, and the service stopped. The new bound is 10x (71 GiB). A measured
  restore took 7.5 s with a +49.7 GiB peak.
- `222c8d0`, `2f50b7e`: tiled prompt prefill (`prefill_moe_chunk = 32`,
  `prefill_tokens = 1024`). Prefill forwards of 256 or more tokens use the
  replay expert tiles; shorter forwards and decode keep sorted rows. See
  "Prefill and decode interleaving" below.
- `8167806`: the epoch plan declares episode and GPU slot counts. Child
  processes import the frozen snapshot. The supervisor checks the service source hashes.
- `service.py`: progress commits to sqlite at most every 30 s.
- Tooling: `scripts/reconstruct_studio_update.py`, `diagnose_flash_numerics.py`,
  `profile_flash_replay.py`, `profile_flash_decode.py`, `bench_flash_moe.py`,
  `bench_flash_ple.py`, `validate_replay_layout.py`,
  `prepare_studio_epoch_root.py`, `smoke_studio_lifecycle.py`.

## Correctness gates

- Production parity: the new code reproduces r3's recorded replay logprobs
  bitwise (`validate-v4`, `validate-v5`).
- Trie replay: logprobs bitwise. Gradient relative L2 is 0.031 against a
  reversed-order noise floor of 0.029, cosine 0.9995.
- Grouped MoE tiles route-flip at BF16, as any layout change does. So a
  gate was declared before the run:
  - Agreement with the sampler is no worse than baseline, within 5%. Measured
    mean |Δ| 0.1205 vs 0.1214, p99 1.625 vs 1.625, max 5.6 vs 7.1.
  - The gradient distance to baseline is no larger than that of the supported
    1024-token segment layout. Measured 0.90 vs 0.93.
- Tests: 28 Flash tests pass, including new trie-shape, tree-parity, and PLE
  bitwise tests.
- Lifecycle smoke: smoke 2 covered rollout, update, save, and publish. Smoke 3
  covered interrupt, restart, restore, and resumed rollouts, with no repeated
  create, update, or optimizer step.

## perf4 host hang and the replay memory fix (2026-10-25)

perf4 committed update 1, then hung the Studio in update 2's forward_backward
(kernel watchdog panic `panic-full-2026-10-25-110157.0002.panic`). The service
held 228 GiB active, with a 240 GiB peak, at branch backward segment 11 of 35.
Free memory was near zero. The batch is saved as
`evidence/perf4-update2-backward.json` (35 sequences, 1.0M tokens, longest 54k).

Root cause, part 1: adapter-gradient copies. The routed-expert LoRA gradient is
7.1 GiB per full copy. `BalancedSums` kept one partial per binary level and
key, and `_add_gradient` also kept an evaluated running total. Trie replay has
two such sums at once: one over the segments of the node in backward, and one
over the finished nodes of the tree. On update 2 this reached 9 full copies
(64 GiB). Admission priced 2 copies, and `mx.set_memory_limit` is soft. The
KV and carry adjoints are small (~1.5 GiB).

Root cause, part 2: resident branch inputs. The engine built each branch input
as one lazy `(1, L, 10240)` tile. The first segment slice evaluated the whole
tile, and it stayed resident until the replay ended. On update 2 that grows to
19 GiB (1.0M tokens × 20 KB). No estimate counted it. The first fixed probe found it:
between trie nodes, memory climbed from 151 to 196 GiB.

The fix (`024222e`, `122b72d`):

1. Lazy totals. The sums keep only partials. The consumer computes the
   total at the point of use, with the same additions in the same order. This
   removes 2 copies (9 → 7 on both update batches).
2. Real trie admission. `tree_gradient_copies` runs the same carry code on
   integer stand-ins in run_tree order. That gives the exact peak copy count.
   Admission then adds that count times the gradient bytes, the head rows,
   and the path estimate.
3. Fallback. If no checkpoint interval fits, admission caps the sum levels
   (4, 3, 2, 1). The two lowest partials merge, so at most 2 × cap copies
   exist. This changes only the BF16 rounding order. The planned fallback, sequential
   sub-groups, did not help: a simulation of 1–6 splits kept 6–8 copies,
   because the long nodes dominate and each split adds an optimizer copy.
4. Guard. At each replay yield, the engine raises `MemoryError` when active
   memory exceeds the admission budget (222.7 GiB) or `kern.memorystatus_level`
   is below 8. The service then aborts the update cleanly.
5. Per-segment inputs. Trie replay embeds each segment from its token-ID
   slice. Each row depends only on its ID, so the values are bitwise the same.
6. Host headroom 16 → 32 GiB (allocator 224 GiB).

Gates (bench `memfix-v1`, `memfix-v2`; inputs: r3 update 1, indices 0 1 11 12):

| Check | Result |
|---|---|
| Old source vs old source (run-to-run floor) | logprobs, loss bitwise; gradient rel. L2 0.0275, cosine 0.99962 |
| New exact sums vs old | logprobs, loss bitwise; rel. L2 0.0276, cosine 0.99962 (the floor) |
| Same recorded layer VJPs, tiny trie (9 nodes) | gradients bitwise equal, old and new |
| Level cap 1 vs exact (declared: rel. L2 ≤ 0.029, cosine ≥ 0.9995) | 0.0276, 0.99962: pass |
| Peak, 4 sequences: old c=2 / new exact c=8 / cap 1 c=8 | 200.6 / 189.0 / 181.8 GiB |

The layer VJPs are not bitwise reproducible from run to run on the GPU, in the old source
too. So the summation change was checked bitwise with recorded VJP outputs
(`scripts/tiny_tree_replay_bits.py`), and against the run-to-run floor on the
real model.

Probe of the perf4 update 2 batch (source 122b72d77393, headroom 32 GiB):

| Measure | Value |
|---|---|
| Admission | interval 10, gradient sums capped at 2 levels |
| Estimated peak | 206.1 GiB (ratio 0.998 to measured) |
| Measured peak | 206.48 GiB (perf4 hung the host above 224 GiB) |
| Replay time | 3,843 s |

The estimate ratio is close to 1 because two errors cancel.
`persistent_bytes` counts the optimizer master and adapter buffers twice (+7 GiB).
The replay model omits the ~9 GiB per-layer-group backward transient.
The update 1 probe and the lifecycle smoke test were skipped to save time.

perf5 (root `studio-flash-hero177-20261026T000627Z-perf5`) halted in update 1
rollouts before any replay. The Codex GM workspace ran out of credits, and
`provider_outage = "halt"` stopped the run. No perf5 throughput exists yet.

## Replay backward performance (2026-10-25, branch `claude/flash-backward-perf`)

Target: replay of the perf4 update 2 batch at ≥70% of its FLOP floor
(20.47 PFLOP, 273 s at 75 TF). The target is not met. End to end, replay runs
2.78x faster, at 19.7% of the floor.

| Probe (`scripts/probe_replay_memory.py`, update 2) | old `122b72d` | new |
|---|---|---|
| Replay time | 3,843 s | 1,383 s |
| % of floor | 7.1% | 19.7% |
| Peak | 206.5 GiB | 195.8 GiB (budget 222.7) |
| Admission | interval 10, 2 sum levels | interval 3, 1 sum level (est. +47.5 GiB, measured +37.7) |
| Forced interval 10 | 3,843 s | 1,355 s (20.2%), peak 194.6 GiB |

With one VJP per group, each layer runs its forward once in the VJP,
whatever the interval. So intervals 3 and 10 cost the same compute
(1,383 s and 1,355 s), and admission can pick for memory alone.

Commits, in order:

| Commit | Change |
|---|---|
| `ffa9b7e` | `scripts/profile_flash_replay.py` per-component profile; exact per-expert sorted MoE segments in training |
| `baaea31` | Sparse QSA attention forward and VJP Metal kernels (only the selected blocks) |
| `b03242f` | One VJP per checkpoint group instead of a forward per layer plus a VJP per layer |
| `109a76e` | Sorted routed-expert dX kernel (`expert_sorted_dx.metal`); the forward stays MLX `gather_qmm` |
| `25f57c5` | Fused hyper-connection norm forward, norm backward, and mix backward kernels |
| `31f835f` | Admission prices the group-VJP transient (576 KiB per token and layer, measured) |
| `55db941` | Window-major sparse QSA backward, so dK/dV atomics stay in the GPU cache |
| `8c73c47` | Explicit-inverse GDN chunk solve (training backend `metal_inverse`; inference unchanged) |
| `05d8cb2` | Profile uses the training kernels; models one VJP per group |
| `4614c5a` | Trie head: one VJP per tile gives logprobs and a unit-cotangent basis; no head recompute |
| `2c3eac9` | GPU scripts wait on the Studio lease (blocking `flock`) |

Per-component model of the update 2 batch (`profile_flash_replay.py`, seconds).
v1 is the baseline and uses the old schedule (2 × forward + VJP per layer).
v6 is the final code and uses one forward + VJP per layer. So the
"modelled" columns are not on the same basis; the probe rows above are the
end-to-end truth.

| Component | v1 fwd / VJP / modelled | v6 fwd / VJP / modelled | Floor |
|---|---|---|---|
| attention HC | 21 / 87 / 130 | 12 / 54 / 66 | 12.4 |
| mixer projections | 40 / 71 / 152 | 41 / 72 / 113 | 107.2 |
| mixer core (GDN, QSA) | 221 / 678 / 1,119 | 57 / 306 / 364 | 25.3 |
| MLP HC | 18 / 86 / 122 | 13 / 51 / 63 | 12.4 |
| routed MoE | 163 / 883 / 1,209 | 91 / 191 / 283 | 96.2 |
| shared expert | 9 / 27 / 45 | 9 / 29 / 38 | 12.1 |
| other (norms, PLE, glue) | 12 / 390 / 413 | 2 / 62 / 65 | 0 |
| head | 19 | 21 | 6.0 |
| Total | 3,324 (8.2% of floor) | 1,108 (24.6%) | 273 |

The probe (1,383 s) is above the v6 model (1,108 s). The in-situ phase times
below put the gap in the layer math, not in trie overhead.

Kernel benchmarks (Studio, one layer at the profile shapes):

| Kernel | Before | After |
|---|---|---|
| Routed MLP VJP | 21.1 ms | 12.1 ms |
| HC VJP | 5.13 ms | 2.23 ms |
| HC forward | ~1.5 ms | 1.03 ms |
| QSA attention backward, 47k keys | 84 ms | 48 ms |
| GDN chunked forward / VJP (local M-series) | 5.7 / 18.9 ms | 4.9 / 16.1 ms |
| Group VJP, 10 layers | 0.913 s (per-layer schedule) | 0.657 s |

Correctness. The kernels change the FP32 sum order, so the forward is not
bitwise. In BF16 that flips MoE routes, and the update gradient moves as much
as between any two valid layouts. Subset gate (sequences 12, 22, 23, interval 10):

| Check | Gradient cosine | rel. L2 | Logprobs |
|---|---|---|---|
| new vs old | 0.523 | 0.999 | mean abs diff 0.09–0.15 |
| control: old, MoE tile 32 → 64 (bitwise forward) | 0.99965 | 0.0265 | bitwise |
| control: old, 1024-token segments (non-bitwise forward) | 0.538 | 0.981 | not bitwise |

The new code sits at the same distance from old as the 1024-token control
from old. Is that layout noise or a bug? Three checks answer it
(`scripts/compare_replay_bitwise.py --fixed-routes`, `--reference-kernels`,
`--fp32-reference`, and `scripts/check_training_kernel_accuracy.py`).

1. Fixed discrete choices. `--fixed-routes` makes the MoE top-k and the QSA
   block top-k fixed functions of layer and position, in both sources. The
   function is then smooth. New vs old goes from 0.523 to 0.929; the 1024-token
   control goes from 0.538 to 0.955. So route and block flips cause almost all
   of the 0.52. The cosine falls from 0.9995 at layer 47 to ~0.85 at layers 0–17,
   because BF16 differences grow through the backward pass.
2. Bisect (fixed routes, sequences 12, 22, 23). One kernel at a time goes back to
   its MLX path. Mean cosine to old over layers 0–17: all new kernels 0.845; HC
   back 0.902; GDN back 0.894; QSA back 0.861; experts back 0.846; all four back
   0.909; 1024-token control 0.903. Only these kernels move the gradient, mostly
   the HC. "Closer to old" is not "more correct", so check 3 is necessary.
3. FP32 reference (fixed routes, sequences 12 and 23). The old source runs with
   FP32 activations, carries, and gradients and the FP32 GDN reference scan.

   | BF16 run | Cosine to FP32 | Layers 0–17 | Layers 18–47 |
   |---|---|---|---|
   | old, 2048-token segments | 0.747 | 0.598 | 0.830 |
   | old, 1024-token segments | 0.735 | 0.578 | 0.837 |
   | new, 2048-token segments | 0.732 | 0.570 | 0.833 |
   | new, 1024-token segments | 0.755 | 0.617 | 0.839 |

   The new values bracket the old values. New vs old (0.951) equals old vs
   old-1024 (0.946). Every BF16 run is far from FP32 in the same way. A probable
   cause is the GDN decay g: BF16 rounds g near 1 to steps of 0.004.
4. Per-kernel accuracy on real layer inputs (sequence 22, layers 0–5, relative
   L2 error to FP32): HC input gradient 0.0042 upstream, 0.0038 new kernel;
   HC outputs equal. GDN substitution and explicit inverse agree to 1% on
   every output and gradient (y 0.0094, dk 0.029, dv 0.020).

Verdict: no bug. The 0.52 is BF16 layout noise. The 0.9996 gate holds only for
changes with a bitwise forward. The acceptance check for a change to the
forward numerics is:

- each changed kernel is no less accurate than the path it replaces, against
  FP32 on real activations (check 4);
- with fixed routes, the gradient cosine to the FP32 reference is inside the
  range of the valid old layouts (check 3; now 0.735–0.747);
- mean |replay − behavior logprob| is no worse than old. Over 5,711 masked
  tokens it is 0.1082 for old and 0.1067 for new; p99 and the maximum are no
  worse.

Speed checks after v6 (subset 12, 22, 23, interval 10, in-situ phase times
from `probe_replay_memory.py --indices`):

| Run | Total | Layer forwards | Group VJPs | Head | Host graph build |
|---|---|---|---|---|---|
| v8 | 179.8 s | 40.9 s | 135.1 s | 0.6 s | — |
| v14 (build timed) | 182.4 s | 41.2 s | 137.3 s | 0.6 s | 2.2 s |
| `MLX_MAX_OPS_PER_BUFFER=20` | 186.7 s | 41.7 s | 140.8 s | 0.7 s | — |
| 4096-token segments | 191.1 s | 50.5 s | 137.5 s | 0.7 s | — |

The layer math is 98% of the replay. Host graph build, heads, sums, and the
objective are ~2%, so asynchronous evaluation cannot help. In situ, a group
VJP costs 84 ms per layer and segment, against 66 ms for the isolated group in
the v6 profile; this, not overhead, is the gap between the probe and the model.
`MLX_MAX_OPS_PER_BUFFER=20` gives nothing for replay. Longer segments do not
lower the cost per token, and they lose the shared 2k prompt segment. Dense BF16
HC weights would save ~2.4% (HC VJP 2.41 → 1.87 ms, forward 1.88 → 0.96 ms).
Most of that comes from the row-invariant injection projection, which keeps the
replay numerics equal to decode, so it is not applied.

What is left, largest first (v6 model):

1. Mixer core VJP, 306 s against a 25 s floor. QSA backward is 70–80%
   fp32 atomics (~123 G/s on the Studio, even in cache). The selection is
   spread evenly (density in used 64×256 tiles 0.22, the same as overall), so
   block-dense tiles waste ~4x. Grouping four adjacent queries cuts the atomics
   2.2x but adds 1.8x compute: ~10 ms of ~60 ms per layer and segment. The GDN
   chunk path is many unfused MLX ops. A fused FLA-style chunk kernel could
   save ~70 s.
2. Routed MoE, 283 s against 96 s. The sorted GEMMs run at 34 TF forward and
   12.8 TF dX in MLX. A custom forward kernel was slower than MLX and was
   dropped. Twice the rows per expert (4096-token segments) did not help.
3. HC VJPs and glue, ~190 s. `mx.compile` does not fuse the reductions.
4. The forward recompute in each group VJP is ~23% of the replay. Only the
   last segment of a leaf (15% of segments) could skip it.

No item above is worth more than ~5%. 70% of the floor (≤390 s) needs about
3.5x: fused whole-layer kernels for GDN, MoE, and attention near peak
throughput. That is a rewrite of the layer math, not a tuning step.

The mixer projections are at their floor. The memory guard and admission from
PR #20 are unchanged; admission now prices the group-VJP transient.

### Fused GDN chunk kernels (2026-10-26, first step of the rewrite)

The GDN training path (`backend="metal_fused"`, used when `training_kernels`
is set) replaces the unfused MLX chunk ops with four Metal kernels in
`numerics/gdn_fused.py`:

- causal conv + SiLU, forward and backward;
- gated RMS norm, forward and backward;
- the intra-chunk transforms, forward and backward.

The chunk forward makes the gates, beta, A = (I + βRkkᵀ)⁻¹, u0, kt, mu0,
q_eff, and the decay terms in one kernel per chunk and head. The chunk
backward is the full VJP of that map in one kernel. The existing Metal scan
still crosses the chunks. Inference is unchanged.

| Probe (update 2) | Total | Layer forwards | Group VJPs | Peak |
|---|---|---|---|---|
| subset 12/22/23, interval 10, before (v14) | 182.4 s | 41.2 s | 137.3 s | 196.0 GiB |
| subset, fused GDN (v18 `0d7b488`) | 167.1 s | 35.5 s | 127.8 s | 192.0 GiB |
| full, interval 10, before | 1,355 s (20.2%) | — | — | 194.6 GiB |
| full, interval 10, fused GDN (v16 `22898df`) | 1,264.6 s (21.6%) | 286.7 s | 924.7 s | 185.8 GiB |

The full run saves 90 s (6.7%), against the ~70 s estimate. v18 changes only
the gate rounding (below), and its subset time equals v16 to 1 s.

Isolated, on one 2048-token segment with 16/48 heads: the chunk transforms
take 0.89 ms forward (old 1.63 ms) and 4.70 ms for the VJP (old 5.87 ms).
The backward kernel itself takes 1.88 ms. Every loop over register
fragments carries `STEEL_PRAGMA_UNROLL`; without it the fragment arrays
spill, and the backward kernel takes 2.54 ms.

Gates. The first version made the gates in FP32 from the logits (v17
`6695818`). It is much closer to FP32: with fixed routes, the gradient
cosine to FP32 rose from 0.73–0.75 to 0.974. This confirms that the BF16
gate g causes the shared BF16 error. But the sampler still makes g in BF16,
so replay moved away from the behavior logprobs. The kernels now round each
gate step to BF16, as `bf16.gdn_inner` does. Only the log and the prefix
sums are FP32 (v18 `0d7b488`).

| Check (sequences 12/22/23; fixed routes 12/23) | old | v17 FP32 gates | v18 sampler gates |
|---|---|---|---|
| mean \|replay − behavior logprob\| (5,711 tokens) | 0.1082 / 0.1067 (pre/post v6) | 0.1572 | 0.1055 |
| p99, max | 1.69–1.75, 4.1–6.3 | 2.38, 7.63 | 1.56, 6.31 |
| fixed-route gradient cosine to FP32 | 0.735–0.747 | 0.974 | 0.768 |

Per-kernel relative L2 error to FP32 on real layer inputs (sequence 22,
layers 0–5; `check_training_kernel_accuracy.py`, which now starts all paths
from the gate logits):

| | y | state | dq | dk | dv | da | db | dstate |
|---|---|---|---|---|---|---|---|---|
| old BF16 path | 0.035 | 0.069 | 0.061 | 0.087 | 0.041 | 0.210 | 0.063 | 0.133 |
| v17 FP32 gates | 0.004 | 0.007 | 0.007 | 0.012 | 0.008 | 0.006 | 0.008 | 0.011 |
| v18 sampler gates | 0.033 | 0.064 | 0.060 | 0.082 | 0.037 | 0.036 | 0.054 | 0.134 |

v18 passes the acceptance check. FP32 gates in both the sampler and replay
would give the v17 gradient accuracy without the sampler gap. That needs a
change to the inference GDN path.

### Overnight kernel pass (2026-10-26/27)

Numerics policy: BF16 is the trained, sampled, and served numerics, so the
pass ran no FP32 reference runs or per-kernel error studies. A change stays
when the tests pass, the loss and logprobs are finite, and the sampler
agreement (mean |replay − behavior logprob| over 5,711 masked tokens of
sequences 12/22/23) does not move by multiples. Then only speed counts.

Snapshots are `bwd/src-vNN-<hash>`. "Subset" is sequences 12/22/23 at
interval 10 (`phases-vNN`); "full" is the whole update 2 batch at interval 10
(`full-vNN-int10`). The floor is 273 s.

| Snapshot | Kept change | Full (s) | % of floor | Subset (s) |
|---|---|---|---|---|
| `e8438d8` (start) | — | 1,264.6 | 21.6% | 167.1 |
| v24 `1f65a49` | QSA backward without atomics, radix select, conv kernels | 1,113.4 | 24.5% | — |
| v25 `2d304eb` | Fused block-score kernel, SwiGLU and combine kernels | 1,038.1 | 26.3% | 137.0 |
| v26 `432fca7` | Injection kernel, up+gate input-gradient kernel | 1,006.4 | 27.1% | 134.2 |
| v26b `b190ef2` | 8 GiB Metal buffer cache | 898.2 | 30.4% | 121.5 |
| v27 `58ef69b` | HC custom VJP, gather-sum permutation VJP | 901.4 | 30.3% | 117.3–119.0 |
| v27b `ed090a5` | HC stream-mix forward kernel | 890.0 | 30.7% | 117.0 |
| v28 `34bf1bc` | PLE n-gram rows kept for the VJP | 825.1 | 33.1% | 110.1 |
| v29 `7affc4a` | Wired limit (residency set) during replay | 653.6 | 41.8% | 86.4 |
| v30 `1a1e415` | Fused conv kernel for the PLE dilated short conv | 651.1 | 41.9% | 86.3 |
| v31 `b8f02c1` | Host builds the next group VJP and layer while the GPU runs | **645.8** | **42.3%** | **85.4** |

All full runs: peak 186.0–186.2 GiB (budget 222.7), no guard stop. The full
runs for v25, v26, v26b, and v27b were made afterwards from frozen snapshots.
v27 (901.4 s) is within run noise of v26b (898.2 s): its subset gain does not
show in the full replay. Sampler
agreement (mean / p99): v23 0.1023 / 1.50, v27 0.1066 / 1.69, v28 0.1058 /
1.63 (v29 has the same forward bits as v28), v30 0.1056 / 1.75 (v31 has the
same bits as v30). The goal, 390 s (70%), is not met; the replay is 1.96x
faster than at the start of the pass.

What changed, in commit order:

1. QSA backward without atomics (`a5dcd42`, `9133015`, `e0a38e3`, `b732cc1`).
   The dQ kernel is query-major. It keeps the K fragments of a slot in
   registers, exchanges the 4-query partial sums with a reduce-scatter, and
   writes P and dS as BF16 in (query, slot, head, key) order. A key-major
   dK/dV kernel takes each 16-key tile, gathers the stored P and dS of the
   queries that selected it, and writes its rows once. QSA VJP on captured
   layer inputs: 60.6 / 68.7 / 66.7 ms → 17.8 / 21.1 / 21.0 ms at
   4k / 18k / 49k keys.
2. QSA selection (`ff0114f`, `5d670cb`). A fused NAX kernel makes the indexer
   block scores (ReLU and the head sum in registers), and a Metal radix select
   takes the top 512 blocks. Select at 18k / 49k keys: 3.9 / 12.5 ms →
   0.63 / 1.18 ms.
3. GDN causal conv (`1f65a49`): four channels per thread with vector loads
   (0.37 → 0.21 ms).
4. Routed MoE glue (`2d304eb`, `58ef69b`): one SwiGLU kernel, one
   score-weighted combine kernel, and one gather-sum kernel for the row
   permutation VJP, each one pass.
5. Routed MoE input gradient (`432fca7`): one kernel writes
   dx = dU Wu + dG Wg + dlow [Au; Ag] for the up and gate projections
   (VJP 7.19 → 3.86 ms). The LoRA scale multiplies the rank-16 rows, not the
   (R, D) rows.
6. Hyper-connections (`b843760`, `8c78441`, `ed090a5`): one kernel for the
   residual injection and its VJP; one custom VJP for the injected connection,
   whose norm kernel sums the three normed-row gradients; one kernel for the
   stream mix forward (0.228 → 0.054 ms), with the upstream BF16 roundings, so
   the result equals the upstream mix.
7. Metal buffer cache during training replay (`b190ef2`): 8 GiB instead of
   2 GiB. The group VJP frees and allocates ~10 GB of buffers per group. With
   2 GiB, MLX returned most of them, and the next group mapped and faulted
   them again (subset 134.2 → 121.5 s; 16 and 32 GiB give the same). MLX frees
   cached buffers before active plus cached memory reaches its GC limit
   (0.95 × the recommended working set), so admission and the replay guard
   are unchanged.
8. PLE n-gram rows kept for the VJP (`34bf1bc`). The n-gram table is ~50 GB,
   and each segment reads ~32k scattered rows. Timed in situ, a lookup takes
   ~0.3 ms in the forward phase but 0.39–0.55 s in the group VJP: under the
   replay's memory load the GPU must make the idle table resident again. The
   VM counters show no compression or page-ins during the replay. The recorded
   forward now keeps the BF16 rows (L × 2560 per PLE layer) in a segment memo,
   and the group VJP takes them once. Results are bitwise unchanged, and
   admission prices the rows. Layer 0–9 group VJP: 0.995 → 0.564 s per
   2048-token segment.
9. Wired limit during replay (`7affc4a`). `forward_backward` sets the MLX
   wired limit (the Metal residency set) to the recommended working set, as
   mlx-lm generation does, and restores the previous limit (0) afterwards.
   Replay uses each layer's weights once per segment and group, so without
   it the GPU made idle buffers resident again at each use. In a micro that
   rotates through the five groups as replay does, a group VJP took
   0.37–0.50 s against 0.32–0.41 s repeated; with the limit, both are equal,
   and the PLE lookup is fast even without the memo. kern.memorystatus_level
   fell to 19% at the subset peak (the guard stops at 8%) and returned to 95%.
10. PLE short conv (`1a1e415`). The GDN conv kernels take a dilation, and the
   replay PLE path uses them for its grouped conv (kernel 4, dilation 3,
   10240 channels): forward 3.42 → 0.16 ms, VJP 7.41 → 0.37 ms.
11. Host and GPU overlap (`b8f02c1`). The group VJP and the forward layer
   start with `mx.async_eval`, and the replay waits for them only after it
   builds the next graph. One group or layer is in flight at a time, so the
   peak does not change (vjp_build 9.7 s and forward_build 3.0 s in the full
   run now overlap the GPU work).

How the gap was found. The profile micros ran ~1.45× faster than the in-situ
replay. It was not throttling (a 10-layer group VJP stays at 0.396 s over
120 s), not memory pressure on the kernels (HC VJP 2.09 ms with 0, 80, or
160 GB of ballast), and not the attention context (0.40 → 0.42 s from 4k to
54k context). A per-call log of the in-situ group VJPs
(`micro/calls/probe_calls.py`) showed that the layer 0–9 group cost 0.94 s
against 0.45–0.63 s for the others. Ablations then priced the PLE at 18 s of
119 s, all of it the n-gram lookup. A micro that rotates through all five
groups (`micro/group_rotate.py`) reproduced the rest of the gap, and the
wired limit removed it.

Tried and dropped: caching the QSA selection between the forward and the VJP
(~1.2 GB outside admission, ~9 s); a pooled-key cache (no gain); a bank-free
exchange layout alone (no gain); 32-key dK/dV tiles (marginal); custom
forward GEMMs for the routed experts in two stage layouts (2.0–3.2 ms against
1.6–1.7 ms for MLX `gather_qmm`); dx prefetch variants (slower, register
pressure); the fused LoRA input gradient for the down projection (no gain);
a combine kernel that adds the down projection's base and LoRA terms
(118.3 s against 117.3 s without it, +0.8 GiB); `gather_qmm` with
`lhs_indices` instead of pre-gathered rows (9.2 against 1.9 ms); 4096-token
segments, which double the rows per expert (admission correctly refuses the
subset: the mandatory checkpoints and scratch exceed the allocator limit). A
routed forward GEMM that dequantizes the 4-bit weights straight into NAX
fragment registers, with no threadgroup memory or barriers (`micro/reg`):
correct, but 6.0 ms against 1.75 ms for `gather_qmm`. Ablations: without the
weight loads it runs at 61 TF (1.1 ms), and dequant from registers alone at
56 TF (1.2 ms), so the per-simdgroup weight loads are the bound. Wider loads
(16 bytes per lane) and a register prefetch stage were slower (4.0–12.8 ms)
from register pressure. Weight loads must be shared across simdgroups, as the
threadgroup-staged kernels and `gather_qmm` do. A threadgroup version of it
(`micro/reg/fwd_tg.metal`: 64 × 128 tiles, a three-slot ring of the raw packed
words, scales, and biases, one barrier per 64-wide group, lane dequant into the
fragments) is correct but takes 4.1 ms. Without the lanes' threadgroup reads
it takes 1.65 ms, only parity with `gather_qmm`, so this structure has no
headroom either. Both were dropped.

In-situ ablations at ~121 s (subset, before the HC VJP and the PLE memo):
routed MoE 48 s, GDN with its projections 26 s, HC 20 s, the QSA sparse
attention core 13 s (selection kept), shared expert 3.6 s.

What is left, largest first:

1. Routed MoE, ~40% of the replay. Per layer and 2048-token segment: forward
   7.0 ms (three `gather_qmm` at ~1.7 ms, ~40 TF), VJP 14 ms. The GEMMs load
   about 40 rows per expert, and the dx kernel is load-latency bound (with no
   MMA or no x loads it still takes 1.29 of 1.5 ms). The rows per expert set the
   rate (`micro/qmm_ceiling.py`, 4-bit up projection, sorted rows):

   | Tokens per call | Rows per expert | `gather_qmm` | 8-bit |
   |---|---|---|---|
   | 2,048 | ~40 | 1.72 ms, 39 TF | 29 TF |
   | 4,096 | ~80 | 2.59 ms, 52 TF | 43 TF |
   | 8,192 | ~160 | 4.62 ms, 58 TF | 52 TF |
   | dense `quantized_matmul` | — | 58.8 TF | 52.0 TF |

   Dense BF16 runs at 93.6 TF. More tokens per expert call is the lever, not
   a better kernel. Longer segments do not fit admission, but sibling trie
   segments are independent once their shared prefix is done, so a replay
   that runs siblings as one batch would give the MoE (and every projection)
   more rows. The update 2 trie bounds this: one shared root segment, 12
   children, then mostly sibling pairs of uneven length (239 forward
   segments). Even if every segment ran in a pair, two 2,048-token GEMMs
   (2 × 1.72 ms) become one 4,096-token GEMM (2.59 ms), 25% of the ~166 s of
   routed GEMMs: at most ~40 s (6%). It also needs ragged attention and GDN
   kernels and about twice the group activations in admission. Measured on a
   real layer (`micro/moe_pair.py`), the whole routed MoE at 4,096 tokens costs
   per 2,048 tokens 6.04 ms forward (7.69 at 2,048) and 12.95 ms module VJP
   (15.58): ~4.3 ms per layer and segment, ~0.21 s per paired segment. Pairing
   sibling leaves only (no ragged kernels: each branch keeps B = 1 and only
   the MoE rows are joined) pairs 48 of 239 segments, ~10 s. It was built on
   branch `claude/flash-pair-leaves` (`1eed76c1f9`): `FlashOps.layer_front` and
   `layer_pair`, a lockstep leaf forward, and a joint group VJP. Tree replay
   with pairing gives bitwise-equal logprobs and close gradients. Admission
   must price the second leaf's state and a second group VJP (>11 GiB), and it
   refuses the subset, whose admission is already at the 222.7 GiB budget
   (158.1 + 64.5); the full batch has ~20 GiB of room. So it was not merged.
2. QSA attention core: 8.8 ms forward and 28 ms VJP per layer and segment at
   26k context, ~12–13 TF. Each query gathers its own 2,048 keys for 12 query
   heads per KV head, so the cache traffic per FLOP is high. On captured
   inputs at 18k / 49k keys (`micro/qsa_parts.py`, v31): forward 4.9 / 5.0 ms
   (~21 TF), dQ 7.0 / 7.0 ms (~22 TF), dK/dV 10.1 / 9.4 ms (~11 TF), tile
   lists 0.5 ms. Adjacent queries share most blocks: the union of a pair is
   1.24x one query's 513 blocks (a group of four, 1.6x). A pair kernel would
   cut the K/V gathers of the forward and dQ by ~38% for ~24% more MMA, about
   5 ms per layer and segment, ~14 s in all. dK/dV cannot share: each
   (16-key tile, query) pair, ~500k per segment, needs that query's own Q and
   dO rows (~12 GB of gathers), and 32-key tiles cut the pairs by only 27%.
   The pair forward was built (`micro/pair/forward_pair.metal`: one
   threadgroup per query pair, union blocks with a 2-bit membership mask, one
   K/V gather for both queries). It is correct (lse within 2e-6) but slower:
   7.1–8.2 ms against 4.6–4.9 ms. The forward is not gather bound; the
   per-chunk score exchange, the online softmax, and register pressure double
   with two queries. It was dropped, and the dQ pair kernel was not built.
3. HC: ~6.5 ms per layer and segment for two connections, near its traffic
   floor except the 8-bit GEMMs.
4. GDN core (v31 micro, VJP including its forward, per layer and segment):
   chunk 0.59 / 2.76 ms, scan 0.31 / 1.55, conv 0.19 / 0.32, gated norm
   0.06 / 0.14. The split and the upstream q/k L2 norm (unfused BF16 ops)
   cost 0.10 / 0.29 ms, ~3.4 s in all; a fused kernel would save ~2 s.
5. The micros now match the replay: in-situ group VJPs (0.29–0.41 s per
   2048-token segment) equal the single-group micro, so profile numbers can
   rank the next work directly.

## Prefill and decode interleaving

The service runs one GPU owner loop. Each step admits requests, runs one
prefill chunk for one request (round robin), then one batched decode step for
all ready rows. Prefill and decode are separate forwards, so a prefill chunk
stalls every decoding row. Before this change a 2048-token chunk took 4.2 s at
a 16k context (`prefill-v2`). In perf2 update 1, about 190 such chunks stalled
decode for ~450 s, about 11% of the rollout window.

| Chunk tokens | Sorted rows | Expert tiles (32) |
|---|---|---|
| 256 | 0.48 s | 0.47 s |
| 512 | 0.85 s | 0.58 s |
| 1024 | 2.32 s | 0.83 s |
| 2048 | 4.22 s | 2.62 s |

The adopted setting is tiles with 1024-token chunks. It is the fastest per
token (1,240 tok/s vs 485 before), and the worst decode stall per step falls
from 4.2 s to 0.83 s. A fused prefill+decode forward was not built; see the limits.

Gate (`numerics-v2`, declared before the run): 12 spans, 1,306 sampled tokens,
forced through the real prefill and decode code. Compared against production
replay (trie + tiles), mean and p99 |sampler − replay| must be no worse than the
current prefill, within 5%.

| Prefill path | mean | p99 | max | Result |
|---|---|---|---|---|
| Sorted, 2048 (before) | 0.128 | 1.875 | 3.25 | baseline |
| Tiles, 2048 | 0.128 | 2.0 | 3.06 | fails p99 by one BF16 step |
| Tiles, 1024 (adopted) | 0.123 | 1.75 | 3.56 | pass |
| Tiles, 512 | 0.123 | 1.75 | 3.13 | pass |

In production (perf3 update 1 vs perf2, service journal, per player sample):

| Per sample | perf2 | perf3 |
|---|---|---|
| Prefill | 4.73 s | 1.18 s |
| Decode rate per request | 7.0 tok/s | 12.5 tok/s |
| Mean prompt / completion tokens | 14.7k / 282 | 18.8k / 333 |

The perf2 figures cover its whole service life (update 1 and part of update 2).

perf3 halted at 7,252 s, before update 1 committed. Codex returned "Selected
model is at capacity" for two GM turns. One 68-turn episode also hit the
7,200 s player-stage timeout. Those three failures tripped the configured halt
mode. The cause is outside the GPU path: GM turns averaged 49.4 s vs 41.7 s in
perf2. perf4 restarts the same snapshot in a new root.

Smoke 4 (`studio-flash-hero177-smoke-20261023T005720Z`) covered rollouts on the
new path, save, publish, interrupt, restart, restore, and dedup. All checks passed.
Replay code is unchanged since smoke 2, which covered the update.

## Sampling-vs-replay discrepancy: root cause

This is not a bug. The Flash forward is chaotic under BF16 execution-layout changes: a
last-bit change flips MoE and QSA routes, and those flips cascade. Replay vs sampler
(mean |Δ| 0.12 nats, max 7 nats here, up to 15 in r3) is the same size as
the gap between two valid replay layouts (0.12 nats, max 6). The mean signed
gap is small and positive (behavior − replay = 0.047 nats).

## Benchmarks (reproducible; inputs are update 1 sequences from R)

| Benchmark | Result | Output |
|---|---|---|
| Replay of 4 sequences: baseline / trie / tiles / trie+tiles | 1012 / 674 / 416 / 268 s | `validate-v4`, `validate-v5` |
| Decode step at 1 / 8 / 16 rows (12k context) | 47 / 94 / 200 ms (21 / 85 / 80 tok/s) | `decode-profile-v2` |
| MoE layer 0 forward / VJP, default vs tiles | 67 / 591 ms vs 14 / 85 ms | `moe-v1` |
| PLE lookup at 2048 rows, sharded vs joined | 63.7 vs 0.3 ms, bitwise | `ple-v1` |
| Prefill chunk at 16k context, sorted vs tiles | 1024 tok: 2.32 vs 0.83 s | `prefill-v2` |
| Forced sampler vs replay, prefill variants | see the gate table | `numerics-v2` |

Each script takes `--sequences`, `--model`, and `--output`, and holds the GPU lease.

## Scheduling experiment (rejected)

Smoke 2 ran 16 episodes over 8 GPU slots. Each episode took 1.9x longer.
Mean GM calls rose from 39.8 s to 53.6 s, and player turns queued for GPU slots.
Collection took 7,107 s vs 4,600 s. Sixteen GPU slots (smoke 1) starved the
prefix cache of reserved memory: the hit rate fell to 19.5%, 8 of 16 episodes timed out,
and the update was refused. perf2 keeps r3's 8 episodes over 8 slots.

## Remaining limits, largest first

1. Rollouts are latency-bound by the slowest episode. A turn is one GM call
   (~40 s) plus player decode at ~10 tok/s per row. Decode is 94 ms per step at 8 rows;
   the weight-read floor is ~5 ms. Per-row attention, MoE, and GDN do not batch,
   so throughput is flat past 8 rows. The fix is batched ragged decode kernels.
2. Replay memory peaks at 248 GiB against a 240 GiB target. The admission
   estimate undercounts the long-sequence replay peak by ≥8 GiB.
3. Prefill and decode still run as separate forwards. A step with a prefill
   chunk stalls decode for up to 0.83 s. A fused ragged forward (prefill rows
   and decode rows in one pass, SGLang style) would remove the stall. It needs
   ragged attention and GDN kernels, as in item 1.
4. A player timeout leaves its service generation running. The leftover reservations can
   block the next update's admission. This predates the changes here.
5. GM provider capacity errors count toward halt mode. Two capacity errors and
   one long-episode timeout halted perf3 before its first update.
6. The pinned Codex token expires about 2026-10-29. The epoch runs past that unless
   the owning `~/.codex` login refreshes it.
7. Replay is still 2.4x above its FLOP floor (645.8 s at interval 10
   against 273 s on perf4 update 2, 42.3%): routed experts (~40%), the QSA
   attention core, and HC. See "Overnight kernel pass (2026-10-26/27)".

## Proposed algorithm changes (not applied)

- Async rollout/train with one-revision staleness: collect update k+1 while
  update k replays. An update would take ≈ max(4,217, 2,952) s instead of the
  sum, about 1.7x more throughput. This changes on-policy semantics and needs
  importance correction to be explicit.
- Gradient noise: two valid BF16 layouts give update gradients with cosine
  ~0.57. FP32 accumulation or deterministic routing in replay would make
  updates reproducible. It costs speed and changes precision.
