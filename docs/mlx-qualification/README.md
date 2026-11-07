# Studio MLX qualification — 2026-10-09

The final R3 numerical qualification passed on `studio-host.local`, a 256 GiB Apple Silicon Studio. This evidence covers the supplied local `Qwen/Qwen3.8-27B` artifact: dense `qwen3_5`, 64 layers, width 5120, affine 8-bit/group-64 weights, rank-16/scale-2 LoRA across all supported text projections. The resident model has 992 adapter factors containing 116,727,808 trainable parameters.

V1 admission requires at least 256 GiB (`274877906944` bytes) of detected physical unified memory, matching the saved [Studio readiness report](studio-runtime-readiness.json). Smaller hosts and unknown capacity fail readiness before model initialization or rollout collection. The fixed 128 GiB MLX allocator and replay budget remain unchanged, preserving host headroom without qualifying a smaller machine. This admission change is included in the current policy source identity; historical checkpoints, evidence, and running jobs retain their original frozen source and identity.

The frozen runtime used Python 3.12.14, MLX 0.32.3, MLX-LM 0.32.0, MLX-VLM 0.7.4, Transformers 5.17.0, and `MLX_ENABLE_TF32=0`. The reference is High Command `10e830e9a5becb9c2387bd37a378fd7dad1d3590`. The R3 numerical source SHA-256 is `b531314a204cbd48ecbce9de57c553a8563303fa643bb63edad2e15135efec4b`; its complete policy identity SHA-256 is `2b78ad759e0bedb7a42f09dace2997c3276fe1379671a92b31822549f934725c`. The report includes hashes of the actual local model, tokenizer, template, and source files; it makes no claim about a remote Hugging Face revision. Current checkpoint identities additionally pin the shared `policy/chat_adapter.py` renderer and `policy/sampler.py` request semantics dependencies and expose the source inventory. They also pin the Pi compaction reserve and retained recent-token settings. R3 checkpoints retain their historical identity and require the original frozen runtime; they cannot be restored or evaluated under the strengthened current identity.

| Gate | Measured result | Evidence |
| --- | --- | --- |
| Independent native FP32 decoder likelihood and every adapter gradient; GDN carry, summed CISPO, BF16 full-prefix replay, zero-gradient Adam, frozen quantized base | 9 tests passed; likelihood tolerances `rtol=1e-5, atol=2e-6`, gradient tolerances `rtol=3e-4, atol=2e-6` | [Native Metal JUnit](tiny-metal-r3.xml), [test source](../../tests/test_mlx_numerics.py) |
| Real-model plain, natural thinking, forced thinking closure, native tool call and EOS | Exact selected IDs and behavior logprobs recorded; forced controls included and identified | [Resident qualification JSON](studio-r3.json) |
| Exact native context admission | 30,752 prompt tokens + requested 8,192 completion tokens rejected before generation; RNG unchanged. An explicit 2-token cap at 32,766 prompt tokens succeeded | [Resident qualification JSON](studio-r3.json) |
| Synthetic full-context update | 24,576 prompt tokens + 8,192 teacher-forced targets, 65 bounded vocabulary-head tiles; likelihood 42.03 s; update 254.13 s | [Resident qualification JSON](studio-r3.json) |
| Memory and frozen base | Maximum measured Metal allocator peak 70,913,049,366 bytes / **66.04 GiB**, below the 128 GiB gate; all 2,180 frozen tensors unchanged after four logical updates | [Resident qualification JSON](studio-r3.json) |
| Fresh Python recovery from step 0 to 1 | Bitwise BF16 adapters and all FP32 optimizer tensors; identical RNG, next selected IDs, and behavior logprobs | [Initial recovery JSON](recovery-r3.json) |
| Fresh Python recovery with populated optimizer state, step 1 to 2 | Loaded nonzero first moments matched exactly; distinct processes produced bitwise-identical next BF16 adapters, FP32 masters/moments, RNG, IDs, and behavior logprobs; fresh phase 20.87 s | [Baseline control](nonzero-control-r3.json), [fresh recovery](nonzero-recovery-r3.json), [reproduction driver](nonzero-recovery-driver.py) |
| Short generation/update smoke | Two additional generation/update/checkpoint/publish cycles, 4.12 s and 4.06 s | [Resident qualification JSON](studio-r3.json) |

The 32K gate is synthetic teacher-forced context and vocabulary-head stress. Its fixed token advantages exercise the resident numerical backend; it does not establish accepted online episode training. The two short cycles are smoke coverage, not a sustained online soak. Reported memory is the MLX Metal allocator peak, not whole-system resident memory.

The [authenticated readiness result](studio-authenticated-readiness.json) records the authorized cached-credential transfer over SSH: dedicated Codex 0.159.2 reported `Logged in using ChatGPT`. Credentials remain outside the checkout and evidence artifacts. The [current readiness result](studio-runtime-readiness.json) additionally checks the approved GM Python runtime and reports `ready: true`, no blockers, and `codex_authenticated: true`.

The [first authenticated canary](studio-live-canary-initial.json) passed four exact native requests, thirteen tool-call correlations, unchanged policy identity, EOS, protocol integrity and factual consistency. Its episode was correctly rejected because the GM used the broken macOS system Python launcher. The GM also misinterpreted empty runtime scratch as absent; the recorded successful directory test proves scratch existed. The job was stopped with zero optimizer updates and a complete baseline checkpoint.

The [actual Studio OS confinement proof](studio-gm-python-confinement.json) passes under the updated adapter's exact frozen profile: approved Python and a copied private venv import JSON/SQLite/SSL, SQLite state persists, and credential opening and runtime writes are denied. The [restarted canary](studio-live-canary-runtime.json), `job-8c40c2bc05b5c650a77d62de`, has six successful approved Python commands, including scratch venv creation, and four aligned native captures. It is collecting episodes. Two accepted live updates, heldout evaluation, an actual confined compaction-and-retry episode, and a sustained online soak remain release gates; no accepted episode or optimizer-update result is claimed here.

The earlier [unauthenticated public readiness result](studio-readiness.json) remains unchanged as historical evidence of the authentication blocker. The public trainer, episode runner, and detached job paths have portable regression coverage; these JSON files do not claim the pending live gates passed.

The earlier [real detached SSH smoke](studio-job-readiness-smoke.json) exercised submission, later status/log retrieval after the submitting process exited, and an identical request retry on the Studio. That child stopped at the expected Codex-authentication readiness failure before model loading; retry returned the same job and attempt 1. This preserved record verifies the negative readiness/job path. Post-authentication regressions passed 2,093 Python tests, 592 subtests, and 14 Node tests, with 19 Python skips. The final pagination hardening passed all 51 job tests separately, including transfers above 32 MiB, historical server compatibility, failed publication rollback and preserving local notes. Compilation and syntax checks passed. Numerical execution code remains unchanged from R3; the identity changes described above do not alter its measured numerical results.

Run the numerical tests and [public qualification script](../../scripts/qualify_mlx.py) on the Studio using the same pinned environment and source. The driver acquires the shared Studio GPU lease and closes the resident model before releasing it. Use fresh output and checkpoint paths for each run:

```sh
studio_python=~/.local/share/shellsmith/studio-v1/env/bin/python
MLX_ENABLE_TF32=0 PYTHONPATH=src "$studio_python" -m pytest tests/test_mlx_numerics.py -q
MLX_ENABLE_TF32=0 PYTHONPATH=src "$studio_python" scripts/qualify_mlx.py \
  --checkpoint-root /tmp/shellsmith-mlx-repeat/checkpoints \
  --output /tmp/shellsmith-mlx-repeat/studio.json \
  --context-gate --completion-tokens 8192 --soak-updates 2
MLX_ENABLE_TF32=0 PYTHONPATH=src "$studio_python" scripts/qualify_mlx.py \
  --proof-source /tmp/shellsmith-mlx-repeat/studio.json \
  --checkpoint-root /tmp/shellsmith-mlx-repeat/recovery-unused \
  --output /tmp/shellsmith-mlx-repeat/recovery.json
```

The populated-state proof was run as two separate Python invocations using the unchanged frozen R3 source bundle at `~/.local/share/shellsmith/qualification-numeric/final-r3`. To reproduce it from a checkout after the main qualification above, run the following baseline invocation, then repeat it with `--phase fresh`. Both phases restore the main run's immutable `first-state` checkpoint at step 1; the fresh process compares against the baseline's step-2 checkpoint, with exact tensor dtype/shape/value checks rather than tolerances:

```sh
MLX_ENABLE_TF32=0 PYTHONPATH=src "$studio_python" \
  docs/mlx-qualification/nonzero-recovery-driver.py --phase baseline \
  --reference /tmp/shellsmith-mlx-repeat/studio.json \
  --control /tmp/shellsmith-mlx-repeat/nonzero-control.json \
  --checkpoint-root /tmp/shellsmith-mlx-repeat/nonzero-checkpoints \
  --output /tmp/shellsmith-mlx-repeat/nonzero-recovery.json
```

Machine-readable files are preserved exactly as emitted on the Studio. Checkpoint paths in them refer to retained qualification artifacts on that host. The source SHA above covers the native MLX modules in the frozen R3 runtime; the standalone evidence driver does not alter that runtime's numerical implementation or policy identity.
