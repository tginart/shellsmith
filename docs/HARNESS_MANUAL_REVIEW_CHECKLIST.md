# Generative-World Harness: Manual Review Checklist

Use this alongside
[`HARNESS_REMEDIATION_PLAN.md`](HARNESS_REMEDIATION_PLAN.md).

This checklist contains only decisions and evidence that deserve human review.
Schema validation, correlation checks, and routine regression tests should
remain automated.

## 1. Approve the architecture before implementation

- [ ] **Native renderer only:** Approve removing the custom text tool-call
  language with no fallback in the shellsmith path.
- [ ] **Invalid sibling policy:** Approve rejecting the complete assistant
  action when any sibling tool call is malformed. No sibling from that message
  reaches the world.
- [ ] **Sibling timing:** Approve sequential world evaluation while withholding
  every result until the complete sibling result set exists.
- [ ] **Thinking mode:** Approve requiring Qwen thinking mode for every RFT
  rollout. Missing or disabled thinking makes the rollout untrainable.
- [ ] **Runtime scope:** Approve Codex and Claude as the only initially
  trainable world runtimes. Pi/Qwen world simulation remains calibration-only.
- [ ] **Same-thread judge:** Approve retaining the Game Master's persistent
  thread for final judging in the repaired baby run, with no independent second
  judge yet.

## 2. Approve the training semantics

- [ ] **Rollout is the weighting unit:** Approve dividing each turn's raw
  advantage by the rollout's total accepted sampled-token count.
- [ ] **Compaction credit:** Approve training a generated compaction summary at
  the tail of its original sequence and treating it as masked context when it
  appears in the next sequence.
- [ ] **Final response credit:** Approve giving the final assistant response
  terminal rollout credit even though it has no turn-level world reward.
- [ ] **Zero-advantage tokens:** Approve counting zero-advantage sampled tokens
  in the rollout denominator even if their all-zero datum is omitted.
- [ ] **Reasoning scores:** Confirm that missing quality/efficiency scores remain
  neutral, while present scores may adjust turn-level advantage.
- [ ] **Contaminated checkpoint:** Approve permanently forbidding training or
  resume from baby v3 while retaining it for research and visualization.

## 3. Review the exact model-visible prompts

Review captured prompts, not merely the source templates.

- [ ] **Player system prompt:** It is neutral and task-oriented.
- [ ] It does not say “expert coding assistant” unless that is deliberately
  restored.
- [ ] It contains no Pi installation paths, real temporary vat paths,
  FCPXML instructions, OpenCode terminology, aliases, plugins, skills,
  or user-level resources.
- [ ] It presents the logical cwd as `/workspace`.
- [ ] It contains the task-specific generated `AGENTS.md` exactly once.
- [ ] Its seven tool names and argument schemas are accurate.
- [ ] **Game Master prompt:** It clearly separates physical vat state from
  virtual `/workspace` state.
- [ ] It allows correction of unpublished malformed artifacts without implying
  the observation was already shown to Player.
- [ ] It says Player-authored files and claims are not authoritative world
  truth.
- [ ] It remains concise enough that simulation, grading, and integrity
  instructions are understandable in one pass.
- [ ] **Sonnet tail:** It asks for faithful difficulty without inventing new
  hidden requirements, coupled failures, or moving success criteria.

## 4. Review one native renderer round trip

Inspect one captured real Pi request and its complete response chain.

- [ ] The request contains all seven real OpenAI function schemas.
- [ ] The rendered Qwen prompt contains the same names, descriptions, required
  fields, and parameter schemas.
- [ ] Historical assistant calls remain structured—not prose.
- [ ] Historical tool results retain the correct call IDs.
- [ ] The sampled output contains genuine Qwen thinking followed by a native
  structured tool call.
- [ ] The parsed call uses the correct tool and arguments, such as
  `read(path=...)`.
- [ ] Pi receives a structured call with the same name, arguments, and ID.
- [ ] No custom parser, alias repair, or transcript recovery was involved.
- [ ] No legacy marker such as `Previous tool-call history` appears anywhere.

## 5. Review thinking propagation

For one real action, compare these five artifacts side by side:

1. Pi's outbound model request.
2. Raw Tinker sample.
3. Parsed Tinker assistant message.
4. Pi `message_end` event.
5. World turn request.

- [ ] All five prove thinking is enabled.
- [ ] The reasoning text is identical after renderer parsing.
- [ ] The world sees the reasoning privately for grading.
- [ ] The reasoning does not leak into visible assistant text or tool output.
- [ ] `run_config.json` and `scenario/model_protocol.json` agree with the wire
  evidence.

## 6. Review sibling behavior

Run and inspect two fixtures.

### Two valid siblings

- [ ] Both calls share the same assistant-message reasoning.
- [ ] Neither sees the other sibling's result as pre-action context.
- [ ] The world evaluates them in assistant-message order.
- [ ] Pi receives neither result until both are ready.
- [ ] Each call produces exactly one world request, response, tool result, and
  ledger entry.

### One valid and one malformed sibling

- [ ] Neither call reaches the world.
- [ ] No simulated state changes.
- [ ] No ledger entry is appended.
- [ ] Player receives an ordinary harness validation error, not simulator
  terminology.
- [ ] The rollout is rejected by protocol integrity rather than assigned a low
  task reward.

## 7. Review Codex and Claude runtime boundaries

- [ ] Inspect both `scenario/gm_runtime_manifest.json` files.
- [ ] **Codex:** Confirm user config and rules are disabled and only episode
  roots are writable.
- [ ] **Claude:** Confirm safe mode, strict MCP config, disabled skills/slash
  commands, no Chrome integration, and the explicit tool list.
- [ ] Confirm Claude exposes no subagent, plugin, memory, cron, worktree, MCP, or
  notification capability.
- [ ] Confirm both runtimes can use the web.
- [ ] Confirm both can write inside the private simulation root.
- [ ] Confirm both are denied access to a sentinel outside approved roots.
- [ ] Confirm only bootstrap may physically access the vat.
- [ ] Confirm the acceptance audit derives its decision from runtime evidence,
  not the string `world_permission_profile`.

If Claude cannot pass the outside-root denial probe, do not approve it for
trainable rollouts.

## 8. Review bootstrap and retry behavior

- [ ] Codex and Claude both write `world/bootstrap/response.json`.
- [ ] Neither uses CLI structured output for bootstrap.
- [ ] The boot artifact contains only status, private initial commitments, and
  failure reason.
- [ ] The orchestrator—not the model—supplies the initial embedded prompt and
  verifies the vat inventory.
- [ ] A malformed boot response is repaired in the same world thread.
- [ ] The consistency ledger is initialized once, after boot acceptance.
- [ ] A provider failure after a valid response artifact does not replay the
  logical action.
- [ ] A provider failure before artifact creation resumes the same thread and
  idempotency key.
- [ ] Ambiguous delivery makes the rollout untrainable.

## 9. Review protocol-integrity rejection

- [ ] Run the new audit over baby v3.
- [ ] Every trajectory containing the injected history marker is rejected.
- [ ] The report names the adapter/protocol violation rather than merely saying
  a manifest is missing.
- [ ] A clean native fixture passes.
- [ ] Manually delete one call/result correlation record and confirm it fails.
- [ ] Change the recorded renderer or thinking mode and confirm it fails.
- [ ] Insert one unparsed native tool call and confirm it fails.
- [ ] Confirm the world model's own `trainable: true` cannot override any of
  these failures.

## 10. Review loss construction before a real update

Inspect the JSON audit for one short rollout and one multi-request or compacted
rollout.

- [ ] Every Tinker datum maps to one captured model request.
- [ ] Prompt tokens have zero advantage.
- [ ] Sampled tokens have the intended raw turn advantage.
- [ ] The rollout denominator includes all accepted sampled tokens.
- [ ] Per-token normalization weights sum to one per rollout.
- [ ] The compaction summary appears in the loss exactly once.
- [ ] Repeated compaction context is masked.
- [ ] A one-sequence and four-subsequence synthetic equivalent produce the same
  total CISPO coefficient.
- [ ] Two groups in one optimizer update do not change relative per-rollout
  weighting.

## 11. Review the live smoke ladder

- [ ] Captured-payload renderer test passes.
- [ ] One-action no-training smoke is manually inspected.
- [ ] Thinking smoke is manually inspected.
- [ ] Valid and invalid sibling smokes are manually inspected.
- [ ] Forced-compaction smoke is manually inspected.
- [ ] Codex runtime-boundary smoke passes.
- [ ] Claude runtime-boundary smoke passes.
- [ ] One disposable Tinker optimizer update completes.
- [ ] The disposable checkpoint is marked unusable for the hero run.

## 12. Final approval for the fresh baby run

- [ ] All remediation gates A–I pass.
- [ ] Focused Python, Node, and complete backend tests pass.
- [ ] The Observatory shows run ID, world runtime/model, renderer, thinking mode,
  protocol-integrity status, reward, and rollout-normalization evidence.
- [ ] The new run uses a fresh directory.
- [ ] The new run uses a fresh Tinker training state from the intended clean
  base model/LoRA.
- [ ] Provider-wide outages stop collection and checkpoint the last completed
  update rather than repeatedly skipping that provider's worlds.
- [ ] You have reviewed every datum from the disposable update and at least one
  complete rollout from each world runtime.

Only after these boxes are checked should the two-update baby run begin. Review
that baby run before approving the full hero slate.
