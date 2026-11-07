# Generative-World Harness: Critical Self-Audit

## Purpose

This document audits the protocol boundaries between:

1. Player running inside Pi.
2. Pi's OpenAI-compatible requests.
3. The local Tinker sampling proxy.
4. Tinker's Qwen renderer.
5. The world-simulator agent running through Codex or Claude.
6. The acceptance audit and Tinker training bridge.

The core conclusion is:

> The current baby-run checkpoint is not a trustworthy training result. The
> harness silently worked around a Pi-to-Tinker incompatibility by inventing a
> plain-text tool-call format. That workaround corrupted real trajectories and
> was not reliably caught before training acceptance.

No additional hero training should run until the P0 items below are corrected
and verified end to end.

## Short version

The immediate problems are:

- Pi sent valid structured OpenAI tool schemas, but the proxy did not pass them
  through Tinker's native Qwen renderer.
- The conductor explicitly enabled an experimental plain-text workaround after
  the proxy had identified structured tools as unsupported.
- That workaround lost the real tool names and argument schemas.
- It inserted `Previous tool-call history (not valid output)` into Player's
  context, which Player sometimes copied as its next output.
- The run claimed Player used high reasoning, but reasoning was actually
  disabled and stripped.
- The Game Master sometimes executed and rewarded tool calls before Pi
  rejected their arguments.
- Fourteen corrupted terminal trajectories were nevertheless accepted for
  training in baby v3.
- The training bridge does not visibly enforce the agreed rule that each
  rollout should have equal total weight regardless of its number of requests
  or compactions.

The correct response to a renderer or protocol mismatch is to stop and report
it. The harness must not invent a substitute wire format without explicit
review.

## Recommended order of work

Work through these in order. Do not combine the first repair pass with unrelated
world-quality or prompt-calibration changes.

### Step 1: Restore native Pi-to-Tinker tool rendering

- Remove `allow_tools_in_tinker_text=True` from the shellsmith conductor.
- Remove the custom `<tool_call name="...">` prompt and parser from this path.
- Convert Pi's actual OpenAI tool schemas into Tinker's native `ToolSpec`
  representation.
- Preserve structured assistant tool calls and structured tool-result messages
  when building the Tinker conversation.
- Use the installed Qwen 3.5 renderer's native Qwen XML format.
- Preserve and explicitly reject `unparsed_tool_calls`.

Exit criterion:

- A captured real Pi request containing all seven tools round-trips through the
  renderer.
- Qwen emits a structured `read(path=...)` call.
- Pi receives and executes the same structured call without any textual
  transcript or custom parser.

### Step 2: Make thinking mode real

- Make Pi send an explicit thinking-mode signal supported by the proxy.
- Do not configure the provider with `supportsReasoningEffort: false` while
  claiming high reasoning is active.
- Preserve Tinker `ThinkingPart` values through the OpenAI response returned to
  Pi.
- Verify Pi records a `thinking` block.
- Verify the Game Master receives that exact thinking block for private
  reasoning assessment.

Exit criterion:

- Run configuration, proxy selection, renderer name, raw sampled output, Pi
  event log, and world request all independently show that thinking is enabled.

### Step 3: Validate tool calls before world simulation

- Keep sibling calls grouped as one assistant action.
- Do not send a tool request to the world merely because it appeared in an
  assistant `message_end` event.
- Allow Pi to validate the tool name and arguments first.
- Once every valid sibling executor has been registered, send the complete
  validated batch to the world.
- Return sibling results to Pi only after the complete batch has been simulated.
- Treat a malformed sibling as an invalid assistant action; do not let it create
  world state.

Exit criterion:

- `read(file_path=...)` never reaches the world.
- `read(path=...)` reaches the world exactly once.
- Two valid sibling calls receive observations only after both world responses
  are ready.

### Step 4: Add protocol-integrity rejection

Reject a rollout before training if it contains:

- Adapter-injected markers.
- Unparsed or malformed tool calls.
- Tool calls simulated by the world but rejected by Pi.
- Missing call/result correlation.
- An unexpected renderer or thinking mode.
- An assistant turn that should have been a tool call but became plain text
  because of parsing failure.

Exit criterion:

- Every known baby-v3 `Previous tool-call history` trajectory is rejected.
- A clean native-renderer fixture is accepted.

### Step 5: Repair Claude bootstrap

- Stop using `claude -p --json-schema` for bootstrap.
- Ask Claude to write `world/bootstrap/response.json`, like normal world turns.
- Validate the file with the repository schema and semantic validator.
- On failure, resume the same thread with an artifact-repair prompt.
- Remove redundant boot fields that the orchestrator already knows or ignores.

Exit criterion:

- Repeated Sonnet boots succeed without `StructuredOutput` field corruption.
- A malformed boot artifact is repaired in the same thread.

### Step 6: Make runtime boundaries honest

- Define the actual isolation guarantees for Codex, Claude, and Pi separately.
- Do not tell every runtime that it is a sandboxed Codex process.
- Prevent or explicitly account for user plugins, skills, memories, and
  unrelated tools in Claude.
- Do not treat `world_permission_profile: game_master` as proof of
  isolation.
- Audit the actual tools and filesystem/network boundary available to each
  runtime.

Exit criterion:

- The runtime log proves the available tool set and isolation settings.
- The acceptance audit fails when the runtime exposes an unapproved capability.

### Step 7: Verify rollout weighting

- Determine Tinker's exact reduction behavior across datums and tokens.
- Normalize subsequences so one long rollout does not automatically outweigh a
  short rollout.
- Preserve downstream terminal credit for a compaction summary.
- Do not apply a second loss to that summary when it appears as context in the
  following subsequence.

Exit criterion:

- A rollout split into four subsequences has the same intended total rollout
  contribution as an equivalent one-subsequence rollout.

### Step 8: Re-run progressively

1. Run renderer unit tests using a captured real Pi payload.
2. Run a one-action no-training smoke.
3. Inspect every wire message manually.
4. Run a multi-action no-training smoke with sibling tools and thinking.
5. Run one disposable optimizer update.
6. Inspect the exact Tinker datums and advantages.
7. Discard the disposable checkpoint.
8. Only then decide whether to restart the baby run.

## P0 findings: training-critical

### P0.1 Native Tinker rendering was deliberately bypassed

The proxy contains an explicit guard:

```text
Tinker mode does not yet support structured OpenAI tool schemas.
```

The shellsmith conductor then configures:

```python
"allow_tools_in_tinker_text": True
```

This converts a known incompatibility into a silent workaround.

Relevant code:

- `backend/research/player/rl/llm_api_proxy/proxy.py`
  - `_tinker_response`
  - `tool_prompt_from_payload`
  - `stringify_message_for_tinker`
- `backend/research/player/simulator/generative_world/campaign/conductor.py`
  - `_proxy_config`

The workaround invents:

```text
<tool_call name="read">
{"path":"..."}
</tool_call>
```

It also flattens historical structured calls into:

```text
Previous tool-call history (not valid output):
- tool=read args={...}
```

However, Tinker's installed Qwen 3.5 renderer already supports:

- Structured tool specifications.
- Qwen-native XML tool calls.
- Structured historical tool calls.
- Structured tool results.
- Thinking blocks.
- Explicit unparsed tool calls.

The correct repair is to integrate those native primitives, not design another
tool language.

### P0.2 Pi's real tool schemas were discarded

Pi sends the standard OpenAI wrapper:

```json
{
  "type": "function",
  "function": {
    "name": "read",
    "description": "Read the contents of a file.",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {"type": "string"}
      },
      "required": ["path"]
    }
  }
}
```

The proxy's `compact_tool_schemas()` and `tool_name_aliases_from_tools()` look
for `name`, `description`, and `parameters` at the top level.

For an actual Pi request, the generated prompt therefore contained seven
entries equivalent to:

```json
{
  "name": "unknown",
  "description": "",
  "properties": {},
  "required": []
}
```

This directly explains invalid calls such as:

```text
read(file_path="...")
```

when Pi required:

```text
read(path="...")
```

The existing unit tests used a handcrafted top-level tool shape rather than a
real captured Pi payload.

### P0.3 High reasoning was silently disabled

The run configuration says:

```json
"player_thinking": "high"
```

Pi's registered provider says:

```js
supportsReasoningEffort: false
```

The actual Pi OpenAI payload contains:

- `messages`
- `tools`
- `model`
- streaming fields
- token limit

It does not contain:

- `reasoning_effort`
- `reasoning`
- `thinking`
- a legacy thinking-mode flag

The proxy interprets the missing signal as `disable_thinking` and selects the
non-thinking renderer.

Furthermore, the proxy calls `get_text_content()` when converting the sampled
message. That extracts ordinary text but strips Tinker's `ThinkingPart`.

Consequences:

- The baby RFT run was not thinking-on.
- The Game Master did not receive Player's actual hidden reasoning.
- Reasoning quality and efficiency scores often assessed only visible
  conversational preambles.
- The planned distinction between thinking-off SFT and thinking-on RFT was not
  implemented.

Relevant code:

- `backend/research/player/simulator/generative_world/episode/vat_runner/pi-session.js`
- `backend/research/player/rl/llm_api_proxy/proxy.py`
  - `resolve_tinker_thinking_mode`
  - `message_to_openai_dict`

### P0.4 The world simulated calls before Pi validated them

Current flow:

```text
Qwen samples assistant message
  -> Pi emits message_end
  -> runner extracts proposed tool calls
  -> bridge immediately sends them to Python/Game Master
  -> Pi later invokes its executor
  -> Pi validates arguments during executor invocation
```

Therefore, a malformed call can create a successful world response and immutable
ledger entry before Pi rejects it.

This happened in Insurance, Packaging, and go-jose trajectories. The world
reported successful `read` operations while Pi rejected the calls because the
required `path` argument was absent.

Relevant code:

- `backend/research/player/simulator/generative_world/episode/vat_runner/runner.js`
- `backend/research/player/simulator/generative_world/episode/vat_runner/tool-bridge.js`

Sibling batching remains correct as a goal. The defect is the point at which a
proposed sibling becomes an authoritative world action.

### P0.5 Corrupted terminal output passed acceptance

Baby v3 contained:

- 33 rollouts that reached an acceptance audit.
- 20 rollouts accepted for training.
- 16 terminal responses containing
  `Previous tool-call history (not valid output)`.
- 14 of those 16 corrupted terminal responses accepted for training.

These endings were caused by the adapter's injected text. Player sometimes
copied the marker and intended call arguments into its next completion. Pi saw
plain text rather than a structured call and ended the agent session.

The final judge often treated this as a low-quality Player response rather than
an infrastructure failure. The Tinker bridge accepts any rollout whose audit
says `accepted_for_training: true`.

This means adapter corruption could reach the optimizer.

### P0.6 Rollout weighting does not visibly enforce the agreed invariant

The intended invariant was:

> Compactions may split one rollout into several training subsequences, but
> those subsequences must be averaged or weighted back into one rollout
> contribution. A rollout with three compactions must not automatically count
> four times more than a short rollout.

Current behavior:

- One Tinker datum is created for every captured model request.
- Each datum receives its own final advantage.
- The same advantage is applied to every sampled-output token in that datum.
- All datums from both accepted groups are submitted together.

There is no explicit per-rollout normalization in the bridge or conductor.

Relevant code:

- `backend/research/player/rl/tinker/to_tinker_bridge.py`
- `backend/research/player/rl/contracts/grpo.py`
- `backend/research/player/simulator/generative_world/campaign/conductor.py`

Tinker's exact datum/token reduction semantics must be confirmed before deciding
the repair.

## P1 findings: world-simulator correctness

### P1.1 The Sonnet response-file diagnosis was partly wrong

Ordinary Sonnet world turns generally wrote valid response files. The recurring
Geospatial failure was mainly in bootstrap.

Bootstrap uses:

```text
claude -p --json-schema ...
```

Claude's `StructuredOutput` sometimes inserted serialized field syntax inside a
neighboring string. The CLI then reported that required fields were missing.

The normal turn-level `response.json` mechanism was substantially more reliable.

Relevant code:

- `backend/research/player/simulator/generative_world/gm_process/`
- `backend/research/player/simulator/generative_world/schemas.py`

### P1.2 Claude's actual environment is broader than claimed

A captured Claude initialization exposed tools including:

- Task and subagent operations.
- Cron and scheduling operations.
- Worktree operations.
- Skills and workflows.
- Remote and notification operations.
- User-installed plugins.
- User-level skills.

The universal prompt nevertheless says:

```text
Your Codex process runs under an episode-only filesystem profile.
```

That is runtime-specific and false for Claude.

The acceptance audit currently considers this configuration value sufficient:

```json
"world_permission_profile": "game_master"
```

That proves only that the value was written to `run_config.json`. It does not
prove the subprocess's isolation.

Relevant code:

- `backend/research/player/simulator/generative_world/prompts/game_master/AGENTS.md`
- `backend/research/player/simulator/generative_world/gm_process/`
- `backend/research/player/simulator/generative_world/episode/acceptance_audit.py`

### P1.3 Sonnet confused the physical vat with the virtual world

In the Geospatial retry, Sonnet answered a virtual `/workspace` listing with the
physical AGENTS-only vat listing.

This contradicted the universal prompt, which explicitly says:

```text
The physical vat contains only AGENTS.md; that is an implementation detail, not
the virtual world's initial filesystem.
```

The same Sonnet thread later judged its own observations consistent and accepted
the rollout.

This demonstrates a correlated-error risk:

```text
one agent creates the evidence
  -> the same agent remembers and evaluates that evidence
  -> its original simulation mistake is rationalized during final judgment
```

An independent judge is one possible mitigation, but the first requirement is
to expose this as a measurable failure rather than silently accepting it.

### P1.4 The artifact-repair prompt misstates the lifecycle

The repair prompt says:

```text
Preserve the world state and public observation you already committed to.
```

An invalid artifact is rejected before:

- Ledger append.
- Delivery to Player.
- Public commitment.

The model should avoid replaying real side effects that may already have
happened, but it must be free to correct an unpublished invalid response.

Relevant file:

- `backend/research/player/simulator/generative_world/prompts/WORLD_ARTIFACT_REPAIR_PROMPT.md`

### P1.5 Claude retries are not fully idempotent

For a transient provider failure, the launcher can resend the identical prompt
to the same Claude thread.

If the first physical attempt changed private files or world state before the
connection failed, replaying the prompt may repeat those changes.

The protocol includes correlation concepts, but it does not programmatically
prevent duplicate simulator side effects. Recovery should be artifact- and
state-aware rather than assuming the previous attempt did nothing.

## Prompt and calibration risks

These are not all confirmed blockers, but they should be intentional decisions.

### The embedded prompt contains unrelated legacy material

Every shellsmith Player request currently receives instructions about:

- OpenCode, although the harness is Pi.
- `propose_fcpxml_project`.
- Video-editing YAML keys such as `position_x`, `font_color`, and `track_name`.

This is unrelated to the generated worlds and should be removed from this path.

### Pi calls Player a coding assistant for every task

Pi's base system prompt describes Player as an expert coding assistant.

That can bias:

- Incident response toward filesystem probing.
- Database tasks toward searching for source files.
- Paper tasks toward code-oriented behavior.

This may be acceptable, but it is a distributional choice rather than a neutral
harness detail.

### Sonnet's difficulty tail may invent extra requirements

The Sonnet tail asks for:

- At least two coupled failure modes.
- Incomplete or version-sensitive tests.
- Several rounds of hypothesis testing.

This may cause Sonnet to add complications that were not committed in the
world file specification and then grade Player against those additions.

Difficulty calibration should control evidence presentation and required
problem-solving depth without changing the world's reward contract.

### The boot contract contains redundant generated fields

The model must generate:

- `initial_prompt`, although the orchestrator ignores it.
- `vat_agents_path`, which is already known.
- `vat_files`, which is already known.
- A private summary.
- Initial commitments.

The extra surface caused real Claude structured-output failures. The contract
should contain only information that the world model uniquely determines.

### The universal world prompt is cognitively heavy

For each action, the world model must simultaneously:

- Simulate the observation.
- Maintain hidden state.
- Grade the turn.
- Assess reasoning.
- Estimate completion.
- Review consistency.
- Review integrity.
- Decide trainability.
- Decide termination.
- Serialize the artifact.

Some of this is necessary, but the instructions should be simplified wherever
the orchestrator already knows or can deterministically compute the answer.

## Design choices that remain sound

The audit does not reject the overall experiment.

These pieces still make sense:

- Pi keeps Player's conversational context.
- The physical vat can remain an empty temporary directory containing only
  `AGENTS.md`.
- The Game Master can provide the virtual files, services, and command
  results.
- A persistent same-thread Game Master gives useful state continuity.
- Turn-level response artifacts provide a good audit surface.
- Same-thread artifact repair is appropriate.
- Sibling tool results should be gathered before any are returned to Player.
- Simulator-model identity must remain part of the rollout-group identity.
- Honest simulator failures should make a trajectory untrainable rather than
  count against Player.

The principal problem is the adapter implementation around those choices.

## Required tests before restarting

### Renderer tests

- Use a real captured Pi OpenAI payload.
- Verify all seven tool names and parameter schemas survive.
- Verify historical tool calls stay structured.
- Verify tool results retain `tool_call_id`.
- Verify multiple sibling calls round-trip.
- Verify malformed native Qwen XML becomes an explicit unparsed call.
- Assert that no text marker or custom tool-call syntax is inserted.

### Thinking tests

- Verify a high-thinking run selects the thinking renderer.
- Verify sampled reasoning becomes a Tinker `ThinkingPart`.
- Verify the OpenAI response carries reasoning to Pi.
- Verify Pi emits a thinking block.
- Verify the world request receives the same reasoning.

### Tool-validation tests

- Invalid arguments never create a world request.
- Valid arguments create exactly one world request.
- Mixed valid and invalid siblings cannot partially mutate the world.
- Valid sibling observations are returned together.

### Acceptance tests

- Every known adapter-contaminated baby trajectory is rejected.
- Missing or mismatched call IDs are rejected.
- Simulated-but-not-executed calls are rejected.
- An unexpected renderer/thinking mode is rejected.

### Claude bootstrap tests

- Bootstrap uses a response file rather than `StructuredOutput`.
- Malformed bootstrap output retries in the same thread.
- A repaired draft creates one accepted artifact.
- Retrying does not repeat already-persisted side effects.

### Training tests

- Confirm the exact Tinker reduction behavior.
- Compare equivalent short and compacted rollouts.
- Confirm rollout-level total weighting.
- Inspect every datum from one disposable optimizer update.

## Status at the time of this audit

- No relevant conductor, Codex, Claude, Pi, or proxy process was running.
- The audit was read-only.
- No harness code was changed as part of the audit.
- Existing working-tree changes and trajectory artifacts were left untouched.
