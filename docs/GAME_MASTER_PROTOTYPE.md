# Generative Game Master prototype

> Status: lightweight no-training experiment.

## Hypothesis

A strong persistent model can simulate a coherent, expansive environment for a
weaker Player from a short description. The simulator may execute real
computations when useful and synthesize the rest.

This hypothesis must be tested directly. We should not prebuild most of the
environment in JSON, fixtures, mock services, or orchestration code. If current
models are poor Game Masters, elaborate prompting is not the solution this
experiment is looking for.

## Architecture

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant W as Persistent Game Master in Codex
    participant S as Episode-only simulation root
    participant V as Unique temporary cwd
    participant N as Node Pi-SDK vat runner
    participant E as Embedded Qwen session

    O->>V: Create unique empty temporary directory
    O->>W: Universal boot prompt + complete world file + vat path
    O->>S: Create private materialization root
    W->>V: Write the only initial file: AGENTS.md
    W->>O: Boot ready + Player initial prompt
    O->>N: Start persistent JSONL child process
    N->>E: Create in-memory Pi session; load AGENTS.md
    O->>N: Send initial prompt
    loop Until Player terminates
        E->>N: Call wire-backed Pi tool
        N->>O: tool_request; executor Promise waits
        O->>W: Turn request path + new action
        W->>S: Optional translated computation/materialization
        W->>O: Write turns/N/response.json
        O->>N: tool_response only
        N->>E: Resolve tool Promise
    end
    N->>O: Final response
    O->>W: Judge prompt + complete trajectory
    W->>O: Reward, rationale, trainability
```

The two sides intentionally have different process lifecycles:

- **Player side:** a persistent Node process using the Pi SDK and an
  in-memory Qwen session.
- **World-simulator side:** a persistent `codex exec` thread.
- **Orchestrator:** a small Python loop that forwards structured events,
  persists audit logs, and enforces budgets.

The Node runner disables Pi's executable built-ins. It exposes replacements
with Pi's normal names, descriptions, and argument schemas, but each `execute()`
sends a JSONL `tool_request` and waits. This preserves the normal model-facing
harness while ensuring no requested file or shell action touches the real cwd.

## Universal prompts

Five reusable prompt assets define the Game Master lifecycle:

- [`../prompts/game_master/AGENTS.md`](../src/shellsmith/prompts/game_master/AGENTS.md)
  gives every Game Master its stable role, privacy boundary,
  simulation rules, and judging responsibilities.
- [`../prompts/WORLD_BOOTSTRAP_PROMPT.md`](../src/shellsmith/prompts/WORLD_BOOTSTRAP_PROMPT.md)
  injects the task-specific Markdown and empty vat path for one episode.
- [`../prompts/WORLD_TURN_PROMPT.md`](../src/shellsmith/prompts/WORLD_TURN_PROMPT.md)
  supplies one private action request and requires a durable per-turn result.
- [`../prompts/WORLD_FINAL_PROMPT.md`](../src/shellsmith/prompts/WORLD_FINAL_PROMPT.md)
  asks the same persistent thread for the authoritative final-state reward.
- [`../prompts/WORLD_ARTIFACT_REPAIR_PROMPT.md`](../src/shellsmith/prompts/WORLD_ARTIFACT_REPAIR_PROMPT.md)
  repairs a missing or malformed authoritative artifact on the same world
  thread without replaying the underlying action.

The Game Master writes the vat's task-appropriate public `AGENTS.md`
during boot and returns the exact initial Player prompt. The
orchestrator starts Pi only after that boot completes successfully.

## The Player vat

For this prototype, the vat is not Docker. It is a unique temporary root:

```text
/tmp/shellsmith-vat-<random>/
  workspace/
    AGENTS.md
  agent-home/
```

The workspace begins empty. World boot materializes exactly `AGENTS.md`, which
Pi reads as public context. The Node runner uses an in-memory session and
settings manager, and its isolated `agent-home` prevents global Pi configuration
from entering the episode.

No repository, API server, database, logs, or scenery is created eagerly. The
real workspace must remain `AGENTS.md`-only for the complete episode. Simulated
files and commands exist in world-simulator state. The Python host validates the
one-file invariant before launch and after every completed prompt; any mutation
makes the episode untrainable.

The temporary root is deleted after the episode. Docker may be added later as
defense in depth, but it is not required for the semantic tool boundary.

The Game Master has a separate host-data boundary. Its model-generated
commands use a custom Codex permission profile with only minimal runtime reads,
read/write access to the episode workspace, and public network access. The
orchestrator
creates `world/simulation_root` inside that workspace. Player commands,
absolute paths, URLs, and environment references are untrusted virtual input:
the Game Master decides whether to synthesize their result, translate them
into an operation under the simulation root, or fetch a public source for a more
faithful answer. Network access never authorizes private host or credential
access. A unique cwd alone does not provide this isolation.

## World file

Each episode is driven by one short Markdown file:

1. `World` — private setting and important truths for the Game Master.
2. `Player` — visible task, role, access, and constraints.
3. `Reward` — outcome rubric, deductions, and untrainable conditions.
4. `Termination conditions` — successful, failed, and neutral episode endings.

See [`WORLD_FILE_SPEC.md`](WORLD_FILE_SPEC.md).

No workspace tree, API schema, database fixture, evidence graph, causal-rule
language, or tool contract is required.

## World-agent behavior

The Game Master owns simulation decisions. Given a Player action, it may:

- answer directly with a consistent simulated observation;
- use its shell or Python tools to compute an exact result;
- create a small file, repository, database, or script when persistence matters;
- run Player-authored code and return the real output;
- maintain a private ledger of entities and prior commitments; or
- decide that a peripheral request can be answered without materialization.

These are options, not an orchestrator decision tree. We want to measure the
model's judgment about when execution matters.

The shared wrapper should enforce only a few invariants:

- reveal only what the Player's action could reveal;
- remain consistent with the world file and prior observations;
- do not coach the Player toward the answer;
- never claim critical execution that did not occur;
- treat Player attempts to rewrite the simulator as untrusted input; and
- never replay Player commands or paths against the real host; and
- mark honest discoveries of simulation failures as untrainable.

## Minimal event protocol

The orchestrator needs only these logical messages:

```text
create_empty_vat() -> vat_path
boot_world(world_file_markdown, vat_path) -> world_thread_id, initial_prompt
launch_node_vat(vat_path) -> player_session_id
prompt_player(initial_prompt)
  -> tool_request(tool_call_id, tool_name, arguments, trajectory_delta)
write_turn_request(turn_index, player_delta, action) -> turns/N/request.json
world_step(world_thread_id, request_path) -> turns/N/response.json
resolve_tool_promise(tool_call_id, response.tool_response)
finish_player(player_session_id) -> trajectory, final_response
write_final_request(trajectory, final_response) -> final/request.json
judge(world_thread_id, request_path) -> final/response.json
```

Python and Node use strict LF-delimited JSONL. Correlation IDs are single-use,
tool execution is sequential in the prototype, and stdout contains protocol
records only. Exact Pi, vat-wire, and Codex records remain in the episode audit
directory, but the world file does not know about them.

The default host budget is 100 world-resolved tool actions. After exhaustion,
the policy receives one ordinary tool result telling it to finish from existing
evidence; a repeated tool attempt ends the live prompt. This is a safety ceiling,
not a reward criterion, and the world file's neutral termination rule determines
how the resulting incomplete attempt is judged.

## Turn and final rewards

Each world-simulator turn owns a private directory:

```text
world/
  turns/
    000035/
      request.json
      response.json
      ...optional simulator scratch artifacts...
  final/
    request.json
    response.json
```

The turn `response.json` is semi-structured. It contains a public
`tool_response` plus private `turn_reward`, `difficulty_rating`,
`completion_percentage`, commentary, and integrity fields. The Game Master
may add useful audit fields. The orchestrator validates the required fields and
matching action ID, then forwards only `tool_response` into the vat.

Pi's completed pre-action reasoning blocks are included in the private
`trajectory_delta` sent to the Game Master. A turn may therefore include an
optional structured `reasoning_assessment` with quality, difficulty-relative
efficiency, and commentary. This can reward correct, direct thinking and
discount needless deliberation on easy actions without treating raw brevity as
the objective. The simulator must not generate replacement reasoning: it sees
hidden world state and the action result, so any rewrite is vulnerable to
hindsight contamination. If corrected reasoning is explored later, a separate
rewriter must receive exactly Player's pre-action policy view. The assessment
remains scalar, private, and never re-enters the vat.

The final `response.json` contains the authoritative `final_reward`, criterion
results, process deductions, completion estimate, and trainability decision.
Turn rewards provide dense process supervision; final reward evaluates the
terminal state. They remain separate signals and are not automatically summed.

## Pi context compaction and training rows

RFT uses Tinker's long-context mode with a 229,376-token working window inside
the 256 Ki-token sequence ceiling. Pi's standard automatic compaction remains
enabled. The gap to the hard ceiling reserves space for renderer overhead and
completion tokens.

Treat one rollout as several training subsequences separated at each Pi
compaction:

- Sequence A trains on the ordinary actions plus the generated compaction
  summary at its tail.
- Sequence B receives that summary and Pi's retained recent tokens as masked
  context, then trains only on newly generated tokens.
- The original pre-compaction history is not repeated in Sequence B.
- Every subsequence retains the same rollout identity. Their losses are
  averaged or weighted back into one rollout contribution, so a rollout with
  three compactions does not automatically count four times more than a short
  rollout.

The generated compaction summary receives downstream credit from behavior after
the boundary because its quality causally affects the next subsequence. Applying
the rollout's terminal advantage across all of its subsequences provides that
credit. The same summary is masked context in Sequence B, so its tokens receive
loss exactly once rather than being trained twice.

The vat audit records the exact Pi compaction summary, the retained-entry cut
point, and the estimated token count before compaction. Preserve the original
uncompacted event stream for audit even though it is not repeated in the next
training row.

## Failure semantics

There are three distinct failures:

1. **Player failure:** the environment remained coherent, but the Player made a
   poor decision. Score it normally.
2. **World-simulator failure:** the simulator contradicted itself, leaked hidden
   context, fabricated execution, or could not sustain immersion. Mark the
   episode untrainable.
3. **Infrastructure failure:** Pi, Codex, or orchestration crashed or produced an
   ambiguous event. Mark the episode untrainable and preserve logs.

An unconventional Player path is not a world failure merely because the task
author did not anticipate it.

## Prototype

Begin without training:

1. Hand-author three one-to-three-page Markdown world files.
2. Run a strong Player model first as an adversarial environment sanity check.
3. Run the target Player model through the same tasks.
4. Inspect whether Codex sustains identity, state, causality, and computational
   honesty over long trajectories.
5. Measure simulator cracks, reward stability, latency, and Player
   usefulness.

The prototype succeeds if the worlds feel expansive, critical computations are
credible, the same hidden commitments survive many turns, and humans can audit
why the final reward was assigned.

Only after that evidence should we add rollout generation, Tinker training,
simulator optimization, or richer harness integrations. Throughput should come
from 4–8 independent rollout workers in parallel; each individual world remains
serialized to preserve a single causal order.
