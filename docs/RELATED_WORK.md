# Shellsmith: related work

Literature survey (two deep-research passes, run 2026-09-29) positioning
Generative World against published work on LLM-simulated environments for agent
training. Papers and preprints from 2023 to mid-2026 were searched; claims were
adversarially verified (two of three reviewers had to refute a claim to kill it).

Shellsmith was called Generative World (GW) when this survey was run; "WorldSim"
below is what Shellsmith now calls the Game Master, and "task-env" is a World
file. Read [`GAME_MASTER_PROTOTYPE.md`](GAME_MASTER_PROTOTYPE.md)
first for the system being compared.

## Summary

- **The core idea is not new.** Using an LLM as the environment for agent RL
  is established (Simia-RL, SynthAgent, DreamGym, SWE-World, Qwen-AgentWorld, ...).
- **The closest prior work is Qwen-AgentWorld** (2606.24597): a general
  agent platform (OpenClaw) trained with RL entirely against a simulated
  environment with hidden state and injected errors.
- **The strongest novelty is that our WorldSim is itself a tool-using agent
  with its own terminal.** Every simulator found is a text predictor.
- No work found combines a real, unmodified coding harness in a full vat,
  an agentic simulator, per-turn rewards from that same simulator, and an
  "untrainable" flag.

"Appears novel" below means no counter-example was found, not that none exists.
Unpublished lab infrastructure is invisible to this search.

## What Generative World is (for comparison)

| Feature | Description |
| --- | --- |
| (a) Vat | Real Pi coding agent; workspace holds only `AGENTS.md`; every bash/read/write/edit is answered by the WorldSim; any real workspace mutation makes the episode untrainable. |
| (b) Agentic WorldSim | Codex CLI with a frontier model. Per action it may invent an observation, compute exactly with its own shell/Python, lazily materialize a file/repo/db, run the agent's code for real, or fetch public web data. Keeps a private ledger. |
| (c) Task-env | One Markdown file: World, Embedded agent, Reward, Termination. No tool schemas or fixtures; the WorldSim writes the agent's `AGENTS.md` and first prompt. |
| (d) Scoring | The same persistent WorldSim thread emits per-turn reward, difficulty, completion estimate, a reasoning-quality assessment, the final reward, and the untrainable flag. |
| (e) Roadblocks | Injected, solvable obstacles ("surmountable roadblocks"). |
| (f) Training | GRPO/CISPO on Tinker over about 150 tasks. |

## Closest work

### Qwen-AgentWorld (arXiv 2606.24597, Jun 2026): closest overall

Trained "language world models" used as environment simulators across 7
domains, including Terminal and SWE. Its **Sim RL** trains a policy agent
entirely in simulation, in three domains:

- **MCP:** Tool Decathlon, MCPMark.
- **Search:** WideSearch.
- **OpenClaw:** an open-source agent platform whose tasks include coding, email,
  file management and browser automation. About 4k environments are synthesized
  from a small set of real traces. Each has initial state (file layout,
  workspace), hidden state (database contents, permissions, service
  availability) and a rubric verifier.

"Controllable simulation" instructions inject errors and withhold results
during training. Reported gains on real benchmarks include +4.3 on Claw-Eval
and +12.3 on MCPMark.

Differences from Generative World:

- The simulator is a trained text-only model. It executes nothing and keeps no
  real files or tools.
- Rewards are trajectory-level from a rubric verifier, not per-turn.
- No explicit consistency ledger was described.
- Terminal and SWE appear only as world-model evaluations, with no simulation-only
  policy RL and no Terminal-Bench or SWE-bench transfer result.

Caveat: read through a summarizing fetch. Re-read the paper before making
strong claims about it.

### SWE-World (arXiv 2602.03419, Feb 2026): partial vat

GRPO training of an SWE agent without Docker. A learned Transition Model
predicts the output of commands that run code (python, pytest), and a Reward
Model simulates the test report and gives a final binary reward. Reports 55.0%
on SWE-bench Verified.

Difference: not a vat. `ls`, `cat`, `grep`, create and `str_replace` run
deterministically on a real repo checkout. Only execution is simulated, by
fine-tuned Qwen2.5-32B/72B models trained on Docker rollouts.

### Simia-RL (arXiv 2511.01824, Nov 2025): purest LLM-as-environment

A prompted reasoning LLM (o4-mini) produces tool outputs and error messages, and
also judges the trajectory for the reward, with no real environment.

Difference: 1/0 whole-episode reward from the simulator, with no separate
judge, no per-turn credit, no hidden-state specs, no ledger, no roadblocks.

### SynthAgent, "Mock Worlds, Real Skills" (arXiv 2601.22511, Jan 2026): most auxiliary features

A teacher LLM writes tasks and tool setups; an LLM answers tool calls and keeps
a per-task table of call-to-response pairs, reusing stored answers for
equivalent calls (a close analogue of our ledger). Rewards come from task-level
rubrics (required subgoals, forbidden behaviors). An LLM user simulator holds
private information.

Difference: simple question-answering tools, not a terminal or OS with hidden
state; tasks are deliberately underspecified so the agent must ask the user.

### Other simulator-based RL

| Work | Approach | Difference |
| --- | --- | --- |
| DreamGym (2511.03773, Meta) | Reasoning "experience model" gives transitions and rewards for GRPO/PPO, plus a curriculum task generator. | Distilled from real offline trajectories; web/embodied targets (WebArena, WebShop, ALFWorld); no separate judge. |
| GenEnv (2512.19682) | Simulator co-trained with the agent as a curriculum policy. | Closer to our adaptive campaign top-ups than to a fixed authored world. Abstract-only check. |
| DynaWeb (2601.22149), WebEvolver (2504.21024) | Learned web world models. | Short episodes: DynaWeb caps imagined rollouts at 5 steps because hallucinations accumulate. |
| GTM (2512.04535) | 1.5B simulator driven by JSON tool templates; RL on search, retrieval, CUDA-kernel tasks. | Schema-defined tool calls only; no bash or filesystem. |
| MTR (2510.06825) | Prompted ToolMaker/ToolActor; SFT then GRPO on simulated observations for multi-hop QA. | Invented schemas and observations; no terminal. |

## Terminals simulated by LLMs, without training

- **shelLM** (2309.00155) and **VelLMes** (2510.06975): LLM-simulated Linux
  shells (also SSH, MySQL, POP3, HTTP) used as honeypots against human
  attackers. Pure generation, no execution, no reward, no agent training. These
  anticipate "an LLM pretends to be a whole OS", not its use as an RL
  environment.

## Evaluation-only and adjacent work

- **ToolEmu** (2309.15817): GPT-4 emulates tools for safety evaluation, with an
  adversarial emulator that builds red-team scenarios. Partially anticipates
  roadblock injection, but for evaluation and failure-seeking.
- **MirrorAPI** (2503.20527): simulated APIs for benchmarking.
- **tau-bench / tau2-bench** (2406.12045, 2506.07982): only the user is
  simulated; tools are deterministic Python over a database.
- **Endless Terminals** (2601.16443): the opposite approach, scaling terminal RL
  with real Docker/Apptainer shells and real tests. Its LLM-driven task and
  container synthesis partially anticipates cheap task authoring.

## Code-executed synthetic environments (opposite runtime)

- **AWM / Agent World Model** (2602.10090): 1,000 LLM-synthesized environments
  with 35,062 tools, run as executable Python on SQLite. The authors argue this
  gives "more reliable and consistent state transitions" than LLM-simulated
  environments, which rejects a WorldSim-style design. Its GPT-5 judge plus
  GRPO training setup is similar to ours.
- **AutoEnv** (2511.19304): a natural-language description (goals, rules, state
  variables, reward conditions) becomes a YAML DSL and Python classes. Maps
  well onto our four Markdown sections, but no LLM answers actions at runtime.
- **CodeGym** (2509.17325) and **EnvGen** (2403.12014): LLM-authored
  environments executed as real code.

## Novelty assessment

| Claim | Verdict |
| --- | --- |
| Real, unmodified coding harness in a full vat; any real workspace change invalidates the episode | **Appears novel as a strict design.** The concept is partly anticipated by Qwen-AgentWorld OpenClaw Sim RL; SWE-World simulates only execution. |
| RL of a coding/terminal agent in a fully simulated OS | **Partially anticipated** by Qwen-AgentWorld and SWE-World. No terminal-specific simulation-only RL found. |
| **WorldSim as a tool-using agent** choosing per action among invent / compute exactly / lazily materialize / run for real / fetch web | **Appears novel. Strongest differentiator.** All simulators found are text-only generators. |
| Private ledger of entities and commitments | **Partially anticipated** by SynthAgent's call-response table; an explicit entity ledger appears novel. |
| One Markdown task-env with no schemas or fixtures; WorldSim writes `AGENTS.md` and the first prompt | **Partially anticipated** by Qwen-AgentWorld, AutoEnv, MTR. The no-schema form appears novel. |
| Same simulator thread gives per-turn reward, difficulty, completion, reasoning-quality assessment | **Appears novel.** Others give trajectory-level or final rewards, from a separate verifier or reward model. |
| "Untrainable" flag when the agent honestly discovers a simulation failure | **Appears novel.** |
| Surmountable roadblocks | **Partially anticipated** by Qwen-AgentWorld's injected errors in training and ToolEmu's adversarial emulator. |
| RL on simulated environments in general | **Anticipated** by Simia-RL, GTM, MTR, SWE-World, Qwen-AgentWorld. |

### Why the agentic WorldSim matters

Text simulators invent compiler errors, test output and arithmetic, and those
errors accumulate over long episodes (see DynaWeb's 5-step cap). An agentic
simulator can ground exactly the observations the agent is likely to check
carefully, without pre-built fixtures or a trained world model, and real files
alongside the ledger give a consistency mechanism the others lack.

## Caveats and open questions

- Novelty judgements rest on not finding counter-examples. Most sources are
  2025 to 2026 arXiv preprints that are not peer-reviewed, and performance
  numbers are the authors' own.
- Not verified: StableToolBench's cached virtual API server, WebDreamer, and
  newer LLM honeypots that might mix in real execution.
- The first-pass works (Simia-RL, DreamGym, SynthAgent, GenEnv, DynaWeb,
  AWM, AutoEnv, WebEvolver) were not re-checked for per-turn simulator rewards,
  ledgers or hybrid execution.
- GenEnv and Qwen-AgentWorld were checked from abstracts or summaries only.
- Open: does simulation-only training of a terminal/coding agent transfer to
  real Terminal-Bench or SWE-bench? Does the ledger plus real materialization
  hold consistency at 20+ turns, where learned simulators degrade at about 5?
- Suggested ablation to support the main claim: the agentic WorldSim versus the
  same model with tools disabled (pure text simulation), on consistency over
  20+ turns and on real-benchmark transfer.
