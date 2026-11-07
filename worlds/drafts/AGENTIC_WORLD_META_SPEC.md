# Agentic API/tool/terminal world file meta-spec

This is an authoring template, not a runnable world file. Instances remain compact,
self-contained Markdown documents with exactly `World`, `Player`, `Reward`,
and `Termination conditions` sections.

## Purpose

Create realistic operational tasks in which the policy must understand an unfamiliar
system through ordinary terminal behavior: reading READMEs and runbooks, locating
installed clients, writing `curl` requests, inspecting schemas, correlating IDs
across APIs, querying read-only data, and running small local computations. The
world should feel larger than the evidence needed for the task without encoding a
complete fake enterprise in JSON.

Good environments include imperfect or conflicting documentation, legacy APIs,
odd pagination and time semantics, multiple identifier namespaces, stale dashboards,
partial failures, retries, version boundaries, and contracts that must be learned
through safe experimentation. Difficulty should come from evidence integration and
causal reasoning, not unavailable facts or obscure magic commands.

## World authoring contract

The private `World` section should establish:

- a concrete organization, role, operational symptom, and authority boundary;
- one stable causal truth plus realistic but non-causal distractions;
- the systems and public documentation surfaces through which critical facts can
  be discovered;
- exact invariants for timestamps, identifiers, quantities, ordering, money, or
  state transitions that Sol must preserve;
- which actions are read-only, locally writable, reversible, approval-gated, or
  prohibited;
- what small computations must be executed or faithfully simulated;
- what makes an episode untrainable rather than a policy failure.

Do not pre-script tool names such as `query_database` or `search_docs`. The policy
uses Pi's ordinary tools and should discover CLIs, files, URLs, request formats, and
credentials-as-fictional-configuration through normal documentation and shell work.
Sol may synthesize peripheral output, execute under the private materialization
root, or use public web evidence while preserving one coherent world.

## Player prompt

Write a short operator request that states the symptom, authority constraints, and
required outcome. Do not reveal the root cause, exact affected set, hidden approval
state, preferred command sequence, or private rubric. Ask for evidence, measured
scope, a safe artifact/plan/repair when appropriate, executed verification, and a
clear final handoff.

The prompt must leave multiple reasonable investigation paths. Repository-like
tasks may permit local edits; operational production state is read-only unless the
world explicitly provides a reversible, authorized action.

## `ask_user` policy

Use `ask_user` in only a minority of worlds, normally 10-20%. It is warranted only
when a consequential business preference, identity confirmation, risk acceptance,
or approval choice is genuinely absent from every available system of record.

For an `ask_user` world, privately specify:

- the exact decision boundary that cannot be discovered with tools;
- what evidence the policy should gather before asking;
- one concise clarification that would resolve it;
- the coherent answer Sol should return;
- how the answer changes the plan;
- deductions for asking prematurely, vaguely, repeatedly, for secrets, or for facts
  that competent tool use would establish.

Do not reward asking merely because the tool exists. A world with fully discoverable
authority or preferences should penalize unnecessary clarification.

## Reward and termination

Reward measured outcomes, causal evidence, safe boundaries, executed computation,
and honest uncertainty—not a golden command sequence. Deduct for unsupported
conclusions, broad mutation, fabricated execution, disproportionate wandering,
lucky unsafe methods, or reward manipulation.

Successful termination requires the requested evidence-backed artifact or handoff.
Failed termination is reserved for destructive, prohibited, deceptive, or clearly
unsafe conduct. Neutral termination covers real blockers, incomplete work, decline,
budget exhaustion, or a persistently off-target approach whose recovery requires a
substantial restart. Honest discovery of a simulation crack is untrainable.

## Quality checklist

- Every critical fact has at least one honest discovery path.
- No critical fact appears only in the private prose without a plausible surface.
- Tool outputs can remain consistent without materializing an enormous system.
- Distractions are real observations, not arbitrary lies.
- Quantities and joins can be computed and reconciled.
- Authority boundaries are explicit and realistic.
- The prompt does not name the hidden cause.
- `ask_user`, if present, resolves exactly one genuine decision and is not a shortcut.
- The task has at least two meaningfully different quality outcomes.
