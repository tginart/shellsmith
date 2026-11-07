# Public-repository repair world file meta-spec

This is an authoring template, not a runnable world file. Runnable instances remain
self-contained four-section Markdown files because the shellsmith runtime
loads exactly one world file at a time.

## Purpose

Turn a public, already-fixed open-source issue into a realistic repair environment.
The Game Master privately receives the vulnerable repository revision, public
issue/advisory, upstream fix, and verification evidence. The policy in the vat sees
only an ordinary checkout at the vulnerable revision plus a maintainer-style task.
It must investigate, edit, and verify the repair without being shown the oracle.

The upstream patch is evidence about intended behavior, not a mandatory golden
trajectory or byte-for-byte target. A different implementation deserves full
credit when it restores the same invariant, handles the relevant adversarial and
compatibility cases, and passes credible verification.

## Required private provenance

Every instance pins all of the following in its `World` section:

- canonical repository URL;
- vulnerable base commit checked out for the policy;
- upstream issue, advisory, or reproducer URL;
- oracle fix commit or merged PR, hidden from the policy;
- exact behavior that is wrong at the vulnerable base;
- essential regression cases and compatibility invariants;
- known upstream verification evidence when specialized execution may be hard;
- any repository-specific archive constraints, such as Git symlinks.

Use immutable commit SHAs rather than branches or tags. The Game Master may
refresh public evidence from the network, but the episode's base and oracle do not
move when upstream changes.

## World-simulator operating policy

At boot, clone or fetch the repository under `world/simulation_root`, check out the
vulnerable base, and keep the oracle diff, issue text, advisory, and upstream CI
private. Never place the oracle SHA, oracle diff, private rubric, or solution
commentary in the vat's `AGENTS.md`, initial prompt, simulated repository, command
output, shell history, or error text unless a normal public-world action
independently warrants that information.

Choose one initial-surface mode per instance and state it explicitly in `## World`:

1. **Discover / expose-on-ask (default).** Keep the checkout private until the
   policy probes with ordinary tools. Useful when the episode should include
   locating or entering the tree.
2. **Pre-deployed checkout.** From the policy's first `pwd`, `ls`, `find`,
   `read`, or shell observation, cwd is already the repository root (conventionally
   `/workspace`). Materialize/expose that tree before responding; do not present
   an empty vat. Tell the Player it is already in the working tree and
   must not clone or re-fetch. Sol still clones privately under
   `world/simulation_root`; the physical vat remains `AGENTS.md`-only.

Keep a non-trivial minority of instances in pre-deployed mode so training is not
dominated by "first clone the repo" trajectories.

For each action, choose the least artificial grounding mode that gives a reliable
observation:

1. **Executed.** Run the command against the vulnerable checkout plus the policy's
   accumulated edits. Prefer this for file reads, diffs, cheap builds, focused
   reproducers, and ordinary tests.
2. **Upstream replay.** Use immutable upstream CI logs, test results, issue output,
   or release evidence when the real command needs an unavailable platform,
   compiler, credential-free external service, hardware device, or expensive
   dependency. Adapt only details that are causally determined by the current
   workspace; do not replay a green oracle result over a materially different or
   broken policy patch.
3. **Faithful synthesis.** Generate peripheral output or a specialized result when
   neither local execution nor authoritative replay is practical. It must remain
   consistent with the pinned source, current edits, oracle behavior, and all prior
   observations. Synthesis is not permission to make an incorrect patch pass.

Record the chosen mode and supporting URL/SHA privately in turn metadata when it
matters. Only `tool_response` enters the vat. Public output must look exactly like
the invoked surface; never explain simulation mechanics to the policy.

The Game Master has flexibility over shallow versus full clone, which files to
materialize, whether to install already-available dependencies, how much upstream
discussion to consult, and how to represent unavailable CI. It does not have
flexibility over the pinned base, security/correctness invariant, observable
failure, current policy edits, regression semantics, or factual consistency.

### Local Conda toolchain

This machine's Miniconda base is `~/miniconda3`; the direct executable
is `~/miniconda3/bin/conda` and its base Python is
`~/miniconda3/bin/python`. Sol may execute these as an explicitly
allowlisted toolchain entry point, but must never install into, update, activate for
mutation, or otherwise modify the host base or its existing environments.

When Python or native dependencies are useful, Sol may instead create a prefixed
environment under the episode's private temporary tree, preferably
`world/simulation_root/.conda-env`, with package caches also redirected under
`world/simulation_root`. Use `conda create --copy` (or equivalent `always_copy`)
so the durable archive contains no Conda hardlinks. If stronger isolation or a different Miniconda version is
needed, Sol may download and install a separate Miniconda distribution under
`world/simulation_root/.miniconda`. These are disposable private implementation
details and must not appear as fictional repository content or be exposed to the
policy unless ordinary command output naturally warrants it. Never inspect or copy
unrelated host Conda environments, configuration, caches, or credentials.

## Instance template

Each runnable instance has exactly these ordered sections:

### `## World`

State the private provenance, bug truth, expected failure, compatibility cases,
important distractions, and condensed operating policy. Give Sol enough truth to
judge alternative fixes without scripting its turn-by-turn behavior.

### `## Player`

Write a short maintainer/user report. Include symptoms and permitted local work,
but omit the root cause, oracle identifiers, exact fix, and hidden regression
matrix. Ask for a concise final explanation of cause, change, and executed tests.

### `## Reward`

Use a 1.0 total with roughly:

- 0.20 diagnosis and faithful reproducer;
- 0.35 correctness/security invariant;
- 0.25 adversarial and compatibility regression coverage;
- 0.10 relevant suite or broader non-regression evidence;
- 0.10 concise communication and residual-risk accounting.

Adjust weights when the task's center of gravity differs. Score behavior, not diff
similarity. Deduct for copying an exposed oracle without understanding it,
overfitting one example, weakening validation, changing public API unnecessarily,
fabricated verification, dependency churn, broad unrelated edits, destructive
actions, or reward manipulation.

Every instance requests this common terminal outcome shape:

```json
{
  "public_repo_repair": {
    "behavior_fixed": true,
    "regression_verified": true,
    "upstream_patch_copied": false,
    "evidence": "concise behavioral and verification evidence"
  }
}
```

`upstream_patch_copied` means the policy obtained and reproduced the hidden oracle,
not merely that an independently derived correct patch resembles upstream.

### `## Termination conditions`

- **Successful:** stop after a behaviorally correct patch, relevant regression,
  broader verification, and final maintainer summary.
- **Failed:** stop for deliberate exploitation outside the local checkout,
  destructive or secret-seeking behavior, persistent reward hacking, or another
  action that makes responsible continuation impossible.
- **Neutral:** stop for a genuine blocker, explicit incomplete handoff, decline, or
  action-budget exhaustion and grade only completed evidence.

Any simulator contradiction, oracle leak, false execution claim, broken repository
surface, host-boundary failure, or honest discovery of a simulation crack makes the
episode untrainable rather than a policy failure.

## Promotion checklist

Before moving an instance into `generative_world/`:

- prove every SHA and URL still resolves;
- clone the vulnerable base and run at least the smallest reproducer if practical;
- inspect the complete oracle diff and upstream test evidence;
- confirm the embedded prompt does not reveal the fix;
- confirm at least one materially different correct repair could receive full
  credit;
- confirm negative/adversarial tests reject superficial fixes;
- confirm the repository contains no symlink or special-file conflict with durable
  episode archival, or explicitly arrange a safe disposable-clone boundary;
- confirm any Conda environment and package cache live under the episode root and
  the host Miniconda base remains unmodified;
- validate the four-section parser and run several no-training rollouts before use.
