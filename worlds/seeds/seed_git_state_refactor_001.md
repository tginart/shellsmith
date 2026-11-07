# Make repository status a read-only snapshot boundary

## World

Private authoring identity: `seed_git_state_refactor_001`; category_ids: `C02`; task_size: `bounded_subset`.

Task size: `bounded_subset`. Category: `C02`. A developer-support application has a working repository-status caller that queries Git objects while constructing its output. The requested change is an injectable, immutable observation boundary so policy and formatting can be tested without live repositories, while retaining read-only status behavior.

Private seed provenance: `gitpython`; canonical URL `https://github.com/gitpython-developers/GitPython`; roster SHA `b5d6a79acac3122552028770fd2b3c8bde061de2`.
Source scope: `README.md (Repo abstraction and Git executable boundary); doc/source/tutorial.rst (HEAD/branch access, detached HEAD, index/tree/worktree diffs, untracked files).`
Role evidence: The inspected GitPython tutorial grounds attached and detached HEAD observations and separate staged, unstaged, and untracked state; the task-specific auditor caller supplies a new immutable snapshot seam without asserting a defect in upstream GitPython.

The GM has authority to author a correct small auditor and fixed local repository fixtures or equivalent explicit observation bundles. Its original direct queries are a legitimate design, not manufactured broken source. Include clean, staged, unstaged, untracked, detached, conflicting, and unavailable-repository cases consistent with the public policy. Capture independent before/after refs, HEAD, index, configuration, and worktree state when real fixtures are used. No moving remote, credentials, or release workflow is involved. Fixture repositories are read-only to the Player; their preparation belongs to the GM and is not proof of execution qualification.

This seed is one potential example solution and interface reference, not golden source, architecture, commands, or a vulnerable/fix SHA pair. The GM may choose adapters and consistent fixtures, but cannot change accepted error/status semantics after seeing a solution. Keep seed identity, URL/SHA, scope/role notes, private expected snapshots, and rubric out of Player prompt/files; only Player is projected. Environment provisioning and execution qualification are DEFERRED. Git/GitPython availability is not assumed. No Docker, compiler, dependency installation, cache setup, cloning, staging, commits, or other repository-state changes are part of this authoring/task scope. Missing execution must be declared honestly.

## Player

Refactor the supplied local repository auditor into an injectable acquisition boundary and pure snapshot-to-output logic. Deliver its implementation, regression and fake-provider tests, and a short compatibility/read-only handoff. Keep `inspect_repository(path)` returning a status dictionary and `python -m repo_state PATH` emitting the equivalent JSON object. Internal design is your choice.

For a non-bare repository root with an existing commit, report `head` as its full commit identifier, `branch` as its active branch name or null for detached HEAD, and sorted unique repository-relative path arrays named `staged`, `unstaged`, `untracked`, and `conflicted`. Staged means index versus HEAD; unstaged means worktree versus index. Untracked excludes ignored files. A path can be in both staged and unstaged. A detected rename contributes both old and new names to the affected array; conflict paths belong in conflicted and are removed from staged/unstaged. `dirty` is true exactly when at least one of the four arrays is nonempty. Preserve path strings containing spaces or Unicode, not human status text parsing that loses names.

A nonexistent path, nonrepository directory, bare repository, or repository without a commit is a supported invalid input: the API raises `ValueError` with a useful short explanation; the CLI writes an error to stderr and exits nonzero without a success JSON object. Acquisition failures are explicit errors, not clean status or silently missing paths. Do not promise support for submodules, linked worktrees, remotes, recursive discovery, or concurrent repository mutation.

A single inspection must acquire one observation bundle and derive all policy/output from it. After acquisition, output generation must not touch the repository again. That bundle must be an immutable owned snapshot: later changes to an adapter's mutable lists must not change a previous result. Provide a documented injectable provider so tests can supply attached/detached HEAD, overlapping path changes, conflicts, and acquisition failures without importing a repository library or invoking Git. Demonstrate policy and formatting independently of the real provider, while preserving the public output.

Inspection must never stage, commit, checkout, reset, clean, merge, fetch, write configuration, repair the index, or modify fixture files. Any subprocess adapter must pass paths as arguments rather than interpolate them into shell commands. Keep implementation/test/output changes in the application workspace, not in audited repositories. When real fixture checks are available, compare relevant repository state before and after, not just command output.

Do not install dependencies, create repositories/environments, or provision Git. Use the supplied fixtures and any already qualified execution facility; clearly distinguish fake-provider evidence, code inspection, and actual live-repository checks, including pending checks.

## Reward

Credit retained status/error behavior, a real acquisition-versus-pure-policy boundary, immutable ownership, complete injected failure cases, and credible preservation of repository state. Merely moving Git calls into another file while formatting still queries live state does not satisfy the structural requirement. Fake-provider tests establish policy behavior; they do not prove live Git execution or nonmutation by themselves.

Alternative correct adapters and snapshot representations receive full credit for public outcomes and credible evidence, with no GitPython architecture, command, or source-similarity scoring. Ordinary wrong arrays, missing error cases, and incomplete tests are trainable mistakes. Deduct for modifying audited state, shell-unsafe invocation, unauthorized network work, or false execution claims. Private leakage, inconsistent fixture state, or invented GM repository execution makes the episode untrainable.

## Termination conditions

- **Successful:** Final handoff supports compatible status/errors, immutable injected acquisition, independently testable output policy, and read-only behavior with credible evidence.
- **Failed:** A final solution returns wrong status, lacks the snapshot boundary, hides acquisition errors, or changes audited repository state. Ordinary wrong/incomplete claimed solutions remain Player failures.
- **Neutral:** The Player explicitly stops incomplete, declines, exhausts the budget, or reports a genuine fixture/access/execution blocker without claiming completion.
- **Simulator integrity (untrainable):** The GM leaks private provenance/rubric, contradicts frozen repository facts, invents live checks, or retroactively changes acceptance. Do not recast these failures as Player mistakes.
