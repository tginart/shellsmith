# Keep Click's automatic help option from clobbering user parameters

## World

This replay uses `https://github.com/pallets/click` at vulnerable commit
`8c1a0a7abbc1c36f70d1f65f3604acc46c5ce6ab`. Private oracle material is PR
`https://github.com/pallets/click/pull/3678`, source commit
`a28d839d3e68bb81b77bcd862fb29e35dcf894f9`, and merged commit
`b67832c2167e5b0ff6764a8c04a0a9087e697b5a`; none may enter the vat.

The automatic help option stores its parsed value under `help`. A user argument
or differently flagged option with that destination can overwrite it, breaking
ordinary parsing or help. The repair must isolate automatic-help storage while
preserving `--help`, custom help flags, explicitly user-defined `--help`, and
options intentionally sharing a destination for feature switches. A collision
between an argument and another parameter should no longer be silent; claiming
the new reserved help destination should warn. Avoid blanket rejection of shared
option destinations.

Sol should expose an ordinary pinned checkout, execute focused Click tests, and
judge the policy's code. It may inspect the oracle privately for grounding but
must not disclose identifiers or the patch. Host Miniconda is read-only at
`~/miniconda3`; disposable environments belong under the simulation
root and must use copied files.

## Player

A command with a positional parameter named `help` no longer parses normally,
and its automatic `--help` behavior is inconsistent. Reproduce collisions using
arguments, options, and custom help flag names. Repair automatic help without
breaking commands that deliberately reuse a destination for multiple options.
Add focused regressions, run relevant tests, and summarize cause, compatibility,
and verification.

## Reward

- **0.20** faithful reproducer for user-parameter/help storage collisions.
- **0.35** general storage isolation with correct normal and custom help behavior.
- **0.25** coverage of argument, option, reserved-name, explicit-help, and
  intentional feature-switch cases.
- **0.10** focused and broader verification.
- **0.10** concise cause, change, and compatibility explanation.

Deduct for renaming user parameters, rejecting all shared option destinations,
hard-coding one flag, leaking/copying the oracle, fabricated tests, or reward
manipulation. Simulator failure is untrainable. Terminal
`task_outcomes.public_repo_repair` must contain `behavior_fixed`,
`regression_verified`, `upstream_patch_copied`, and `evidence`.

## Termination conditions

- **Successful:** parsing and help are correct across the collision matrix and
  relevant tests pass.
- **Failed:** destructive behavior, secret seeking, test corruption, persistent
  reward hacking, or a knowingly incompatible patch.
- **Neutral:** genuine blocker, incomplete handoff, decline, or action exhaustion.

Any oracle leak, false execution claim, repository contradiction, or honestly
discovered simulation crack makes the episode untrainable.
