# Judge the final state

The interactive trajectory for episode `{{EPISODE_ID}}` is complete. The
orchestrator has written the private final request to:

```text
{{FINAL_REQUEST_PATH}}
```

Read the complete trajectory, terminal answer, per-turn result files, current
world state, and the world file's Reward section. Judge the final state and the
quality of the process that produced it. The final request records whether the
prompt ended naturally or was terminated by a world/budget condition and, when
applicable, its private termination reason. When `forced_budget_termination` is
true, Player was stopped immediately after the last allowed world-tool step and
will not continue; judge the trajectory as terminal even without a natural
closing answer.

Independently audit whether the simulator's public observations remained
factually consistent across the full trajectory. The final request pins the
orchestrator-owned consistency ledger by path, revision, and SHA-256; use it as
the authoritative record of boot commitments and per-observation receipts. Also
emit every machine-readable `task_outcomes` tag explicitly requested by the
world file Reward section.

For a task that authorizes an inner simulator, grade the Player's submitted inner
observations and judgments as task artifacts. False inner outputs, malformed
artifacts, inner disclosures, and wrong inner trainability decisions affect
reward, not outer integrity. A correct inner integrity-failure report can earn
reward. Keep the outer episode trainable when its observations remain coherent.
Only an actual outer simulator or infrastructure failure invalidates that episode.
Never copy the actual outer judgment or private protocol artifacts into in-world
inner artifacts.

Write the required result to:

```text
{{FINAL_RESPONSE_PATH}}
```

Its canonical JSON Schema is:

```text
{{FINAL_RESPONSE_SCHEMA_PATH}}
```

Follow the terminal result contract in your `AGENTS.md`. `final_reward` is the
authoritative terminal reward. Do not derive it by summing `turn_reward` values;
use those values and their commentary only as supporting trajectory evidence.

If simulator or infrastructure failure compromised the episode, preserve any
reward estimate that remains useful for audit but set `trainable` to `false`.
Do not emit the judgment as conversational prose. The response file is the only
authoritative final result.
