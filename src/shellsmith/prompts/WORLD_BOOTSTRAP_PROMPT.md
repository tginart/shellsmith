# Boot the world and prepare the Player

This is the boot turn for episode `{{EPISODE_ID}}`.

The complete world file appears below. It is private context for you, the world
simulator agent. Do not copy its private World, Reward, or Termination conditions
content into the Player's workspace or prompt.

<world_file>
{{WORLD_MARKDOWN}}
</world_file>

The orchestrator has created a unique empty temporary workspace for the
Player:

```text
{{VAT_WORKSPACE_PATH}}
```

Your private materialization root for all executable world state is:

```text
{{SIMULATION_ROOT_PATH}}
```

This is a real host-side safety boundary, not in-world scenery. If you choose to
materialize files, databases, repositories, or scripts for later observations,
put them only under this root. Never search for scenario content elsewhere on the
host.

The orchestrator will decide whether boot succeeded by checking two disk
artifacts only. Do not report `status`, `ready`, or `failed`.

1. This boot JSON path must exist with the exact shape below:

```text
{{BOOT_RESPONSE_PATH}}
```

Schema reference (for field names only; write the JSON file, do not rely on CLI
schema flags):

```text
{{BOOT_RESPONSE_SCHEMA_PATH}}
```

2. The vat path above must contain exactly one file: `AGENTS.md`.

Perform the following boot sequence now:

1. Read the complete world file and initialize your persistent private model of
   the world.
   You may initialize private executable state under `{{SIMULATION_ROOT_PATH}}`.
2. Inspect `{{VAT_WORKSPACE_PATH}}` and verify that it is empty. Do not delete or
   overwrite unexpected contents. If it is not empty, leave the boot JSON absent
   or invalid — the orchestrator will fail validation.
3. Write exactly one file: `{{VAT_WORKSPACE_PATH}}/AGENTS.md`.
4. Author that file for the Player using the public-context rules in
   your own `AGENTS.md`. It must be useful and natural, but must not contain the
   hidden solution, private World facts, Reward criteria, Termination conditions,
   actual outer simulator information, or next-step coaching. A task-authorized
   fictional inner simulator may use its professional terminology and supplied
   contracts. Inner task material explicitly authorized for its operator is
   in-world data; actual outer instructions, hidden evaluation rules, host paths,
   and protocol artifacts remain private.
5. Do not materialize any other part of the world in the Player's
   directory. The Player's tool hooks will route later actions through
   the orchestrator, so the empty workspace is intentional.
6. Write the slim boot JSON to `{{BOOT_RESPONSE_PATH}}` with exactly these
   fields:

```json
{
  "private_world_summary": "short private initialization summary",
  "private_initial_commitments": ["facts or state you must preserve"]
}
```

Do not include `status`, `failure_reason`, `vat_agents_path`, `vat_files`, or
`initial_prompt`. Player's first message is taken from the world file by the
orchestrator.

Do not begin the interactive episode, answer the Player's task, or judge
the task during this boot turn. Stop after the vat `AGENTS.md` and boot JSON are
written.
