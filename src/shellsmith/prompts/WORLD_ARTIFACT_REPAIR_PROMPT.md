# Repair the authoritative world artifact

Your previous `{{STAGE}}` turn completed, but the orchestrator could not accept
the authoritative artifact at:

```text
{{RESPONSE_PATH}}
```

Validation failed with:

```text
{{VALIDATION_ERROR}}
```

This is a repair of the same turn, not a new Player action. Re-read the
original request at `{{REQUEST_PATH}}` and your private instructions. Preserve
the world state and public observation you already committed to. Do not replay
an external mutation merely because the result file was missing or malformed.

Write or overwrite the required artifact at `{{RESPONSE_PATH}}`, satisfying the
same contract and correlation IDs. The artifact is the only authoritative
result. Do not substitute a conversational explanation.
