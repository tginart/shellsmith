# Game Master prompt assets

- [`game_master/AGENTS.md`](game_master/AGENTS.md) is installed as the
  persistent Game Master's universal project instruction file.
- [`WORLD_BOOTSTRAP_PROMPT.md`](WORLD_BOOTSTRAP_PROMPT.md) is rendered once per
  episode with the world file Markdown, episode ID, and empty vat path.
- [`WORLD_TURN_PROMPT.md`](WORLD_TURN_PROMPT.md) sends one intercepted action to
  the persistent thread, inlines a short Player thinking/actions preview, and
  still requires reading `request.json` plus a private `turns/<index>/response.json`.
- [`WORLD_FINAL_PROMPT.md`](WORLD_FINAL_PROMPT.md) asks the same thread to write
  the authoritative final-state reward to `final/response.json`.
- [`WORLD_ARTIFACT_REPAIR_PROMPT.md`](WORLD_ARTIFACT_REPAIR_PROMPT.md) retries a
  missing or malformed boot, turn, or final artifact on that same thread without
  replaying the underlying action.

The boot turn writes vat `AGENTS.md` plus private
`world/bootstrap/response.json`. The orchestrator takes Player's first prompt
from the world file Player section. On each subsequent turn, the
orchestrator forwards only `tool_response` into the vat; rewards and assessment
fields remain private audit and training data.
