"""Default models and pinned tool versions for Shellsmith, in one place.

Architecture map
----------------
    Game Master (GM) defaults      Codex CLI ``gpt-6.1-sol``, effort medium, ChatGPT OAuth
        GM_PROVIDER_BY_RUNTIME     metadata provider label for each GM runtime
    Player defaults (eval-only)    OpenRouter ``qwen/qwen3.8-27b`` through Pi
    Player defaults (training)     Tinker ``Qwen/Qwen3.8-27B`` LoRA 64, Qwen3.8 xhigh renderer
    Pinned CLIs                    Codex 0.159.2 (verified with gpt-6.1-sol), Pi 0.84.1

Every CLI (``episode.smoke``, ``campaign.rollout_campaign``,
``campaign.checkout_batch``) and every GM runtime dataclass reads its default
from here, so changing a default is a one-line edit.

Why this exists
---------------
In the original monorepo these defaults were scattered across three CLIs and the Codex
dataclass, and they disagreed (the CLI said effort "low", the dataclass said
"high").
"""

from __future__ import annotations

# --- Game Master -----------------------------------------------------------
DEFAULT_GM_RUNTIME = "codex"
DEFAULT_GM_MODEL = "gpt-6.1-sol"
DEFAULT_GM_REASONING_EFFORT = "medium"
DEFAULT_GM_PROVIDER = "codex-cli"
DEFAULT_GM_TURN_TIMEOUT_SECONDS = 900.0
DEFAULT_GM_API_IDLE_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_WORLD_STEPS = 256
DEFAULT_PLAYER_PROMPT_TIMEOUT_SECONDS = 43_200.0
# Only the Pi GM runtime talks to an API directly; these apply to it.
DEFAULT_GM_API_KEY_ENV = "OPENROUTER_API_KEY"

# Provider label recorded for each GM runtime (metadata and outage remaps).
GM_PROVIDER_BY_RUNTIME = {
    "codex": "codex-cli",
    "claude": "claude-cli",
    "pi": "openrouter",
}

# --- Player (eval-only, OpenAI-compatible) ---------------------------------
DEFAULT_PLAYER_PROVIDER = "openrouter"
DEFAULT_PLAYER_MODEL = "qwen/qwen3.8-27b"
DEFAULT_PLAYER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_PLAYER_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_PLAYER_THINKING = "high"

# --- Player (training): Tinker -----------------------------------------------
# Qwen/Qwen3.8-27B LoRA. Tinker serves the plain name with only 65,536 tokens of
# context; the ":peft:262144" entry is the same model with the 262,144-token
# LoRA context Shellsmith's 224 KiB working window needs.
DEFAULT_TINKER_BASE_MODEL = "Qwen/Qwen3.8-27B:peft:262144"
# The Player renderer and acceptance audit are fixed to Qwen3.8. Exact names
# only; an allowed name must still pass the service's actual context check.
SUPPORTED_TINKER_BASE_MODELS = (
    DEFAULT_TINKER_BASE_MODEL,
    "Qwen/Qwen3.8-27B",
)
DEFAULT_LORA_RANK = 64
# Qwen3.8's template offers xhigh (its default), medium and low reasoning; there
# is no "high". tinker-cookbook >= 0.5.7 recommends the xhigh renderer.
DEFAULT_PLAYER_THINKING_RENDERER = "qwen3_8_xhigh_reasoning"
DEFAULT_PLAYER_PLAIN_RENDERER = "qwen3_8_disable_thinking"
# Env var through which Pi receives the Shellsmith endpoint's bearer token.
PLAYER_ENDPOINT_KEY_ENV = "SHELLSMITH_PLAYER_ENDPOINT_KEY"

# --- Pinned agent CLIs (also the docker/ build defaults) ---------------------
CODEX_CLI_VERSION = "0.159.2"
PI_CODING_AGENT_VERSION = "0.84.1"
