"""Create Tinker sampling clients for the Player, after checking the model fits.

Why this exists:
    Shellsmith's renderer and acceptance audit are fixed to Qwen3.8, so other
    model names are rejected before creating an authenticated service client.
    Shellsmith also advertises a 224 KiB working window to Pi. A Tinker model
    with a smaller limit would only fail deep into an episode, so this checks
    the service's reported context length before any sampling starts.

Main callers:
    ``shellsmith episode --player tinker`` (sync, before its event loop);
    the trainer and ``shellsmith campaign`` (async).
"""

from __future__ import annotations

from typing import Any

from shellsmith.config import SUPPORTED_TINKER_BASE_MODELS
from shellsmith.policy.sampler import CONTEXT_WINDOW_TOKENS


def check_long_context(capabilities: Any, *, model_name: str, required_tokens: int = CONTEXT_WINDOW_TOKENS) -> None:
    """Fail unless Tinker lists ``model_name`` exactly once with enough context."""
    supported = getattr(capabilities, "supported_models", None)
    if not isinstance(supported, list):
        raise RuntimeError("Tinker capabilities did not contain a supported_models list.")
    limits = [getattr(entry, "max_context_length", None) for entry in supported
              if getattr(entry, "model_name", None) == model_name]
    if len(limits) != 1 or not isinstance(limits[0], int) or isinstance(limits[0], bool):
        raise RuntimeError(f"Tinker must list {model_name!r} exactly once with a context limit; found {limits}.")
    if limits[0] < required_tokens:
        raise RuntimeError(f"Tinker serves {model_name!r} with {limits[0]} tokens; Shellsmith needs {required_tokens}.")


async def require_long_context_async(service_client: Any, *, model_name: str) -> None:
    """``check_long_context`` from inside an event loop (trainer, campaign).

    Tinker's sync methods warn (and can deadlock) when called from async code.
    """
    check_long_context(await service_client.get_server_capabilities_async(), model_name=model_name)


def base_sampling_client(deps: Any, *, model_name: str) -> Any:
    """A sampling client on supported base weights, after renderer/context checks."""
    if model_name not in SUPPORTED_TINKER_BASE_MODELS:
        raise ValueError(
            f"Unsupported Tinker base model {model_name!r}; Shellsmith's fixed Qwen3.8 "
            f"renderer/audit contract supports only {SUPPORTED_TINKER_BASE_MODELS!r}."
        )
    service_client = deps.tinker.ServiceClient()
    check_long_context(service_client.get_server_capabilities(), model_name=model_name)
    return service_client.create_sampling_client(base_model=model_name)
