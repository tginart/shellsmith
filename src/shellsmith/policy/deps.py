"""Load the optional Tinker stack (``pip install -e ".[tinker]"``) as one ``deps`` object.

Why this exists:
    The episode, campaign and credit code is stdlib-only. Everything that
    needs Tinker receives this namespace instead of importing it, so tests can
    pass fakes and the core installs without Tinker.

Main callers:
    ``policy.server`` setup in ``shellsmith episode --player tinker`` and the
    trainer (M4); ``credit`` and ``train.durable_capture`` read it.
"""

from __future__ import annotations

from types import SimpleNamespace


def load_tinker_deps() -> SimpleNamespace:
    """Import Tinker, torch and the Cookbook renderers, or fail with the install command."""
    try:
        import tinker
        from tinker import types
        from tinker.types.tensor_data import TensorData
        import torch
        from tinker_cookbook import renderers
        from tinker_cookbook.renderers import Message, TextPart, ThinkingPart, ToolCall
        from tinker_cookbook.tokenizer_utils import get_tokenizer
    except ImportError as error:
        raise RuntimeError(
            'Tinker dependencies are missing. Install them with pip install -e ".[tinker]" '
            f"(tinker-cookbook >= 0.5.7 for the Qwen3.8 renderers). Import error: {error}"
        ) from error
    return SimpleNamespace(
        tinker=tinker, types=types, TensorData=TensorData, torch=torch, renderers=renderers,
        Message=Message, TextPart=TextPart, ThinkingPart=ThinkingPart, ToolCall=ToolCall,
        get_tokenizer=get_tokenizer,
    )
