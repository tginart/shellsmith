"""Optional native MLX training backend.

Importing this package never imports MLX, downloads a model, or probes Metal.
Numerical modules are loaded only after execution-host readiness succeeds.
"""

from __future__ import annotations


async def create_session(config, *, checkpoint=None, checkpoint_root):
    """Construct the optional resident session after explicit backend selection."""
    from .backend import create_session as create

    return await create(config, checkpoint=checkpoint, checkpoint_root=checkpoint_root)

