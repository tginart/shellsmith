"""Many tasks / many episodes: catalog, slate, campaign queue, stream, outage.

Scope invariant:
    This package schedules and aggregates independent episode runs. It calls
    ``shellsmith.episode`` (smoke / audit) but does not own the Player ↔
    GM turn loop, optimizer steps, or full training-run scripting
    (those live in ``shellsmith.train``).

Public surfaces:
    ``rollout_campaign`` — one-shot multi-task tryout queue + scoreboard.
    ``stream_campaign`` — open / wave / finalize for durable multi-wave enclosures.
    ``provider_outage`` — consecutive-fail health tracker + Sol↔Opus / terra↔Sonnet map.

One-episode runtime lives in ``shellsmith.episode``.
Shared world file Markdown grammar lives in ``shellsmith.world_file``.
"""
