"""One Player ↔ GM episode.

Scope invariant:
    Everything in this package owns a single rollout directory: boot → tool
    turns → final judge → optional acceptance audit. It does not schedule many
    tasks, build hero slates, or hand off to Tinker.

Cross-episode orchestration lives in ``shellsmith.campaign``.
Shared world file Markdown grammar lives in ``shellsmith.world_file``.
"""
