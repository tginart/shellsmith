# Historical Shellsmith hero slates

The `mlx_qualification_train_v2.json` and `mlx_qualification_eval_v2.json` slates
pin the current 177-world catalog and their current selected world bytes.
The live qualification example TOMLs use these successors.
`mlx_qualification_evidence_v2.json` records their **not-executed** plan;
historical v1 execution evidence does not qualify the new inputs.

All v1 JSON remains unchanged. The pinned v1 qualification slates deliberately
fail current-manifest checks after catalog promotion. Reproduce them from their
historical source revision; do not replace their pins to make them load now.
The older `first_hero_*` slates retain their original selections and models.

The older `first_hero_*` JSON files preserve the original monorepo selections.
They are selection data; the original conductor is **not shipped** in Shellsmith.
The standalone trainer now consumes compatible slates through
`shellsmith train --config ...`; see the root README for supported commands.
The original conductor's execution details below remain historical context.

`first_hero_v1.json` is the immutable selection, simulator assignment, and
explicit 111-world execution order for the first paid run. `shuffle_seed`
records how that order was originally sampled; the trainer reads
`world_order` directly and does not reshuffle it. The loader fails unless the
order is an exact permutation of the selected worlds and the simulator mix is:

- 56 Claude Sonnet Low
- 33 Codex Terra Low
- 22 Codex Sol Low

`first_hero_baby_v1.json` is the same contract sliced to the first eight worlds
of that order (4 Sonnet / 2 Terra / 2 Sol). It uses `catalog.include` so the
loader does not expand the full draft pile. With `groups_per_update: 2` and
`--steps 2`, the ceiling is two CISPO updates (four trainable groups), with a
few spare worlds if some skip.

Each slate is a fixed world queue with a GM assignment per world. The legacy
conductor that ran these slates (with their checkpoint cadence and replay-based
resume) was not ported. `shellsmith train` (milestone M4)
consumes slates through `campaign/hero_slate.py` and the
ported per-world adaptive policy in `train/stream_policy.py`: two trainable
groups per CISPO update; a world that cannot supply four accepted rollouts after
its bounded top-ups is skipped and the next world fills the slot.
