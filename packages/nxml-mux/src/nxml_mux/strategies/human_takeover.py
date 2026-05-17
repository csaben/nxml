"""All-or-nothing human takeover.

If any human source is actively contributing *any* index (a button down,
a stick deflected past its deadzone), the merged action is **only** the
human's contribution — every other source is dropped for that tick. As
soon as the human stops touching everything, non-human sources take
over wholesale.

Compared to :class:`HumanPriority` (per-index merge: human claims just
the indices they're actually touching, AI fills the rest), this is the
mode you want when even a *partial* human input — e.g. holding a stick
while the AI fires buttons — feels wrong. Side effect: AI cannot
contribute anything until the human fully releases.

A human ``ActionSnapshot`` with ``mask=None`` is treated as actively
contributing every index (i.e. the human is always-on). That generally
means the source isn't telling the strategy enough — prefer real masks.
"""

from __future__ import annotations

import numpy as np
from nx_packets import ACTION_DIM

from nxml_mux.source import ActionSnapshot


class HumanTakeover:
    def __init__(self, human_source_ids: set[str] | list[str]) -> None:
        self.human_source_ids = set(human_source_ids)

    def merge(self, snapshots: list[ActionSnapshot]) -> np.ndarray:
        merged = np.zeros(ACTION_DIM, dtype=np.float32)

        human_snaps: list[ActionSnapshot] = []
        ai_snaps: list[ActionSnapshot] = []
        for snap in snapshots:
            (human_snaps if snap.source_id in self.human_source_ids else ai_snaps).append(snap)

        human_active = False
        for snap in human_snaps:
            mask = snap.mask if snap.mask is not None else np.ones(ACTION_DIM, dtype=bool)
            if mask.any():
                human_active = True
                break

        if human_active:
            for snap in human_snaps:
                mask = snap.mask if snap.mask is not None else np.ones(ACTION_DIM, dtype=bool)
                merged[mask] = snap.action[mask]
            return merged

        claimed = np.zeros(ACTION_DIM, dtype=bool)
        for snap in ai_snaps:
            mask = snap.mask if snap.mask is not None else np.ones(ACTION_DIM, dtype=bool)
            avail = mask & ~claimed
            merged[avail] = snap.action[avail]
            claimed |= avail
        return merged
