"""Field-level human-priority merge.

For each of the 26 action dimensions, the human's actively-contributed
value wins. AI/macro/policy sources fill in indices the human is *not*
actively touching, so they run non-blockingly alongside live play (the
nxml-coplay use case).

"Actively contributing" is whatever the source chose to mark in
``ActionSnapshot.mask``. ``EvdevReader`` marks buttons it sees pressed and
stick axes outside the deadzone. A source that emits ``mask=None`` is
treated as contributing every index — useful for a baseline/idle source
but obviously a bad fit for "human" if you want this strategy to make
sense.

Order of resolution (later sources fall through earlier ones):
  1. All sources whose ``source_id`` is in ``human_source_ids``: their
     active indices win and lock those slots.
  2. All other sources, in the order they were given: contributions are
     written only to slots no human has claimed; later non-human sources
     fall through earlier ones the same way.
"""

from __future__ import annotations

import numpy as np
from nx_packets import ACTION_DIM

from nxml_mux.source import ActionSnapshot


class HumanPriority:
    def __init__(self, human_source_ids: set[str] | list[str]) -> None:
        self.human_source_ids = set(human_source_ids)

    def merge(self, snapshots: list[ActionSnapshot]) -> np.ndarray:
        merged = np.zeros(ACTION_DIM, dtype=np.float32)
        claimed = np.zeros(ACTION_DIM, dtype=bool)

        for snap in snapshots:
            if snap.source_id not in self.human_source_ids:
                continue
            mask = (
                snap.mask
                if snap.mask is not None
                else np.ones(ACTION_DIM, dtype=bool)
            )
            avail = mask & ~claimed
            merged[avail] = snap.action[avail]
            claimed |= avail

        for snap in snapshots:
            if snap.source_id in self.human_source_ids:
                continue
            mask = (
                snap.mask
                if snap.mask is not None
                else np.ones(ACTION_DIM, dtype=bool)
            )
            avail = mask & ~claimed
            merged[avail] = snap.action[avail]
            claimed |= avail

        return merged
