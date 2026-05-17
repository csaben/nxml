"""ActionSource protocol + a synchronous callable adapter.

An ``ActionSource`` produces ``ActionSnapshot``s on demand. Sources are
polled by :class:`nxml_mux.mux.ControllerMux` at whatever cadence the app
drives. Each source carries an opaque ``source_id`` so strategies can
distinguish humans from policies from macros.

The optional ``mask`` field marks which of the 26 action dimensions this
source is *actively contributing*. A human-controller reader sets ``True``
for buttons it sees pressed and stick axes outside the deadzone, ``False``
elsewhere — this is what lets a "human-priority" strategy fall through to
an AI/macro source for indices the human isn't touching. Sources that
always emit a full-frame contribution can leave ``mask=None`` and the
strategy treats every index as actively contributed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from nx_packets import ACTION_DIM


@dataclass(frozen=True, slots=True)
class ActionSnapshot:
    action: np.ndarray  # (ACTION_DIM,) float32
    timestamp: float
    source_id: str
    mask: np.ndarray | None = None  # (ACTION_DIM,) bool, optional


@runtime_checkable
class ActionSource(Protocol):
    source_id: str

    def latest(self) -> ActionSnapshot | None:
        """Return the most recent snapshot, or ``None`` if not yet ready."""

    def start(self) -> None:
        """Begin producing snapshots (e.g. spin up a reader thread)."""

    def stop(self) -> None:
        """Stop producing snapshots and release resources."""


class CallableActionSource:
    """Synchronous: ``get_action`` is invoked on each ``latest()`` call.

    Use for cheap, in-process callers — pre-recorded macros that read from
    a buffer, mocks in tests, in-process policy networks small enough to
    inference-block at the mux's tick rate.

    For slow callables (any real network/GPU policy), wrap your callable in
    its own thread and have it return a cached recent value, so ``latest()``
    never blocks the mux loop.

    The callable returns ``(action, mask)`` where ``mask`` may be ``None``
    to mean "every index is actively contributed".
    """

    def __init__(
        self,
        source_id: str,
        get_action: Callable[[], tuple[np.ndarray, np.ndarray | None]],
    ) -> None:
        self.source_id = source_id
        self._get = get_action

    def latest(self) -> ActionSnapshot | None:
        action, mask = self._get()
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"action shape {action.shape} != ({ACTION_DIM},)")
        if mask is not None and mask.shape != (ACTION_DIM,):
            raise ValueError(f"mask shape {mask.shape} != ({ACTION_DIM},)")
        return ActionSnapshot(
            action=action.astype(np.float32, copy=False),
            timestamp=time.time(),
            source_id=self.source_id,
            mask=mask,
        )

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None
