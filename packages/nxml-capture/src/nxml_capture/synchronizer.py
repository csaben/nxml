"""Frame ↔ action time alignment.

The orchestrator broadcasts controller state at its update rate (default 120
Hz) and the capture device produces frames at camera FPS (typically 30-60).
For recording we want one ``(frame, action)`` pair per frame: take each
frame as it arrives, then attach the most recent action snapshot. The action
snapshot is the truth at the moment of capture — this is what the human
hand was doing when the pixels were emitted.

If no action snapshot exists yet, ``Synchronizer.frames`` waits for one
(bounded by ``initial_timeout``) before emitting.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from nxml_capture.controller_subscribe import ControllerSubscription
from nxml_capture.source import CaptureSource, Frame


@dataclass(frozen=True, slots=True)
class SyncedFrame:
    timestamp: float
    frame: np.ndarray  # (H, W, C) uint8, BGR
    action: np.ndarray  # (26,) float32
    action_age: float  # seconds between frame ts and action ts (>= 0)


class Synchronizer:
    def __init__(
        self,
        source: CaptureSource,
        controller: ControllerSubscription,
        *,
        max_action_age: float = 0.5,
        initial_timeout: float = 5.0,
    ) -> None:
        self.source = source
        self.controller = controller
        self.max_action_age = max_action_age
        self.initial_timeout = initial_timeout

    def frames(self) -> Iterator[SyncedFrame]:
        if not self.controller.wait_for_first(timeout=self.initial_timeout):
            raise TimeoutError(
                f"no controller snapshot within {self.initial_timeout}s — "
                "is nxbt-orchestrator running and connected?"
            )

        for frame in self.source.frames():
            synced = self._pair(frame)
            if synced is not None:
                yield synced

    def _pair(self, frame: Frame) -> SyncedFrame | None:
        snapshot = self.controller.latest()
        if snapshot is None:
            return None
        age = max(0.0, frame.timestamp - snapshot.timestamp)
        if age > self.max_action_age:
            # State stream lagging hard; skip rather than emit stale data.
            return None
        return SyncedFrame(
            timestamp=frame.timestamp,
            frame=frame.image,
            action=snapshot.action,
            action_age=age,
        )

    def latest(self) -> SyncedFrame | None:
        frame = self.source.latest()
        if frame is None:
            return None
        return self._pair(frame)

    def __enter__(self) -> Synchronizer:
        self.source.start()
        self.controller.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.source.stop()
        self.controller.stop()


def now() -> float:
    """Wall-clock seconds, matching ``Frame.timestamp`` and ``ControllerSnapshot.timestamp``."""
    return time.time()
