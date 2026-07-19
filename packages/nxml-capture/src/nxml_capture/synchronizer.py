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
from typing import Literal, Protocol

import numpy as np
from nx_packets import ACTION_DIM, neutral_action

from nxml_capture.controller_subscribe import ControllerSnapshot
from nxml_capture.source import CaptureSource, Frame


class ControllerStateSource(Protocol):
    @property
    def is_connected(self) -> bool: ...

    def latest(self) -> ControllerSnapshot | None: ...

    def latest_at(
        self, *, timestamp: float, monotonic_ns: int | None = None
    ) -> ControllerSnapshot | None: ...

    def wait_for_first(self, timeout: float = 5.0) -> bool: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SyncedFrame:
    timestamp: float
    frame: np.ndarray  # (H, W, C) uint8, BGR
    action: np.ndarray  # (26,) float32; compatibility alias for applied_action
    action_age: float  # seconds between frame ts and action ts (>= 0)
    frame_monotonic_ns: int | None = None
    action_timestamp: float | None = None
    action_monotonic_ns: int | None = None
    human_action: np.ndarray | None = None
    human_mask: np.ndarray | None = None
    policy_action: np.ndarray | None = None
    ownership: np.ndarray | None = None
    controller_id: str | None = None
    active_driver: str | None = None
    policy_id: str | None = None
    policy_revision: str | None = None
    policy_digest: str | None = None
    human_monotonic_ns: int | None = None
    policy_monotonic_ns: int | None = None
    policy_observation_monotonic_ns: int | None = None
    muted_policy_action: np.ndarray | None = None
    mute_mask: np.ndarray | None = None
    mute_mask_version: str | None = None
    ownership_source: str | None = None
    mode: str | None = None
    takeover: bool = False
    takeover_reason: str | None = None
    takeover_release_remaining_ns: int = 0
    proposal_valid: bool = False
    proposal_fresh: bool = False
    proposal_sequence: int | None = None
    proposal_age_ns: int | None = None
    gap_state: str = "none"
    gap_reason: str | None = None
    gap_duration_ns: int = 0
    boundary_sequence: int | None = None
    boundary_acknowledged: bool = False
    valid: bool = True
    invalid_reasons: tuple[str, ...] = ()

    @property
    def applied_action(self) -> np.ndarray:
        return self.action


class Synchronizer:
    def __init__(
        self,
        source: CaptureSource,
        controller: ControllerStateSource,
        *,
        max_action_age: float = 0.5,
        initial_timeout: float = 5.0,
        driver: Literal["human", "unknown"] = "unknown",
    ) -> None:
        self.source = source
        self.controller = controller
        self.max_action_age = max_action_age
        self.initial_timeout = initial_timeout
        self.driver = driver
        self.invalid_samples = 0

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
        latest_at = getattr(self.controller, "latest_at", None)
        snapshot = (
            latest_at(timestamp=frame.timestamp, monotonic_ns=frame.monotonic_ns)
            if callable(latest_at)
            else self.controller.latest()
        )
        if snapshot is None:
            return self._invalid(frame, "no_prior_controller_sample")
        if frame.monotonic_ns is not None and snapshot.monotonic_ns is not None:
            age = (frame.monotonic_ns - snapshot.monotonic_ns) / 1e9
        else:
            age = frame.timestamp - snapshot.timestamp
        if age < 0:
            return self._invalid(frame, "future_controller_sample", snapshot=snapshot, age=age)
        if not self.controller.is_connected:
            return self._invalid(frame, "controller_disconnected", snapshot=snapshot, age=age)
        if age > self.max_action_age:
            return self._invalid(frame, "stale_controller_sample", snapshot=snapshot, age=age)
        if snapshot.action.shape != (ACTION_DIM,):
            return self._invalid(frame, "wrong_action_spec", snapshot=snapshot, age=age)
        if snapshot.action_source == "inference":
            return self._invalid(frame, "inference_provenance", snapshot=snapshot, age=age)
        if self.driver != "human":
            return self._invalid(frame, "unknown_provenance", snapshot=snapshot, age=age)

        human_mask = np.ones(ACTION_DIM, dtype=bool)
        return SyncedFrame(
            timestamp=frame.timestamp,
            frame=frame.image,
            action=snapshot.action,
            action_age=age,
            frame_monotonic_ns=frame.monotonic_ns,
            action_timestamp=snapshot.timestamp,
            action_monotonic_ns=snapshot.monotonic_ns,
            human_action=snapshot.action.copy(),
            human_mask=human_mask,
            policy_action=neutral_action(),
            ownership=np.ones(ACTION_DIM, dtype=np.uint8),
            controller_id="nxbt-orchestrator:switch_packets.v1",
            active_driver="human",
        )

    def _invalid(
        self,
        frame: Frame,
        reason: str,
        *,
        snapshot: object | None = None,
        age: float = 0.0,
    ) -> SyncedFrame:
        self.invalid_samples += 1
        action = neutral_action()
        return SyncedFrame(
            timestamp=frame.timestamp,
            frame=frame.image,
            action=action.copy(),
            action_age=age,
            frame_monotonic_ns=frame.monotonic_ns,
            action_timestamp=getattr(snapshot, "timestamp", None),
            action_monotonic_ns=getattr(snapshot, "monotonic_ns", None),
            human_action=neutral_action(),
            human_mask=np.zeros(ACTION_DIM, dtype=bool),
            policy_action=neutral_action(),
            ownership=np.zeros(ACTION_DIM, dtype=np.uint8),
            controller_id="nxbt-orchestrator:switch_packets.v1",
            active_driver="none",
            valid=False,
            invalid_reasons=(reason,),
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
