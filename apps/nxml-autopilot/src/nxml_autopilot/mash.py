"""Time-bounded A-mash state machine.

A :class:`MashController` is a small state object queried once per action-loop
tick: it returns an A-press or neutral 26-dim vector based on
``frames_per_phase`` toggling until ``duration_sec`` elapses, then deactivates
and yields ``None`` so the action loop resumes normal AI/human merging.

State on disk is just the :class:`TriggerSpec`'s ``mash_duration_sec`` and
``mash_frames_per_phase`` — no macro file. The controller's ``is_active``
mirrors :class:`nx_macros.MacroPlayer`'s ``is_playing`` so callers can treat
both as "scripted overrides" with one guard.
"""

from __future__ import annotations

import time

import numpy as np
from nx_packets import ACTION_DIM, BUTTON_INDEX

DEFAULT_MASH_DURATION_SEC = 40.0
DEFAULT_MASH_FRAMES_PER_PHASE = 3


def _a_press_action() -> np.ndarray:
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    a[BUTTON_INDEX["A"]] = 1.0
    return a


def _neutral_action() -> np.ndarray:
    return np.zeros(ACTION_DIM, dtype=np.float32)


class MashController:
    """Time-bounded alternating-A state machine.

    Drive it from the action loop: call ``next_action()`` once per tick
    and post the returned vector when it's non-None. Inactive ticks
    return ``None`` so the caller can fall through to its normal
    mux/AI path.
    """

    def __init__(
        self,
        *,
        duration_sec: float = DEFAULT_MASH_DURATION_SEC,
        frames_per_phase: int = DEFAULT_MASH_FRAMES_PER_PHASE,
    ) -> None:
        self.duration_sec = float(duration_sec)
        self.frames_per_phase = int(frames_per_phase)
        self._a_press = _a_press_action()
        self._neutral = _neutral_action()
        self._active = False
        self._started_at: float = 0.0
        self._counter: int = 0
        # Per-fire overrides — set on start() so a trigger can supply
        # its own duration / phase length without mutating the
        # controller's defaults.
        self._fire_duration: float = self.duration_sec
        self._fire_frames_per_phase: int = self.frames_per_phase

    def start(
        self,
        *,
        duration_sec: float | None = None,
        frames_per_phase: int | None = None,
        source: str | None = None,
    ) -> None:
        """Begin a fresh mash window. Restarts cleanly if already active."""
        self._fire_duration = (
            float(duration_sec) if duration_sec is not None else self.duration_sec
        )
        self._fire_frames_per_phase = (
            int(frames_per_phase)
            if frames_per_phase is not None
            else self.frames_per_phase
        )
        self._started_at = time.time()
        self._counter = 0
        self._active = True
        src = f" from {source!r}" if source else ""
        print(
            f"[mash] start{src} → {self._fire_duration:.1f}s "
            f"(frames_per_phase={self._fire_frames_per_phase} at action-tick rate)",
            flush=True,
        )

    def stop(self) -> None:
        """Force-cancel an in-flight mash."""
        if self._active:
            print("[mash] stopped early", flush=True)
        self._active = False

    @property
    def is_active(self) -> bool:
        """True while a mash window is open. Auto-deactivates on deadline."""
        if self._active and (time.time() - self._started_at) >= self._fire_duration:
            self._active = False
            print(
                f"[mash] {self._fire_duration:.1f}s elapsed → release",
                flush=True,
            )
        return self._active

    def next_action(self) -> np.ndarray | None:
        """Return the action vector for this tick, or ``None`` if inactive.

        Cadence is inferred from the call rate: at 30 Hz ticks with
        ``frames_per_phase=3`` the mash runs at 5 Hz.
        """
        if not self.is_active:
            return None
        press = (self._counter // self._fire_frames_per_phase) % 2 == 0
        self._counter += 1
        return self._a_press if press else self._neutral
