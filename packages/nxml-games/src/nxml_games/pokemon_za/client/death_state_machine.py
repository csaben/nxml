"""Death state machine: PLAYING ↔ MASHING_A.

Two states:

  - ``PLAYING``: the policy is driving the Switch. After
    ``play_debounce_sec`` of being in this state, *and* an end-screen or
    connection-lost detection fires, transition to ``MASHING_A``.
  - ``MASHING_A``: ignore the policy entirely. Toggle a script that holds
    A for ``mash_frames_per_phase`` frames, releases for the same number,
    and loops for ``mash_duration_sec``. When the timer expires,
    transition back to ``PLAYING`` and start a fresh debounce window.

The state machine produces a ``(action, mask, is_active)`` triple per tick
that callers feed into ``ControllerMux``. When inactive, the override
contributes nothing and the policy's action passes through.
"""

from __future__ import annotations

import time
from enum import Enum

import numpy as np
from nx_packets import ACTION_DIM, BUTTON_INDEX

DEFAULT_PLAY_DEBOUNCE_SEC = 60.0
DEFAULT_MASH_DURATION_SEC = 40.0
DEFAULT_MASH_FRAMES_PER_PHASE = 3


class DeathState(Enum):
    PLAYING = "playing"
    MASHING_A = "mashing_a"


def _press_a_action() -> np.ndarray:
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    a[BUTTON_INDEX["A"]] = 1.0
    return a


def _neutral_action() -> np.ndarray:
    return np.zeros(ACTION_DIM, dtype=np.float32)


class DeathStateMachine:
    """Stateful detector → mash-A scripted overlay.

    Single-threaded: ``tick(frame_bgr)`` is called once per consumer tick
    and updates internal state. ``current_action()`` returns
    ``(action, mask, active)`` reflecting the most recent tick.
    """

    def __init__(
        self,
        *,
        end_screen_detector=None,
        connection_lost_detector=None,
        play_debounce_sec: float = DEFAULT_PLAY_DEBOUNCE_SEC,
        mash_duration_sec: float = DEFAULT_MASH_DURATION_SEC,
        mash_frames_per_phase: int = DEFAULT_MASH_FRAMES_PER_PHASE,
        verbose: bool = True,
    ) -> None:
        self._end_screen = end_screen_detector
        self._conn_lost = connection_lost_detector
        self.play_debounce_sec = play_debounce_sec
        self.mash_duration_sec = mash_duration_sec
        self.mash_frames_per_phase = mash_frames_per_phase
        self.verbose = verbose

        self.state = DeathState.PLAYING
        self._state_entered_at = time.time()
        self._mash_frame_counter = 0
        self._last_action = _neutral_action()
        self._last_active = False

    def tick(self, frame_bgr: np.ndarray | None) -> None:
        """Advance the state machine using the latest captured frame.

        Pass ``None`` to skip detection (e.g. when no frame has arrived
        this tick); the state machine still advances its mash timer.
        """
        now = time.time()
        elapsed = now - self._state_entered_at

        is_end = bool(
            frame_bgr is not None
            and self._end_screen is not None
            and self._end_screen.detect(frame_bgr)
        )
        is_conn_lost = bool(
            frame_bgr is not None
            and self._conn_lost is not None
            and self._conn_lost.detect(frame_bgr)
        )

        if self.state is DeathState.PLAYING:
            if elapsed < self.play_debounce_sec:
                if (is_end or is_conn_lost) and self.verbose:
                    kind = "end-screen" if is_end else "conn-lost"
                    print(
                        f"[{kind}] ignored during play debounce "
                        f"({self.play_debounce_sec - elapsed:.1f}s left)"
                    )
                self._last_action = _neutral_action()
                self._last_active = False
                return
            if is_end or is_conn_lost:
                self.state = DeathState.MASHING_A
                self._state_entered_at = now
                self._mash_frame_counter = 0
                if self.verbose:
                    kind = "end-screen" if is_end else "conn-lost"
                    print(
                        f"[{kind}] detected after debounce → MASHING_A "
                        f"for {self.mash_duration_sec}s"
                    )
                self._last_action = _press_a_action()
                self._last_active = True
                return
            self._last_action = _neutral_action()
            self._last_active = False
            return

        # MASHING_A state
        if elapsed >= self.mash_duration_sec:
            self.state = DeathState.PLAYING
            self._state_entered_at = now
            if self.verbose:
                print("[mash] timed mash done → PLAYING")
            self._last_action = _neutral_action()
            self._last_active = False
            return

        self._mash_frame_counter += 1
        press = (self._mash_frame_counter // self.mash_frames_per_phase) % 2 == 0
        self._last_action = _press_a_action() if press else _neutral_action()
        self._last_active = True

    def current(self) -> tuple[np.ndarray, np.ndarray, bool]:
        """Return ``(action, mask, active)`` for the most recent ``tick()``.

        When ``active`` is False the override should not contribute (mask
        is all-False). When ``active`` is True the override claims every
        index (mask is all-True), so any merging strategy treating this
        as priority replaces the policy's action wholesale.
        """
        if not self._last_active:
            return _neutral_action(), np.zeros(ACTION_DIM, dtype=bool), False
        mask = np.ones(ACTION_DIM, dtype=bool)
        return self._last_action.copy(), mask, True
