"""HTTP-fed gamepad reader.

A push-driven :class:`ActionSource` for a browser-side gamepad. Whoever
serves the page calls :meth:`WebGamepadReader.push_action` whenever a new
26-dim vector arrives from the page; ``latest()`` returns the most recent
snapshot with an "active" mask derived the same way as
:class:`EvdevReader` (sticks past the deadzone, buttons ≥ 0.5).

Two safeguards the evdev path doesn't need:

  - **Stick deadzone**. Browser ``Gamepad`` axes report controller drift
    as tiny non-zero floats. Without a deadzone, ``HumanTakeover`` would
    treat resting drift as "actively deflected" and never let the AI
    drive.
  - **Staleness timeout**. If the page goes silent (tab blurred → ``rAF``
    throttled, network blip, browser closed) the cached action is no
    longer trustworthy. After ``stale_after_s`` of no updates,
    ``latest()`` returns ``None`` so strategies treat the human as
    absent — exactly what we want when the user has stopped driving.

This deliberately knows nothing about HTTP — the transport (FastAPI,
WebSocket, raw sockets) lives in the host app. That keeps nxml-mux free
of web-server deps.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from nx_packets import ACTION_DIM, BUTTON_DIMS, BUTTON_RANGE, STICK_RANGE

from nxml_mux.source import ActionSnapshot


class WebGamepadReader:
    def __init__(
        self,
        *,
        source_id: str = "web:gamepad",
        stick_deadzone: float = 0.15,
        stale_after_s: float = 0.3,
    ) -> None:
        self.source_id = source_id
        self._stick_deadzone = stick_deadzone
        self._stale_after_s = stale_after_s
        self._action = np.zeros(ACTION_DIM, dtype=np.float32)
        # Edge-latch for buttons. Any button observed pressed in any
        # ``push_action`` since the last ``latest()`` call is OR-merged
        # into the snapshot returned by the next ``latest()``, then
        # cleared. Without this, a brief tap (browser sees A=1 then A=0
        # within 33ms) gets clobbered before the consumer's next read
        # and the press is lost. Sticks aren't latched — they're
        # continuous and the most-recent sample is what the consumer
        # wants.
        self._buttons_latched = np.zeros(BUTTON_DIMS, dtype=bool)
        self._lock = threading.Lock()
        self._latest_ts = 0.0

    def start(self) -> None:
        return None

    def stop(self) -> None:
        with self._lock:
            self._action.fill(0.0)
            self._buttons_latched.fill(False)
            self._latest_ts = 0.0

    def push_action(self, action: np.ndarray) -> None:
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"action shape {action.shape} != ({ACTION_DIM},)")
        cleaned = action.astype(np.float32, copy=True)
        sticks = cleaned[STICK_RANGE]
        sticks[np.abs(sticks) < self._stick_deadzone] = 0.0
        cleaned[STICK_RANGE] = sticks
        with self._lock:
            self._buttons_latched |= cleaned[BUTTON_RANGE] >= 0.5
            np.copyto(self._action, cleaned)
            self._latest_ts = time.time()

    def latest(self) -> ActionSnapshot | None:
        now = time.time()
        with self._lock:
            if self._latest_ts == 0.0:
                return None
            if now - self._latest_ts > self._stale_after_s:
                return None
            action = self._action.copy()
            if self._buttons_latched.any():
                action[BUTTON_RANGE] = np.maximum(
                    action[BUTTON_RANGE],
                    self._buttons_latched.astype(np.float32),
                )
                self._buttons_latched.fill(False)
            ts = self._latest_ts
        mask = np.zeros(ACTION_DIM, dtype=bool)
        mask[STICK_RANGE] = action[STICK_RANGE] != 0.0
        mask[BUTTON_RANGE] = action[BUTTON_RANGE] >= 0.5
        return ActionSnapshot(action=action, timestamp=ts, source_id=self.source_id, mask=mask)
