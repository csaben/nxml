"""Thread-safe macro recorder.

The producer (e.g. nxml-coplay's action loop) calls :meth:`MacroRecorder.append`
once per tick with the action that was just dispatched and the timestamp the
tick started. ``stop()`` returns the in-memory :class:`Macro`; persisting it
is the caller's choice (see :class:`nx_macros.MacroStore`).
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from nx_macros.schema import Macro, MacroFrame


class MacroRecorder:
    def __init__(self, *, tick_hz: float) -> None:
        if tick_hz <= 0:
            raise ValueError(f"tick_hz must be > 0, got {tick_hz}")
        self._tick_hz = float(tick_hz)
        self._lock = threading.Lock()
        self._frames: list[MacroFrame] = []
        self._name: str | None = None
        self._metadata: dict[str, Any] = {}
        self._last_t: float | None = None

    def start(self, name: str, *, metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            if self._name is not None:
                raise RuntimeError(f"recording already active: {self._name!r}")
            self._name = name
            self._metadata = dict(metadata or {})
            self._frames = []
            self._last_t = None

    def append(self, action: np.ndarray, t: float) -> None:
        """Append a frame. ``t`` is wall-clock seconds (e.g. ``time.time()``).

        ``dt`` is computed as ``t - last_t`` (clamped to ``>= 0`` to defend
        against clock weirdness). The first frame's ``dt`` is ``0.0``.
        """
        with self._lock:
            if self._name is None:
                return
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
            dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
            self._frames.append(MacroFrame(action=arr.tolist(), dt=dt))
            self._last_t = t

    def stop(self) -> Macro:
        with self._lock:
            if self._name is None:
                raise RuntimeError("recording not active")
            macro = Macro(
                name=self._name,
                tick_hz=self._tick_hz,
                frames=self._frames,
                metadata=self._metadata,
            )
            self._name = None
            self._metadata = {}
            self._frames = []
            self._last_t = None
            return macro

    def cancel(self) -> None:
        """Discard the in-progress recording without producing a macro."""
        with self._lock:
            self._name = None
            self._metadata = {}
            self._frames = []
            self._last_t = None

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._name is not None

    @property
    def frame_count(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def name(self) -> str | None:
        with self._lock:
            return self._name
