"""Macro player.

A :class:`MacroPlayer` walks a :class:`Macro`'s frames and calls a
``poster`` callable for each one, sleeping ``dt`` seconds between frames.
The poster is whatever the caller wants to drive — typically an HTTP
``POST /action`` to nxbt-orchestrator, but it could equally be a mux
source's ``feed()`` or a test spy.

``play()`` is blocking; ``play_async()`` runs the same logic in a daemon
thread and returns immediately. Either way ``stop()`` interrupts a running
playback within roughly one ``dt``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np

from nx_macros.schema import Macro

Poster = Callable[[np.ndarray], None]


class MacroPlayer:
    def __init__(self, poster: Poster) -> None:
        self._poster = poster
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._playing_name: str | None = None
        self._playing_started_at: float = 0.0
        self._playing_total_sec: float | None = None  # None when looping

    def play(self, macro: Macro, *, loop: bool = False) -> None:
        """Blocking playback. Returns when the macro ends (or ``stop()`` is called)."""
        self._stop.clear()
        with self._lock:
            self._playing_name = macro.name
            self._playing_started_at = time.time()
            self._playing_total_sec = None if loop else _total_duration_sec(macro)
        try:
            self._run(macro, loop=loop)
        finally:
            with self._lock:
                self._playing_name = None
                self._playing_total_sec = None

    def play_async(self, macro: Macro, *, loop: bool = False) -> None:
        """Non-blocking playback. Raises if a previous playback is still running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("playback already in progress")
            self._stop.clear()
            self._playing_name = macro.name
            self._playing_started_at = time.time()
            self._playing_total_sec = None if loop else _total_duration_sec(macro)
            self._thread = threading.Thread(
                target=self._run_async,
                args=(macro,),
                kwargs={"loop": loop},
                daemon=True,
                name=f"nx-macros-player:{macro.name}",
            )
            self._thread.start()

    def _run_async(self, macro: Macro, *, loop: bool) -> None:
        try:
            self._run(macro, loop=loop)
        finally:
            with self._lock:
                self._playing_name = None
                self._playing_total_sec = None

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            t = self._thread
        if t is not None:
            t.join(timeout=2.0)
        with self._lock:
            self._thread = None
            self._playing_name = None
            self._playing_total_sec = None

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def playing_info(self) -> dict | None:
        """Return ``{name, started_at, elapsed_sec, total_sec, remaining_sec}``
        while a macro is playing, else ``None``. ``total_sec`` is ``None`` when
        the macro is looping (no fixed duration)."""
        with self._lock:
            name = self._playing_name
            if name is None or self._thread is None or not self._thread.is_alive():
                return None
            started = self._playing_started_at
            total = self._playing_total_sec
        elapsed = max(0.0, time.time() - started)
        remaining = max(0.0, total - elapsed) if total is not None else None
        return {
            "name": name,
            "started_at": started,
            "elapsed_sec": elapsed,
            "total_sec": total,
            "remaining_sec": remaining,
        }

    def _run(self, macro: Macro, *, loop: bool) -> None:
        if not macro.frames:
            return
        # Pre-stack to avoid re-allocating numpy arrays each frame.
        actions = macro.actions()
        dts = [f.dt for f in macro.frames]

        while not self._stop.is_set():
            target = time.time()
            for action, dt in zip(actions, dts, strict=True):
                if dt > 0.0:
                    target += dt
                    remaining = target - time.time()
                    if remaining > 0:
                        # Use the stop event's wait so we can interrupt promptly.
                        if self._stop.wait(remaining):
                            return
                    elif self._stop.is_set():
                        return
                try:
                    self._poster(action)
                except Exception as e:
                    print(f"[nx-macros] poster failed: {type(e).__name__}: {e}", flush=True)
            if not loop:
                return


def _total_duration_sec(macro: Macro) -> float:
    return float(sum(f.dt for f in macro.frames))
