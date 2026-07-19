"""Toggle-able recording controller for nxml-autopilot.

Wraps a :class:`VideoParquetEpisodeWriter` so the action loop can call
``append()`` every tick without caring whether recording is currently
active. ``start()`` / ``stop()`` are safe to call from any thread (the web
UI's uvicorn handler thread, in practice) and use a single lock to keep
the writer state consistent.

Path policy:

  - Evdev mode: runner calls ``start()`` once with ``record_dir`` → one
    episode per session.
  - Web mode: runner starts inactive; each button press calls ``start()``
    with a fresh ``<record_dir>/<timestamp>/`` subdir.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from nxml_capture import SyncedFrame, VideoParquetEpisodeWriter


class RecordingController:
    def __init__(
        self,
        *,
        fps: float,
        codec: str = "ffv1",
        game: str | None = None,
        config: dict[str, object] | None = None,
        lineage: dict[str, object] | None = None,
    ) -> None:
        self._fps = fps
        self._codec = codec
        self._game = game
        self._config = dict(config or {})
        self._lineage = dict(lineage or {})
        self._lock = threading.Lock()
        self._writer: VideoParquetEpisodeWriter | None = None
        self._frames = 0
        self._current_path: Path | None = None
        self._last_finalized: Path | None = None

    def start(self, target: Path) -> Path:
        with self._lock:
            if self._writer is not None:
                raise RuntimeError("recording already active")
            target.mkdir(parents=True, exist_ok=True)
            self._writer = VideoParquetEpisodeWriter(
                target,
                codec=self._codec,
                fps=self._fps,
                game=self._game,
                config=self._config,
                lineage=self._lineage,
            )
            self._frames = 0
            self._current_path = target
            return target

    def stop(self) -> Path | None:
        with self._lock:
            if self._writer is None:
                return None
            out = self._writer.close()
            self._writer = None
            self._current_path = None
            self._last_finalized = out
            return out

    def append(self, frame: SyncedFrame) -> None:
        with self._lock:
            if self._writer is None:
                return
            self._writer.append(frame)
            self._frames += 1

    def append_event(
        self,
        kind: str,
        *,
        timestamp: float,
        monotonic_ns: int | None = None,
        source: str = "autopilot",
        payload: dict[str, object] | None = None,
    ) -> None:
        with self._lock:
            if self._writer is not None:
                self._writer.append_event(
                    kind,
                    timestamp=timestamp,
                    monotonic_ns=monotonic_ns,
                    source=source,
                    payload=payload,
                )

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._writer is not None

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": self._writer is not None,
                "frames": self._frames,
                "path": str(self._current_path) if self._current_path else None,
                "last_finalized": str(self._last_finalized) if self._last_finalized else None,
            }

    def close(self) -> Path | None:
        return self.stop()


def fresh_episode_path(root: Path) -> Path:
    return root / time.strftime("%Y%m%dT%H%M%S")
