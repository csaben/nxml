"""Episode writer that flushes a buffer of synced frames to a single ``.npz``.

Layout per episode:

  ``{name}.npz``::
      frames:     (T, H, W, C) uint8     # BGR, native cv2 order
      actions:    (T, A)        float32  # canonical 26-dim vector
      timestamps: (T,)          float64  # wall-clock seconds, monotonic-ish

  ``{name}.manifest.json``::
      schema_version, action_spec, action_dim, frame_count, fps_estimate, created_at_utc
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from nx_packets import ACTION_DIM

from nxml_capture.synchronizer import SyncedFrame

SCHEMA_VERSION = 1
ACTION_SPEC_NAME = "switch_packets.v1"


class NpzEpisodeWriter:
    """Buffer ``SyncedFrame`` rows in memory; flush a single ``.npz`` on close.

    A single episode is small enough that streaming to ``.npz`` isn't worth
    the complexity.
    """

    def __init__(self, output_dir: str | Path, *, episode_name: str | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_name = episode_name or _default_episode_name()
        self._frames: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._timestamps: list[float] = []
        self._closed = False

    def append(self, synced: SyncedFrame) -> None:
        if self._closed:
            raise RuntimeError("writer already closed")
        if synced.action.shape != (ACTION_DIM,):
            raise ValueError(
                f"action shape {synced.action.shape} != ({ACTION_DIM},)"
            )
        self._frames.append(synced.frame)
        self._actions.append(synced.action.astype(np.float32, copy=False))
        self._timestamps.append(synced.timestamp)

    def __len__(self) -> int:
        return len(self._frames)

    def close(self) -> Path | None:
        if self._closed:
            return None
        self._closed = True
        if not self._frames:
            return None
        npz_path = self.output_dir / f"{self.episode_name}.npz"
        manifest_path = self.output_dir / f"{self.episode_name}.manifest.json"
        frames = np.stack(self._frames, axis=0)
        actions = np.stack(self._actions, axis=0)
        timestamps = np.asarray(self._timestamps, dtype=np.float64)
        np.savez_compressed(
            npz_path,
            frames=frames,
            actions=actions,
            timestamps=timestamps,
        )
        manifest_path.write_text(
            json.dumps(
                _manifest(frames, actions, timestamps),
                indent=2,
            )
        )
        return npz_path

    def __enter__(self) -> NpzEpisodeWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _default_episode_name() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")


def _manifest(
    frames: np.ndarray,
    actions: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, object]:
    if timestamps.size > 1:
        duration = float(timestamps[-1] - timestamps[0])
        fps = float(timestamps.size - 1) / duration if duration > 0 else 0.0
    else:
        fps = 0.0
    return {
        "schema_version": SCHEMA_VERSION,
        "action_spec": ACTION_SPEC_NAME,
        "action_dim": int(actions.shape[1]),
        "frame_count": int(frames.shape[0]),
        "frame_shape": list(frames.shape[1:]),
        "fps_estimate": fps,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
    }
