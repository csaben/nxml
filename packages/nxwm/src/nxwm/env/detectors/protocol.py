"""Detector protocol for per-frame reward-target signals.

A :class:`Detector` is split into two stages so the same implementation can
serve three modes without code duplication:

  - **Live tuning** — call ``signals`` on each new frame, append to a ring
    buffer, call ``decide`` to get the current detected/streak state.
  - **Offline replay** — load a recorded ``signals`` track from a session
    file, call ``decide`` end-to-end with whatever ``params`` the user is
    sweeping. No CV.
  - **Variant comparison** — re-run ``signals`` over recorded ``frames_jpeg``
    with a different detector, then ``decide``.

``signals`` is pure (no streak state); ``decide`` is the only place where
thresholds + temporal smoothing live, and it operates on a list rather than
a single frame so re-evaluation over a session is a single call.

``params`` are the *threshold-like* knobs that can be tuned without re-running
CV (template-match threshold, sat threshold, min_consecutive_hits, …).
*Computation-changing* config (template path, Canny edges, ROI crop) is fixed
at construction and persisted via ``static_meta`` so a recorded session can be
faithfully reproduced.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict

import numpy as np


class ParamSchema(TypedDict, total=False):
    """One slider's worth of metadata for the UI."""

    type: str  # "float" | "int"
    min: float
    max: float
    step: float
    label: str  # human-readable slider title


class DetectorState(TypedDict, total=False):
    """What ``decide`` returns for a given history end-state."""

    detected: bool
    streak: int


class Detector(Protocol):
    """Protocol every detector implementation must satisfy.

    Implementations should be cheap to construct and idempotent to ``reset``.
    The split between ``signals`` (raw, pure) and ``decide`` (thresholds,
    streak) is load-bearing for offline replay — keep them honest.
    """

    name: str
    """Stable string id, e.g. ``pokemon_za:target_ui``. Persisted in sessions."""

    def signals(self, frame_bgr: np.ndarray) -> dict[str, float]:
        """Run CV on a single BGR uint8 frame; return raw measurements.

        No state is mutated. Output keys must be stable across calls so a
        session's recorded signals form a regular structured array.
        """
        ...

    def decide(self, history: list[dict[str, float]]) -> DetectorState:
        """Apply current ``params`` to a list of raw signals, end-to-end.

        ``history[-1]`` is the most recent frame. Returns the detector state
        *after* observing the full history. Stateless w.r.t. the detector
        instance — can be called over any prefix without disturbing
        live-tuning's running streak.
        """
        ...

    def params(self) -> dict[str, Any]:
        """Current threshold-like parameters."""
        ...

    def update_params(self, **kwargs: Any) -> None:
        """Patch params in place. Caller resets the live streak separately."""
        ...

    def schema(self) -> dict[str, ParamSchema]:
        """Per-param metadata for the UI: type, min/max/step, label."""
        ...

    def static_meta(self) -> dict[str, Any]:
        """Computation-fixing config (template path, Canny lo/hi, ROI crop, …).

        Recorded into session files so playback knows which detector produced
        the signals. Not editable through ``update_params``.
        """
        ...

    def debug_image(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Optional: return a single BGR uint8 image visualizing the detector's
        view of this frame (e.g. the cropped ROI with edges overlaid). The UI
        renders it next to the readout. Return ``None`` to opt out.
        """
        ...

    def reset(self) -> None:
        """Drop any internal counters. Called on reseed and on threshold edits
        so a stale streak doesn't leak across changes.
        """
        ...
