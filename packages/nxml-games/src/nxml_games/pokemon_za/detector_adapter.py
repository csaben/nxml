"""Adapter exposing :class:`TargetUIDetector` through the nxwm Detector protocol.

Imported for its registration side-effect:

    from nxml_games.pokemon_za import detector_adapter  # noqa: F401

After import, ``nxwm.env.detectors.detector_registry["pokemon_za:target_ui"]``
returns a factory that builds the adapter from a template path + threshold
overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from nxwm.env.detectors import ParamSchema, register_detector

from .target_ui_detector import TargetUIDetector


class TargetUIAdapter:
    """Wraps :class:`TargetUIDetector`, splitting raw CV from threshold logic.

    ``signals`` runs the (cached) edge-based template match and returns
    ``{"score", "sat"}`` without touching the streak. ``decide`` re-walks any
    history under the current ``params`` — so changing a threshold updates
    the readout retroactively over the recent buffer.
    """

    name = "pokemon_za:target_ui"

    def __init__(
        self,
        *,
        template_path: str | Path,
        threshold: float = 0.15,
        sat_threshold: float = 68.0,
        min_consecutive_hits: int = 5,
    ):
        self._inner = TargetUIDetector(
            template_path=template_path,
            threshold=threshold,
            sat_threshold=sat_threshold,
            min_consecutive_hits=min_consecutive_hits,
        )

    # ------------------------------------------------------------------
    # signals/decide split
    # ------------------------------------------------------------------

    def signals(self, frame_bgr: np.ndarray) -> dict[str, float]:
        score, sat = self._inner.raw_signals(frame_bgr)
        return {"score": score, "sat": sat}

    def decide(self, history: list[dict[str, float]]) -> dict[str, Any]:
        """Replay the streak rule over ``history`` under the current params.

        Stateless w.r.t. ``self._inner._streak`` — we recompute the streak
        from scratch so the same call shape works for live (history grows by
        one each frame) and offline (history is the whole session).
        """
        threshold = self._inner.threshold
        sat_threshold = self._inner.sat_threshold
        min_hits = self._inner.min_consecutive_hits
        streak = 0
        for s in history:
            sat_ok = sat_threshold <= 0 or s.get("sat", 0.0) >= sat_threshold
            if s.get("score", 0.0) >= threshold and sat_ok:
                streak += 1
            else:
                streak = 0
        return {"detected": streak >= min_hits, "streak": streak}

    # ------------------------------------------------------------------
    # params + schema
    # ------------------------------------------------------------------

    def params(self) -> dict[str, Any]:
        return {
            "threshold": float(self._inner.threshold),
            "sat_threshold": float(self._inner.sat_threshold),
            "min_consecutive_hits": int(self._inner.min_consecutive_hits),
        }

    def update_params(self, **kwargs: Any) -> None:
        if "threshold" in kwargs:
            self._inner.set_threshold(float(kwargs["threshold"]))
        if "sat_threshold" in kwargs:
            self._inner.set_sat_threshold(float(kwargs["sat_threshold"]))
        if "min_consecutive_hits" in kwargs:
            self._inner.set_min_consecutive_hits(int(kwargs["min_consecutive_hits"]))

    def schema(self) -> dict[str, ParamSchema]:
        return {
            "threshold": ParamSchema(
                type="float", min=0.0, max=1.0, step=0.01, label="match score"
            ),
            "sat_threshold": ParamSchema(
                type="float", min=0.0, max=255.0, step=1.0, label="patch saturation"
            ),
            "min_consecutive_hits": ParamSchema(
                type="int", min=1, max=30, step=1, label="consecutive hits"
            ),
        }

    def static_meta(self) -> dict[str, Any]:
        return {
            "kind": self.name,
            "template_path": self._inner.template_path,
            "canny_lo": self._inner.canny_lo,
            "canny_hi": self._inner.canny_hi,
            "roi_right_exclude_frac": self._inner.roi_right_exclude_frac,
            "template_mean_sat": self._inner.template_mean_sat,
        }

    # ------------------------------------------------------------------
    # debug visualization + reset
    # ------------------------------------------------------------------

    def debug_image(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Side-by-side ROI / ROI-edges / template-edges as a single BGR image."""
        viz = self._inner.debug_visualize(frame_bgr)
        roi = viz["roi"]
        roi_edges = cv2.cvtColor(viz["roi_edges"], cv2.COLOR_GRAY2BGR)
        tpl_edges = cv2.cvtColor(viz["tpl_edges"], cv2.COLOR_GRAY2BGR)
        # Pad shorter images vertically so they stack cleanly.
        h = max(roi.shape[0], roi_edges.shape[0], tpl_edges.shape[0])
        def _pad(img: np.ndarray) -> np.ndarray:
            pad = h - img.shape[0]
            if pad <= 0:
                return img
            return cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return cv2.hconcat([_pad(roi), _pad(roi_edges), _pad(tpl_edges)])

    def reset(self) -> None:
        self._inner.reset()


def _factory(**kwargs: Any) -> TargetUIAdapter:
    return TargetUIAdapter(**kwargs)


register_detector(TargetUIAdapter.name, _factory)
