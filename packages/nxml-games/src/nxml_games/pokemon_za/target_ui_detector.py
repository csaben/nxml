"""Detect the in-game target-lock UI in the bottom-right of a decoded frame.

Uses normalized cross-correlation against a reference template. Designed to
tolerate the slight blur/distortion that the world model introduces vs. real
game frames.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class TargetUIDetector:
    """Template-matches the targeting popup in the bottom-right cell of a frame."""

    def __init__(
        self,
        template_path: str | Path,
        threshold: float = 0.15,
        sat_threshold: float = 68.0,
        template_right_exclude_frac: float = 0.32,
        roi_right_exclude_frac: float = 0.30,
        canny_lo: int = 40,
        canny_hi: int = 120,
        min_consecutive_hits: int = 5,
    ):
        tpl = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if tpl is None:
            raise FileNotFoundError(f"Template not found: {template_path}")
        # The reference template includes a pink target-lock circle on the right
        # which is a SEPARATE UI element and should be ignored. Crop it off.
        if template_right_exclude_frac > 0:
            keep_w = int(tpl.shape[1] * (1.0 - template_right_exclude_frac))
            tpl = tpl[:, :keep_w]
        self.template_path = str(template_path)
        self.template_color = tpl  # BGR
        self.template_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        self.th, self.tw = self.template_gray.shape
        self.threshold = threshold
        self.roi_right_exclude_frac = roi_right_exclude_frac
        self.canny_lo = canny_lo
        self.canny_hi = canny_hi
        # Mean S-channel value of the template region — false positives in the
        # game world (street, walls, sky) tend to be much less saturated.
        tpl_hsv = cv2.cvtColor(tpl, cv2.COLOR_BGR2HSV)
        self.template_mean_sat = float(tpl_hsv[:, :, 1].mean())
        self.sat_threshold = sat_threshold
        self.min_consecutive_hits = min_consecutive_hits
        self._streak: int = 0
        # Pre-compute the template's edge map (dilated, for tolerance).
        self._tpl_edges_cache: dict[tuple[int, int], np.ndarray] = {}

    def reset(self) -> None:
        """Clear streak — call on scene reseed."""
        self._streak = 0

    @property
    def streak(self) -> int:
        return self._streak

    def crop_roi(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Bottom-right cell of a 3x3 grid over the frame, with the rightmost
        slice excluded so the pink target-lock circle is never in the search
        window.
        """
        h, w = frame_bgr.shape[:2]
        y0 = (2 * h) // 3
        x0 = (2 * w) // 3
        x1 = w - int((w - x0) * self.roi_right_exclude_frac)
        return frame_bgr[y0:h, x0:x1]

    def _get_tpl_edges(self, size: tuple[int, int]) -> np.ndarray:
        """Cached: resize template, Canny, dilate."""
        if size in self._tpl_edges_cache:
            return self._tpl_edges_cache[size]
        new_w, new_h = size
        tpl_gray = cv2.resize(self.template_gray, (new_w, new_h))
        edges = cv2.Canny(tpl_gray, self.canny_lo, self.canny_hi)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
        self._tpl_edges_cache[size] = edges
        return edges

    def raw_signals(self, frame_bgr: np.ndarray) -> tuple[float, float]:
        """Pure CV pass: NCC score + mean S of the matched patch. No streak.

        Returned alone so callers that only want the raw measurements (live
        tuning, offline replay, sweep) don't perturb the streak counter.
        Out-of-range frames (template can't fit the ROI) return ``(0.0, 0.0)``.
        """
        roi = self.crop_roi(frame_bgr)
        rh, rw = roi.shape[:2]

        scale = min(rh / self.th, rw / self.tw)
        if scale <= 0:
            return 0.0, 0.0
        new_w = max(1, min(rw, int(self.tw * scale)))
        new_h = max(1, min(rh, int(self.th * scale)))
        if new_h < 4 or new_w < 4:
            return 0.0, 0.0

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_edges = cv2.Canny(roi_gray, self.canny_lo, self.canny_hi)
        roi_edges = cv2.dilate(roi_edges, np.ones((2, 2), np.uint8))
        tpl_edges = self._get_tpl_edges((new_w, new_h))

        if tpl_edges.shape[0] > roi_edges.shape[0] or tpl_edges.shape[1] > roi_edges.shape[1]:
            return 0.0, 0.0

        result = cv2.matchTemplate(roi_edges, tpl_edges, cv2.TM_CCOEFF_NORMED)
        _, score, _, max_loc = cv2.minMaxLoc(result)
        x, y = max_loc

        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        patch_sat = float(roi_hsv[y : y + new_h, x : x + new_w, 1].mean())
        return float(score), patch_sat

    def detect(self, frame_bgr: np.ndarray) -> tuple[bool, float, float]:
        """Run edge-based template matching on the bottom-right popup ROI.

        Returns:
            (detected, score, patch_sat) — score is the edge-map match score,
            patch_sat is the mean HSV S in the matched patch (informational; the
            sat gate only fires if sat_threshold > 0).
        """
        score, patch_sat = self.raw_signals(frame_bgr)
        sat_ok = self.sat_threshold <= 0 or patch_sat >= self.sat_threshold
        raw_hit = score >= self.threshold and sat_ok

        if raw_hit:
            self._streak += 1
        else:
            self._streak = 0
        detected = self._streak >= self.min_consecutive_hits

        return detected, score, patch_sat

    def set_min_consecutive_hits(self, n: int) -> None:
        self.min_consecutive_hits = max(1, int(n))
        self._streak = 0

    def debug_visualize(self, frame_bgr: np.ndarray) -> dict[str, np.ndarray]:
        """Return ROI, ROI edges, and template edges for offline inspection."""
        roi = self.crop_roi(frame_bgr)
        rh, rw = roi.shape[:2]
        scale = min(rh / self.th, rw / self.tw)
        new_w = max(1, min(rw, int(self.tw * scale)))
        new_h = max(1, min(rh, int(self.th * scale)))
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_edges = cv2.Canny(roi_gray, self.canny_lo, self.canny_hi)
        roi_edges = cv2.dilate(roi_edges, np.ones((2, 2), np.uint8))
        tpl_edges = self._get_tpl_edges((new_w, new_h))
        return {"roi": roi, "roi_edges": roi_edges, "tpl_edges": tpl_edges}

    def set_threshold(self, t: float) -> None:
        self.threshold = float(t)

    def set_sat_threshold(self, t: float) -> None:
        self.sat_threshold = float(t)
