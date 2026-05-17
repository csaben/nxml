"""End-screen detector.

Mean-pixel-difference template match against a small reference image of
the post-battle end screen. Designed to be cheap (8x4 grayscale resize
followed by a single ``np.abs`` subtract), so we can run it every frame.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEFAULT_DETECT_SIZE = (64, 32)
DEFAULT_SIMILARITY_THRESHOLD = 0.85


class EndScreenDetector:
    """Mean-abs-diff template match for the ZA end-screen overlay."""

    def __init__(
        self,
        template_path: str | Path,
        *,
        detect_size: tuple[int, int] = DEFAULT_DETECT_SIZE,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        verbose: bool = True,
    ) -> None:
        ref = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if ref is None:
            raise FileNotFoundError(f"end-screen template not found: {template_path}")
        self.template_path = str(template_path)
        self.template = cv2.resize(ref, detect_size)
        self.detect_size = detect_size
        self.similarity_threshold = similarity_threshold
        self.verbose = verbose
        self._last_score: float = 0.0

    @property
    def last_score(self) -> float:
        return self._last_score

    def detect(self, frame_bgr: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, self.detect_size).astype(np.float32)
        ref = self.template.astype(np.float32)
        diff = float(np.mean(np.abs(small - ref))) / 255.0
        score = 1.0 - diff
        self._last_score = score
        matched = score >= self.similarity_threshold
        if matched and self.verbose:
            print(
                f"[end-screen] match score={score:.3f} "
                f"(threshold={self.similarity_threshold})"
            )
        return matched
