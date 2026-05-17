"""Connection-lost popup detector.

The Switch shows a centered grey dialog box when the Bluetooth connection
to nxbt drops. We grayscale-template-match the dialog's center crop with
``cv2.TM_CCOEFF_NORMED`` after resizing the input frame to a fixed
processing size; default thresholds are tuned for the ZA game UI.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEFAULT_PROC_SIZE = (256, 128)
DEFAULT_CROP = (52, 38, 204, 90)  # (x0, y0, x1, y1) of the dialog box in proc-size space
DEFAULT_THRESHOLD = 0.75


class ConnectionLostDetector:
    """Template-match the Switch's "connection lost" popup."""

    def __init__(
        self,
        template_path: str | Path,
        *,
        proc_size: tuple[int, int] = DEFAULT_PROC_SIZE,
        crop: tuple[int, int, int, int] = DEFAULT_CROP,
        threshold: float = DEFAULT_THRESHOLD,
        verbose: bool = True,
    ) -> None:
        ref = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if ref is None:
            raise FileNotFoundError(f"connection-lost template not found: {template_path}")
        ref = cv2.resize(ref, proc_size)
        x0, y0, x1, y1 = crop
        self.template_path = str(template_path)
        self.template = ref[y0:y1, x0:x1]
        self.proc_size = proc_size
        self.threshold = threshold
        self.verbose = verbose
        self._last_score: float = 0.0

    @property
    def last_score(self) -> float:
        return self._last_score

    def detect(self, frame_bgr: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        proc = cv2.resize(gray, self.proc_size)
        result = cv2.matchTemplate(proc, self.template, cv2.TM_CCOEFF_NORMED)
        score = float(result.max())
        self._last_score = score
        matched = score >= self.threshold
        if matched and self.verbose:
            print(
                f"[conn-lost] match score={score:.3f} (threshold={self.threshold})"
            )
        return matched
