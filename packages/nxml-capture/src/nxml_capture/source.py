"""``CaptureSource`` protocol shared by all frame backends.

A ``CaptureSource`` is a thread-managed object that produces raw ``uint8``
frames (``H, W, C``, BGR per OpenCV convention) tagged with a wall-clock
timestamp. Backends own their own thread; consumers read the most recent
frame via :meth:`latest` or iterate via :meth:`frames`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class Frame:
    timestamp: float
    image: np.ndarray  # (H, W, C) uint8, BGR


@runtime_checkable
class CaptureSource(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    @property
    def is_open(self) -> bool: ...

    def latest(self) -> Frame | None:
        """Return the most recent frame, or ``None`` until one has been read."""
        ...

    def frames(self) -> Iterator[Frame]:
        """Yield frames as they arrive until :meth:`stop` is called."""
        ...
