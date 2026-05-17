"""Name → factory map for detector adapters.

nxwm itself ships no concrete detectors — they live in game packages. The
CLI imports the relevant game module before reading ``--detector NAME`` so
the registration side-effect runs first. Same pattern as
``nxwm.architectures``.
"""

from __future__ import annotations

from typing import Any, Callable

from .protocol import Detector

DetectorFactory = Callable[..., Detector]

detector_registry: dict[str, DetectorFactory] = {}


def register_detector(name: str, factory: DetectorFactory) -> None:
    """Register ``factory`` under ``name`` (idempotent — overwrites)."""
    detector_registry[name] = factory


def build_detector(name: str, **kwargs: Any) -> Detector:
    """Instantiate a registered detector. Raises ``KeyError`` if unknown."""
    if name not in detector_registry:
        known = ", ".join(sorted(detector_registry)) or "(none registered)"
        raise KeyError(f"unknown detector {name!r}; registered: {known}")
    return detector_registry[name](**kwargs)
