"""Bundled reference images for Pokémon ZA detectors.

Resolved via :mod:`importlib.resources` so they survive wheel installs.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any, TypedDict


def default_end_screen_path() -> Path:
    """Path to the bundled end-screen template (post-battle dialog).

    Seeded by ``nxml-autopilot``'s default Triggers on first launch.
    """
    return Path(str(files("nxml_games.pokemon_za.assets") / "end_screen.png"))


def default_connection_lost_path() -> Path:
    """Path to the bundled connection-lost popup template (grey center dialog)."""
    return Path(str(files("nxml_games.pokemon_za.assets") / "connection_lost.png"))


class DefaultTriggerSeed(TypedDict, total=False):
    name: str
    image_path: Path
    image_filename: str
    macro_name: str
    similarity_threshold: float
    debounce_sec: float
    cooldown_sec: float
    detector_kind: str
    detector_params: dict[str, Any]
    action_kind: str
    mash_duration_sec: float
    mash_frames_per_phase: int


# Cooldown is post-fire (not post-mash): with mash_duration_sec=40 and
# cooldown_sec=80, the effective re-trigger lockout after the mash ends
# is 40 s, so a still-visible dialog flicker can't re-arm immediately.
# Debounce stays at autopilot's default 1.0 s (anti-flicker).
#
# Both bundled triggers fire the in-memory MashController (no Macro file
# needed): 40 s of alternating A-press/neutral packets at 30 Hz × 3-frame  # noqa: RUF003
# phases (5 Hz mash).
#
# end_screen → mean-abs-diff on the whole downsampled frame (cheap; the
# end screen is a full-screen overlay so MAD score collapses cleanly).
# connection_lost → ``cv2.matchTemplate(TM_CCOEFF_NORMED)`` on a centered
# crop of the dialog box.
def default_triggers() -> list[DefaultTriggerSeed]:
    """Return the trigger specs seeded into an empty autopilot trigger store."""
    return [
        {
            "name": "end_screen",
            "image_path": default_end_screen_path(),
            "image_filename": "end_screen.png",
            "macro_name": "",
            "similarity_threshold": 0.85,
            "debounce_sec": 1.0,
            "cooldown_sec": 80.0,
            "detector_kind": "mad",
            "detector_params": {},
            "action_kind": "mash_a",
            "mash_duration_sec": 40.0,
            "mash_frames_per_phase": 3,
        },
        {
            "name": "connection_lost",
            "image_path": default_connection_lost_path(),
            "image_filename": "connection_lost.png",
            "macro_name": "",
            "similarity_threshold": 0.75,
            "debounce_sec": 1.0,
            "cooldown_sec": 80.0,
            "detector_kind": "template_crop",
            # proc_size/crop default to the constants in
            # nxml_autopilot.triggers (256x128, (52, 38, 204, 90)); leave
            # detector_params empty so a single source of truth wins.
            "detector_params": {},
            "action_kind": "mash_a",
            "mash_duration_sec": 40.0,
            "mash_frames_per_phase": 3,
        },
    ]


__all__ = [
    "DefaultTriggerSeed",
    "default_connection_lost_path",
    "default_end_screen_path",
    "default_triggers",
]
