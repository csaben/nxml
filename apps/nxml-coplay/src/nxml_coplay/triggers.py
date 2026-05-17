"""Visual-trigger → macro playback.

A :class:`TriggerWatcher` polls the capture source on its own daemon
thread and, when an armed :class:`TriggerSpec`'s reference image matches
the live frame for ``debounce_sec`` continuously, fires the bound macro
through :class:`nx_macros.MacroPlayer` (or the :class:`MashController`
for ``action_kind="mash_a"`` triggers).

The action loop already yields the orchestrator POST while
``macro_player.is_playing`` is True (see ``CoplayRunner.run``), so a
fired macro automatically displaces the AI without any mux-strategy
plumbing.

Multiple triggers can be armed concurrently — each carries its own
detector + per-trigger cooldown. The macro player itself is single-shot,
so when two triggers match in the same tick the second is suppressed by
the global ``is_playing`` gate.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from nx_macros import MacroPlayer, MacroStore, sanitize_name
from pydantic import BaseModel, Field

from nxml_coplay.mash import (
    DEFAULT_MASH_DURATION_SEC,
    DEFAULT_MASH_FRAMES_PER_PHASE,
    MashController,
)

DEFAULT_DETECT_SIZE = (64, 32)
DEFAULT_SIMILARITY_THRESHOLD = 0.85
DEFAULT_DEBOUNCE_SEC = 1.0
DEFAULT_COOLDOWN_SEC = 30.0
DEFAULT_WATCHER_HZ = 10.0

# Defaults for the template_crop detector (connection-lost dialog box).
DEFAULT_TEMPLATE_PROC_SIZE = (256, 128)
DEFAULT_TEMPLATE_CROP = (52, 38, 204, 90)  # (x0, y0, x1, y1) in proc-size space
DEFAULT_TEMPLATE_THRESHOLD = 0.75


class _FrameSource(Protocol):
    def latest(self) -> Any: ...  # SyncedFrame | None, with .image and .timestamp


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class ImageMatchDetector:
    """Mean-abs-diff template match against a small reference image.

    Cheap by design: both the live frame and the template are resized
    to ``detect_size`` grayscale (default 64x32), so a single np.abs
    subtract per call. Robust to small UI position jitter and codec
    noise; not invariant to translation or scale changes.

    Score is ``1.0 - mean(|live - template|) / 255``; ``detect()``
    returns True when score >= ``similarity_threshold``.
    """

    def __init__(
        self,
        template_path: str | Path,
        *,
        detect_size: tuple[int, int] = DEFAULT_DETECT_SIZE,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        ref = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if ref is None:
            raise FileNotFoundError(f"template not loadable: {template_path}")
        self.template_path = str(template_path)
        self.template = cv2.resize(ref, detect_size).astype(np.float32)
        self.detect_size = detect_size
        self.similarity_threshold = float(similarity_threshold)
        self._last_score: float = 0.0

    @property
    def last_score(self) -> float:
        return self._last_score

    def detect(self, frame_bgr: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, self.detect_size).astype(np.float32)
        diff = float(np.mean(np.abs(small - self.template))) / 255.0
        score = 1.0 - diff
        self._last_score = score
        return score >= self.similarity_threshold


class TemplateCropDetector:
    """``cv2.matchTemplate(TM_CCOEFF_NORMED)`` on a center crop of the frame.

    Selective for small dialog boxes (e.g. the Switch connection-lost popup)
    where MAD on the whole frame is dominated by background pixels and can't
    tell popup-on from popup-off apart. Resizes the frame to ``proc_size``
    grayscale, then template-matches the ``crop``-sized window of the
    reference against the full resized frame.
    """

    def __init__(
        self,
        template_path: str | Path,
        *,
        proc_size: tuple[int, int] = DEFAULT_TEMPLATE_PROC_SIZE,
        crop: tuple[int, int, int, int] = DEFAULT_TEMPLATE_CROP,
        similarity_threshold: float = DEFAULT_TEMPLATE_THRESHOLD,
    ) -> None:
        ref = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if ref is None:
            raise FileNotFoundError(f"template not loadable: {template_path}")
        ref = cv2.resize(ref, proc_size)
        x0, y0, x1, y1 = crop
        self.template_path = str(template_path)
        self.template = ref[y0:y1, x0:x1]
        self.proc_size = proc_size
        self.crop = crop
        self.similarity_threshold = float(similarity_threshold)
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
        return score >= self.similarity_threshold


class _Detector(Protocol):
    similarity_threshold: float

    @property
    def last_score(self) -> float: ...

    def detect(self, frame_bgr: np.ndarray) -> bool: ...


def _build_detector(spec: TriggerSpec, image_path: Path) -> _Detector:
    kind = spec.detector_kind
    params = spec.detector_params or {}
    if kind == "mad":
        detect_size = tuple(params.get("detect_size", DEFAULT_DETECT_SIZE))
        return ImageMatchDetector(
            image_path,
            detect_size=detect_size,
            similarity_threshold=spec.similarity_threshold,
        )
    if kind == "template_crop":
        return TemplateCropDetector(
            image_path,
            proc_size=tuple(params.get("proc_size", DEFAULT_TEMPLATE_PROC_SIZE)),
            crop=tuple(params.get("crop", DEFAULT_TEMPLATE_CROP)),
            similarity_threshold=spec.similarity_threshold,
        )
    raise ValueError(f"unknown detector_kind: {kind!r} (expected 'mad' or 'template_crop')")


# ---------------------------------------------------------------------------
# Spec + store
# ---------------------------------------------------------------------------


class TriggerSpec(BaseModel):
    name: str
    image_filename: str
    # Macro to play when ``action_kind == "macro"``. Empty allowed for
    # mash_a triggers, which don't need a backing Macro file.
    macro_name: str = ""
    similarity_threshold: float = Field(
        default=DEFAULT_SIMILARITY_THRESHOLD, ge=0.0, le=1.0
    )
    debounce_sec: float = Field(default=DEFAULT_DEBOUNCE_SEC, ge=0.0)
    cooldown_sec: float = Field(default=DEFAULT_COOLDOWN_SEC, ge=0.0)
    loop: bool = False
    # Detector algorithm. "mad" = mean-abs-diff on the whole frame
    # downsampled to 64×32 (cheap, robust for full-screen overlays).
    # "template_crop" = TM_CCOEFF_NORMED on a center-cropped reference
    # (selective for small dialog boxes that MAD can't tell from background).
    detector_kind: str = Field(default="mad")
    detector_params: dict[str, Any] = Field(default_factory=dict)
    # What happens when the trigger fires. "macro" plays the named
    # MacroPlayer macro. "mash_a" runs the in-memory MashController
    # state machine for ``mash_duration_sec`` (no Macro file required).
    action_kind: str = Field(default="macro")
    mash_duration_sec: float = Field(default=DEFAULT_MASH_DURATION_SEC, gt=0.0)
    mash_frames_per_phase: int = Field(default=DEFAULT_MASH_FRAMES_PER_PHASE, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TriggerStore:
    """Directory-of-JSON trigger store with sibling image files.

    Layout::

        <root>/<name>.json          # TriggerSpec
        <root>/images/<filename>    # reference templates uploaded by UI
    """

    IMAGE_SUBDIR = "images"

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def images_dir(self) -> Path:
        return self._root / self.IMAGE_SUBDIR

    def ensure_dirs(self) -> None:
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self._root / f"{sanitize_name(name)}.json"

    def list(self) -> list[str]:
        if not self._root.exists():
            return []
        return sorted(p.stem for p in self._root.glob("*.json") if p.is_file())

    def exists(self, name: str) -> bool:
        return self._path(name).is_file()

    def save(self, spec: TriggerSpec) -> Path:
        self.ensure_dirs()
        p = self._path(spec.name)
        p.write_text(spec.model_dump_json(indent=2))
        return p

    def load(self, name: str) -> TriggerSpec:
        p = self._path(name)
        if not p.is_file():
            raise FileNotFoundError(f"trigger not found: {name!r} (looked at {p})")
        return TriggerSpec.model_validate_json(p.read_text())

    def delete(self, name: str) -> bool:
        p = self._path(name)
        if not p.is_file():
            return False
        p.unlink()
        return True

    def image_path(self, spec: TriggerSpec) -> Path:
        return self.images_dir / spec.image_filename

    def save_image(self, filename: str, data: bytes) -> Path:
        # Image filenames carry extensions, so the macro-name sanitizer
        # (which rejects leading dots and path separators but allows
        # the dot in ``.png``) is too strict. Apply just the separator/
        # null/whitespace checks here.
        cleaned = filename.strip()
        if not cleaned:
            raise ValueError("image filename must not be empty")
        if "/" in cleaned or "\\" in cleaned or "\x00" in cleaned:
            raise ValueError(f"image filename has forbidden characters: {filename!r}")
        if cleaned.startswith("."):
            raise ValueError(f"image filename must not start with '.': {filename!r}")
        self.ensure_dirs()
        dest = self.images_dir / cleaned
        dest.write_bytes(data)
        return dest


# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------


class _ArmedTrigger:
    """Per-trigger runtime state. One per name in ``TriggerWatcher._armed``."""

    __slots__ = (
        "cooldown_until",
        "detector",
        "fire_count",
        "last_fire_at",
        "match_started_at",
        "spec",
    )

    def __init__(self, spec: TriggerSpec, detector: _Detector) -> None:
        self.spec = spec
        self.detector = detector
        self.match_started_at: float | None = None
        self.cooldown_until: float = 0.0
        self.fire_count: int = 0
        self.last_fire_at: float = 0.0


class TriggerWatcher:
    """Background watcher that fires any number of armed triggers.

    Lifecycle:

      - ``start()`` spins up the daemon thread; it idles when nothing
        is armed.
      - ``arm(name)`` loads the spec + its reference image, validates
        that the bound macro exists, and adds it to the set of armed
        triggers (re-arming the same name refreshes its detector and
        clears its cooldown). Returns the loaded spec.
      - ``disarm(name)`` removes a single trigger; ``disarm_all()``
        clears everything. Neither interrupts an in-flight macro.
      - ``stop()`` joins the thread.

    Firing: each trigger tracks its own match-persistence timer and
    post-fire cooldown. When the match has persisted for
    ``debounce_sec`` (anti-flicker) the watcher calls
    ``MacroPlayer.play_async``; the trigger then ignores further matches
    for its own ``cooldown_sec``. While the macro player is already
    playing, the watcher skips evaluation entirely so a second trigger
    cannot clobber an in-flight macro.
    """

    def __init__(
        self,
        *,
        source: _FrameSource,
        store: TriggerStore,
        macro_store: MacroStore,
        macro_player: MacroPlayer,
        mash_controller: MashController,
        watcher_hz: float = DEFAULT_WATCHER_HZ,
    ) -> None:
        self._source = source
        self._store = store
        self._macro_store = macro_store
        self._macro_player = macro_player
        self._mash_controller = mash_controller
        self._period = 1.0 / max(watcher_hz, 1.0)

        self._lock = threading.Lock()
        self._armed: dict[str, _ArmedTrigger] = {}

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def armed_names(self) -> list[str]:
        with self._lock:
            return sorted(self._armed.keys())

    def is_armed(self, name: str) -> bool:
        with self._lock:
            return name in self._armed

    def status(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            armed = sorted(self._armed.keys())
            states: dict[str, dict[str, Any]] = {}
            for name, at in self._armed.items():
                states[name] = {
                    "macro": at.spec.macro_name,
                    "loop": bool(at.spec.loop),
                    "threshold": at.spec.similarity_threshold,
                    "debounce_sec": at.spec.debounce_sec,
                    "cooldown_sec": at.spec.cooldown_sec,
                    "score": at.detector.last_score,
                    "in_cooldown": now < at.cooldown_until,
                    "cooldown_remaining": max(0.0, at.cooldown_until - now),
                    "fire_count": at.fire_count,
                    "last_fire_at": at.last_fire_at,
                }
        return {
            "armed": armed,
            "states": states,
            "playing": self._macro_player.playing_info(),
        }

    def arm(self, name: str) -> TriggerSpec:
        spec = self._store.load(name)
        if spec.action_kind == "macro":
            if not spec.macro_name:
                raise ValueError(
                    f"trigger {name!r} has action_kind='macro' but no macro_name set"
                )
            if not self._macro_store.exists(spec.macro_name):
                raise ValueError(
                    f"trigger {name!r} references macro {spec.macro_name!r}, "
                    "but no such macro exists in the macro store"
                )
        elif spec.action_kind != "mash_a":
            raise ValueError(
                f"trigger {name!r} has unknown action_kind={spec.action_kind!r} "
                "(expected 'macro' or 'mash_a')"
            )
        image_path = self._store.image_path(spec)
        detector = _build_detector(spec, image_path)
        with self._lock:
            self._armed[name] = _ArmedTrigger(spec, detector)
        action_desc = (
            f"macro {spec.macro_name!r}"
            if spec.action_kind == "macro"
            else f"mash_a {spec.mash_duration_sec:.1f}s"
        )
        print(
            f"[trigger] armed {spec.name!r} → {action_desc} "
            f"(detector={spec.detector_kind} threshold={spec.similarity_threshold:.2f} "
            f"debounce={spec.debounce_sec:.1f}s cooldown={spec.cooldown_sec:.0f}s)",
            flush=True,
        )
        return spec

    def disarm(self, name: str) -> bool:
        """Disarm a single trigger. Returns True if it was armed."""
        with self._lock:
            removed = self._armed.pop(name, None) is not None
        if removed:
            print(f"[trigger] disarmed {name!r}", flush=True)
        return removed

    def update_armed_spec(self, name: str, **updates: Any) -> bool:
        """Mutate the in-memory spec of an armed trigger.

        Returns True if the trigger was armed (and its spec was updated).
        No-op if the trigger isn't currently armed — the JSON change on
        disk is enough for the next ``arm()`` to pick up the new value.
        Detector params are not re-applied here; the detector instance
        keeps its construction-time threshold. Re-arm to rebuild it.
        """
        with self._lock:
            at = self._armed.get(name)
            if at is None:
                return False
            for k, v in updates.items():
                if hasattr(at.spec, k):
                    setattr(at.spec, k, v)
        print(
            f"[trigger] live-updated {name!r}: "
            + ", ".join(f"{k}={v!r}" for k, v in updates.items()),
            flush=True,
        )
        return True

    def disarm_all(self) -> None:
        with self._lock:
            names = list(self._armed.keys())
            self._armed.clear()
        for n in names:
            print(f"[trigger] disarmed {n!r}", flush=True)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="coplay-trigger-watcher"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        last_frame_ts: float | None = None
        while not self._stop.is_set():
            t_start = time.time()
            with self._lock:
                armed_snapshot = list(self._armed.items())

            if (
                not armed_snapshot
                or self._macro_player.is_playing
                or self._mash_controller.is_active
            ):
                if self._stop.wait(self._period):
                    return
                continue

            frame = self._source.latest()
            if frame is None or frame.timestamp == last_frame_ts:
                if self._stop.wait(self._period):
                    return
                continue
            last_frame_ts = frame.timestamp

            now = time.time()
            for name, at in armed_snapshot:
                if now < at.cooldown_until:
                    continue
                try:
                    matched = at.detector.detect(frame.image)
                except Exception as e:
                    print(
                        f"[trigger] {name!r} detector error: "
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )
                    continue
                if matched:
                    if at.match_started_at is None:
                        at.match_started_at = now
                    elif (now - at.match_started_at) >= at.spec.debounce_sec:
                        self._fire(at)
                        at.match_started_at = None
                        # Macro is now playing; subsequent triggers in this
                        # snapshot would be suppressed by is_playing anyway.
                        break
                else:
                    at.match_started_at = None

            elapsed = time.time() - t_start
            if elapsed < self._period:
                if self._stop.wait(self._period - elapsed):
                    return

    def _fire(self, at: _ArmedTrigger) -> None:
        spec = at.spec
        if spec.action_kind == "mash_a":
            try:
                self._mash_controller.start(
                    duration_sec=spec.mash_duration_sec,
                    frames_per_phase=spec.mash_frames_per_phase,
                    source=spec.name,
                )
            except Exception as e:  # defensive — controller is stateless-ish
                print(f"[trigger] mash start failed: {e}", flush=True)
                return
            action_desc = f"mash_a {spec.mash_duration_sec:.1f}s"
        else:
            try:
                macro = self._macro_store.load(spec.macro_name)
            except FileNotFoundError as e:
                print(f"[trigger] {spec.name!r}: {e} — disarming", flush=True)
                self.disarm(spec.name)
                return
            try:
                self._macro_player.play_async(macro, loop=spec.loop)
            except RuntimeError as e:
                print(f"[trigger] play_async failed: {e}", flush=True)
                return
            action_desc = f"macro {spec.macro_name!r}"
        now = time.time()
        with self._lock:
            at.last_fire_at = now
            at.cooldown_until = now + spec.cooldown_sec
            at.fire_count += 1
        print(
            f"[trigger] fired {spec.name!r} → {action_desc} "
            f"(cooldown {spec.cooldown_sec:.0f}s)",
            flush=True,
        )
