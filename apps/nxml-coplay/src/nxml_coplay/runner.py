"""Coplay runner: capture + AI inference + mux + POST + record.

Runs two threads:

  - **AI inference loop** (``_AiInference._loop``) — pulls the latest frame
    from the V4L2 source, VAE-encodes it, slides a latent window of length
    ``policy.sequence_length``, and once the window is full, calls
    ``policy_client.predict()``. The result is cached atomically. Runs at
    whatever rate the GPU sustains — could be 5 Hz for a heavy policy, 30+
    Hz for a small one.

  - **Action loop** (main thread) — at a fixed tick rate (default 30 Hz),
    polls the human ``EvdevReader`` and the ``CachedAiSource``, runs them
    through ``ControllerMux`` + ``HumanPriority``, POSTs the merged 26-dim
    action to ``nxbt-orchestrator /action``, optionally appends to the
    recorder.

The two are decoupled by ``CachedAiSource``: the action loop never blocks
on inference. If the AI hasn't produced anything yet (window not full,
inference still warming up), the human's input drives the Switch on its
own — exactly what "non-blocking AI" means.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from nxml_capture import SyncedFrame
from nxml_capture.backends.v4l2 import V4L2Source
from nxml_mux import ActionSnapshot, ControllerMux, HumanPriority, HumanTakeover
from nxml_mux.input_devices.auto_detect import detect_mapper_for_device, detect_mapper_for_name
from nxml_mux.input_devices.mappers.base import Mapper
from nxml_mux.input_devices.readers import EvdevReader, WebGamepadReader
from nxml_mux.input_devices.registry import get as get_mapper
from nxml_mux.input_devices.registry import load_bundled_mappers
from nx_packets import ACTION_DIM


@dataclass
class CoplayConfig:
    game: str
    policy_uri: str
    controller_url: str  # http://host:port (orchestrator)
    controller_input: str = "auto"  # mapper id, or "auto"
    device_path: str | None = None  # /dev/input/eventN; None = auto-pick
    camera_id: int = 0
    capture_width: int = 1920
    capture_height: int = 1080
    input_source: str = "evdev"  # "evdev" | "web"
    web_host: str = "0.0.0.0"
    web_port: int = 8080
    web_token: str = ""
    web_stick_deadzone: float = 0.15
    mode: str = "human-priority"
    record_dir: Path | None = None
    macro_dir: Path | None = None
    trigger_dir: Path | None = None
    vae_path: str | None = None  # default: stabilityai/sd-vae-ft-mse
    device: str | None = None  # cuda / cpu (default: cuda if available)
    tick_hz: float = 30.0
    inference_min_period_s: float = 0.0  # 0 = run as fast as the GPU allows
    extra_metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AI inference
# ---------------------------------------------------------------------------


class _AiInference:
    """Background-threaded VAE encode + policy predict.

    Source frames feed an N-deep sliding latent window; once the window is
    full each iteration calls ``policy_client.predict`` and caches the
    result. ``latest_action()`` returns the most recent cached numpy
    array, never blocking.
    """

    def __init__(
        self,
        *,
        policy_client,
        vae,
        sequence_length: int,
        latent_shape: tuple[int, int, int],
        source: V4L2Source,
        device,
        min_period_s: float = 0.0,
    ) -> None:
        self._client = policy_client
        self._vae = vae
        self._seq_len = sequence_length
        self._latent_shape = latent_shape
        self._source = source
        self._device = device
        self._min_period_s = min_period_s

        self._lock = threading.Lock()
        self._cached_action: np.ndarray | None = None
        self._cached_at: float = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._inferences = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="coplay-ai")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def latest_action(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            if self._cached_action is None:
                return None, 0.0
            return self._cached_action.copy(), self._cached_at

    @property
    def inference_count(self) -> int:
        return self._inferences

    def _loop(self) -> None:
        from nxwm.inference.vae import encode_image

        # Sliding latent window — keep on CPU as numpy (the predict client
        # uploads to GPU itself for in-process, or wire-serializes for ZMQ).
        window: deque[np.ndarray] = deque(maxlen=self._seq_len)
        last_frame_ts: float | None = None

        while not self._stop.is_set():
            try:
                t_start = time.time()
                frame = self._source.latest()
                if frame is None or frame.timestamp == last_frame_ts:
                    time.sleep(0.01)
                    continue
                last_frame_ts = frame.timestamp

                # cv2 capture is BGR HWC; nxwm encode_image expects RGB HWC uint8.
                img_rgb = frame.image[:, :, ::-1]
                latent_t = encode_image(np.ascontiguousarray(img_rgb), self._vae, self._device)
                window.append(latent_t.detach().cpu().numpy())

                if len(window) < self._seq_len:
                    continue

                latents = np.stack(window, axis=0).astype(np.float32, copy=False)
                action = self._client.predict(latents)

                with self._lock:
                    self._cached_action = action.astype(np.float32, copy=False)
                    self._cached_at = time.time()
                self._inferences += 1

                elapsed = time.time() - t_start
                if elapsed < self._min_period_s:
                    time.sleep(self._min_period_s - elapsed)
            except Exception as e:
                # Log and continue so a transient server hiccup or VAE OOM
                # doesn't silently kill the daemon thread.
                print(f"[coplay-ai] iteration failed: {type(e).__name__}: {e}", flush=True)
                time.sleep(0.05)


class CachedAiSource:
    """``ActionSource`` adapter wrapping a cached-action provider.

    Both :class:`_AiInference` (local VAE + remote/in-process policy) and
    :class:`_RemoteFrameAiInference` (server-side VAE + window) expose
    ``latest_action()`` and start/stop, so the mux side stays uniform.

    The ``enabled`` flag lets the runtime suppress the AI source without
    tearing down inference — flipping it back on resumes from the cached
    action immediately.
    """

    source_id = "ai:policy"

    def __init__(self, ai) -> None:
        self._ai = ai
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)

    def latest(self) -> ActionSnapshot | None:
        if not self._enabled:
            return None
        action, ts = self._ai.latest_action()
        if action is None:
            return None
        return ActionSnapshot(
            action=action,
            timestamp=ts,
            source_id=self.source_id,
            mask=None,  # AI contributes the full vector; HumanPriority fills only what human didn't claim.
        )

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Remote-VAE inference (frame mode — no local GPU/VAE)
# ---------------------------------------------------------------------------


class _RemoteFrameAiInference:
    """Background loop that ships JPEGs to a frame-mode policy server.

    Mirrors :class:`_AiInference`'s contract (``start`` / ``stop`` /
    ``latest_action``) but does no VAE work locally — the server keeps the
    sliding latent window and runs encode + predict. Useful when the play
    box has no GPU.
    """

    def __init__(
        self,
        *,
        client,
        source: V4L2Source,
        jpeg_quality: int = 85,
        resize_to: tuple[int, int] | None = (256, 128),
        min_period_s: float = 0.0,
    ) -> None:
        self._client = client
        self._source = source
        self._jpeg_quality = int(jpeg_quality)
        self._resize_to = resize_to
        self._min_period_s = min_period_s

        self._lock = threading.Lock()
        self._cached_action: np.ndarray | None = None
        self._cached_at: float = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._inferences = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="coplay-ai-frame"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def latest_action(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            if self._cached_action is None:
                return None, 0.0
            return self._cached_action.copy(), self._cached_at

    @property
    def inference_count(self) -> int:
        return self._inferences

    def _loop(self) -> None:
        import cv2

        last_frame_ts: float | None = None
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]

        while not self._stop.is_set():
            try:
                t_start = time.time()
                frame = self._source.latest()
                if frame is None or frame.timestamp == last_frame_ts:
                    time.sleep(0.01)
                    continue
                last_frame_ts = frame.timestamp

                # cv2 capture is BGR; cv2.imencode wants BGR. Pre-resize to
                # the VAE input size so the wire is ~5KB/frame instead of
                # ~150KB at 1080p — the server resizes again but it's a no-op.
                img = frame.image
                if self._resize_to is not None:
                    img = cv2.resize(img, self._resize_to)
                ok, jpeg = cv2.imencode(".jpg", img, encode_params)
                if not ok:
                    continue
                action = self._client.predict_frame(jpeg.tobytes())
                if action is None:
                    # Server still warming up its window.
                    continue
                with self._lock:
                    self._cached_action = action.astype(np.float32, copy=False)
                    self._cached_at = time.time()
                self._inferences += 1

                elapsed = time.time() - t_start
                if elapsed < self._min_period_s:
                    time.sleep(self._min_period_s - elapsed)
            except Exception as e:
                print(
                    f"[coplay-ai-frame] iteration failed: {type(e).__name__}: {e}",
                    flush=True,
                )
                time.sleep(0.05)


# ---------------------------------------------------------------------------
# Mapper picking
# ---------------------------------------------------------------------------


def _pick_device_and_mapper(
    *, controller_input: str, device_path: str | None
) -> tuple[str, Mapper]:
    """Resolve a (device_path, mapper) pair from the CLI flags.

    Strategy:
      - ``controller_input == "auto"``:
          - if ``device_path`` set, dispatch via ``detect_mapper_for_device``.
          - else scan ``/dev/input/event*`` and pick the first whose name matches
            any registered mapper.
      - Otherwise ``controller_input`` is a mapper id; load it from the registry.
        Caller must supply ``device_path``.
    """
    load_bundled_mappers()

    if controller_input == "auto":
        if device_path is not None:
            mapper = detect_mapper_for_device(device_path)
            if mapper is None:
                raise RuntimeError(f"no mapper matched device at {device_path}")
            return device_path, mapper
        # Enumerate evdev devices, pick the first claimed by a registered mapper.
        import evdev

        for dev_path in sorted(evdev.list_devices()):
            try:
                dev = evdev.InputDevice(dev_path)
            except OSError:
                continue
            mapper = detect_mapper_for_name(dev.name, autoload_bundled=False)
            if mapper is not None:
                return dev_path, mapper
        raise RuntimeError(
            "no evdev device matched any registered mapper. Pass --device-path explicitly."
        )

    # Explicit mapper id.
    mapper = get_mapper(controller_input)
    if device_path is None:
        raise RuntimeError(f"--controller-input={controller_input!r} requires --device-path")
    return device_path, mapper


# ---------------------------------------------------------------------------
# CoplayRunner
# ---------------------------------------------------------------------------


class CoplayRunner:
    def __init__(self, config: CoplayConfig) -> None:
        from nxrl.serve import build_client

        self.config = config

        if config.mode not in ("human-priority", "human-takeover"):
            raise NotImplementedError(
                f"unsupported mode {config.mode!r}; expected 'human-priority' or 'human-takeover'"
            )

        # Frame mode: server runs the VAE, no torch needed locally.
        self._frame_mode = config.policy_uri.startswith(("zmq+frames://", "tcp+frames://"))

        if self._frame_mode:
            self._device = None
        else:
            import torch

            self._device = torch.device(
                config.device or ("cuda" if torch.cuda.is_available() else "cpu")
            )

        # Capture
        self._source = V4L2Source(
            camera_id=config.camera_id,
            width=config.capture_width,
            height=config.capture_height,
        )

        # Policy client (in-process, remote zmq, or remote zmq+frames)
        client_kwargs: dict[str, Any] = (
            {} if self._frame_mode else {"device": str(self._device)}
        )
        self._client = build_client(config.policy_uri, **client_kwargs)
        info = self._client.info()
        self._sequence_length = int(info["sequence_length"])
        self._latent_shape = tuple(info["latent_shape"])
        print(
            f"[coplay] policy {info['architecture']} seq_len={self._sequence_length} "
            f"latent={self._latent_shape} algo={info.get('algorithm')}"
            f"{' (frame mode — remote VAE)' if self._frame_mode else ''}"
        )

        if self._frame_mode:
            self._vae = None
            self._ai = _RemoteFrameAiInference(
                client=self._client,
                source=self._source,
                min_period_s=config.inference_min_period_s,
            )
        else:
            from nxwm.inference.vae import load_vae

            self._vae = load_vae(
                config.vae_path or "stabilityai/sd-vae-ft-mse", device=self._device
            )
            self._vae.eval()
            self._ai = _AiInference(
                policy_client=self._client,
                vae=self._vae,
                sequence_length=self._sequence_length,
                latent_shape=self._latent_shape,  # type: ignore[arg-type]
                source=self._source,
                device=self._device,
                min_period_s=config.inference_min_period_s,
            )
        self._ai_source = CachedAiSource(self._ai)

        # Human (evdev or browser-fed)
        self._web_server = None
        if config.input_source == "web":
            self._human = WebGamepadReader(stick_deadzone=config.web_stick_deadzone)
            from nxml_coplay.web import CoplayWebServer

            self._web_server = CoplayWebServer(
                self._human,
                self._source,
                host=config.web_host,
                port=config.web_port,
                token=config.web_token,
            )
            display_host = "localhost" if config.web_host in ("0.0.0.0", "::") else config.web_host
            url = f"http://{display_host}:{config.web_port}/"
            if config.web_token:
                url += f"?token={config.web_token}"
            print(f"[coplay] human input: web — open {url}")
            if config.web_host in ("0.0.0.0", "::"):
                print(
                    f"[coplay]   (also reachable from LAN at http://<this-host-ip>:{config.web_port}/"
                    f"{'?token=' + config.web_token if config.web_token else ''})"
                )
        elif config.input_source == "evdev":
            device_path, mapper = _pick_device_and_mapper(
                controller_input=config.controller_input, device_path=config.device_path
            )
            print(f"[coplay] human input: {device_path} ({mapper.id})")
            self._human = EvdevReader(device_path, mapper)
        else:
            raise ValueError(f"unknown input_source {config.input_source!r}")

        # Mux. Strategy is swappable at runtime via ``set_mode`` so the web
        # UI can flip between human-priority and human-takeover live.
        self._human_ids = {self._human.source_id}
        self._mode = config.mode
        self._mux = ControllerMux(
            sources=[self._human, self._ai_source],
            strategy=self._build_strategy(self._mode),
        )

        # Orchestrator HTTP client
        self._post_url = config.controller_url.rstrip("/") + "/action"
        self._http = httpx.Client(timeout=2.0)

        # Recorder controller (toggle-able from web UI; auto-start in evdev mode).
        from nxml_coplay.recording import RecordingController

        self._recorder_ctl = RecordingController(fps=config.tick_hz)
        if config.input_source == "evdev" and config.record_dir is not None:
            config.record_dir.mkdir(parents=True, exist_ok=True)
            self._recorder_ctl.start(config.record_dir)
            print(f"[coplay] recording → {config.record_dir}")
        elif config.input_source == "web" and config.record_dir is not None:
            config.record_dir.mkdir(parents=True, exist_ok=True)
            print(f"[coplay] recording root: {config.record_dir} (use the toggle in the web UI)")

        # Macro recorder + player + store (always present; store is None when
        # --macro-dir wasn't passed). Recording into a no-op recorder is safe,
        # and the web UI surfaces "not configured" when the store is missing.
        from nx_macros import MacroPlayer, MacroRecorder, MacroStore

        self._macro_recorder = MacroRecorder(tick_hz=config.tick_hz)
        self._macro_store: MacroStore | None = (
            MacroStore(config.macro_dir) if config.macro_dir is not None else None
        )
        if self._macro_store is not None:
            self._seed_default_macros(self._macro_store, config.game)
        self._macro_player = MacroPlayer(poster=self._post_action)

        # Synthetic A-mash controller for trigger-driven death-screen handling.
        # The action loop drives it (one ``next_action()`` per tick) so cadence
        # is ``tick_hz`` × ``frames_per_phase``.
        from nxml_coplay.mash import MashController

        self._mash_controller = MashController()

        # Trigger store + watcher. The watcher's daemon thread idles when
        # nothing is armed, so it's cheap to always-on.
        from nxml_coplay.triggers import TriggerStore, TriggerWatcher

        self._trigger_store: TriggerStore | None = (
            TriggerStore(config.trigger_dir) if config.trigger_dir is not None else None
        )
        if self._trigger_store is not None:
            self._seed_default_triggers(self._trigger_store, config.game)
        self._trigger_watcher: TriggerWatcher | None = None
        if self._trigger_store is not None and self._macro_store is not None:
            self._trigger_watcher = TriggerWatcher(
                source=self._source,
                store=self._trigger_store,
                macro_store=self._macro_store,
                macro_player=self._macro_player,
                mash_controller=self._mash_controller,
            )

        # Hand the controller to the web server if there is one.
        if self._web_server is not None:
            self._web_server.attach_recorder(
                self._recorder_ctl, root=config.record_dir
            )
            self._web_server.attach_macros(
                recorder=self._macro_recorder,
                store=self._macro_store,
                player=self._macro_player,
                game=config.game,
            )
            self._web_server.attach_triggers(
                store=self._trigger_store,
                watcher=self._trigger_watcher,
            )
            self._web_server.attach_runtime(self)

        self._stop_flag = threading.Event()

    def _post_action(self, action: np.ndarray) -> None:
        try:
            payload = {"vector": action.tolist(), "source": "inference"}
            self._http.post(self._post_url, json=payload)
        except httpx.HTTPError as e:
            print(f"[coplay] orchestrator POST failed: {e}")

    SUPPORTED_MODES = ("human-priority", "human-takeover")

    @staticmethod
    def _seed_default_macros(store: Any, game: str) -> None:
        """Populate ``store`` with the game's bundled default macros.

        Idempotent — skips any macro whose JSON already exists. Today this
        only seeds ``mash_a`` for Pokémon ZA: a 40-second alternating-A
        press/release at 5 Hz.
        """
        from nx_macros import Macro, MacroFrame

        normalized = game.replace("-", "_")
        if normalized != "pokemon_za":
            return
        if store.exists("mash_a"):
            return

        # 26-dim action vector: indices 0..3 sticks, 4..5 stick PRESSED,
        # 6..9 dpad, 10..13 L/ZL/R/ZR, 14..17 joycon SR/SL, 18..21
        # PLUS/MINUS/HOME/CAPTURE, 22..25 Y/X/B/A. A is index 25.
        neutral = [0.0] * 26
        a_press = [0.0] * 25 + [1.0]
        phase_dt = 0.1  # 100ms per phase -> 5Hz mash rate
        cycles = 200  # 200 cycles * 2 frames * 0.1s = 40s
        frames = []
        for _ in range(cycles):
            frames.append(MacroFrame(dt=phase_dt, action=a_press))
            frames.append(MacroFrame(dt=phase_dt, action=neutral))
        store.save(Macro(name="mash_a", tick_hz=1.0 / phase_dt, frames=frames))
        print(
            f"[coplay] seeded default macro 'mash_a' "
            f"({len(frames)} frames, ~{cycles * 2 * phase_dt:.0f}s)",
            flush=True,
        )

    @staticmethod
    def _seed_default_triggers(store: Any, game: str) -> None:
        """Populate ``store`` with the game's bundled default triggers.

        Idempotent: skips any trigger whose JSON spec already exists, so
        user edits to threshold/debounce survive subsequent restarts. The
        bundled reference image is copied into ``store.images_dir`` only
        when missing.
        """
        from nxml_coplay.triggers import TriggerSpec

        # CLI accepts "pokemon-za" or "pokemon_za"; normalize to the
        # importable package name.
        normalized = game.replace("-", "_")
        if normalized != "pokemon_za":
            return

        try:
            from nxml_games.pokemon_za.assets import default_triggers
        except ImportError:
            return

        store.ensure_dirs()
        for seed in default_triggers():
            dest_image = store.images_dir / seed["image_filename"]
            if not dest_image.is_file():
                dest_image.write_bytes(Path(seed["image_path"]).read_bytes())
            spec_kwargs: dict[str, Any] = {
                "name": seed["name"],
                "image_filename": seed["image_filename"],
                "macro_name": seed.get("macro_name", ""),
                "similarity_threshold": seed["similarity_threshold"],
                "debounce_sec": seed["debounce_sec"],
                "cooldown_sec": seed["cooldown_sec"],
                "detector_kind": seed.get("detector_kind", "mad"),
                "detector_params": dict(seed.get("detector_params", {})),
                "action_kind": seed.get("action_kind", "macro"),
            }
            if "mash_duration_sec" in seed:
                spec_kwargs["mash_duration_sec"] = seed["mash_duration_sec"]
            if "mash_frames_per_phase" in seed:
                spec_kwargs["mash_frames_per_phase"] = seed["mash_frames_per_phase"]
            spec = TriggerSpec(**spec_kwargs)
            if store.exists(seed["name"]):
                # One-time migration: an existing JSON saved before the
                # action_kind field was introduced gets overwritten so
                # the bundled trigger picks up the right detector +
                # action. User edits to a trigger that already has
                # action_kind are preserved.
                try:
                    existing_raw = (
                        (store.root / f"{seed['name']}.json").read_text()
                    )
                except OSError:
                    existing_raw = ""
                if '"action_kind"' in existing_raw:
                    continue
                store.save(spec)
                print(
                    f"[coplay] migrated stale trigger {seed['name']!r} "
                    f"→ detector={spec.detector_kind} action={spec.action_kind} "
                    f"threshold={spec.similarity_threshold:.2f}",
                    flush=True,
                )
                continue
            store.save(spec)
            print(
                f"[coplay] seeded default trigger {seed['name']!r} "
                f"(detector={spec.detector_kind} action={spec.action_kind} "
                f"threshold={seed['similarity_threshold']:.2f} "
                f"cooldown={seed['cooldown_sec']:.0f}s)",
                flush=True,
            )

    def _build_strategy(self, mode: str):
        if mode == "human-takeover":
            return HumanTakeover(self._human_ids)
        if mode == "human-priority":
            return HumanPriority(self._human_ids)
        raise ValueError(
            f"unsupported mode {mode!r}; expected one of {self.SUPPORTED_MODES}"
        )

    def set_mode(self, mode: str) -> None:
        """Swap the mux strategy live. Raises ``ValueError`` for unknown modes."""
        strategy = self._build_strategy(mode)
        self._mux.strategy = strategy
        self._mode = mode

    @property
    def mode(self) -> str:
        return self._mode

    def set_ai_enabled(self, enabled: bool) -> None:
        self._ai_source.enabled = enabled

    @property
    def ai_enabled(self) -> bool:
        return self._ai_source.enabled

    def runtime_status(self) -> dict[str, Any]:
        return {
            "mode": self._mode,
            "supported_modes": list(self.SUPPORTED_MODES),
            "ai_enabled": self._ai_source.enabled,
        }

    def request_stop(self) -> None:
        self._stop_flag.set()

    def run(self) -> None:
        config = self.config
        period = 1.0 / max(config.tick_hz, 1.0)

        self._source.start()
        self._human.start()
        self._ai.start()
        if self._trigger_watcher is not None:
            self._trigger_watcher.start()
        if self._web_server is not None:
            self._web_server.start()
            self._web_server.wait_ready(timeout=5.0)

        print(f"[coplay] running (tick={config.tick_hz}Hz, post={self._post_url})")
        ticks = 0
        try:
            while not self._stop_flag.is_set():
                t_start = time.time()

                # Synthetic A-mash takes priority over the normal mux.
                # Macro playback also bypasses the mux but is driven by
                # MacroPlayer's own thread; mash is driven from this loop.
                mash_action = self._mash_controller.next_action()
                if mash_action is not None:
                    self._post_action(mash_action)
                elif not self._macro_player.is_playing:
                    # Sample sources directly (don't call ``mux.tick()``, which
                    # would re-call each source's ``latest()`` — and the web
                    # reader's ``latest()`` is destructive: it clears the
                    # button-press latch). Merge with the current strategy.
                    human_snap = self._human.latest()
                    ai_snap = self._ai_source.latest()
                    snapshots = [s for s in (human_snap, ai_snap) if s is not None]
                    action = self._mux.strategy.merge(snapshots)
                    self._mux._latest = action

                    self._post_action(action)

                    # Record the human's intent only — what the user
                    # actively pressed/deflected. AI fills are excluded
                    # so a 1s pause replays as a 1s pause, not as
                    # whatever the policy was suggesting at that moment.
                    rec_action = np.zeros(ACTION_DIM, dtype=np.float32)
                    if human_snap is not None:
                        if human_snap.mask is not None:
                            rec_action[human_snap.mask] = human_snap.action[human_snap.mask]
                        else:
                            rec_action[:] = human_snap.action
                    self._macro_recorder.append(rec_action, t_start)

                    if self._recorder_ctl.is_active:
                        frame = self._source.latest()
                        if frame is not None:
                            self._recorder_ctl.append(
                                SyncedFrame(
                                    timestamp=t_start,
                                    frame=frame.image,
                                    action=action,
                                    action_age=0.0,
                                )
                            )

                ticks += 1
                if ticks % int(config.tick_hz * 5) == 0:
                    print(f"[coplay] ticks={ticks} ai_inferences={self._ai.inference_count}")

                elapsed = time.time() - t_start
                if elapsed < period:
                    time.sleep(period - elapsed)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._stop_flag.set()
        if self._trigger_watcher is not None:
            with contextlib.suppress(Exception):
                self._trigger_watcher.stop()
        with contextlib.suppress(Exception):
            self._macro_player.stop()
        with contextlib.suppress(Exception):
            self._macro_recorder.cancel()
        try:
            self._ai.stop()
        finally:
            try:
                self._human.stop()
            finally:
                if self._web_server is not None:
                    with contextlib.suppress(Exception):
                        self._web_server.stop()
                self._source.stop()
                out = self._recorder_ctl.close()
                if out is not None:
                    print(f"[coplay] wrote {out}")
                with contextlib.suppress(Exception):
                    self._http.close()
