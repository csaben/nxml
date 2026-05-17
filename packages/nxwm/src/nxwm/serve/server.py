"""Transport-agnostic world model inference server.

The server holds *shared* resources (model weights, VAE, sampler, data_path)
and hands out :class:`Session` objects, each of which owns the *per-client*
rollout state (history, episode seed, detector buffer). The single-session
shape — one ``WorldModelServer`` exposes ``step``/``reseed``/``init_from_frames``
directly — is preserved via a default ``self._session`` so the in-process
client and the ZMQ transport see no change.

Transports (ZMQ, HTTP, in-process) wrap this class for delivery.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from nxml_core.checkpoint import load_checkpoint
from nxml_core.uri import resolve_model_uri

import nxwm.architectures  # noqa: F401  triggers architecture registration
from nxwm.core.registry import architecture_registry
from nxwm.env.detectors import Detector
from nxwm.env.episode_seed import EpisodeSeed
from nxwm.inference.flow_matching import FlowMatchingSampler
from nxwm.inference.vae import (
    LATENT_SCALE,
    decode_to_jpeg,
    decode_to_jpeg_and_bgr,
    encode_jpeg,
    load_vae,
)

DEFAULT_HISTORY_LENGTH = 10
DEFAULT_GOAL_OFFSET = 30
# How many past signals frames to retain for retroactive `decide` after a
# threshold edit. ~20s @ 30fps is enough to see whether a slider change
# would have flipped the readout in the recent past without storing the whole
# session.
DEFAULT_SIGNAL_BUFFER = 600


@dataclass
class ServerInfo:
    """Snapshot of server state. Returned by :meth:`WorldModelServer.info`."""

    current_model_path: str
    architecture: str
    config: dict[str, Any]
    history_length: int
    goal_offset: int
    flow_steps: int
    cfg_scale: float
    latent_scale: float
    current_episode_frame: int | None
    current_episode_file: str | None
    available_episodes: list[str] = field(default_factory=list)
    available_checkpoints: list[str] = field(default_factory=list)


class Session:
    """Per-client rollout state on top of a shared :class:`WorldModelServer`.

    All torch work routes back through the parent's model/vae/sampler. The
    parent's :attr:`WorldModelServer.lock` (an `asyncio.Lock` placeholder
    overridden by the HTTP transport) serializes steps across sessions, so
    a single GPU process can host many clients without weight contention.
    """

    def __init__(
        self,
        server: "WorldModelServer",
        *,
        detector: Detector | None = None,
        signal_buffer_size: int = DEFAULT_SIGNAL_BUFFER,
    ) -> None:
        self._server = server
        self.state: Any = None  # arch-specific RolloutState | None
        self.episode_seed: EpisodeSeed | None = None
        self.current_episode_file: str | None = None
        self.last_latent: torch.Tensor | None = None
        self.last_frame_bgr: np.ndarray | None = None
        self.detector = detector
        self._signal_buffer_size = signal_buffer_size
        self._signal_buffer: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step_latent(self, action: np.ndarray) -> torch.Tensor:
        """Advance one frame; return the predicted latent (C, H, W) on device."""
        srv = self._server
        if self.state is None:
            raise RuntimeError("session not seeded; call init_from_frames or reseed first")
        if action.shape != (srv.model.action_dims,):
            raise ValueError(
                f"action must be shape ({srv.model.action_dims},), got {action.shape}"
            )

        action_t = torch.from_numpy(action).float().to(srv.device)
        self.state, predicted = srv.model.step_rollout(
            self.state,
            action_t,
            sampler=srv.sampler,
            flow_steps=srv.flow_steps,
            cfg_scale=srv.cfg_scale,
        )
        if self.episode_seed is not None:
            new_goal = self.episode_seed.advance()
            self.state = srv.model.update_goal(self.state, new_goal)

        self.last_latent = predicted
        return predicted

    @torch.no_grad()
    def step(self, action: np.ndarray) -> bytes:
        """Advance one frame; return predicted frame as JPEG bytes."""
        latent = self.step_latent(action)
        vae = self._server._ensure_vae()
        return decode_to_jpeg(latent.unsqueeze(0), vae)

    @torch.no_grad()
    def step_with_telemetry(
        self, action: np.ndarray
    ) -> tuple[bytes, dict[str, Any]]:
        """Step + run the configured detector; return (jpeg, telemetry)."""
        latent = self.step_latent(action)
        vae = self._server._ensure_vae()

        sigs: dict[str, float] = {}
        telemetry: dict[str, Any] = {}
        if self.detector is not None:
            bgr, jpeg = decode_to_jpeg_and_bgr(latent.unsqueeze(0), vae)
            self.last_frame_bgr = bgr
            sigs = self.detector.signals(bgr)
            self._signal_buffer.append(sigs)
            if len(self._signal_buffer) > self._signal_buffer_size:
                del self._signal_buffer[: len(self._signal_buffer) - self._signal_buffer_size]
            state = self.detector.decide(self._signal_buffer)
            telemetry = {
                "name": self.detector.name,
                "signals": sigs,
                "state": state,
                "params": self.detector.params(),
            }
        else:
            jpeg = decode_to_jpeg(latent.unsqueeze(0), vae)

        return jpeg, telemetry

    def init_from_frames(
        self, frames_jpeg: list[bytes], goal_jpeg: bytes | None = None
    ) -> None:
        """Initialize from N JPEG frames + optional goal JPEG."""
        srv = self._server
        if not (1 <= len(frames_jpeg) <= srv.history_length):
            raise ValueError(
                f"num_frames must be in [1, {srv.history_length}], got {len(frames_jpeg)}"
            )
        vae = srv._ensure_vae()

        latents: list[torch.Tensor] = [
            encode_jpeg(b, vae, srv.device) for b in frames_jpeg
        ]
        # Pad to history_length by repeating the oldest frame at the front.
        while len(latents) < srv.history_length:
            latents.insert(0, latents[0].clone())

        action_dim = srv.model.action_dims
        actions = torch.zeros(srv.history_length, action_dim, device=srv.device)
        history = torch.stack(latents)

        goal_latent = (
            encode_jpeg(goal_jpeg, vae, srv.device)
            if goal_jpeg is not None
            else latents[-1].clone()
        )

        self.state = srv.model.init_rollout_state(history, actions, goal_latent)
        self.episode_seed = None
        self.current_episode_file = None
        self.last_frame_bgr = None
        if self.detector is not None:
            self.detector.reset()
        self._signal_buffer.clear()

    def reseed(self, episode_filename: str, start_frame: int = 100) -> None:
        """Reseed from a named episode under ``server.data_path``."""
        srv = self._server
        if srv.data_path is None:
            raise RuntimeError("data_path is not configured; cannot reseed by filename")
        npz_files = sorted(srv.data_path.glob("*.npz"))
        match = next((f for f in npz_files if f.name == episode_filename), None)
        if match is None:
            raise FileNotFoundError(f"episode not found in {srv.data_path}: {episode_filename}")
        self._seed_from_npz(match, start_frame=start_frame)

    def invalidate(self) -> None:
        """Drop rollout state (e.g., after the server hot-swaps its model)."""
        self.state = None
        self.episode_seed = None
        self.current_episode_file = None
        self.last_latent = None
        self.last_frame_bgr = None
        if self.detector is not None:
            self.detector.reset()
        self._signal_buffer.clear()

    # ------------------------------------------------------------------
    # Detector
    # ------------------------------------------------------------------

    def detector_config(self) -> dict[str, Any]:
        if self.detector is None:
            return {}
        return {
            "name": self.detector.name,
            "params": self.detector.params(),
            "schema": self.detector.schema(),
            "static_meta": self.detector.static_meta(),
        }

    def apply_detector_params(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.detector is None:
            raise RuntimeError("no detector configured on this session")
        self.detector.update_params(**params)
        # Streak meaning changed under the user's feet — clear it. Retroactive
        # `decide` over the ring buffer gives the slider feedback immediately;
        # the next step rebuilds the live streak from there.
        self.detector.reset()
        state = self.detector.decide(self._signal_buffer)
        return {"state": state, "params": self.detector.params()}

    def reset_detector(self) -> None:
        if self.detector is None:
            raise RuntimeError("no detector configured on this session")
        self.detector.reset()
        self._signal_buffer.clear()

    def detector_debug_image(self) -> np.ndarray | None:
        if self.detector is None or self.last_frame_bgr is None:
            return None
        return self.detector.debug_image(self.last_frame_bgr)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _seed_from_npz(self, npz_path: Path, *, start_frame: int) -> None:
        srv = self._server
        seed, init_lat, init_act = EpisodeSeed.from_npz(
            npz_path,
            start_frame=start_frame,
            history_length=srv.history_length,
            goal_offset=srv.goal_offset,
            device=srv.device,
        )
        goal_latent = seed.current_goal()
        self.state = srv.model.init_rollout_state(init_lat, init_act, goal_latent)
        self.episode_seed = seed
        self.current_episode_file = npz_path.name
        self.last_latent = None
        self.last_frame_bgr = None
        if self.detector is not None:
            self.detector.reset()
        self._signal_buffer.clear()

    def _auto_seed_from_data_path(self) -> None:
        srv = self._server
        if srv.data_path is None or not srv.data_path.exists():
            return
        npz_files = sorted(srv.data_path.glob("*.npz"))
        if not npz_files:
            return
        self._seed_from_npz(npz_files[0], start_frame=0)


class WorldModelServer:
    def __init__(
        self,
        *,
        model_path: str | Path,
        device: str | torch.device = "cuda",
        flow_steps: int = 5,
        cfg_scale: float = 1.0,
        history_length: int = DEFAULT_HISTORY_LENGTH,
        goal_offset: int = DEFAULT_GOAL_OFFSET,
        data_path: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        vae_path: str | Path | None = None,
        load_vae_eagerly: bool = True,
        detector: Detector | None = None,
        signal_buffer_size: int = DEFAULT_SIGNAL_BUFFER,
    ):
        self.device = torch.device(device)
        self.flow_steps = flow_steps
        self.cfg_scale = cfg_scale
        self.history_length = history_length
        self.goal_offset = goal_offset
        self.data_path = Path(data_path) if data_path is not None else None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        # VAE path resolution order: explicit kwarg → VAE_PATH env
        # (lowercase ``vae_path`` honored too) → HuggingFace default in load_vae.
        self.vae_path = self._resolve_vae_path(vae_path)

        resolved = resolve_model_uri(str(model_path))
        self.current_model_path = str(resolved.resolve())
        self.model, self.config, self._ckpt_meta = load_checkpoint(
            resolved, architecture_registry, device=self.device
        )
        self.model.eval()
        self.architecture: str = self._ckpt_meta["architecture"]

        self.sampler = FlowMatchingSampler()
        self._vae = None
        if load_vae_eagerly:
            self._ensure_vae()

        # Default session — preserves the byte-identical surface for in-process
        # and ZMQ callers. HTTP transport creates additional sessions via
        # `new_session()`; they're tracked weakly so reload() can invalidate
        # them without keeping them alive past the WS lifetime.
        self._default_signal_buffer_size = signal_buffer_size
        self._sessions: weakref.WeakSet[Session] = weakref.WeakSet()
        self._session = self.new_session(
            detector=detector, signal_buffer_size=signal_buffer_size
        )

        if self.data_path is not None:
            self._session._auto_seed_from_data_path()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def new_session(
        self,
        *,
        detector: Detector | None = None,
        detector_factory: Callable[[], Detector | None] | None = None,
        signal_buffer_size: int | None = None,
    ) -> Session:
        """Build a fresh per-client session. The HTTP transport calls this
        once per WebSocket connection; ZMQ + in-process keep using
        :attr:`self._session`.

        ``detector_factory`` is preferred over a literal ``detector``
        argument when the caller wants per-session isolation — a single
        detector instance shared across sessions would have a shared
        ``_signal_buffer`` on the detector itself.
        """
        if detector is None and detector_factory is not None:
            detector = detector_factory()
        sess = Session(
            self,
            detector=detector,
            signal_buffer_size=signal_buffer_size or self._default_signal_buffer_size,
        )
        self._sessions.add(sess)
        return sess

    # ------------------------------------------------------------------
    # Back-compat surface — delegates to the default session.
    # ------------------------------------------------------------------

    def step_latent(self, action: np.ndarray) -> torch.Tensor:
        return self._session.step_latent(action)

    def step(self, action: np.ndarray) -> bytes:
        return self._session.step(action)

    def step_with_telemetry(
        self, action: np.ndarray
    ) -> tuple[bytes, dict[str, Any]]:
        return self._session.step_with_telemetry(action)

    def init_from_frames(
        self, frames_jpeg: list[bytes], goal_jpeg: bytes | None = None
    ) -> None:
        self._session.init_from_frames(frames_jpeg, goal_jpeg=goal_jpeg)

    def reseed(self, episode_filename: str, start_frame: int = 100) -> None:
        self._session.reseed(episode_filename, start_frame=start_frame)

    def detector_config(self) -> dict[str, Any]:
        return self._session.detector_config()

    def apply_detector_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._session.apply_detector_params(params)

    def reset_detector(self) -> None:
        self._session.reset_detector()

    def detector_debug_image(self) -> np.ndarray | None:
        return self._session.detector_debug_image()

    # Properties so existing tests that read attributes still work.
    @property
    def state(self) -> Any:
        return self._session.state

    @state.setter
    def state(self, value: Any) -> None:
        self._session.state = value

    @property
    def episode_seed(self) -> EpisodeSeed | None:
        return self._session.episode_seed

    @property
    def current_episode_file(self) -> str | None:
        return self._session.current_episode_file

    @property
    def last_latent(self) -> torch.Tensor | None:
        return self._session.last_latent

    @property
    def last_frame_bgr(self) -> np.ndarray | None:
        return self._session.last_frame_bgr

    @property
    def detector(self) -> Detector | None:
        return self._session.detector

    # ------------------------------------------------------------------
    # Server-level operations
    # ------------------------------------------------------------------

    def reload(self, model_path: str | Path) -> None:
        """Hot-swap the model. Invalidates *all* live sessions."""
        resolved = resolve_model_uri(str(model_path))
        new_model, new_config, ckpt_meta = load_checkpoint(
            resolved, architecture_registry, device=self.device
        )
        new_model.eval()
        self.model = new_model
        self.config = new_config
        self._ckpt_meta = ckpt_meta
        self.architecture = ckpt_meta["architecture"]
        self.current_model_path = str(resolved.resolve())
        for sess in list(self._sessions):
            sess.invalidate()
        # If a data_path is configured, re-seed the default session from the
        # first available episode (back-compat behavior).
        if self.data_path is not None:
            self._session._auto_seed_from_data_path()

    def info(self) -> ServerInfo:
        from dataclasses import asdict

        # load_checkpoint always returns an instance of the registered config_cls
        # (a dataclass), so asdict is safe.
        config_dict: dict[str, Any] = asdict(self.config)  # type: ignore[arg-type]
        episodes = self._list_episodes()
        checkpoints = self._list_checkpoints()
        return ServerInfo(
            current_model_path=self.current_model_path,
            architecture=self.architecture,
            config=config_dict,
            history_length=self.history_length,
            goal_offset=self.goal_offset,
            flow_steps=self.flow_steps,
            cfg_scale=self.cfg_scale,
            latent_scale=LATENT_SCALE,
            current_episode_frame=self._session.episode_seed.current_frame
            if self._session.episode_seed is not None
            else None,
            current_episode_file=self._session.current_episode_file,
            available_episodes=episodes,
            available_checkpoints=checkpoints,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_vae(self) -> Any:
        if self._vae is None:
            kwargs: dict[str, Any] = {"device": self.device}
            if self.vae_path is not None:
                kwargs["path"] = str(self.vae_path)
            self._vae = load_vae(**kwargs)
        return self._vae

    @staticmethod
    def _resolve_vae_path(arg: str | Path | None) -> str | None:
        if arg is not None:
            return str(arg)
        import os

        for env_key in ("VAE_PATH", "vae_path"):
            v = os.environ.get(env_key)
            if v:
                return v
        return None

    def _list_episodes(self) -> list[str]:
        if self.data_path is None or not self.data_path.exists():
            return []
        return sorted(f.name for f in self.data_path.glob("*.npz"))

    def _list_checkpoints(self) -> list[str]:
        if self.checkpoint_dir is None or not self.checkpoint_dir.exists():
            return []
        return sorted(str(f) for f in self.checkpoint_dir.rglob("best*.pt"))
