"""Gamepad-playable browser session backed by the RAEv2 codec.

Implements nxwm's ``WorldModelClient`` protocol so nxwm's existing gamepad UI (Xbox /
Standard Gamepad -> 26-dim Switch action mapping, play loop, reseed picker) drives it
unchanged. Until the MIRA world-model port lands, the stepper is a **codec roundtrip**:
each step advances through a real episode and streams the *decoded* frame, so the whole
browser <- JPEG <- decode <- GPU loop is exercised end-to-end (actions are captured and
echoed in telemetry, but don't influence dynamics yet). The world model later replaces
``_advance`` with real action-conditioned latent dynamics; everything else stays.

The UI's "reload model" button hot-swaps checkpoints, so a running session can follow
training progress (point it at checkpoints/codec/run_001/checkpoint-<step>).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from nxwm_mira.codec.codec_model import VideoCodec, preprocess_video
from nxwm_mira.data.actions import SwitchActions
from nxwm_mira.data.batch import VideoActionBatch
from nxwm_mira.data.za_dataset import (
    EpisodeInfo,
    _decode_clip,
    load_actions,
    load_episodes,
    pool_actions,
)
from nxwm_mira.training.checkpoints import resolve_checkpoint

logger = logging.getLogger(__name__)


def _jpeg_frames(frames01: torch.Tensor, quality: int) -> list[bytes]:
    """(T, 3, H, W) float in [0,1] RGB -> list of JPEG bytes."""
    frames = (frames01.float().clamp(0, 1) * 255).to(torch.uint8)
    out = []
    for frame in frames:
        bgr = frame.permute(1, 2, 0).cpu().numpy()[..., ::-1]
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            out.append(buf.tobytes())
    return out


class CodecPlayClient:
    """nxwm ``WorldModelClient`` implementation streaming codec-roundtrip frames."""

    def __init__(
        self,
        checkpoint: str | Path,
        data_root: str | Path,
        device: str | None = None,
        jpeg_quality: int = 85,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.data_root = Path(data_root)
        self.jpeg_quality = jpeg_quality
        self._lock = threading.Lock()

        self.episodes: list[EpisodeInfo] = [
            ep for ep in load_episodes(self.data_root) if ep.frame_count >= 64
        ]
        self._by_name = {ep.video_path.name: ep for ep in self.episodes}

        self.checkpoint_path = str(resolve_checkpoint(checkpoint))
        self.codec = self._load_codec(self.checkpoint_path)

        self.episode: EpisodeInfo = self.episodes[0]
        self.cursor = 100
        self.last_action: np.ndarray | None = None
        self._buffer: list[bytes] = []  # decoded JPEG frames, ready to serve

    def _load_codec(self, path: str) -> VideoCodec:
        codec = VideoCodec.load_from_checkpoint(path, device=self.device)
        codec.eval()
        logger.info(f"Loaded codec {path} on {self.device}")
        return codec

    # ---- stepping ------------------------------------------------------------

    @property
    def _frame_stride(self) -> int:
        return max(1, round(30 / self.codec.config.encoder.video.fps))

    def _advance(self) -> None:
        """Refill the frame buffer: decode the next codec window of the episode.

        This is the seam the world model replaces: instead of encode->decode of real
        frames, it will roll latent dynamics forward from the action history.
        """
        clip_len = self.codec.config.encoder.video.timesteps
        stride = self._frame_stride
        span = clip_len * stride
        if self.cursor + span >= self.episode.frame_count:
            self.cursor = 0  # loop the episode
        indices = list(range(self.cursor, self.cursor + span, stride))
        self.cursor += span

        video = _decode_clip(self.episode.video_path, indices)  # (T, 3, H, W) uint8
        with torch.no_grad():
            batch = preprocess_video(
                video[None].to(self.device),
                target_h=self.codec.config.encoder.video.height,
                target_w=self.codec.config.encoder.video.width,
                aspect_mode=self.codec.config.encoder.aspect_mode,
            )
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, enc = self.codec.encode(batch)
                    decoded = self.codec.decode(enc.z)
            else:
                _, enc = self.codec.encode(batch)
                decoded = self.codec.decode(enc.z)

        frames = ((decoded[0].float() * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
        for frame in frames:  # (3, H, W) RGB -> JPEG (BGR for cv2)
            bgr = frame.permute(1, 2, 0).cpu().numpy()[..., ::-1]
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                self._buffer.append(buf.tobytes())

    def step(self, action: np.ndarray) -> bytes:
        return self.step_with_telemetry(action)[0]

    def step_with_telemetry(self, action: np.ndarray) -> tuple[bytes, dict[str, Any]]:
        with self._lock:
            self.last_action = np.asarray(action, dtype=np.float32)
            if not self._buffer:
                self._advance()
            jpeg = self._buffer.pop(0)
            telemetry = {
                "mode": "codec-roundtrip (world model pending)",
                "episode": self.episode.video_path.name,
                "frame": self.cursor - len(self._buffer) * self._frame_stride,
                "sticks": [round(float(x), 2) for x in self.last_action[:4]],
                "buttons_down": int((self.last_action[4:] > 0.5).sum()),
            }
            return jpeg, telemetry

    # ---- session management ---------------------------------------------------

    def reseed(self, file: str, start_frame: int = 100) -> None:
        with self._lock:
            episode = self._by_name.get(file)
            if episode is None:
                raise ValueError(f"Unknown episode {file!r}")
            self.episode = episode
            self.cursor = min(start_frame, max(0, episode.frame_count - 64))
            self._buffer.clear()

    def reload(self, model_path: str) -> None:
        with self._lock:
            path = str(resolve_checkpoint(model_path))
            self.codec = self._load_codec(path)
            self.checkpoint_path = path
            self._buffer.clear()

    def info(self) -> dict[str, Any]:
        return {
            "current_model_path": self.checkpoint_path,
            "available_episodes": sorted(self._by_name),
            "available_checkpoints": [],
            "current_episode_file": self.episode.video_path.name,
            "current_episode_frame": self.cursor,
            "architecture": "raev2-codec-roundtrip",
            "config": {"device": self.device, "fps": self.codec.config.encoder.video.fps},
        }

    def init_from_frames(
        self, frames_jpeg: list[bytes], goal_jpeg: bytes | None = None
    ) -> None:
        raise NotImplementedError("codec-roundtrip sessions seed from episodes (reseed)")

    # ---- detector protocol (none configured) -----------------------------------

    def detector_config(self) -> dict[str, Any]:
        return {"detector": None}

    def apply_detector_params(self, params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("no detector configured")

    def reset_detector(self) -> None:
        raise RuntimeError("no detector configured")

    def detector_debug_image(self) -> np.ndarray | None:
        return None

    def close(self) -> None:
        pass


class WorldModelPlayClient(CodecPlayClient):
    """The real thing: gamepad actions drive the world model's latent dynamics.

    Seeds a context window from a recorded episode (frames + its recorded actions), then
    each latent step feeds the live controller state into ``streaming_inference_step``
    (kv-cached) and decodes the generated latent to frames. One latent step consumes
    ``action_temporal_downsampling`` actions and yields ``temporal_downsampling`` frames,
    so the held control is repeated across the chunk (the MIRA inference convention).
    """

    def __init__(
        self,
        wm_checkpoint: str | Path,
        data_root: str | Path,
        device: str | None = None,
        jpeg_quality: int = 85,
        n_diffusion_steps: int = 10,
    ) -> None:
        from nxwm_mira.world_model.config import WorldModelInferenceConfig
        from nxwm_mira.world_model.latent_world_model import LatentWorldModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.data_root = Path(data_root)
        self.jpeg_quality = jpeg_quality
        self._lock = threading.Lock()

        self.checkpoint_path = str(resolve_checkpoint(wm_checkpoint))
        self.wm = LatentWorldModel.load_from_checkpoint(self.checkpoint_path, device=self.device)
        self.wm.eval()
        self.codec = self.wm.codec
        self.inference_config = WorldModelInferenceConfig(n_diffusion_steps=n_diffusion_steps)
        logger.info(f"Loaded world model {self.checkpoint_path} on {self.device}")

        min_frames = self.wm.n_context_frames * self._frame_stride + 1
        self.episodes = [
            ep for ep in load_episodes(self.data_root) if ep.frame_count >= min_frames + 64
        ]
        self._by_name = {ep.video_path.name: ep for ep in self.episodes}
        self._actions_by_episode = load_actions(self.data_root)

        self.episode = self.episodes[0]
        self.cursor = 100
        self.last_action: np.ndarray | None = None
        self._buffer: list[bytes] = []
        self._z: torch.Tensor | None = None
        self._kv = None
        self._actions_history: torch.Tensor | None = None  # (1, T, 26) at video fps
        self.reseed(self.episode.video_path.name, self.cursor)

    @property
    def _frame_stride(self) -> int:
        return max(1, round(30 / self.wm.config.video.fps))

    def reseed(self, file: str, start_frame: int = 100) -> None:
        with self._lock:
            episode = self._by_name.get(file)
            if episode is None:
                raise ValueError(f"Unknown episode {file!r}")
            stride = self._frame_stride
            n_ctx = self.wm.n_context_frames
            start_frame = min(start_frame, max(0, episode.frame_count - n_ctx * stride - 1))
            indices = list(range(start_frame, start_frame + n_ctx * stride, stride))

            video = _decode_clip(episode.video_path, indices)[None]  # (1, T, 3, H, W)
            actions = pool_actions(
                self._actions_by_episode[episode.episode_index], indices, stride
            )[None]  # (1, T, 26)

            with torch.no_grad(), self._autocast():
                batch = VideoActionBatch(
                    video=video, actions=SwitchActions(actions)
                ).to(self.device)
                self.wm.codec.preprocess_batch(batch)
                self._z = self.wm.encode_video(batch).clone()
            self._kv = None
            self._actions_history = actions.to(self.device)
            self.episode = episode
            self.cursor = start_frame + n_ctx * stride
            self._buffer.clear()

    def _autocast(self):
        if self.device.startswith("cuda"):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        import contextlib

        return contextlib.nullcontext()

    def step_with_telemetry(self, action: np.ndarray) -> tuple[bytes, dict[str, Any]]:
        with self._lock:
            self.last_action = np.asarray(action, dtype=np.float32)
            if not self._buffer:
                self._latent_step()
            jpeg = self._buffer.pop(0)
            telemetry = {
                "mode": "world-model",
                "episode": self.episode.video_path.name,
                "sticks": [round(float(x), 2) for x in self.last_action[:4]],
                "buttons_down": int((self.last_action[4:] > 0.5).sum()),
                "context_latents": int(self._z.shape[1]) if self._z is not None else 0,
            }
            return jpeg, telemetry

    def _latent_step(self) -> None:
        assert self._z is not None and self._actions_history is not None
        atd = self.wm.action_temporal_downsampling
        live = torch.from_numpy(self.last_action).to(self.device).view(1, 1, -1)
        self._actions_history = torch.cat(
            [self._actions_history, live.expand(1, atd, -1)], dim=1
        )[:, -4096:]

        with torch.no_grad(), self._autocast():
            z_next, self._kv = self.wm.streaming_inference_step(
                self._z,
                SwitchActions(self._actions_history),
                streaming_kv_cache=self._kv,
                config=self.inference_config,
            )
            frames = self.wm.decode_to_video(z_next[:, -1:])[0]  # (td, 3, H, W) in [0,1]
        self._z = z_next
        self._buffer.extend(_jpeg_frames(frames, self.jpeg_quality))

    def reload(self, model_path: str) -> None:
        from nxwm_mira.world_model.latent_world_model import LatentWorldModel

        with self._lock:
            path = str(resolve_checkpoint(model_path))
            self.wm = LatentWorldModel.load_from_checkpoint(path, device=self.device)
            self.wm.eval()
            self.codec = self.wm.codec
            self.checkpoint_path = path
        # Re-seed outside the lock (reseed takes it) so the new model gets fresh context.
        self.reseed(self.episode.video_path.name, max(0, self.cursor - 200))

    def info(self) -> dict[str, Any]:
        return {
            "current_model_path": self.checkpoint_path,
            "available_episodes": sorted(self._by_name),
            "available_checkpoints": [],
            "current_episode_file": self.episode.video_path.name,
            "current_episode_frame": self.cursor,
            "architecture": "mira-latent-world-model",
            "config": {
                "device": self.device,
                "fps": self.wm.config.video.fps,
                "n_context_frames": self.wm.n_context_frames,
                "flow_steps": self.inference_config.n_diffusion_steps,
            },
        }
