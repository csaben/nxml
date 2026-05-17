"""Transport-agnostic policy inference server.

Holds a loaded policy + cached metadata. Methods:

  - :meth:`predict` — run the policy on a sequence of latents, return a
    26-dim action vector. Stateless: callers manage their own sliding window.
  - :meth:`reload` — swap to a different policy checkpoint at runtime.
  - :meth:`info` — describe the loaded policy (architecture, sequence
    length, expected latent shape, action dim).

The class is fully usable without any transport — it's just Python objects
with methods. Transports (ZMQ, HTTP, in-process) wrap it for delivery.
Operationally parallels ``nxwm.serve.server.WorldModelServer``.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from nxml_core.checkpoint import load_checkpoint
from nxml_core.uri import resolve_model_uri

import nxrl  # noqa: F401  triggers policy registration
from nxrl.core.registry import policy_registry


def _derive_latent_shape(policy) -> tuple[int, int, int]:
    """Pull ``(C, H, W)`` from whichever config the policy carries.

    BC architectures store ``latent_channels``/``latent_height``/``latent_width``
    on ``policy.config``; ``ppo_policy_v1`` exposes the same fields under
    ``policy.base_config``.
    """
    cfg = getattr(policy, "config", None)
    if cfg is not None and hasattr(cfg, "latent_channels"):
        return (int(cfg.latent_channels), int(cfg.latent_height), int(cfg.latent_width))
    base = getattr(policy, "base_config", None)
    if base is not None and hasattr(base, "latent_channels"):
        return (int(base.latent_channels), int(base.latent_height), int(base.latent_width))
    # bc_lstm_v1 / bc_transformer_v1 ship with these on their dataclass; PPO
    # ships them on the wrapped base config. If we land here, the policy's
    # config schema changed — surface the failure rather than silently guess.
    raise ValueError(
        f"could not derive latent shape from {type(policy).__name__}'s config"
    )


@dataclass
class PolicyServerInfo:
    current_model_path: str
    architecture: str
    config: dict[str, Any]
    sequence_length: int
    latent_shape: tuple[int, int, int]
    action_dim: int = 26
    algorithm: str | None = None
    available_checkpoints: list[str] = field(default_factory=list)


class PolicyServer:
    def __init__(
        self,
        *,
        model_path: str | Path,
        device: str | torch.device = "cuda",
        checkpoint_dir: str | Path | None = None,
        enable_frame_mode: bool = False,
        vae_path: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        resolved = resolve_model_uri(str(model_path))
        self.current_model_path = str(resolved.resolve())
        self._load(resolved)

        # Frame-mode (MODE_PREDICT_FRAME): server owns the VAE and a sliding
        # latent window. Single-client by design — see transports/zmq.py.
        self._frame_mode = enable_frame_mode
        self._vae = None
        self._vae_path = vae_path or "stabilityai/sd-vae-ft-mse"
        self._frame_window: deque[torch.Tensor] | None = None
        self._frame_lock = threading.Lock()
        if enable_frame_mode:
            self._init_frame_mode()

    def _init_frame_mode(self) -> None:
        from nxwm.inference.vae import load_vae

        self._vae = load_vae(self._vae_path, device=self.device)
        self._vae.eval()
        self._frame_window = deque(maxlen=self.sequence_length)

    def _load(self, ckpt_path: Path) -> None:
        model, config, ckpt = load_checkpoint(ckpt_path, policy_registry, device=self.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.config = config
        self._ckpt_meta = ckpt
        self.architecture = ckpt["architecture"]
        self.algorithm = ckpt.get("algorithm")
        self.sequence_length = int(model.sequence_length)
        self.latent_shape = _derive_latent_shape(model)

    @torch.no_grad()
    def predict(self, latents: np.ndarray | torch.Tensor) -> np.ndarray:
        """Run the policy on a single latent window. Returns a ``(26,)``
        ``float32`` numpy action vector. For PPO policies the call is
        deterministic (mean stick + thresholded buttons via
        :func:`forward`), since the server is meant for live deployment
        rather than exploration; callers that want sampling can use
        :meth:`predict_with_sampling`.
        """
        x = self._coerce_input(latents)
        out = self.model(x)
        # ``out`` is either ``(B, 26)`` (BC) or ``(action_out, value)`` (PPO).
        if isinstance(out, tuple):
            out = out[0]
        return self._post_process(out)

    @torch.no_grad()
    def predict_with_sampling(self, latents: np.ndarray | torch.Tensor) -> np.ndarray:
        """For PPO policies: sample stick from the learned Gaussian, sample
        buttons from the Bernoulli, return the resulting action. BC policies
        fall back to deterministic ``predict`` since they have no sampler."""
        if not hasattr(self.model, "get_action_and_value"):
            return self.predict(latents)
        x = self._coerce_input(latents)
        action, _, _, _ = self.model.get_action_and_value(x)
        return action.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    @torch.no_grad()
    def predict_frame(self, jpeg_bytes: bytes) -> np.ndarray | None:
        """Frame-mode entry point: decode JPEG, VAE-encode, append to the
        sliding window, predict if the window is full. Returns ``None``
        while warming up (window not yet at ``sequence_length``).

        Single-client by construction: the window is shared server-wide,
        so two clients alternating frames would corrupt each other's
        history. The transport keeps this honest by serializing requests.
        """
        if not self._frame_mode or self._vae is None or self._frame_window is None:
            raise RuntimeError(
                "frame mode not enabled on this server (start with enable_frame_mode=True)"
            )
        from nxwm.inference.vae import encode_jpeg

        latent = encode_jpeg(jpeg_bytes, self._vae, self.device)
        with self._frame_lock:
            self._frame_window.append(latent)
            if len(self._frame_window) < self.sequence_length:
                return None
            window = torch.stack(list(self._frame_window), dim=0)  # (T, C, H, W)
        x = window.unsqueeze(0).to(self.device, dtype=torch.float32)
        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        return self._post_process(out)

    def reset_frame_window(self) -> None:
        """Drop any buffered latents — call between sessions or on reconnect."""
        if self._frame_window is not None:
            with self._frame_lock:
                self._frame_window.clear()

    def reload(self, model_path: str | Path) -> None:
        resolved = resolve_model_uri(str(model_path))
        self._load(resolved)
        self.current_model_path = str(resolved.resolve())
        # sequence_length may have changed — resize window if frame mode is on.
        if self._frame_mode and self._frame_window is not None:
            with self._frame_lock:
                self._frame_window = deque(maxlen=self.sequence_length)

    def info(self) -> PolicyServerInfo:
        from dataclasses import asdict, is_dataclass

        if is_dataclass(self.config) and not isinstance(self.config, type):
            cfg_dict: dict[str, Any] = asdict(self.config)
        else:
            cfg_dict = dict(self.config)  # type: ignore[arg-type]
        ckpts: list[str] = []
        if self.checkpoint_dir is not None and self.checkpoint_dir.is_dir():
            ckpts = sorted(p.name for p in self.checkpoint_dir.glob("*.pt"))
        return PolicyServerInfo(
            current_model_path=self.current_model_path,
            architecture=self.architecture,
            config=cfg_dict,
            sequence_length=self.sequence_length,
            latent_shape=self.latent_shape,
            algorithm=self.algorithm,
            available_checkpoints=ckpts,
        )

    # ------------------------------------------------------------------

    def _coerce_input(self, latents: np.ndarray | torch.Tensor) -> torch.Tensor:
        x = torch.from_numpy(latents) if isinstance(latents, np.ndarray) else latents
        if x.dim() == 4:
            x = x.unsqueeze(0)
        if x.dim() != 5:
            raise ValueError(f"latents must be (T,C,H,W) or (B,T,C,H,W); got {tuple(x.shape)}")
        _b, t, c, h, w = x.shape
        if (c, h, w) != self.latent_shape:
            raise ValueError(
                f"latent shape {(c, h, w)} != policy expects {self.latent_shape}"
            )
        if t != self.sequence_length:
            raise ValueError(
                f"sequence length {t} != policy expects {self.sequence_length}"
            )
        return x.to(self.device, dtype=torch.float32)

    def _post_process(self, raw_out: torch.Tensor) -> np.ndarray:
        """Convert raw policy output (sticks tanh'd, buttons logits) to the
        canonical 26-dim action vector with buttons thresholded to {0, 1}.
        """
        out = raw_out[0].detach().cpu()
        sticks = out[:4].clamp(-1.0, 1.0)
        buttons = (torch.sigmoid(out[4:]) > 0.5).float()
        return torch.cat([sticks, buttons], dim=0).numpy().astype(np.float32, copy=False)
