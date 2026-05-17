"""Episode-backed seed for world-model rollouts.

Loads a real episode from an mmap-backed ``.npz`` and exposes the initial
history (first ``history_length`` frames) plus a receding goal latent at
``current_frame + goal_offset``. Latents are pre-scaled to match the
training-time convention (``LATENT_SCALE``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from nxwm.inference.vae import LATENT_SCALE


@dataclass
class EpisodeSeed:
    """Stateful pointer into a single ``.npz`` episode.

    The episode contains ``latents`` (T, C, H, W) and ``actions`` (T, A) arrays.
    """

    latents: np.ndarray  # mmap view, (T, C, H, W)
    actions: np.ndarray  # mmap view, (T, A)
    current_frame: int  # index of the most recent frame in the rolling history
    goal_offset: int
    device: torch.device | str

    @classmethod
    def from_npz(
        cls,
        path: Path,
        *,
        start_frame: int,
        history_length: int,
        goal_offset: int,
        device: torch.device | str = "cpu",
    ) -> tuple[EpisodeSeed, torch.Tensor, torch.Tensor]:
        """Open ``path`` mmap'd, build initial latent + action history.

        Returns ``(seed, initial_latents, initial_actions)`` where the tensors
        have shape ``(history_length, ...)`` and have been moved to ``device``.
        Latents are scaled by ``LATENT_SCALE``; actions are unscaled.
        """
        data = np.load(path, mmap_mode="r")
        latents = data["latents"]
        actions = data["actions"]

        total_frames = latents.shape[0]
        min_required = history_length + goal_offset + 1
        if total_frames < min_required:
            raise ValueError(
                f"{path}: episode has {total_frames} frames, need at least {min_required}"
            )
        if start_frame < 0 or start_frame > total_frames - min_required:
            raise ValueError(
                f"{path}: start_frame={start_frame} out of range "
                f"[0, {total_frames - min_required}]"
            )

        history_slice_latents = latents[start_frame : start_frame + history_length]
        history_slice_actions = actions[start_frame : start_frame + history_length]

        initial_latents = (
            torch.from_numpy(history_slice_latents.copy()).float().to(device) * LATENT_SCALE
        )
        initial_actions = torch.from_numpy(history_slice_actions.copy()).float().to(device)

        seed = cls(
            latents=latents,
            actions=actions,
            current_frame=start_frame + history_length - 1,
            goal_offset=goal_offset,
            device=device,
        )
        return seed, initial_latents, initial_actions

    def current_goal(self) -> torch.Tensor:
        """Latent at ``current_frame + goal_offset``, scaled, on device.

        Clamps to the last valid frame if we've run past the episode end.
        """
        goal_idx = self.current_frame + self.goal_offset
        if goal_idx >= self.latents.shape[0]:
            goal_idx = self.latents.shape[0] - 1
        return (
            torch.from_numpy(self.latents[goal_idx].copy()).float().to(self.device) * LATENT_SCALE
        )

    def advance(self) -> torch.Tensor:
        """Increment current_frame and return the new goal latent."""
        self.current_frame += 1
        return self.current_goal()
