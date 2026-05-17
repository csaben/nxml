from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class DiTRolloutState:
    """Immutable rollout state for the dit_v1 world model.

    Tensors are unbatched (no leading batch dim) and live on the model's device:
      - ``latent_history``: (T, C, H, W) — past T frames in scaled-latent space.
      - ``action_history``: (T, A) — actions paired with each history frame, where
        action_history[t] is the action that produced latent_history[t+1] (or, for
        the most recent slot, the action that produced the latest predicted frame).
      - ``goal_latent``: (C, H, W) — goal frame in scaled-latent space.
    """

    latent_history: torch.Tensor
    action_history: torch.Tensor
    goal_latent: torch.Tensor

    def with_goal(self, goal_latent: torch.Tensor) -> DiTRolloutState:
        return replace(self, goal_latent=goal_latent)
