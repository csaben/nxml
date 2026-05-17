"""Reward protocol used by nxrl PPO and other consumers of world-model rollouts.

A ``Reward`` is invoked once per rollout step with the action the policy
just emitted, the world model's predicted next latent (batch dim 1), and an
``info`` dict carrying step metadata (currently ``{"step": int, "n_frames":
int}``; specific reward implementations may add their own keys).

The return is ``(reward_value, components)`` where ``components`` is a flat
dict of named contributions for logging. Implementations can also stuff
``"__terminal__": True`` into ``components`` to end the rollout early (the
"lock-on" pattern in the pokemon_za stack).

Game-specific rewards live in ``nxml-games/<game>/rewards/``; the generic
ones in ``nxwm.env.rewards.generic`` are usable by any consumer.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch


class Reward(Protocol):
    def __call__(
        self,
        action: torch.Tensor,
        predicted_latent: torch.Tensor,
        info: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        ...
