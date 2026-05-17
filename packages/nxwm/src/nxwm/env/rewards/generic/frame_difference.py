"""Frame-difference reward — game-agnostic shaping that rewards the policy
for *changing* the world. Returns ``weight * ||L_t - L_{t-1}||_2``.

Useful as a baseline anti-stagnation signal: a policy that does nothing
keeps the latent constant and gets zero reward. Combine with a goal
reward elsewhere; on its own this just encourages "do anything."

Factory shape matches the launcher's ``_load_callable`` contract — call
``make_reward_fn(weight=...)`` to get the runtime callable.
"""

from __future__ import annotations

from typing import Any

import torch


def make_reward_fn(*, weight: float = 1.0):
    prev_latent: dict[str, torch.Tensor | None] = {"x": None}

    def _reward(
        action: torch.Tensor,
        predicted_latent: torch.Tensor,
        info: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        del action  # this reward only looks at frames
        prev = prev_latent["x"]
        if prev is None or info.get("step", 0) == 0:
            prev_latent["x"] = predicted_latent.detach()
            return 0.0, {"frame_diff": 0.0}
        diff = (predicted_latent - prev).norm().item()
        prev_latent["x"] = predicted_latent.detach()
        return weight * diff, {"frame_diff": diff}

    return _reward
