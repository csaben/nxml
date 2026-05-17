from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FlowMatchingSampler:
    """Standalone flow matching denoising loop."""

    @torch.no_grad()
    def sample(
        self,
        *,
        model,
        obs: torch.Tensor,  # (B, T, C, H, W)
        actions: torch.Tensor,  # (B, T, A)
        goal: torch.Tensor,  # (B, C, H, W)
        flow_steps: int = 5,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        device = obs.device
        B = obs.shape[0]  # noqa: N806
        latent_shape = model.latent_shape  # (C, H, W)

        use_cfg = abs(cfg_scale - 1.0) > 1e-6
        xt = torch.randn(B, *latent_shape, device=device)
        dt = 1.0 / flow_steps

        for i in range(flow_steps):
            t = torch.full((B,), i / flow_steps, device=device)
            v_cond = model(xt, t, obs, actions, goal)
            if use_cfg:
                null_goal = torch.zeros_like(goal)
                v_uncond = model(xt, t, obs, actions, null_goal)
                v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_pred = v_cond
            xt = xt + v_pred * dt

        return xt
