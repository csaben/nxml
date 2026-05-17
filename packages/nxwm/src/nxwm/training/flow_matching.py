"""Flow matching primitives shared between trainer and eval.

Pure functions over ``(model, batch, hyperparams)`` — no optimizer, no scaler,
no logging, no autocast. The caller is responsible for placing inputs on the
right device and wrapping in ``torch.autocast`` if needed.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint as grad_checkpoint


def sample_logit_normal(B: int, device: torch.device | str) -> torch.Tensor:
    """Sample t ~ sigmoid(N(0,1)) — verbatim port of WorldModelTrainer.sample_logit_normal."""
    m = torch.randn(B, device=device)
    return torch.sigmoid(m)


def flow_match_loss(
    model: torch.nn.Module,
    x1: torch.Tensor,
    obs: torch.Tensor,
    actions: torch.Tensor,
    goal: torch.Tensor,
    *,
    cfg_dropout_prob: float,
    gradient_checkpoint: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single flow matching step: returns ``(v_pred, mse_loss, x0)``.

    The order of RNG consumption is fixed (cfg-dropout rand → x0 randn →
    t-sampling randn) for determinism.
    """
    B = x1.shape[0]
    device = x1.device

    # CFG: randomly zero out goal for unconditional learning.
    if cfg_dropout_prob > 0:
        drop_mask = torch.rand(B, device=device) < cfg_dropout_prob
        goal = goal * (~drop_mask).float().view(B, 1, 1, 1)

    x0 = torch.randn_like(x1)
    t = sample_logit_normal(B, device)
    t_view = t.view(B, 1, 1, 1)

    xt = t_view * x1 + (1 - t_view) * x0
    v_target = x1 - x0

    if gradient_checkpoint:
        v_pred = grad_checkpoint(model, xt, t, obs, actions, goal, use_reentrant=False)
    else:
        v_pred = model(xt, t, obs, actions, goal)

    mse_loss = torch.mean((v_pred - v_target) ** 2)
    return v_pred, mse_loss, x0


def fa_dit_flow_match_loss(
    model: torch.nn.Module,
    x1: torch.Tensor,
    past_latents: torch.Tensor,
    past_poses: torch.Tensor,
    future_latents: torch.Tensor,
    future_poses: torch.Tensor,
    target_pose: torch.Tensor,
    *,
    cfg_dropout_prob: float,
    gradient_checkpoint: bool = False,
    past_taus: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single rectified-flow step for fa_dit. Returns ``(v_pred, plan, mse_loss, x0)``.

    Same noise/interpolation schedule as :func:`flow_match_loss` (logit-normal
    ``t``, linear interp ``xt = t·x1 + (1-t)·x0``, target ``v = x1 − x0``), but
    routes through fa_dit's multi-arg forward and unpacks both the velocity
    head and the multi-hypothesis plan head from the model output.

    ``taus`` layout matches the model's frame order
    ``[future_anchor | past | target]``:

      - anchor slots are always 0 (anchor stays clean per paper).
      - past slots default to 0; pass ``past_taus`` of shape ``(B, T_past)``
        to tell the model the noise level applied to each past frame (the
        paper's "noise-aug-aware" past — pairs with
        :func:`apply_noise_augmentation_with_taus`).
      - target slot carries the sampled ``t`` for this step.

    CFG dropout zeros the future-anchor latents (and paired poses) per-sample
    so the model learns to predict without anchor conditioning too.
    """
    B = x1.shape[0]
    device = x1.device

    if cfg_dropout_prob > 0:
        drop_mask = torch.rand(B, device=device) < cfg_dropout_prob
        keep = (~drop_mask).float()
        future_latents = future_latents * keep.view(B, 1, 1, 1, 1)
        future_poses = future_poses * keep.view(B, 1, 1)

    x0 = torch.randn_like(x1)
    t = sample_logit_normal(B, device)
    t_view = t.view(B, 1, 1, 1)
    xt = t_view * x1 + (1 - t_view) * x0
    v_target = x1 - x0

    F_a = future_latents.shape[1]
    T_past = past_latents.shape[1]
    total_seq_len = F_a + T_past + 1
    taus = torch.zeros(B, total_seq_len, device=device)
    if past_taus is not None:
        taus[:, F_a : F_a + T_past] = past_taus
    taus[:, -1] = t

    args = (xt, past_latents, past_poses, future_latents, future_poses, target_pose, taus)
    if gradient_checkpoint:
        v_pred, plan = grad_checkpoint(model, *args, use_reentrant=False)
    else:
        v_pred, plan = model(*args)

    mse_loss = torch.mean((v_pred - v_target) ** 2)
    return v_pred, plan, mse_loss, x0
