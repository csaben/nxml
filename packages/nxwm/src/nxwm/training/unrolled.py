"""K-step unrolled flow matching forward pass.

The TBPTT detach is **load-bearing**:

  ``steps_from_end = (K - 1) - k``
  if ``tbptt_window is not None`` and ``steps_from_end >= tbptt_window``, then
  the predicted clean latent ``x1_hat`` is detached before being shifted into
  history. This means gradients from later steps cannot flow back into earlier
  ones beyond ``tbptt_window`` predictions.

  ``tbptt_window=None`` preserves full BPTT.

The function is pure over ``(model, batch, hyperparams)`` plus an optional
``extra_step_loss`` callback for k==0 LPIPS.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .flow_matching import flow_match_loss

ExtraStepLoss = Callable[..., torch.Tensor | None]


def unrolled_forward(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    K: int,
    *,
    cfg_dropout_prob: float,
    tbptt_window: int | None = None,
    extra_step_loss: ExtraStepLoss | None = None,
) -> dict[str, Any]:
    """K-step unrolled flow matching loss.

    ``batch`` keys: ``observations``, ``actions``, ``targets``, ``goals``,
    ``future_actions`` (already on the right device).
    """
    obs = batch["observations"]
    actions = batch["actions"]
    targets = batch["targets"]
    goals = batch["goals"]
    future_actions = batch["future_actions"]

    history = obs
    act_buf = actions

    total_loss = torch.tensor(0.0, device=obs.device)
    total_mse = torch.tensor(0.0, device=obs.device)
    total_extra = torch.tensor(0.0, device=obs.device)
    extra_count = 0

    for k in range(K):
        x1 = targets[:, k]
        goal = goals[:, k]

        v_pred, mse_loss, x0 = flow_match_loss(
            model,
            x1,
            history,
            act_buf,
            goal,
            cfg_dropout_prob=cfg_dropout_prob,
            gradient_checkpoint=(K > 1),
        )
        step_loss = mse_loss
        total_mse = total_mse + mse_loss

        if extra_step_loss is not None:
            extra = extra_step_loss(k=k, v_pred=v_pred, x0=x0, x1=x1)
            if extra is not None:
                step_loss = step_loss + extra
                total_extra = total_extra + extra
                extra_count += 1

        total_loss = total_loss + step_loss

        if k < K - 1:
            x1_hat = x0 + v_pred  # predicted clean frame (has grad)
            if tbptt_window is not None:
                steps_from_end = (K - 1) - k
                if steps_from_end >= tbptt_window:
                    x1_hat = x1_hat.detach()

            history = torch.cat([history[:, 1:], x1_hat.unsqueeze(1)], dim=1)
            fa = future_actions[:, k : k + 1]
            act_buf = torch.cat([act_buf[:, 1:], fa], dim=1)

    return {
        "loss": total_loss / K,
        "mse": total_mse / K,
        "extra": total_extra,
        "extra_count": extra_count,
        "K": K,
    }


def unrolled_forward_fa_dit(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    K: int,
    *,
    cfg_dropout_prob: float,
    tbptt_window: int | None = None,
    plan_loss_weight: float = 0.0,
) -> dict[str, Any]:
    """K-step unrolled rectified-flow loss for fa_dit.

    Same TBPTT semantics as :func:`unrolled_forward` (predicted clean latents
    feed back into history with a configurable detach window) but routes
    through fa_dit's forward signature. Future-anchor latents/poses are read
    once from the batch and stay constant across the unroll.

    Plan-head MHP-Laplace NLL is added at k=0 only (mirrors the LPIPS
    economy). Set ``plan_loss_weight=0`` to skip the plan-head loss entirely
    — the plan head still gets evaluated, just untrained.
    """
    from nxwm.architectures.fa_dit.model import mhp_laplace_nll

    from .flow_matching import fa_dit_flow_match_loss

    obs = batch["observations"]
    actions = batch["actions"]
    targets = batch["targets"]
    future_actions = batch["future_actions"]
    future_latents = batch["future_latents"]
    future_poses = batch["future_poses"]
    # Per-frame noise levels for past slots (from the noise-aug-aware
    # augmenter). May be missing if the trainer didn't run noise aug.
    past_taus = batch.get("past_taus")

    history = obs
    act_buf = actions

    total_loss = torch.tensor(0.0, device=obs.device)
    total_mse = torch.tensor(0.0, device=obs.device)
    total_plan = torch.tensor(0.0, device=obs.device)
    plan_count = 0

    for k in range(K):
        x1 = targets[:, k]
        target_pose = future_actions[:, k]

        v_pred, plan, mse_loss, x0 = fa_dit_flow_match_loss(
            model,
            x1,
            history,
            act_buf,
            future_latents,
            future_poses,
            target_pose,
            cfg_dropout_prob=cfg_dropout_prob,
            gradient_checkpoint=(K > 1),
            past_taus=past_taus,
        )
        step_loss = mse_loss
        total_mse = total_mse + mse_loss

        if k == 0 and plan_loss_weight > 0 and "plan_target" in batch:
            plan_loss = mhp_laplace_nll(plan, batch["plan_target"])
            step_loss = step_loss + plan_loss_weight * plan_loss
            total_plan = total_plan + plan_loss
            plan_count += 1

        total_loss = total_loss + step_loss

        if k < K - 1:
            x1_hat = x0 + v_pred
            if tbptt_window is not None:
                steps_from_end = (K - 1) - k
                if steps_from_end >= tbptt_window:
                    x1_hat = x1_hat.detach()
            history = torch.cat([history[:, 1:], x1_hat.unsqueeze(1)], dim=1)
            fa = future_actions[:, k : k + 1]
            act_buf = torch.cat([act_buf[:, 1:], fa], dim=1)
            # Shift past_taus: oldest frame drops; newly-predicted frame is
            # clean (tau=0).
            if past_taus is not None:
                B = past_taus.shape[0]
                new_tau = torch.zeros(B, 1, device=past_taus.device)
                past_taus = torch.cat([past_taus[:, 1:], new_tau], dim=1)

    return {
        "loss": total_loss / K,
        "mse": total_mse / K,
        "plan": total_plan,
        "plan_count": plan_count,
        # Keys below kept for trainer-side log compatibility with the dit_v1 path.
        "extra": torch.tensor(0.0, device=obs.device),
        "extra_count": 0,
        "K": K,
    }
