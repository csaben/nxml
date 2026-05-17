"""LPIPS perceptual-loss helper for the unrolled trainer.

Invoked only on the first unroll step (k==0) and on a small ``n_subset``
slice of the batch to keep VRAM/compute manageable.
"""

from __future__ import annotations

import torch


def lpips_step_loss(
    v_pred: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    vae,
    lpips_fn,
    *,
    n_subset: int,
    autocast_dtype: torch.dtype,
) -> torch.Tensor:
    """LPIPS loss over a subset of the batch's first-step predictions.

    ``v_pred + x0`` reconstructs the predicted clean latent. Both predicted and
    GT are decoded outside autocast to keep gradient through ``vae.decode`` in
    fp32.
    """
    B = x1.shape[0]
    n = min(n_subset, B)
    x1_pred = x0[:n] + v_pred[:n]
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=False):
        decoded_pred = vae.decode(x1_pred.float() / 0.18215).sample  # type: ignore[arg-type, attr-defined]
        with torch.no_grad():
            decoded_gt = vae.decode(x1[:n].float() / 0.18215).sample  # type: ignore[arg-type, attr-defined]
        return lpips_fn(decoded_pred, decoded_gt.detach()).mean()
