"""Noise augmentation on initial history obs.

Verbatim port of the noise-aug block at the top of ``train_epoch``: consumes
exactly one ``torch.rand`` to decide whether to apply, then ``randn_like(obs)``
plus ``randn(B,T,1,1,1)`` for ``tau`` blending.
"""

from __future__ import annotations

import torch


def apply_noise_augmentation(
    obs: torch.Tensor,
    *,
    prob: float,
    scale: float,
) -> torch.Tensor:
    """Returns ``obs`` (possibly noised). Always consumes 1 rand when ``prob > 0``."""
    if prob <= 0:
        return obs
    if torch.rand(1).item() >= prob:
        return obs
    B, T = obs.shape[0], obs.shape[1]
    noise = torch.randn_like(obs)
    tau = torch.sigmoid(torch.randn(B, T, 1, 1, 1, device=obs.device) * scale)
    return (1 - tau) * obs + tau * noise


def apply_noise_augmentation_with_taus(
    obs: torch.Tensor,
    *,
    prob: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same blend as :func:`apply_noise_augmentation` but also returns the
    per-frame ``tau`` so the caller can pass it into a model whose forward
    accepts an explicit noise-level conditioning vector (fa_dit's
    ``past_taus``).

    Returns ``(possibly_noised_obs, taus)`` where ``taus`` has shape ``(B, T)``
    and ``taus[b, t] == 0`` means frame ``(b, t)`` was not noised (either
    the augmentation didn't fire this batch, or that sample drew τ≈0).
    """
    B, T = obs.shape[0], obs.shape[1]
    zero_taus = torch.zeros(B, T, device=obs.device)
    if prob <= 0:
        return obs, zero_taus
    if torch.rand(1).item() >= prob:
        return obs, zero_taus
    noise = torch.randn_like(obs)
    tau = torch.sigmoid(torch.randn(B, T, 1, 1, 1, device=obs.device) * scale)
    noised = (1 - tau) * obs + tau * noise
    return noised, tau.view(B, T)
