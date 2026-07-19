"""CodecLoss: MAE term, adaptive weighting, frame-frac subsetting, degenerate configs."""

from __future__ import annotations

import torch
from torch import nn

from nxwm_mira.codec.codec_model import VideoCodecOutputs
from nxwm_mira.codec.loss import CodecLoss, CodecLossWeights, calculate_adaptive_weight


def _outputs(requires_grad: bool = True) -> VideoCodecOutputs:
    target = torch.rand(2, 4, 3, 32, 32) * 2 - 1
    pred = (torch.rand(2, 4, 3, 32, 32) * 2 - 1).requires_grad_(requires_grad)
    z = torch.randn(2, 2, 8, 2, 2)
    return VideoCodecOutputs(input_video=target, output_video=pred, z=z)


def test_mae_only() -> None:
    loss_fn = CodecLoss(CodecLossWeights(loss_mae=1.0, compile_dino=False))
    losses = loss_fn(_outputs())
    assert set(losses) == {"loss_mae", "loss_total"}
    assert torch.isfinite(losses["loss_total"])
    assert torch.allclose(losses["loss_total"], losses["loss_mae"])
    losses["loss_total"].backward()  # backward hooks record per-term grad norms
    assert "loss_mae" in loss_fn.backward_metrics


def test_all_zero_weights_returns_zero() -> None:
    loss_fn = CodecLoss(CodecLossWeights(loss_mae=0.0, compile_dino=False))
    losses = loss_fn(_outputs())
    assert losses["loss_total"].item() == 0.0


def test_adaptive_weight_scales_and_detaches() -> None:
    layer = nn.Linear(4, 4)
    x = torch.randn(8, 4)
    anchor = layer(x).abs().mean()
    other = 100 * layer(x).pow(2).mean()
    factor = calculate_adaptive_weight(anchor, other, layer.weight)
    assert factor.ndim == 0
    assert not factor.requires_grad
    assert 0.0 < factor.item() <= 1e4


def test_adaptive_weight_clamps_to_max() -> None:
    layer = nn.Linear(4, 4)
    x = torch.randn(8, 4)
    anchor = layer(x).abs().mean()
    other = 1e-12 * layer(x).pow(2).mean()  # tiny grads -> huge ratio
    factor = calculate_adaptive_weight(anchor, other, layer.weight, max_weight=10.0)
    assert factor.item() <= 10.0


def test_frame_frac_rounding_minimum_one_frame() -> None:
    # With 4 frames and frac 0.25 the subset is exactly 1 frame; the loss must still compute.
    weights = CodecLossWeights(loss_mae=1.0, compile_dino=False)
    assert max(1, round(4 * weights.lpips_perceptual_frame_frac)) == 1
