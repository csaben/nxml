"""preprocess_video: pad vs stretch on the ZA source shape (480x640 -> 288x512)."""

from __future__ import annotations

import torch

from nxwm_mira.codec.codec_model import preprocess_video

from .conftest import random_video


def test_stretch_resizes_anisotropically() -> None:
    video = random_video(batch=2, frames=4, height=480, width=640)
    out = preprocess_video(video, target_h=288, target_w=512, aspect_mode="stretch")
    assert out.shape == (2, 4, 3, 288, 512)
    assert out.dtype == torch.float32
    assert 0.0 <= out.min().item() and out.max().item() <= 1.0


def test_pad_letterboxes_4x3_into_16x9() -> None:
    # A bright 4:3 source padded into 16:9 gets black columns on the right:
    # 480x640 -> right pad to 853 wide -> resize. The rightmost ~25% is padding.
    video = torch.full((1, 2, 3, 480, 640), 255, dtype=torch.uint8)
    out = preprocess_video(video, target_h=288, target_w=512, aspect_mode="pad")
    assert out.shape == (1, 2, 3, 288, 512)
    right_band = out[..., :, 500:]
    left_band = out[..., :, :384]
    assert right_band.mean().item() < 0.05  # black letterbox
    assert left_band.mean().item() > 0.95  # original content


def test_stretch_fills_frame_completely() -> None:
    video = torch.full((1, 2, 3, 480, 640), 255, dtype=torch.uint8)
    out = preprocess_video(video, target_h=288, target_w=512, aspect_mode="stretch")
    assert out.min().item() > 0.95  # no black bars anywhere


def test_noop_at_target_resolution() -> None:
    video = random_video(batch=1, frames=2, height=288, width=512)
    out = preprocess_video(video, target_h=288, target_w=512, aspect_mode="stretch")
    assert torch.allclose(out, video / 255.0)
