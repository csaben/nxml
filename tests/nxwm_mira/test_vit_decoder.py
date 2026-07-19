"""ViT decoder: shape contract and auto_weight's last-layer hook."""

from __future__ import annotations

import torch

from nxwm_mira.codec.vit_decoder import ViTVideoDecoder

from .conftest import tiny_raev2_config


def _tiny_decoder(**overrides: object) -> ViTVideoDecoder:
    config = tiny_raev2_config().decoder.model_copy(update=overrides)
    return ViTVideoDecoder(config)


def test_forward_shape() -> None:
    decoder = _tiny_decoder()
    # 64x64 input, /32 latent grid -> 2x2; 4 frames, td=2 -> 2 latent frames; latent_dim 8.
    z = torch.randn(1, 2, 8, 2, 2)
    out = decoder(z)
    assert out.shape == (1, 4, 3, 64, 64)
    assert out.min().item() >= -1.0 and out.max().item() <= 1.0  # tanh-bounded


def test_last_layer_weight_exposed() -> None:
    decoder = _tiny_decoder()
    weight = decoder.last_layer_weight
    assert isinstance(weight, torch.Tensor)
    assert weight.requires_grad


def test_activation_checkpointing_matches() -> None:
    torch.manual_seed(0)
    plain = _tiny_decoder(activation_checkpointing=False)
    ckpt = _tiny_decoder(activation_checkpointing=True)
    ckpt.load_state_dict(plain.state_dict())
    ckpt.train()
    plain.train()
    z = torch.randn(1, 2, 8, 2, 2)
    assert torch.allclose(plain(z), ckpt(z), atol=1e-6)
