"""VideoCodecConfig validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nxwm_mira.codec import RAEEncoderConfig, VideoCodecConfig
from nxwm_mira.ml import ImageConfig

from .conftest import tiny_raev2_config


def test_tiny_config_validates() -> None:
    config = tiny_raev2_config()
    assert config.encoder.latent_dim == config.decoder.latent_dim == 8
    assert config.decoder.patch_size_t == config.encoder.bottleneck.temporal_stride == 2


def test_aspect_mode_default_is_pad() -> None:
    encoder = RAEEncoderConfig(
        latent_dim=8,
        rae_model="dinov3_vitb16",
        video=ImageConfig(height=64, width=64, channels=3, timesteps=4, fps=15),
    )
    assert encoder.aspect_mode == "pad"


def test_aspect_mode_rejects_unknown() -> None:
    with pytest.raises(ValidationError):
        RAEEncoderConfig(
            latent_dim=8,
            rae_model="dinov3_vitb16",
            aspect_mode="crop",
            video=ImageConfig(height=64, width=64, channels=3, timesteps=4, fps=15),
        )


def test_extra_fields_forbidden() -> None:
    config = tiny_raev2_config().model_dump()
    config["encoder"]["unknown_field"] = 1
    with pytest.raises(ValidationError):
        VideoCodecConfig.model_validate(config)


def test_roundtrip_through_dump() -> None:
    config = tiny_raev2_config(aspect_mode="stretch")
    restored = VideoCodecConfig.model_validate(config.model_dump())
    assert restored == config
    assert restored.encoder.aspect_mode == "stretch"
