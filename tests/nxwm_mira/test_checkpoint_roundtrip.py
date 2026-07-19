"""Checkpoint contract: CheckpointManager save -> VideoCodec.load_from_checkpoint.

This is the contract the future world-model port depends on: EMA-swapped weights in
``checkpoint-{step}/checkpoint.pth``, ``latent_mean_std`` in the extra data, and a
``codec_config.yaml`` discoverable by walking parent directories. Also proves MIRA-layout
configs (``model.architecture.config`` with hydra ``_target_`` keys) still load.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from nxwm_mira.codec.codec_model import VideoCodec

from .conftest import build_codec_or_skip, tiny_batch, tiny_raev2_config


def _write_our_config(output_dir: Path, config) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / VideoCodec.CONFIG_FILENAME).write_text(
        yaml.safe_dump({"model": {"config": config.model_dump()}})
    )


def test_save_load_roundtrip(tmp_path: Path) -> None:
    config = tiny_raev2_config(aspect_mode="stretch")
    codec = build_codec_or_skip(config)
    _write_our_config(tmp_path, config)

    ckpt_dir = tmp_path / "checkpoint-10"
    ckpt_dir.mkdir()
    codec.save_checkpoint(
        ckpt_dir / "checkpoint.pth",
        extra_data={"latent_mean_std": [0.12, 1.34], "iter_num": 10},
    )

    loaded = VideoCodec.load_from_checkpoint(ckpt_dir / "checkpoint.pth")
    assert loaded.config == config
    assert loaded.info_from_checkpoint is not None
    assert loaded.info_from_checkpoint["latent_mean_std"] == [0.12, 1.34]
    for (name_a, p_a), (name_b, p_b) in zip(
        codec.state_dict().items(), loaded.state_dict().items(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(p_a, p_b)

    # Loaded codec round-trips a batch with the same shapes.
    batch = tiny_batch()
    with torch.no_grad():
        outputs = loaded(batch)
    assert outputs.output_video.shape == outputs.input_video.shape
    assert outputs.z.shape == (1, 2, 8, 2, 2)


def test_mira_layout_config_loads(tmp_path: Path) -> None:
    """A codec_config.yaml written by MIRA (hydra _target_ keys, architecture nesting) loads."""
    config = tiny_raev2_config()
    codec = build_codec_or_skip(config)

    config_dict = config.model_dump()
    config_dict["_target_"] = "mira.codec.VideoCodec"
    config_dict["encoder"]["_target_"] = "mira.codec.config.RAEEncoderConfig"
    config_dict["encoder"]["is_audio_model"] = False  # removed field, at its no-op value
    del config_dict["encoder"]["aspect_mode"]  # MIRA configs predate this field
    (tmp_path / VideoCodec.CONFIG_FILENAME).write_text(
        yaml.safe_dump({"model": {"architecture": {"config": config_dict}}})
    )

    ckpt_dir = tmp_path / "checkpoint-5"
    ckpt_dir.mkdir()
    codec.save_checkpoint(ckpt_dir / "checkpoint.pth")

    loaded = VideoCodec.load_from_checkpoint(ckpt_dir / "checkpoint.pth")
    assert loaded.config.encoder.aspect_mode == "pad"  # default preserved for MIRA checkpoints
    assert loaded.config.encoder.rae_model == config.encoder.rae_model


def test_save_without_config_raises(tmp_path: Path) -> None:
    codec = build_codec_or_skip()
    ckpt_dir = tmp_path / "checkpoint-1"
    ckpt_dir.mkdir()
    try:
        codec.save_checkpoint(ckpt_dir / "checkpoint.pth")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError without codec_config.yaml")
