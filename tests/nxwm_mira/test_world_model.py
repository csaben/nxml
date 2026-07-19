"""LatentWorldModel: training forward, rollout, streaming step, checkpoint round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from nxwm_mira.codec.codec_model import VideoCodec
from nxwm_mira.data.actions import SwitchActions
from nxwm_mira.data.batch import VideoActionBatch

from .conftest import build_codec_or_skip, random_video, tiny_raev2_config


@pytest.fixture(scope="module")
def codec_ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny codec checkpoint (random backbone) for the world model to load."""
    root = tmp_path_factory.mktemp("codec")
    config = tiny_raev2_config(aspect_mode="stretch")
    codec = build_codec_or_skip(config)
    (root / VideoCodec.CONFIG_FILENAME).write_text(
        yaml.safe_dump({"model": {"config": config.model_dump()}})
    )
    ckpt_dir = root / "checkpoint-1"
    ckpt_dir.mkdir()
    codec.save_checkpoint(
        ckpt_dir / "checkpoint.pth", extra_data={"latent_mean_std": [0.0, 1.0]}
    )
    return ckpt_dir / "checkpoint.pth"


def _tiny_wm_config(codec_ckpt: Path) -> dict:
    return {
        "actions": {"source_fps": 30, "target_fps": 15},
        "video": {"height": 64, "width": 64, "channels": 3, "timesteps": 8, "fps": 15},
        "codec_checkpoint": str(codec_ckpt),
        "n_context_frames": 6,
        "hidden_dim": 64,
        "n_head": 4,
        "n_kv_head": 2,
        "n_layers": 2,
        "time_attention_every": 1,
        "use_clean_past": True,
        "learned_temporal_pool": True,
        "dropout_action_prob": 0.1,
        "causal": True,
    }


def _tiny_wm(codec_ckpt: Path):
    from nxwm_mira.world_model import LatentWorldModelConfig
    from nxwm_mira.world_model.latent_world_model import LatentWorldModel

    config = LatentWorldModelConfig.model_validate(_tiny_wm_config(codec_ckpt))
    return LatentWorldModel(config)


def _tiny_batch(batch: int = 2, frames: int = 8) -> VideoActionBatch:
    return VideoActionBatch(
        video=random_video(batch=batch, frames=frames, height=64, width=64),
        actions=SwitchActions(torch.rand(batch, frames, 26) * 2 - 1),
    )


def test_training_forward(codec_ckpt: Path) -> None:
    wm = _tiny_wm(codec_ckpt)
    losses = wm(_tiny_batch())
    assert torch.isfinite(losses["loss_total"])
    losses["loss_total"].backward()
    # The codec must stay frozen: no codec grads, transformer has grads.
    assert all(p.grad is None for p in wm.codec.parameters())
    assert any(p.grad is not None for p in wm.world_model.parameters())


def test_inference_rollout(codec_ckpt: Path) -> None:
    from nxwm_mira.world_model import WorldModelInferenceConfig

    wm = _tiny_wm(codec_ckpt).eval()
    with torch.no_grad():
        outputs = wm.inference(
            _tiny_batch(batch=1),
            config=WorldModelInferenceConfig(n_diffusion_steps=2),
            progress_bar=False,
        )
    # 8 frames -> 4 latents; context 3 latents + 1 generated, rolled to the end.
    assert outputs.output_video.shape == (1, 8, 3, 64, 64)
    assert torch.isfinite(outputs.output_video).all()
    viz = wm.visualize(outputs)
    assert viz["viz_video"].dtype == torch.uint8
    assert viz["viz_video"].shape[-2] == 128  # pred stacked over GT


def test_streaming_inference_step(codec_ckpt: Path) -> None:
    from nxwm_mira.world_model import WorldModelInferenceConfig

    wm = _tiny_wm(codec_ckpt).eval()
    batch = _tiny_batch(batch=1, frames=6)  # context window: 3 latents
    with torch.no_grad():
        z = wm.init_streaming_inference(VideoActionBatch(video=batch.video, actions=batch.actions))
        z_t = torch.cat([z, torch.randn_like(z[:, :1])], dim=1)[:, 1:]  # roll one step
        actions = SwitchActions(torch.rand(1, 8, 26) * 2 - 1)
        z_next, kv = wm.streaming_inference_step(
            torch.cat([z, torch.randn_like(z[:, :1])], dim=1)[:, -z.shape[1] - 0 :],
            actions,
            config=WorldModelInferenceConfig(n_diffusion_steps=2),
        )
    assert torch.isfinite(z_next).all()
    assert kv is not None
    del z_t


def test_wm_checkpoint_roundtrip(codec_ckpt: Path, tmp_path: Path) -> None:
    from nxwm_mira.world_model.latent_world_model import LatentWorldModel

    wm = _tiny_wm(codec_ckpt)
    (tmp_path / LatentWorldModel.CONFIG_FILENAME).write_text(
        yaml.safe_dump({"model": {"config": _tiny_wm_config(codec_ckpt)}})
    )
    ckpt_dir = tmp_path / "checkpoint-3"
    ckpt_dir.mkdir()
    wm.save_checkpoint(ckpt_dir / "checkpoint.pth", extra_data={"iter_num": 3})

    loaded = LatentWorldModel.load_from_checkpoint(ckpt_dir / "checkpoint.pth")
    for (name_a, p_a), (name_b, p_b) in zip(
        wm.state_dict().items(), loaded.state_dict().items(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(p_a, p_b)
