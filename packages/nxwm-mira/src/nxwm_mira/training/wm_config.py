"""Training config for the latent world model (`nxwm-mira train-wm`).

Reuses the codec trainer's run/wandb/data/validation/optim sections; the model section
holds a :class:`LatentWorldModelConfig` (which includes the frozen ``codec_checkpoint``).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from nxwm_mira.training.train_config import (
    DataConfig,
    OptimConfig,
    RunConfig,
    ValidationConfig,
    WandbConfig,
)
from nxwm_mira.world_model.config import LatentWorldModelConfig


class WMVizConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_diffusion_steps: int = 10
    n_samples: int = 2  # rollout clips per viz event
    # Extra source-window frames on VAL clips beyond video.timesteps, giving the viz
    # rollout room to generate multiple latent steps (0 = single-step prediction only).
    extra_rollout_frames: int = 32


class WMModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: LatentWorldModelConfig


class WMTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    wandb: WandbConfig = Field(default_factory=lambda: WandbConfig(project="nxwm-mira-wm-za"))
    model: WMModelConfig
    data: DataConfig = Field(default_factory=DataConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    viz: WMVizConfig = Field(default_factory=WMVizConfig)

    @property
    def frame_stride(self) -> int:
        stride = self.data.source_fps / self.model.config.video.fps
        if abs(stride - round(stride)) > 1e-6 or stride < 1:
            raise ValueError(
                f"data.source_fps ({self.data.source_fps}) must be an integer multiple of "
                f"video.fps ({self.model.config.video.fps})"
            )
        return round(stride)
