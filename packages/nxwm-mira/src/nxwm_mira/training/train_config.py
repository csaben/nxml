"""The training config schema: a plain-YAML replacement for MIRA's Hydra composition.

``nxwm-mira train configs/codec/za_3090.yaml`` loads the YAML into :class:`TrainConfig`.
Model configs stay the ported pydantic :class:`~nxwm_mira.codec.config.VideoCodecConfig`;
the cross-field equalities Hydra interpolation used to guarantee are handled by a validator
(``decoder.video`` defaults to ``encoder.video`` when omitted) plus ``VideoCodec.__init__``'s
existing mismatch checks.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nxwm_mira.codec.config import VideoCodecConfig
from nxwm_mira.codec.loss import CodecLossWeights


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1


class SchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    warmup_steps: int = 1000
    constant_steps: int = 0
    decay_steps: int = 0  # 0 disables the cosine decay phase
    min_lr: float = 1e-6


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    model_ema_decay: float = 0.9999


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["za", "fake"] = "za"
    root: str = "data/za-mp4"
    num_workers: int = 6
    # Source-frame step between clip starts; None = non-overlapping (clip span).
    clip_spacing: int | None = None
    # Source fps / model fps; e.g. 30fps corpus at video.fps=15 -> stride 2.
    source_fps: float = 30.0
    holdout_per_folder: int = 1
    # Explicit val episode indices; overrides holdout_per_folder when set.
    val_episodes: list[int] | None = None
    fake_n_clips: int = 64


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 28
    steps: int
    batch_size: int
    output_dir: str | None = None  # None -> auto-increment checkpoints/codec/run_NNN
    checkpoint_every: int = 2500
    checkpoint_keep_recent: int = 2
    checkpoint_keep_permanent_every: int = 25_000
    log_every: int | str = 50
    viz_every: int = 500
    viz_keep_every: int = 10  # keep every Nth viz GIF permanently (+ the most recent 20)
    latent_diagnostics: bool = False  # plotly latent correlation/std panels (needs plotly)
    compile: bool = False  # torch>2.8 inductor is known-broken for this model; keep off
    continue_from: str | None = None
    finetune_from: str | None = None
    latents_ema_decay: float = 0.99
    require_dino_weights: bool = True  # false -> random frozen backbone (smoke runs only)


class WandbConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str = "nxwm-mira-codec-za"
    entity: str | None = None
    name: str | None = None  # None -> "<run_dir_name>-<yymmdd-HHMM>"
    group: str | None = None
    mode: Literal["online", "offline", "disabled"] = "online"


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int | None = None  # None -> run.batch_size
    val_every: int | str = 2500
    val_first: bool = True
    val_n_samples: int = 96


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: VideoCodecConfig
    loss: CodecLossWeights

    @model_validator(mode="before")
    @classmethod
    def _default_decoder_video(cls, values: dict) -> dict:
        """Let YAML omit decoder.video; it must equal encoder.video anyway."""
        config = values.get("config")
        if (
            isinstance(config, dict)
            and isinstance(config.get("encoder"), dict)
            and isinstance(config.get("decoder"), dict)
            and "video" not in config["decoder"]
            and "video" in config["encoder"]
        ):
            config["decoder"]["video"] = config["encoder"]["video"]
        return values


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    model: ModelConfig
    data: DataConfig = Field(default_factory=DataConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)

    @property
    def frame_stride(self) -> int:
        """Source-frame subsampling to hit the model's fps (e.g. 30 -> 15 fps = stride 2)."""
        stride = self.data.source_fps / self.model.config.encoder.video.fps
        if abs(stride - round(stride)) > 1e-6 or stride < 1:
            raise ValueError(
                f"data.source_fps ({self.data.source_fps}) must be an integer multiple of "
                f"encoder.video.fps ({self.model.config.encoder.video.fps})"
            )
        return round(stride)
