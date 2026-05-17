from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class TrainingConfig:
    """All hyperparameters for WorldModelTrainer."""

    lr: float = 1e-4
    weight_decay: float = 0.01
    cfg_dropout_prob: float = 0.1
    noise_aug_prob: float = 0.3
    noise_aug_scale: float = 0.25
    lpips_weight: float = 0.0
    lpips_subset_size: int = 4
    # fa_dit only: weight on the multi-hypothesis-planning Laplace NLL
    # auxiliary loss (paper §4.2.2). 0 disables the plan head's training.
    # Ignored by architectures without a plan head (e.g. dit_v1).
    plan_loss_weight: float = 0.0
    unroll_steps: int = 1
    unroll_min: int = 1
    unroll_ramp_epochs: int = 0
    tbptt_window: int | None = None
    precision: Literal["fp16", "bf16"] = "fp16"
    compile_model: bool = False
    grad_clip: float = 1.0
    project_name: str = "nxwm"
    run_dir: Path | str = "checkpoints"
    # Path to a self-describing checkpoint to resume from. Loaded by the
    # launcher (weights + optimizer state + epoch counter). Overridden by
    # ``--resume`` on the CLI when set. ``None`` / empty = train from scratch.
    resume_from: str | None = None

    # Eval-rollout GIFs logged to wandb (and saved to {run_dir}/rollouts/).
    # Set ``eval_gif_every_n_epochs > 0`` to turn it on; the launcher will
    # auto-load a VAE if not already needed for LPIPS.
    eval_gif_every_n_epochs: int = 0
    # ``repeat_last``: seed from one val batch and hold its goal constant —
    # cheapest, fine for "is the WM collapsing?" checks. ``from_mmap``: pull
    # latents/actions/goals fresh from the val .npz each step (receding goal,
    # actions follow the recorded human) — apples-to-apples with ``nxwm rollout``.
    eval_gif_mode: Literal["repeat_last", "from_mmap"] = "repeat_last"
    eval_gif_frames: int = 90
    eval_gif_flow_steps: int = 5
    eval_gif_cfg_scale: float = 1.0
    # Index into ``val_loader.dataset.files`` for ``from_mmap`` mode.
    eval_gif_episode_idx: int = 0
    # Fixed start frame for ``from_mmap`` mode (so diffs across epochs are
    # comparable). Ignored in ``repeat_last`` mode (uses the first batch).
    eval_gif_start_frame: int = 0
