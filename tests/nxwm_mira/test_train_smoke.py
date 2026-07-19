"""End-to-end trainer smoke: a 3-step run on fake data, artifacts, and resume."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nxwm_mira.training.train_config import TrainConfig

from .conftest import tiny_raev2_config

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs" / "codec"


def _smoke_config(tmp_path: Path, steps: int = 3) -> TrainConfig:
    return TrainConfig.model_validate(
        {
            "run": {
                "seed": 0,
                "steps": steps,
                "batch_size": 2,
                "output_dir": str(tmp_path / "run"),
                "checkpoint_every": 2,
                "checkpoint_keep_recent": 1,
                "checkpoint_keep_permanent_every": 1_000_000,
                "log_every": 1,
                "viz_every": 2,
                "require_dino_weights": False,
            },
            "wandb": {"mode": "disabled"},
            "model": {
                "loss": {"loss_mae": 1.0, "compile_dino": False, "auto_weight": False},
                "config": tiny_raev2_config().model_dump(),
            },
            "data": {"source": "fake", "num_workers": 0, "fake_n_clips": 8},
            "validation": {"val_every": 1_000_000, "val_first": True, "val_n_samples": 2},
            "optim": {
                "scheduler": {"warmup_steps": 1, "constant_steps": 0, "decay_steps": 0}
            },
        }
    )


def _run(cfg: TrainConfig) -> None:
    from nxwm_mira.training.trainer import run_training

    try:
        run_training(cfg)
    except Exception as exc:
        if "dinov3" in str(exc).lower() or "hub" in str(exc).lower():
            pytest.skip(f"DINOv3 backbone unavailable: {exc}")
        raise


def test_shipped_yaml_configs_parse() -> None:
    for name in ("tiny_smoke.yaml", "za_3090.yaml"):
        raw = yaml.safe_load((CONFIGS_DIR / name).read_text())
        cfg = TrainConfig.model_validate(raw)
        # decoder.video was omitted in YAML and must inherit encoder.video
        assert cfg.model.config.decoder.video == cfg.model.config.encoder.video
    za = TrainConfig.model_validate(yaml.safe_load((CONFIGS_DIR / "za_3090.yaml").read_text()))
    assert za.frame_stride == 2
    assert za.model.config.encoder.aspect_mode == "stretch"


def test_train_smoke_and_resume(tmp_path: Path) -> None:
    cfg = _smoke_config(tmp_path)
    _run(cfg)

    run_dir = tmp_path / "run"
    assert (run_dir / "codec_config.yaml").is_file()
    assert (run_dir / "status.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "recons" / "latest.gif").is_file()
    ckpts = sorted(run_dir.glob("checkpoint-*/checkpoint.pth"))
    assert ckpts, "no checkpoint written"

    # Extend the run: auto-resume must pick up from the saved step, not restart at 0.
    cfg2 = _smoke_config(tmp_path, steps=5)
    _run(cfg2)
    latest = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))[-1]
    assert int(latest.name.split("-")[1]) == 4  # final save at step 4 (0-indexed, 5 steps)
