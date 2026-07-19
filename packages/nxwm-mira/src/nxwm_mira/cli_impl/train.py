"""`nxwm-mira train`: YAML -> TrainConfig -> run_training, with run-dir auto-increment."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


def next_run_dir(base: Path) -> Path:
    """Auto-increment ``base/run_NNN`` (mirrors nxwm's launcher convention)."""
    base.mkdir(parents=True, exist_ok=True)
    existing = [
        int(m.group(1))
        for p in base.iterdir()
        if (m := re.fullmatch(r"run_(\d+)", p.name)) and p.is_dir()
    ]
    return base / f"run_{max(existing, default=0) + 1:03d}"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()  # RS_DINO_WEIGHTS_DIR, WANDB_* etc.
    except ImportError:
        pass


def _resolve_run_dir(cfg, resume: str | None, output_dir: str | None, base: Path) -> None:
    if resume is not None:
        cfg.run.continue_from = resume
    if output_dir is not None:
        cfg.run.output_dir = output_dir
    elif cfg.run.output_dir is None:
        import os

        if os.environ.get("RANK") not in (None, "0"):
            raise SystemExit(
                "Under torchrun, pass --output-dir explicitly so all ranks agree on the run dir."
            )
        cfg.run.output_dir = str(next_run_dir(base))


def run_train(config_path: str, resume: str | None, output_dir: str | None) -> None:
    _load_dotenv()

    from nxwm_mira.training.train_config import TrainConfig

    cfg = TrainConfig.model_validate(yaml.safe_load(Path(config_path).read_text()))
    _resolve_run_dir(cfg, resume, output_dir, Path("checkpoints/codec"))

    from nxwm_mira.training.trainer import run_training

    run_training(cfg)


def run_train_wm(config_path: str, resume: str | None, output_dir: str | None) -> None:
    _load_dotenv()

    from nxwm_mira.training.wm_config import WMTrainConfig

    cfg = WMTrainConfig.model_validate(yaml.safe_load(Path(config_path).read_text()))
    _resolve_run_dir(cfg, resume, output_dir, Path("checkpoints/wm-mira"))

    from nxwm_mira.training.wm_trainer import run_training

    run_training(cfg)
