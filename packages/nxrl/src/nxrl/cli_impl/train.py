"""Implementation of ``nxrl train``."""

from __future__ import annotations

from pathlib import Path


def run_train(*, config_path: str, resume: str | None, world_size: int | None) -> None:
    from nxrl.training.launcher import launch

    launch(
        Path(config_path),
        world_size=world_size,
        resume_override=Path(resume) if resume else None,
    )
