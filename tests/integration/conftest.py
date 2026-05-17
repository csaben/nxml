from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = WORKSPACE_ROOT / "tests" / "fixtures"


@pytest.fixture
def tiny_episode_path() -> Path:
    p = FIXTURE_DIR / "tiny_episode.npz"
    assert p.exists(), f"missing fixture: {p}"
    return p


@pytest.fixture
def tiny_dit_v1_config():
    from nxwm.architectures.dit_v1 import DiTV1Config

    return DiTV1Config(embed_dim=64, depth=2, num_heads=4, patch_size=2, seq_len=10)


@pytest.fixture
def tiny_dit_v1_checkpoint(tiny_dit_v1_config, tmp_path) -> Path:
    """Build a tiny ``dit_v1`` model and save it as a self-describing checkpoint
    loadable via ``nxml_core.load_checkpoint``.
    """
    from nxml_core.checkpoint import save_checkpoint
    from nxwm.architectures.dit_v1 import DiTWorldModel

    torch.manual_seed(42)
    model = DiTWorldModel(tiny_dit_v1_config)
    out = tmp_path / "tiny_dit_v1.pt"
    save_checkpoint(
        architecture="dit_v1",
        config=asdict(tiny_dit_v1_config),
        state_dict=model.state_dict(),
        path=out,
    )
    return out
