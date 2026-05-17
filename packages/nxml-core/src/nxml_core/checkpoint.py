from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

NXML_VERSION = "0.1.0"


def save_checkpoint(
    *,
    architecture: str,
    config: Any,
    state_dict: dict,
    path: Path,
    extra: dict | None = None,
):
    config_dict = asdict(config) if is_dataclass(config) else dict(config)
    payload = {
        "architecture": architecture,
        "config": config_dict,
        "state_dict": state_dict,
        "nxml_version": NXML_VERSION,
        **(extra or {}),
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, registry, *, device: str | torch.device = "cpu"):
    """Load any registered architecture from a self-describing checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "architecture" not in ckpt:
        raise ValueError(f"{path} is not a self-describing checkpoint (missing 'architecture' key)")
    arch_name = ckpt["architecture"]
    cls = registry.get(arch_name)
    config_cls = registry.get_config(arch_name)
    config = config_cls(**ckpt["config"])
    model = cls(config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, config, ckpt
