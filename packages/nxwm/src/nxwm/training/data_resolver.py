"""Resolve config-style data path entries (strings, dirs, .npz files) to lists.

``resolve_data_paths(data_paths, val_paths)`` returns ``(train_files,
val_files)``: bare basenames in ``val_paths`` are filtered from
``data_paths``; anything else is taken literally and resolved.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def _resolve(entries: Iterable) -> list[Path]:
    out: list[Path] = []
    for e in entries:
        p = Path(e)
        if p.is_dir():
            out.extend(sorted(p.glob("*.npz")))
        elif p.suffix == ".npz":
            out.append(p)
        else:
            raise ValueError(f"data entry is neither a directory nor a .npz file: {e}")
    return out


def resolve_data_paths(
    data_paths: str | list, val_paths: list | None = None
) -> tuple[list[Path], list[Path]]:
    """Return ``(train_files, val_files)`` from raw config entries."""
    raw = [data_paths] if isinstance(data_paths, str) else list(data_paths)
    all_files = list(dict.fromkeys(_resolve(raw)))

    val_entries = list(val_paths or [])
    basename_only = [
        v
        for v in val_entries
        if "/" not in str(v) and Path(v).suffix == ".npz" and not Path(v).exists()
    ]
    explicit_val = [v for v in val_entries if v not in basename_only]

    val_files = list(
        dict.fromkeys(
            _resolve(explicit_val) + [f for f in all_files if f.name in set(basename_only)]
        )
    )
    val_set = set(val_files)
    train_files = [f for f in all_files if f not in val_set]
    return train_files, val_files
