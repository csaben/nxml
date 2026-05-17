"""MacroStore directory operations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from nx_macros import MacroStore, macro_from_arrays, sanitize_name
from nx_packets import ACTION_DIM


def _macro(name: str):
    return macro_from_arrays(
        name=name,
        tick_hz=30.0,
        actions=np.zeros((1, ACTION_DIM), dtype=np.float32),
        dts=[0.0],
    )


def test_store_save_load_round_trip(tmp_path: Path):
    store = MacroStore(tmp_path)
    store.save(_macro("walk"))
    assert store.list() == ["walk"]
    assert store.exists("walk")
    loaded = store.load("walk")
    assert loaded.name == "walk"


def test_store_list_empty_when_root_missing(tmp_path: Path):
    store = MacroStore(tmp_path / "nope")
    assert store.list() == []


def test_store_delete(tmp_path: Path):
    store = MacroStore(tmp_path)
    store.save(_macro("walk"))
    assert store.delete("walk") is True
    assert store.delete("walk") is False
    assert store.list() == []


def test_store_load_missing_raises(tmp_path: Path):
    store = MacroStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load("nope")


def test_sanitize_name_rejects_separators():
    for bad in ["", "  ", ".hidden", "a/b", "a\\b", "a\x00b"]:
        with pytest.raises(ValueError):
            sanitize_name(bad)


def test_sanitize_name_allows_unicode_and_hyphens():
    assert sanitize_name("walk-and-jump") == "walk-and-jump"
    assert sanitize_name("マクロ") == "マクロ"
