"""Schema round-trip and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from nx_macros import Macro, MacroFrame, macro_from_arrays
from nx_packets import ACTION_DIM, BUTTON_INDEX, neutral_action


def test_macro_frame_rejects_wrong_length():
    with pytest.raises(ValueError):
        MacroFrame(action=[0.0] * (ACTION_DIM - 1), dt=0.0)


def test_macro_frame_rejects_non_finite():
    bad = [0.0] * ACTION_DIM
    bad[0] = float("nan")
    with pytest.raises(ValueError):
        MacroFrame(action=bad, dt=0.0)


def test_macro_round_trip_preserves_action_values(tmp_path: Path):
    a0 = neutral_action()
    a1 = neutral_action()
    a1[BUTTON_INDEX["A"]] = 1.0
    a1[0] = 0.5  # L stick X

    m = macro_from_arrays(
        name="press-a",
        tick_hz=30.0,
        actions=np.stack([a0, a1]),
        dts=[0.0, 0.033],
        metadata={"game": "test"},
    )
    p = m.save(tmp_path / "press-a.json")
    loaded = Macro.load(p)

    assert loaded.name == "press-a"
    assert loaded.tick_hz == 30.0
    assert loaded.metadata == {"game": "test"}
    np.testing.assert_array_equal(loaded.actions(), m.actions())
    assert [f.dt for f in loaded.frames] == [0.0, 0.033]


def test_macro_to_packet_data_button_threshold():
    a = neutral_action()
    a[BUTTON_INDEX["A"]] = 1.0
    m = macro_from_arrays(name="x", tick_hz=30.0, actions=a[None], dts=[0.0])
    pd = m.to_packet_data()
    assert len(pd.packets) == 1
    assert pd.packets[0].A is True
    assert pd.packets[0].B is False


def test_macro_from_arrays_rejects_wrong_shape():
    with pytest.raises(ValueError):
        macro_from_arrays(
            name="bad",
            tick_hz=30.0,
            actions=np.zeros((3, 25), dtype=np.float32),
            dts=[0.0, 0.0, 0.0],
        )


def test_macro_from_arrays_rejects_dt_length_mismatch():
    with pytest.raises(ValueError):
        macro_from_arrays(
            name="bad",
            tick_hz=30.0,
            actions=np.zeros((3, ACTION_DIM), dtype=np.float32),
            dts=[0.0, 0.0],
        )
