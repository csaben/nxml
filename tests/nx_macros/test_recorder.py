"""MacroRecorder semantics."""

from __future__ import annotations

import numpy as np
import pytest
from nx_macros import MacroRecorder
from nx_packets import ACTION_DIM, BUTTON_INDEX, neutral_action


def test_recorder_full_cycle():
    rec = MacroRecorder(tick_hz=30.0)
    assert not rec.is_active

    rec.start("walk")
    assert rec.is_active
    assert rec.name == "walk"

    a = neutral_action()
    a[BUTTON_INDEX["A"]] = 1.0

    rec.append(neutral_action(), t=100.0)
    rec.append(a, t=100.05)
    rec.append(a, t=100.10)

    macro = rec.stop()
    assert not rec.is_active
    assert macro.name == "walk"
    assert macro.tick_hz == 30.0
    assert len(macro.frames) == 3
    assert macro.frames[0].dt == 0.0
    assert macro.frames[1].dt == pytest.approx(0.05, abs=1e-6)
    assert macro.frames[2].dt == pytest.approx(0.05, abs=1e-6)
    assert macro.frames[1].action[BUTTON_INDEX["A"]] == 1.0


def test_recorder_double_start_raises():
    rec = MacroRecorder(tick_hz=30.0)
    rec.start("a")
    with pytest.raises(RuntimeError):
        rec.start("b")


def test_recorder_stop_without_start_raises():
    rec = MacroRecorder(tick_hz=30.0)
    with pytest.raises(RuntimeError):
        rec.stop()


def test_recorder_append_when_inactive_is_noop():
    rec = MacroRecorder(tick_hz=30.0)
    rec.append(neutral_action(), t=1.0)  # silently dropped
    assert rec.frame_count == 0


def test_recorder_clamps_negative_dt():
    rec = MacroRecorder(tick_hz=30.0)
    rec.start("a")
    rec.append(neutral_action(), t=10.0)
    rec.append(neutral_action(), t=9.5)  # clock went backwards
    macro = rec.stop()
    assert macro.frames[1].dt == 0.0


def test_recorder_cancel_drops_frames():
    rec = MacroRecorder(tick_hz=30.0)
    rec.start("a")
    rec.append(neutral_action(), t=1.0)
    rec.cancel()
    assert not rec.is_active
    assert rec.frame_count == 0


def test_recorder_normalizes_action_dtype():
    rec = MacroRecorder(tick_hz=30.0)
    rec.start("a")
    rec.append(np.zeros(ACTION_DIM, dtype=np.float64), t=1.0)
    macro = rec.stop()
    assert len(macro.frames[0].action) == ACTION_DIM
