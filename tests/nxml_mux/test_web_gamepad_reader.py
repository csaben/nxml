"""WebGamepadReader — basic snapshot + button-press latch."""

from __future__ import annotations

import numpy as np
import pytest
from nxml_mux.input_devices.readers import WebGamepadReader
from nx_packets import ACTION_DIM, BUTTON_INDEX, BUTTON_RANGE, neutral_action


def _press_a():
    a = neutral_action()
    a[BUTTON_INDEX["A"]] = 1.0
    return a


def test_latest_none_until_first_push():
    r = WebGamepadReader()
    assert r.latest() is None


def test_basic_round_trip():
    r = WebGamepadReader()
    r.push_action(_press_a())
    snap = r.latest()
    assert snap is not None
    assert snap.action[BUTTON_INDEX["A"]] == 1.0
    assert snap.mask is not None
    assert bool(snap.mask[BUTTON_INDEX["A"]]) is True


def test_held_button_persists_across_reads():
    r = WebGamepadReader()
    r.push_action(_press_a())
    s1 = r.latest()
    assert s1 is not None
    assert s1.action[BUTTON_INDEX["A"]] == 1.0
    # Without a re-push the cached state still says A=1.
    r.push_action(_press_a())
    s2 = r.latest()
    assert s2 is not None
    assert s2.action[BUTTON_INDEX["A"]] == 1.0


def test_latch_preserves_brief_tap():
    """Press → release within one read window: tap survives one read.

    Mirrors the real-world failure: the browser pushes A=1, then 16ms
    later pushes A=0, before the runner's 33ms tick reads ``latest()``.
    With the latch, that read should see A=1; the *next* read sees A=0.
    """
    r = WebGamepadReader()
    r.push_action(_press_a())
    r.push_action(neutral_action())  # released before mux read
    snap = r.latest()
    assert snap is not None
    assert snap.action[BUTTON_INDEX["A"]] == 1.0, "latch should preserve the tap"
    # Latch is one-shot — next read returns the current (released) state.
    snap2 = r.latest()
    assert snap2 is not None
    assert snap2.action[BUTTON_INDEX["A"]] == 0.0


def test_latch_only_affects_buttons_not_sticks():
    """Sticks are continuous — they should reflect current state, not latch."""
    r = WebGamepadReader()
    a1 = neutral_action()
    a1[0] = 0.8  # L stick X
    r.push_action(a1)
    a2 = neutral_action()
    a2[0] = 0.0
    r.push_action(a2)
    snap = r.latest()
    assert snap is not None
    assert snap.action[0] == 0.0, "sticks must NOT latch"


def test_multiple_button_taps_collapse_to_one_pressed_read():
    """Many fast taps within one read window come out as a single 'pressed' read."""
    r = WebGamepadReader()
    for _ in range(5):
        r.push_action(_press_a())
        r.push_action(neutral_action())
    snap = r.latest()
    assert snap is not None
    assert snap.action[BUTTON_INDEX["A"]] == 1.0
    snap2 = r.latest()
    assert snap2 is not None
    assert snap2.action[BUTTON_INDEX["A"]] == 0.0


def test_stop_clears_latch():
    r = WebGamepadReader()
    r.push_action(_press_a())
    r.stop()
    assert r.latest() is None


def test_push_rejects_wrong_shape():
    r = WebGamepadReader()
    with pytest.raises(ValueError):
        r.push_action(np.zeros(ACTION_DIM - 1, dtype=np.float32))


def test_stick_deadzone_applied():
    r = WebGamepadReader(stick_deadzone=0.2)
    a = neutral_action()
    a[0] = 0.1  # below deadzone
    a[1] = 0.5  # above
    r.push_action(a)
    snap = r.latest()
    assert snap is not None
    assert snap.action[0] == 0.0
    assert snap.action[1] == 0.5


def test_mask_reflects_latched_button():
    r = WebGamepadReader()
    r.push_action(_press_a())
    r.push_action(neutral_action())
    snap = r.latest()
    assert snap is not None
    assert snap.mask is not None
    # The mask should mark A as actively claimed even though the latest
    # underlying push was neutral.
    assert bool(snap.mask[BUTTON_INDEX["A"]]) is True
    # Neutral buttons stay unmasked.
    assert not bool(snap.mask[BUTTON_INDEX["B"]])


def test_buttons_at_other_indices_unaffected_by_latch():
    r = WebGamepadReader()
    r.push_action(_press_a())
    r.push_action(neutral_action())
    snap = r.latest()
    assert snap is not None
    others = list(snap.action[BUTTON_RANGE])
    # Only A (index 25 in the absolute vector → BUTTON_RANGE[-1]) was pushed.
    a_local_idx = BUTTON_INDEX["A"] - 4
    for i, val in enumerate(others):
        if i == a_local_idx:
            assert val == 1.0
        else:
            assert val == 0.0
