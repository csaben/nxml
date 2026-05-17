"""The 26-dim action contract — layout + roundtrip invariants.

These tests guard the canonical ordering. If they fail, existing checkpoints'
``action_encoder`` weights are incompatible with the new layout — that's a v2
of the action spec and you need a migration story, not a fix.
"""

from __future__ import annotations

import numpy as np
import pytest
from nx_packets import (
    ACTION_DIM,
    BUTTON_DIMS,
    BUTTON_INDEX,
    BUTTON_NAMES,
    BUTTON_RANGE,
    STICK_DIMS,
    STICK_RANGE,
    Packet,
    StickData,
    action_to_packet,
    neutral_action,
    packet_to_action,
)


def test_dimensions():
    assert ACTION_DIM == 26
    assert STICK_DIMS == 4
    assert BUTTON_DIMS == 22
    assert STICK_DIMS + BUTTON_DIMS == ACTION_DIM


def test_button_index_layout():
    """The exact index for each button — load-bearing, do not change."""
    assert BUTTON_INDEX["L_STICK_PRESSED"] == 4
    assert BUTTON_INDEX["R_STICK_PRESSED"] == 5
    assert BUTTON_INDEX["DPAD_UP"] == 6
    assert BUTTON_INDEX["DPAD_LEFT"] == 7
    assert BUTTON_INDEX["DPAD_RIGHT"] == 8
    assert BUTTON_INDEX["DPAD_DOWN"] == 9
    assert BUTTON_INDEX["L"] == 10
    assert BUTTON_INDEX["ZL"] == 11
    assert BUTTON_INDEX["R"] == 12
    assert BUTTON_INDEX["ZR"] == 13
    assert BUTTON_INDEX["JCL_SR"] == 14
    assert BUTTON_INDEX["JCL_SL"] == 15
    assert BUTTON_INDEX["JCR_SR"] == 16
    assert BUTTON_INDEX["JCR_SL"] == 17
    assert BUTTON_INDEX["PLUS"] == 18
    assert BUTTON_INDEX["MINUS"] == 19
    assert BUTTON_INDEX["HOME"] == 20
    assert BUTTON_INDEX["CAPTURE"] == 21
    assert BUTTON_INDEX["Y"] == 22
    assert BUTTON_INDEX["X"] == 23
    assert BUTTON_INDEX["B"] == 24
    assert BUTTON_INDEX["A"] == 25


def test_button_names_unique_and_sized():
    assert len(BUTTON_NAMES) == BUTTON_DIMS
    assert len(set(BUTTON_NAMES)) == BUTTON_DIMS
    assert slice(0, 4) == STICK_RANGE
    assert slice(4, 26) == BUTTON_RANGE


def test_neutral_action_shape_and_zero():
    a = neutral_action()
    assert a.shape == (ACTION_DIM,)
    assert a.dtype == np.float32
    assert (a == 0).all()


def test_packet_to_action_roundtrip_full_state():
    """Roundtrip a Packet with every field exercised."""
    pkt = Packet(
        L_STICK=StickData(PRESSED=True, X_VALUE=-50, Y_VALUE=75),
        R_STICK=StickData(PRESSED=False, X_VALUE=100, Y_VALUE=-100),
        DPAD_UP=True,
        DPAD_LEFT=False,
        DPAD_RIGHT=True,
        DPAD_DOWN=False,
        L=True,
        ZL=False,
        R=True,
        ZR=False,
        JCL_SR=True,
        JCL_SL=False,
        JCR_SR=True,
        JCR_SL=False,
        PLUS=True,
        MINUS=False,
        HOME=True,
        CAPTURE=False,
        Y=True,
        X=False,
        B=True,
        A=False,
    )
    arr = packet_to_action(pkt)
    assert arr.shape == (ACTION_DIM,)
    assert arr.dtype == np.float32

    # Sticks: -50/100=-0.5, 75/100=0.75, 100/100=1.0, -100/100=-1.0
    np.testing.assert_allclose(arr[:4], [-0.5, 0.75, 1.0, -1.0], atol=1e-6)

    # Round-trip back to a Packet — values must match.
    pkt2 = action_to_packet(arr)
    assert pkt2.model_dump(exclude={"L_STICK", "R_STICK"}) == pkt.model_dump(
        exclude={"L_STICK", "R_STICK"}
    )
    # Stick PRESSED + X/Y survive.
    assert pkt2.L_STICK.PRESSED == pkt.L_STICK.PRESSED
    assert pkt2.L_STICK.X_VALUE == pkt.L_STICK.X_VALUE
    assert pkt2.L_STICK.Y_VALUE == pkt.L_STICK.Y_VALUE
    assert pkt2.R_STICK.PRESSED == pkt.R_STICK.PRESSED
    assert pkt2.R_STICK.X_VALUE == pkt.R_STICK.X_VALUE
    assert pkt2.R_STICK.Y_VALUE == pkt.R_STICK.Y_VALUE


def test_action_to_packet_thresholding():
    """Sub-threshold button probabilities decode to False; over decode to True."""
    a = neutral_action()
    a[BUTTON_INDEX["A"]] = 0.6
    a[BUTTON_INDEX["B"]] = 0.4
    pkt = action_to_packet(a, button_threshold=0.5)
    assert pkt.A is True
    assert pkt.B is False


def test_action_to_packet_rejects_bad_shape():
    with pytest.raises(ValueError, match="action must be"):
        action_to_packet(np.zeros(25, dtype=np.float32))


def test_action_to_packet_accepts_torch_tensor():
    """Torch tensors should be detached + pulled to CPU before conversion."""
    torch = pytest.importorskip("torch")
    t = torch.zeros(ACTION_DIM)
    t[BUTTON_INDEX["A"]] = 1.0
    pkt = action_to_packet(t)
    assert pkt.A is True


def test_auxiliary_stick_fields_default_off_and_dont_affect_action():
    """LS_UP/LS_DOWN/etc are aux signals — packet_to_action must not consult them."""
    pkt = Packet(
        L_STICK=StickData(LS_UP=True, LS_DOWN=True, LS_LEFT=True, LS_RIGHT=True),
        R_STICK=StickData(RS_UP=True, RS_DOWN=True, RS_LEFT=True, RS_RIGHT=True),
    )
    arr = packet_to_action(pkt)
    # All zeros — LS_*/RS_* aux signals must not bleed into the 26-dim vector.
    assert (arr == 0).all()
