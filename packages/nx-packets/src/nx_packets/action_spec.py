"""Canonical 26-dim action contract for Nintendo Switch input.

The 26-dim float32 action vector is the contract between:
  - World models (``action_encoder = nn.Linear(26, embed_dim)`` consumes it)
  - Policies (output it)
  - UIs and clients (serialize it)
  - nxbt-orchestrator (converts it to ``Packet`` for Bluetooth send)

Layout (indices into the 26-dim vector):

  ===  ====================  ===================================================
   0   L stick X             continuous, ``[-1, 1]``
   1   L stick Y             continuous, ``[-1, 1]``
   2   R stick X             continuous, ``[-1, 1]``
   3   R stick Y             continuous, ``[-1, 1]``
   4   L_STICK.PRESSED       binary
   5   R_STICK.PRESSED       binary
   6   DPAD_UP               binary
   7   DPAD_LEFT             binary
   8   DPAD_RIGHT            binary
   9   DPAD_DOWN             binary
  10   L                     binary
  11   ZL                    binary
  12   R                     binary
  13   ZR                    binary
  14   JCL_SR                binary
  15   JCL_SL                binary
  16   JCR_SR                binary
  17   JCR_SL                binary
  18   PLUS                  binary
  19   MINUS                 binary
  20   HOME                  binary
  21   CAPTURE               binary
  22   Y                     binary
  23   X                     binary
  24   B                     binary
  25   A                     binary
  ===  ====================  ===================================================

Sticks are stored on the wire as int8-ish in ``[-100, 100]`` inside ``StickData``,
but as float in ``[-1, 1]`` in the 26-dim vector. Conversion: ``action[0:4] * 100 = stick value``.

Buttons in the 26-dim vector are continuous in ``[0, 1]`` (post-sigmoid for
policies, or 0/1 for ground truth). Threshold at ``0.5`` to recover binary state.

This module is the **single source of truth**. Do not duplicate this layout
elsewhere; existing checkpoints' ``action_encoder`` weights depend on this
exact ordering.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np

from nx_packets.packet import Packet, StickData

ACTION_DIM: Final[int] = 26
STICK_DIMS: Final[int] = 4
BUTTON_DIMS: Final[int] = 22

STICK_RANGE: Final[slice] = slice(0, 4)
BUTTON_RANGE: Final[slice] = slice(4, 26)

# Names for indices 4..25 (button range). Add 4 for the absolute index.
BUTTON_NAMES: Final[list[str]] = [
    "L_STICK_PRESSED",
    "R_STICK_PRESSED",
    "DPAD_UP",
    "DPAD_LEFT",
    "DPAD_RIGHT",
    "DPAD_DOWN",
    "L",
    "ZL",
    "R",
    "ZR",
    "JCL_SR",
    "JCL_SL",
    "JCR_SR",
    "JCR_SL",
    "PLUS",
    "MINUS",
    "HOME",
    "CAPTURE",
    "Y",
    "X",
    "B",
    "A",
]
# Absolute (0..25) index into the 26-dim vector for each button name.
BUTTON_INDEX: Final[dict[str, int]] = {
    name: 4 + i for i, name in enumerate(BUTTON_NAMES)
}


def packet_to_action(packet: Packet) -> np.ndarray:
    """Convert a :class:`Packet` to the canonical 26-dim float32 vector."""
    arr = np.zeros(ACTION_DIM, dtype=np.float32)
    arr[0] = packet.L_STICK.X_VALUE / 100.0
    arr[1] = packet.L_STICK.Y_VALUE / 100.0
    arr[2] = packet.R_STICK.X_VALUE / 100.0
    arr[3] = packet.R_STICK.Y_VALUE / 100.0

    bits = [
        packet.L_STICK.PRESSED,
        packet.R_STICK.PRESSED,
        packet.DPAD_UP,
        packet.DPAD_LEFT,
        packet.DPAD_RIGHT,
        packet.DPAD_DOWN,
        packet.L,
        packet.ZL,
        packet.R,
        packet.ZR,
        packet.JCL_SR,
        packet.JCL_SL,
        packet.JCR_SR,
        packet.JCR_SL,
        packet.PLUS,
        packet.MINUS,
        packet.HOME,
        packet.CAPTURE,
        packet.Y,
        packet.X,
        packet.B,
        packet.A,
    ]
    arr[4:] = np.array(bits, dtype=np.float32)
    return arr


def action_to_packet(
    action: Any,
    *,
    button_threshold: float = 0.5,
) -> Packet:
    """Convert a 26-dim action vector to a :class:`Packet`.

    Accepts a numpy array or torch tensor (anything with ``.detach()`` is
    treated as a tensor and pulled to CPU). Sticks: ``action[0:4]`` in
    ``[-1, 1]`` → int ``[-100, 100]`` in the Packet. Buttons: ``action[4:26]``
    are interpreted as probabilities (or binary 0/1) and thresholded at
    ``button_threshold``.

    The caller is responsible for any sigmoid: pass already-sigmoided
    probabilities, or use ``button_threshold=0.0`` to threshold raw logits.
    """
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    arr = np.asarray(action)
    if arr.shape != (ACTION_DIM,):
        raise ValueError(f"action must be ({ACTION_DIM},), got {arr.shape}")

    sticks = (arr[STICK_RANGE] * 100.0).astype(np.int32).tolist()
    buttons = (arr[BUTTON_RANGE] > button_threshold).tolist()

    return Packet(
        L_STICK=StickData(
            PRESSED=bool(buttons[0]), X_VALUE=int(sticks[0]), Y_VALUE=int(sticks[1])
        ),
        R_STICK=StickData(
            PRESSED=bool(buttons[1]), X_VALUE=int(sticks[2]), Y_VALUE=int(sticks[3])
        ),
        DPAD_UP=bool(buttons[2]),
        DPAD_LEFT=bool(buttons[3]),
        DPAD_RIGHT=bool(buttons[4]),
        DPAD_DOWN=bool(buttons[5]),
        L=bool(buttons[6]),
        ZL=bool(buttons[7]),
        R=bool(buttons[8]),
        ZR=bool(buttons[9]),
        JCL_SR=bool(buttons[10]),
        JCL_SL=bool(buttons[11]),
        JCR_SR=bool(buttons[12]),
        JCR_SL=bool(buttons[13]),
        PLUS=bool(buttons[14]),
        MINUS=bool(buttons[15]),
        HOME=bool(buttons[16]),
        CAPTURE=bool(buttons[17]),
        Y=bool(buttons[18]),
        X=bool(buttons[19]),
        B=bool(buttons[20]),
        A=bool(buttons[21]),
    )


def neutral_action() -> np.ndarray:
    """All-zero action vector — no sticks, no buttons."""
    return np.zeros(ACTION_DIM, dtype=np.float32)
