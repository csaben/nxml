"""Pydantic models for Nintendo Switch input.

The ``Packet`` model is the structured form of one frame of controller state.
The 26-dim action vector contract — used by world models and policies — is
defined in :mod:`nx_packets.action_spec`. This module is the *structured*
side of that contract; ``action_spec`` is the *array* side.

Notes:
  - ``StickData.LS_UP``/``LS_LEFT``/``LS_RIGHT``/``LS_DOWN`` and the ``RS_*``
    counterparts are **auxiliary** binary direction signals consumed by the
    downstream nxbt/switch-emulation tools. They are NOT part of the 26-dim
    action vector — keep them defaulted to ``False`` when synthesizing actions
    from a model.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from pydantic import BaseModel, Field


class StickData(BaseModel):
    """Analog stick state. See module docstring re: auxiliary ``LS_*``/``RS_*`` fields."""

    PRESSED: bool = False
    X_VALUE: int = Field(default=0, ge=-100, le=100)
    Y_VALUE: int = Field(default=0, ge=-100, le=100)
    # Auxiliary direction signals — not part of the 26-dim contract.
    LS_UP: bool = False
    LS_LEFT: bool = False
    LS_RIGHT: bool = False
    LS_DOWN: bool = False
    RS_UP: bool = False
    RS_LEFT: bool = False
    RS_RIGHT: bool = False
    RS_DOWN: bool = False


class Packet(BaseModel):
    """One frame of controller state."""

    L_STICK: StickData
    R_STICK: StickData
    DPAD_UP: bool = False
    DPAD_LEFT: bool = False
    DPAD_RIGHT: bool = False
    DPAD_DOWN: bool = False
    L: bool = False
    ZL: bool = False
    R: bool = False
    ZR: bool = False
    JCL_SR: bool = False
    JCL_SL: bool = False
    JCR_SR: bool = False
    JCR_SL: bool = False
    PLUS: bool = False
    MINUS: bool = False
    HOME: bool = False
    CAPTURE: bool = False
    Y: bool = False
    X: bool = False
    B: bool = False
    A: bool = False


class PacketData(BaseModel):
    """A timed sequence of ``Packet``s — used for replay scripts."""

    timing: str
    packets: list[Packet]

    _current_index: int = -1
    _timing_value: float = -1.0

    def __post_init__(self) -> None:
        try:
            self._timing_value = float(Fraction(self.timing))
        except ValueError as e:
            raise ValueError(f"Invalid timing format: {self.timing}") from e

    def __iter__(self):
        self._current_index = -1
        return self

    def __next__(self) -> Packet:
        self._current_index += 1
        if self._current_index < len(self.packets):
            return self.packets[self._current_index]
        raise StopIteration

    def next(self) -> Packet:
        return self.__next__()

    @property
    def sleep_duration(self) -> float:
        if self._timing_value == -1.0:
            self.__post_init__()
        return self._timing_value

    def save_to_file(self, file_path: str | Path) -> None:
        Path(file_path).write_text(self.model_dump_json(indent=4))
