"""Macro on-disk schema.

A ``Macro`` is a list of ``MacroFrame``s. Each frame carries a 26-dim
float action vector and a wall-clock ``dt`` (seconds since the previous
frame; the first frame's ``dt`` is ``0.0``). Replay sleeps on ``dt``,
so a macro recorded at variable rate replays at the same rate without
any resampling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator
from nx_packets import ACTION_DIM, PacketData, action_to_packet


class MacroFrame(BaseModel):
    action: list[float] = Field(min_length=ACTION_DIM, max_length=ACTION_DIM)
    dt: float = Field(ge=0.0)

    @field_validator("action")
    @classmethod
    def _finite(cls, v: list[float]) -> list[float]:
        if not all(np.isfinite(v)):
            raise ValueError("action contains non-finite values")
        return v


class Macro(BaseModel):
    name: str
    tick_hz: float = Field(gt=0.0)
    frames: list[MacroFrame]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return sum(f.dt for f in self.frames)

    def actions(self) -> np.ndarray:
        """Return frames stacked as a ``(N, 26)`` float32 array."""
        if not self.frames:
            return np.zeros((0, ACTION_DIM), dtype=np.float32)
        return np.asarray([f.action for f in self.frames], dtype=np.float32)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> Macro:
        return cls.model_validate_json(Path(path).read_text())

    def to_packet_data(self, *, button_threshold: float = 0.5) -> PacketData:
        """Bridge to ``nx_packets.PacketData`` for consumers that want the
        structured ``Packet`` form (e.g. nxbt's macro DSL).

        ``PacketData.timing`` is a single-rate string, so we use the macro's
        nominal ``tick_hz`` here. Per-frame ``dt`` is lost — if the consumer
        cares about variable timing, replay through ``MacroPlayer`` instead.
        """
        timing = f"1/{self.tick_hz:g}"
        packets = [
            action_to_packet(np.asarray(f.action, dtype=np.float32), button_threshold=button_threshold)
            for f in self.frames
        ]
        return PacketData(timing=timing, packets=packets)


def macro_from_arrays(
    *,
    name: str,
    tick_hz: float,
    actions: np.ndarray,
    dts: np.ndarray | list[float],
    metadata: dict[str, Any] | None = None,
) -> Macro:
    """Build a ``Macro`` from a ``(N, 26)`` array + length-N ``dt`` array.

    Convenience for callers that already have arrays in hand (tests,
    conversions from external formats).
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"actions must have shape (N, {ACTION_DIM}), got {actions.shape}")
    dts = list(dts)
    if len(dts) != actions.shape[0]:
        raise ValueError(f"len(dts)={len(dts)} != len(actions)={actions.shape[0]}")
    frames = [MacroFrame(action=a.tolist(), dt=float(dt)) for a, dt in zip(actions, dts, strict=True)]
    return Macro(name=name, tick_hz=tick_hz, frames=frames, metadata=metadata or {})
