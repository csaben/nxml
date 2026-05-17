"""Mapper YAML schema + loader.

A mapper translates one physical controller's evdev events into
``nx_packets`` action-vector contributions. The YAML schema::

    id: xbox_one
    extends: null                # optional relative path to a parent mapper
    device_match:
      name_regex: "Microsoft.*Xbox.*"
    buttons:
      # evdev BTN_* string -> switch button name (from nx_packets.BUTTON_NAMES)
      BTN_SOUTH: B
      BTN_EAST: A
    axes:
      ABS_X:
        kind: stick
        target: L_STICK_X        # one of L_STICK_X, L_STICK_Y, R_STICK_X, R_STICK_Y
        deadzone: 0.1
        invert: false
      ABS_Z:
        kind: trigger
        target: ZL               # any switch button name
        threshold: 0.5
        invert: false
      ABS_HAT0X:
        kind: hat_pair
        targets: [DPAD_LEFT, DPAD_RIGHT]    # negative-value, positive-value

Inheritance: ``extends:`` points at another mapper file (relative path from
the current file's directory). Child overrides parent on per-key collisions
in ``buttons`` and ``axes``; ``device_match`` and ``id`` come from the
child.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from nx_packets import BUTTON_INDEX

STICK_AXIS_NAMES: dict[str, int] = {
    "L_STICK_X": 0,
    "L_STICK_Y": 1,
    "R_STICK_X": 2,
    "R_STICK_Y": 3,
}


@dataclass(frozen=True)
class StickAxisMap:
    axis_index: int  # 0..3 in nx_packets vector
    deadzone: float
    invert: bool


@dataclass(frozen=True)
class TriggerAxisMap:
    button_index: int  # absolute index 4..25
    threshold: float
    invert: bool


@dataclass(frozen=True)
class HatAxisMap:
    neg_button_index: int  # written when value < 0
    pos_button_index: int  # written when value > 0


@dataclass(frozen=True)
class Mapper:
    id: str
    name_regex: re.Pattern[str] | None
    # evdev string code -> action info
    button_map: dict[str, int] = field(default_factory=dict)
    stick_map: dict[str, StickAxisMap] = field(default_factory=dict)
    trigger_map: dict[str, TriggerAxisMap] = field(default_factory=dict)
    hat_map: dict[str, HatAxisMap] = field(default_factory=dict)

    def matches(self, device_name: str) -> bool:
        if self.name_regex is None:
            return False
        return self.name_regex.search(device_name) is not None


def _resolve_button_index(name: str, ctx: str) -> int:
    if name not in BUTTON_INDEX:
        raise ValueError(f"{ctx}: unknown switch button {name!r}")
    return BUTTON_INDEX[name]


def _merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Shallow merge of mapper raws; child sub-keys override parent's."""
    out: dict[str, Any] = dict(parent)
    for k, v in child.items():
        if k in {"buttons", "axes"} and isinstance(v, dict) and isinstance(out.get(k), dict):
            merged = dict(out[k])
            merged.update(v)
            out[k] = merged
        else:
            out[k] = v
    return out


def _load_raw(path: Path) -> dict[str, Any]:
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(raw).__name__}")
    extends = raw.get("extends")
    if extends:
        parent_path = (path.parent / extends).resolve()
        parent = _load_raw(parent_path)
        raw = _merge(parent, raw)
        raw.pop("extends", None)
    return raw


def load_mapper(path: str | Path) -> Mapper:
    """Load a single mapper YAML, resolving ``extends:`` recursively."""
    p = Path(path)
    raw = _load_raw(p)

    mapper_id = raw.get("id") or p.stem
    device_match = raw.get("device_match") or {}
    name_regex_str = device_match.get("name_regex")
    name_regex = re.compile(name_regex_str) if name_regex_str else None

    button_map: dict[str, int] = {}
    for evdev_code, target in (raw.get("buttons") or {}).items():
        button_map[evdev_code] = _resolve_button_index(target, f"{p}: button {evdev_code}")

    stick_map: dict[str, StickAxisMap] = {}
    trigger_map: dict[str, TriggerAxisMap] = {}
    hat_map: dict[str, HatAxisMap] = {}

    for evdev_code, axis in (raw.get("axes") or {}).items():
        ctx = f"{p}: axis {evdev_code}"
        if not isinstance(axis, dict) or "kind" not in axis:
            raise ValueError(f"{ctx}: axis must be a mapping with a 'kind' field")
        kind = axis["kind"]
        if kind == "stick":
            target = axis.get("target")
            if target not in STICK_AXIS_NAMES:
                raise ValueError(f"{ctx}: stick target must be one of {sorted(STICK_AXIS_NAMES)}")
            stick_map[evdev_code] = StickAxisMap(
                axis_index=STICK_AXIS_NAMES[target],
                deadzone=float(axis.get("deadzone", 0.0)),
                invert=bool(axis.get("invert", False)),
            )
        elif kind == "trigger":
            target = axis.get("target")
            if not isinstance(target, str):
                raise ValueError(f"{ctx}: trigger target must be a string")
            trigger_map[evdev_code] = TriggerAxisMap(
                button_index=_resolve_button_index(target, ctx),
                threshold=float(axis.get("threshold", 0.5)),
                invert=bool(axis.get("invert", False)),
            )
        elif kind == "hat_pair":
            targets = axis.get("targets")
            if not (isinstance(targets, list) and len(targets) == 2):
                raise ValueError(f"{ctx}: hat_pair needs exactly 2 targets [neg, pos]")
            hat_map[evdev_code] = HatAxisMap(
                neg_button_index=_resolve_button_index(targets[0], ctx),
                pos_button_index=_resolve_button_index(targets[1], ctx),
            )
        else:
            raise ValueError(f"{ctx}: unknown axis kind {kind!r}")

    return Mapper(
        id=mapper_id,
        name_regex=name_regex,
        button_map=button_map,
        stick_map=stick_map,
        trigger_map=trigger_map,
        hat_map=hat_map,
    )


def load_mapper_dir(directory: str | Path) -> list[Mapper]:
    """Load every ``.yaml`` mapper in a directory (non-recursive)."""
    d = Path(directory)
    return [load_mapper(p) for p in sorted(d.glob("*.yaml"))]
