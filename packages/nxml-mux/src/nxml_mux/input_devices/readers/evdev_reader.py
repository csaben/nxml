"""evdev-backed gamepad reader.

Runs a background thread that reads events from a Linux evdev device,
applies a :class:`Mapper` to translate codes/values into the canonical
``nx_packets`` 26-dim vector, and exposes the current state via
``latest()`` as an :class:`ActionSnapshot` with a derived "active" mask
(buttons currently pressed; stick axes currently outside the deadzone).

Threading model:
  - One thread per device, started in ``start()``.
  - State updates take a short lock; ``latest()`` copies the state under
    the same lock so the consumer always sees a consistent frame.
  - ``stop()`` closes the device, which raises ``OSError`` inside
    ``read_loop`` and breaks us out cleanly.
"""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path

import evdev
import numpy as np
from evdev import ecodes
from nx_packets import ACTION_DIM, BUTTON_RANGE, STICK_RANGE

from nxml_mux.input_devices.mappers.base import Mapper
from nxml_mux.source import ActionSnapshot


class EvdevReader:
    def __init__(self, device_path: str, mapper: Mapper, *, source_id: str | None = None) -> None:
        self._device_path = device_path
        self._mapper = mapper
        self.source_id = source_id or f"evdev:{Path(device_path).name}"

        self._action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._lock = threading.Lock()
        self._latest_ts = 0.0
        self._device: evdev.InputDevice | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # Populated in start():
        self._axis_norm: dict[int, tuple[float, float]] = {}
        self._code_to_str: dict[int, str] = {}

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        device = evdev.InputDevice(self._device_path)
        self._device = device

        all_codes = (
            set(self._mapper.button_map.keys())
            | set(self._mapper.stick_map.keys())
            | set(self._mapper.trigger_map.keys())
            | set(self._mapper.hat_map.keys())
        )
        for code_str in all_codes:
            code_int = ecodes.ecodes.get(code_str)
            if code_int is None:
                continue
            self._code_to_str[code_int] = code_str

        # Axis normalizers: read absinfo for each axis code present on the device.
        axis_codes = (
            set(self._mapper.stick_map.keys())
            | set(self._mapper.trigger_map.keys())
            | set(self._mapper.hat_map.keys())
        )
        for code_str in axis_codes:
            code_int = ecodes.ecodes.get(code_str)
            if code_int is None:
                continue
            try:
                absinfo = device.absinfo(code_int)
            except OSError:
                # Axis not present on this device; events for it just won't fire.
                continue
            mid = (absinfo.max + absinfo.min) / 2.0
            half = (absinfo.max - absinfo.min) / 2.0 or 1.0
            self._axis_norm[code_int] = (mid, half)

        self._thread = threading.Thread(target=self._loop, daemon=True, name=self.source_id)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._device is not None:
            with contextlib.suppress(Exception):
                self._device.close()
            self._device = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def latest(self) -> ActionSnapshot | None:
        with self._lock:
            if self._latest_ts == 0.0:
                return None
            action = self._action.copy()
            ts = self._latest_ts
        mask = np.zeros(ACTION_DIM, dtype=bool)
        mask[STICK_RANGE] = action[STICK_RANGE] != 0.0
        mask[BUTTON_RANGE] = action[BUTTON_RANGE] >= 0.5
        return ActionSnapshot(action=action, timestamp=ts, source_id=self.source_id, mask=mask)

    def _loop(self) -> None:
        device = self._device
        if device is None:
            return
        try:
            for event in device.read_loop():
                if self._stop.is_set():
                    break
                self._handle(event)
        except OSError:
            # Triggered by stop() closing the device — clean exit.
            return

    def _handle(self, event: evdev.InputEvent) -> None:
        code_str = self._code_to_str.get(event.code)
        if code_str is None:
            return

        if event.type == ecodes.EV_KEY:
            idx = self._mapper.button_map.get(code_str)
            if idx is None:
                return
            with self._lock:
                self._action[idx] = 1.0 if event.value else 0.0
                self._latest_ts = event.timestamp()
            return

        if event.type != ecodes.EV_ABS:
            return

        mid, half = self._axis_norm.get(event.code, (0.0, 1.0))
        normalized = (event.value - mid) / half
        normalized = max(-1.0, min(1.0, normalized))

        if code_str in self._mapper.stick_map:
            m = self._mapper.stick_map[code_str]
            v = -normalized if m.invert else normalized
            if abs(v) < m.deadzone:
                v = 0.0
            with self._lock:
                self._action[m.axis_index] = v
                self._latest_ts = event.timestamp()
            return

        if code_str in self._mapper.trigger_map:
            m = self._mapper.trigger_map[code_str]
            # Triggers commonly rest at min and grow toward max — normalized is
            # then ~[-1, 1] with rest at -1. Remap to [0, 1] pressure.
            pressure = (-normalized + 1.0) / 2.0 if m.invert else (normalized + 1.0) / 2.0
            pressed = pressure >= m.threshold
            with self._lock:
                self._action[m.button_index] = 1.0 if pressed else 0.0
                self._latest_ts = event.timestamp()
            return

        if code_str in self._mapper.hat_map:
            m = self._mapper.hat_map[code_str]
            with self._lock:
                if event.value < 0:
                    self._action[m.neg_button_index] = 1.0
                    self._action[m.pos_button_index] = 0.0
                elif event.value > 0:
                    self._action[m.neg_button_index] = 0.0
                    self._action[m.pos_button_index] = 1.0
                else:
                    self._action[m.neg_button_index] = 0.0
                    self._action[m.pos_button_index] = 0.0
                self._latest_ts = event.timestamp()
