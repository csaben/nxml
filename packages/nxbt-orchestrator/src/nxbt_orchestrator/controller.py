"""NXBT wrapper: controller setup, shared state, fixed-rate update loop, recording.

This module is transport-agnostic. The HTTP/WS server lives in :mod:`server`
and drives the controller via the :class:`NxbtController` API:

  - :meth:`apply_packet` — write a full controller frame, with optional
    human-override gating for inference traffic.
  - :meth:`apply_action_vector` — same, but accepts the canonical 26-dim
    float vector from :mod:`nx_packets`.
  - :meth:`fire_macro` / :meth:`toggle_recording` / :meth:`set_recording_path`.
  - :meth:`subscribe_state` / :meth:`unsubscribe_state` — register a
    thread-safe callback fired each update tick, used by the WS state stream.
  - :meth:`snapshot_state` — one-shot read of the current state.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Literal

import nxbt
from nx_packets import Packet, action_to_packet

StateCallback = Callable[[dict[str, Any]], None]
ActionSource = Literal["human", "inference"]

_BUTTON_NAMES = (
    "A", "B", "X", "Y",
    "L", "R", "ZL", "ZR",
    "PLUS", "MINUS", "HOME", "CAPTURE",
    "DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT",
    "JCL_SR", "JCL_SL", "JCR_SR", "JCR_SL",
)


class NxbtController:
    def __init__(
        self,
        *,
        controller_type: int = nxbt.PRO_CONTROLLER,
        reconnect_address: str | None = None,
        update_rate: int = 120,
        override_window: float = 0.3,
        recording_output_path: str | Path = "recorded_macro.json",
        debug: bool = False,
    ) -> None:
        self.update_rate = update_rate
        self.override_window = override_window
        self.debug = debug

        self.nx = nxbt.Nxbt()
        self._running = Event()
        self._connected = Event()

        self.controller_idx = self.nx.create_controller(
            controller_type,
            reconnect_address=reconnect_address,
        )

        self._state_lock = Lock()
        self._controller_state: dict[str, Any] = self.nx.create_input_packet()
        self._init_sticks()

        # Tracks the wall-clock time of the most recent human-source action;
        # inference packets are dropped if the elapsed time is below
        # `override_window`.
        self._last_human_input_time: float = 0.0

        self._recording_active = False
        self._packet_history: list[dict[str, Any]] = []
        self._recording_output_path = Path(recording_output_path)

        self._subscribers: list[StateCallback] = []
        self._subscribers_lock = Lock()

        self._update_thread = Thread(
            target=self._continuous_update_loop,
            daemon=True,
            name="nxbt-update-loop",
        )

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self, *, wait_for_connection: bool = True) -> None:
        if wait_for_connection:
            print("[orchestrator] waiting for Switch to connect over Bluetooth...", flush=True)
            self.nx.wait_for_connection(self.controller_idx)
            print("[orchestrator] Switch connected", flush=True)
            try:
                addrs = self.nx.get_switch_addresses()
                if addrs:
                    print(
                        f"[orchestrator] paired Switch address(es): {', '.join(addrs)} — "
                        f"pass to --reconnect-address next run to skip the pairing dance",
                        flush=True,
                    )
            except Exception as e:
                if self.debug:
                    print(f"[orchestrator] could not enumerate paired Switches: {e}", flush=True)
        self._connected.set()
        self._running.set()
        self._update_thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._recording_active:
            self._save_recording()
        self._update_thread.join(timeout=2.0)
        self.nx.remove_controller(self.controller_idx)

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    # ── stick init (preserves nxbt quirks) ────────────────────────────

    def _init_sticks(self) -> None:
        for stick_key in ("L_STICK", "R_STICK"):
            prefix = "LS" if stick_key == "L_STICK" else "RS"
            if stick_key not in self._controller_state:
                self._controller_state[stick_key] = {
                    "PRESSED": False,
                    f"{prefix}_UP": False,
                    f"{prefix}_DOWN": False,
                    f"{prefix}_LEFT": False,
                    f"{prefix}_RIGHT": False,
                    "X_VALUE": 0,
                    "Y_VALUE": 0,
                    "ANALOG_X": 0,
                    "ANALOG_Y": 0,
                }
            else:
                for field in ("ANALOG_X", "ANALOG_Y"):
                    if field not in self._controller_state[stick_key]:
                        self._controller_state[stick_key][field] = 0

    # ── update loop (sends to Switch, fans out to subscribers) ────────

    def _continuous_update_loop(self) -> None:
        sleep_time = 1.0 / self.update_rate

        while self._running.is_set():
            with self._state_lock:
                snapshot = self._build_send_packet_locked()
                if self._recording_active:
                    self._packet_history.append(_deep_copy_packet(snapshot))

            self.nx.set_controller_input(self.controller_idx, snapshot)

            self._notify_subscribers(snapshot)
            time.sleep(sleep_time)

    def _build_send_packet_locked(self) -> dict[str, Any]:
        """Make a deep copy of state, mirroring ANALOG_* into X_VALUE/Y_VALUE.

        Must be called while holding ``self._state_lock``.
        """
        packet = dict(self._controller_state)
        packet["L_STICK"] = dict(self._controller_state["L_STICK"])
        packet["R_STICK"] = dict(self._controller_state["R_STICK"])

        packet["L_STICK"]["X_VALUE"] = self._controller_state["L_STICK"]["ANALOG_X"]
        packet["L_STICK"]["Y_VALUE"] = self._controller_state["L_STICK"]["ANALOG_Y"]
        packet["R_STICK"]["X_VALUE"] = self._controller_state["R_STICK"]["ANALOG_X"]
        packet["R_STICK"]["Y_VALUE"] = self._controller_state["R_STICK"]["ANALOG_Y"]
        return packet

    # ── action ingress ────────────────────────────────────────────────

    def apply_packet(
        self,
        packet: Packet | dict[str, Any],
        *,
        source: ActionSource = "inference",
    ) -> bool:
        """Apply one frame of controller state.

        Returns ``False`` if the packet was suppressed by the human-override
        window; ``True`` otherwise.
        """
        if source == "inference" and self._is_human_active():
            if self.debug:
                print("[orchestrator] inference packet suppressed (human active)")
            return False

        packet_dict = packet.model_dump() if isinstance(packet, Packet) else dict(packet)

        if source == "human":
            self._last_human_input_time = time.time()

        with self._state_lock:
            self._merge_packet_into_state_locked(packet_dict)
        return True

    def apply_action_vector(
        self,
        action: list[float] | tuple[float, ...],
        *,
        source: ActionSource = "inference",
        button_threshold: float = 0.5,
    ) -> bool:
        """Apply a 26-dim float action vector from ``nx_packets``."""
        packet = action_to_packet(action, button_threshold=button_threshold)
        return self.apply_packet(packet, source=source)

    def _is_human_active(self) -> bool:
        if self._last_human_input_time == 0.0:
            return False
        return (time.time() - self._last_human_input_time) < self.override_window

    def _merge_packet_into_state_locked(self, packet_dict: dict[str, Any]) -> None:
        for name in _BUTTON_NAMES:
            if name in self._controller_state:
                self._controller_state[name] = bool(packet_dict.get(name, False))

        l_stick = packet_dict.get("L_STICK") or {}
        r_stick = packet_dict.get("R_STICK") or {}

        self._controller_state["L_STICK"]["PRESSED"] = bool(l_stick.get("PRESSED", False))
        self._controller_state["R_STICK"]["PRESSED"] = bool(r_stick.get("PRESSED", False))

        # Accept either ANALOG_* (orchestrator-internal) or X_VALUE/Y_VALUE
        # (the wire form used by Packet) — they mean the same thing.
        self._controller_state["L_STICK"]["ANALOG_X"] = int(
            l_stick.get("ANALOG_X", l_stick.get("X_VALUE", 0))
        )
        self._controller_state["L_STICK"]["ANALOG_Y"] = int(
            l_stick.get("ANALOG_Y", l_stick.get("Y_VALUE", 0))
        )
        self._controller_state["R_STICK"]["ANALOG_X"] = int(
            r_stick.get("ANALOG_X", r_stick.get("X_VALUE", 0))
        )
        self._controller_state["R_STICK"]["ANALOG_Y"] = int(
            r_stick.get("ANALOG_Y", r_stick.get("Y_VALUE", 0))
        )

    def apply_button_set(self, buttons: list[str], *, source: ActionSource = "human") -> None:
        """Apply a set-of-button-names update (TCP `button` command shape)."""
        if source == "human":
            self._last_human_input_time = time.time()
        pressed = set(buttons)
        with self._state_lock:
            for name in list(self._controller_state.keys()):
                if name not in ("L_STICK", "R_STICK"):
                    self._controller_state[name] = name in pressed
            self._controller_state["L_STICK"]["PRESSED"] = "L_STICK_PRESS" in pressed
            self._controller_state["R_STICK"]["PRESSED"] = "R_STICK_PRESS" in pressed

    def apply_stick(
        self,
        stick: Literal["LEFT_STICK", "RIGHT_STICK"],
        x: int,
        y: int,
        *,
        source: ActionSource = "human",
    ) -> None:
        if source == "human":
            self._last_human_input_time = time.time()
        x = max(-100, min(100, int(x)))
        y = max(-100, min(100, int(y)))
        key = "L_STICK" if stick == "LEFT_STICK" else "R_STICK"
        with self._state_lock:
            self._controller_state[key]["ANALOG_X"] = x
            self._controller_state[key]["ANALOG_Y"] = y

    # ── macros ────────────────────────────────────────────────────────

    def fire_macro(self, macro: str, *, block: bool = False) -> int:
        return self.nx.macro(self.controller_idx, macro, block=block)

    # ── recording ─────────────────────────────────────────────────────

    @property
    def recording_active(self) -> bool:
        return self._recording_active

    def set_recording_path(self, path: str | Path) -> None:
        self._recording_output_path = Path(path)

    def toggle_recording(self) -> bool:
        """Toggle recording. Returns the new ``recording_active`` value."""
        if self._recording_active:
            self._recording_active = False
            self._save_recording()
            self._packet_history = []
        else:
            self._packet_history = []
            self._recording_active = True
        return self._recording_active

    def _save_recording(self) -> None:
        if not self._packet_history:
            return
        data = {
            "timing": f"1/{self.update_rate}",
            "packets": self._packet_history,
        }
        self._recording_output_path.write_text(json.dumps(data, indent=4))

    # ── state subscribers (for WS stream) ─────────────────────────────

    def subscribe_state(self, callback: StateCallback) -> None:
        with self._subscribers_lock:
            self._subscribers.append(callback)

    def unsubscribe_state(self, callback: StateCallback) -> None:
        with self._subscribers_lock, contextlib.suppress(ValueError):
            self._subscribers.remove(callback)

    def _notify_subscribers(self, packet: dict[str, Any]) -> None:
        with self._subscribers_lock:
            subscribers = list(self._subscribers)
        if not subscribers:
            return
        # The packet object is mutated by subsequent ticks; subscribers must
        # serialize/copy promptly. The state-stream module wraps it before
        # crossing the asyncio boundary.
        for cb in subscribers:
            try:
                cb(packet)
            except Exception as e:
                if self.debug:
                    print(f"[orchestrator] subscriber error: {e}")

    def snapshot_state(self) -> dict[str, Any]:
        with self._state_lock:
            return _deep_copy_packet(self._build_send_packet_locked())


def _deep_copy_packet(packet: dict[str, Any]) -> dict[str, Any]:
    out = dict(packet)
    if "L_STICK" in out:
        out["L_STICK"] = dict(out["L_STICK"])
    if "R_STICK" in out:
        out["R_STICK"] = dict(out["R_STICK"])
    return out
