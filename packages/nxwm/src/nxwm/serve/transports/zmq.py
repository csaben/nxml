"""ZMQ transport for :class:`WorldModelServer`.

Binary REP-socket protocol. Model/state work is delegated to
:class:`WorldModelServer`; this file is purely message-shape adaptation.

Wire protocol summary
---------------------

  ===========  =======================================================
  ``MODE_LEGACY``  Naked 104-byte action (no leading byte). Backwards-compat.
  ``MODE_ACTION``  ``0x00`` + 104-byte action. Returns JPEG bytes (or ``b"ERROR..."``).
  ``MODE_INIT``    ``0x01`` + payload. Sets up state from client-supplied JPEGs.
  ``MODE_RESEED``  ``0x02`` + JSON ``{"file": str, "start_frame": int}``. Returns ``b"OK"``.
  ``MODE_RELOAD``  ``0x03`` + JSON ``{"model_path": str}``. Returns ``b"OK"``.
  ``MODE_INFO``    ``0x04`` (single byte). Returns JSON with legacy-compat keys.
  ===========  =======================================================

Errors are returned as ``b"ERROR: <message>"`` byte strings. Exceptions never
escape ``run`` — they're caught in ``_handle`` and turned into error responses
so a misbehaving client can't kill the server.
"""

from __future__ import annotations

import json
import struct
from typing import Any, Final

import numpy as np
import zmq

from nxwm.serve.server import WorldModelServer

# Protocol modes.
MODE_ACTION: Final[int] = 0x00
MODE_INIT: Final[int] = 0x01
MODE_RESEED: Final[int] = 0x02
MODE_RELOAD: Final[int] = 0x03
MODE_INFO: Final[int] = 0x04
MODE_LEGACY: Final[int] = 0xFF
ACTION_SIZE: Final[int] = 26 * 4  # 104 bytes


def parse_message(msg: bytes) -> tuple[int, bytes]:
    """Parse one inbound message into ``(mode, payload)``."""
    if len(msg) == ACTION_SIZE:
        return MODE_LEGACY, msg
    if len(msg) == ACTION_SIZE + 1 and msg[0] == MODE_ACTION:
        return MODE_ACTION, msg[1:]
    if len(msg) > 1 and msg[0] == MODE_INIT:
        return MODE_INIT, msg[1:]
    if len(msg) > 1 and msg[0] == MODE_RESEED:
        return MODE_RESEED, msg[1:]
    if len(msg) > 1 and msg[0] == MODE_RELOAD:
        return MODE_RELOAD, msg[1:]
    if len(msg) == 1 and msg[0] == MODE_INFO:
        return MODE_INFO, b""
    return MODE_LEGACY, msg


def _parse_init_payload(data: bytes) -> tuple[list[bytes], bytes | None]:
    """Parse the ``MODE_INIT`` payload into ``(frame_jpegs, optional_goal_jpeg)``.

    Format (verbatim):
        u8 num_frames
        u8 has_goal
        for i in range(num_frames):
            u32_le frame_size
            byte[frame_size] jpeg
        if has_goal:
            u32_le goal_size
            byte[goal_size] goal_jpeg

    Raises ``ValueError`` (prefixed ``"ERROR: ..."``) on any framing error.
    """
    if len(data) < 2:
        raise ValueError("ERROR: Init message too short")
    num_frames = data[0]
    has_goal = data[1]
    offset = 2

    frames: list[bytes] = []
    for i in range(num_frames):
        if offset + 4 > len(data):
            raise ValueError(f"ERROR: EOF at frame {i}")
        (frame_size,) = struct.unpack("<I", data[offset : offset + 4])
        offset += 4
        if offset + frame_size > len(data):
            raise ValueError(f"ERROR: Frame {i} size mismatch")
        frames.append(bytes(data[offset : offset + frame_size]))
        offset += frame_size

    goal_jpeg: bytes | None = None
    if has_goal:
        if offset + 4 > len(data):
            raise ValueError("ERROR: Missing goal size")
        (goal_size,) = struct.unpack("<I", data[offset : offset + 4])
        offset += 4
        goal_jpeg = bytes(data[offset : offset + goal_size])

    return frames, goal_jpeg


def info_to_legacy_wire(info) -> dict[str, Any]:
    """Translate the modern ``ServerInfo`` into the legacy info JSON shape."""
    return {
        "current_model": info.current_model_path,
        "npz_files": info.available_episodes,
        "checkpoints": info.available_checkpoints,
        "flow_steps": info.flow_steps,
        "cfg_scale": info.cfg_scale,
        "history_length": info.history_length,
        "goal_offset": info.goal_offset,
        "latent_scale": info.latent_scale,
        "current_episode_frame": info.current_episode_frame,
    }


class ZMQTransport:
    """Wraps a :class:`WorldModelServer` in the binary ZMQ REP-socket protocol."""

    def __init__(
        self,
        server: WorldModelServer,
        *,
        port: int = 5556,
        host: str = "*",
        context: zmq.Context | None = None,
    ):
        self.server = server
        self.port = port
        self.host = host
        self._owns_context = context is None
        self.context = context if context is not None else zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self, *, poll_timeout_ms: int = 1000) -> None:
        """Main loop. Blocks until ``stop()`` is called or KeyboardInterrupt."""
        self._running = True
        try:
            while self._running:
                if self.socket.poll(timeout=poll_timeout_ms) == 0:
                    continue
                msg = self.socket.recv()
                response = self._handle(msg)
                self.socket.send(response)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def stop(self) -> None:
        self._running = False

    def shutdown(self) -> None:
        self._running = False
        if not self.socket.closed:
            self.socket.close()
        if self._owns_context and not self.context.closed:
            self.context.term()

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _handle(self, msg: bytes) -> bytes:
        try:
            mode, data = parse_message(msg)
            if mode == MODE_INFO:
                return json.dumps(info_to_legacy_wire(self.server.info())).encode()
            if mode == MODE_RESEED:
                params = json.loads(data.decode())
                self.server.reseed(
                    params["file"], int(params.get("start_frame", 100))
                )
                return b"OK"
            if mode == MODE_RELOAD:
                params = json.loads(data.decode())
                self.server.reload(params["model_path"])
                return b"OK"
            if mode == MODE_INIT:
                try:
                    frames_jpeg, goal_jpeg = _parse_init_payload(data)
                except ValueError as e:
                    return str(e).encode()
                self.server.init_from_frames(frames_jpeg, goal_jpeg)
                return b"OK"
            if mode in (MODE_ACTION, MODE_LEGACY):
                if len(data) != ACTION_SIZE:
                    return b"ERROR: Invalid action size"
                action = np.frombuffer(data, dtype=np.float32)
                # frombuffer returns a read-only view; copy so server can use it.
                return self.server.step(action.copy())
            return b"ERROR: Unknown mode"
        except FileNotFoundError as e:
            return f"ERROR: {e}".encode()
        except Exception as e:
            return f"ERROR: {e}".encode()
