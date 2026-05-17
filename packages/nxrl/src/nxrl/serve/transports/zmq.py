"""ZMQ transport — policy-inference wire protocol.

  =====================  =====================================================
  ``MODE_PREDICT``       ``0x00`` + ``<u32 T><u32 C><u32 H><u32 W>`` + ``T*C*H*W*4`` float32 bytes.
                         Stateless. Returns ``104`` bytes (26 float32 action).
  ``MODE_PREDICT_FRAME`` ``0x01`` + raw JPEG bytes (single frame).
                         Stateful — server keeps a sliding latent window and
                         runs VAE encode itself. Returns ``104`` bytes
                         (action) once the window is full, or ``WARMING`` (1
                         byte ``\\x00``) while warming up.
  ``MODE_RELOAD``        ``0x03`` + JSON ``{"model_path": str}``. Returns ``b"OK"``.
  ``MODE_INFO``          ``0x04`` (single byte). Returns JSON.
  =====================  =====================================================

Errors are returned as ``b"ERROR: <msg>"`` byte strings; ``_handle`` never
lets exceptions escape so a misbehaving client can't kill the server.

``MODE_PREDICT_FRAME`` exists so a client without a local GPU/VAE can hand
off raw frames and get actions back. The single-window assumption pins us
to one client at a time per server — see ``server.predict_frame``.
"""

from __future__ import annotations

import json
import struct
from typing import Final

import numpy as np
import zmq

from nxrl.serve.server import PolicyServer

MODE_PREDICT: Final[int] = 0x00
MODE_PREDICT_FRAME: Final[int] = 0x01
MODE_RELOAD: Final[int] = 0x03
MODE_INFO: Final[int] = 0x04
ACTION_SIZE: Final[int] = 26 * 4
HEADER_SIZE: Final[int] = 1 + 4 * 4  # mode + 4 u32 dims
WARMING_RESPONSE: Final[bytes] = b"\x00"  # frame mode: window not full yet


def encode_predict_request(latents: np.ndarray) -> bytes:
    """Build the ``MODE_PREDICT`` payload. ``latents`` must be ``(T, C, H, W)``
    or ``(1, T, C, H, W)`` float32. The leading batch dim is squeezed.
    """
    arr = np.asarray(latents, dtype=np.float32)
    if arr.ndim == 5:
        if arr.shape[0] != 1:
            raise ValueError(f"batched predict not supported on the wire; got B={arr.shape[0]}")
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"latents must be (T,C,H,W); got {arr.shape}")
    t, c, h, w = arr.shape
    return bytes([MODE_PREDICT]) + struct.pack("<IIII", t, c, h, w) + arr.tobytes(order="C")


def parse_predict_payload(data: bytes) -> np.ndarray:
    """Parse the body following the MODE_PREDICT byte. Returns a
    ``(T, C, H, W)`` float32 numpy array."""
    if len(data) < 16:
        raise ValueError("predict payload too short for header")
    t, c, h, w = struct.unpack("<IIII", data[:16])
    expected = t * c * h * w * 4
    if len(data) - 16 != expected:
        raise ValueError(
            f"predict payload size mismatch: header says {expected} bytes, got {len(data) - 16}"
        )
    arr = np.frombuffer(data[16:], dtype=np.float32).reshape(t, c, h, w)
    return arr


def encode_predict_frame_request(jpeg_bytes: bytes) -> bytes:
    """Build the ``MODE_PREDICT_FRAME`` payload — opcode + raw JPEG."""
    return bytes([MODE_PREDICT_FRAME]) + jpeg_bytes


def info_to_wire(info) -> dict:  # pyright: ignore[reportMissingTypeArgument]
    return {
        "current_model_path": info.current_model_path,
        "architecture": info.architecture,
        "config": info.config,
        "sequence_length": info.sequence_length,
        "latent_shape": list(info.latent_shape),
        "action_dim": info.action_dim,
        "algorithm": info.algorithm,
        "available_checkpoints": info.available_checkpoints,
    }


class ZMQTransport:
    def __init__(
        self,
        server: PolicyServer,
        *,
        port: int = 5557,
        host: str = "*",
        context: zmq.Context | None = None,  # pyright: ignore[reportMissingTypeArgument]
    ) -> None:
        self.server = server
        self.port = port
        self.host = host
        self._owns_context = context is None
        self.context = context if context is not None else zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._running = False

    def run(self, *, poll_timeout_ms: int = 1000) -> None:
        self._running = True
        try:
            while self._running:
                if self.socket.poll(timeout=poll_timeout_ms) == 0:
                    continue
                msg = self.socket.recv()
                self.socket.send(self._handle(msg))
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

    def _handle(self, msg: bytes) -> bytes:
        try:
            if not msg:
                return b"ERROR: empty message"
            mode = msg[0]
            if mode == MODE_INFO and len(msg) == 1:
                return json.dumps(info_to_wire(self.server.info())).encode()
            if mode == MODE_RELOAD:
                params = json.loads(msg[1:].decode())
                self.server.reload(params["model_path"])
                return b"OK"
            if mode == MODE_PREDICT:
                latents = parse_predict_payload(msg[1:])
                action = self.server.predict(latents)
                return action.tobytes(order="C")
            if mode == MODE_PREDICT_FRAME:
                action = self.server.predict_frame(msg[1:])
                if action is None:
                    return WARMING_RESPONSE
                return action.tobytes(order="C")
            return b"ERROR: Unknown mode"
        except FileNotFoundError as e:
            return f"ERROR: {e}".encode()
        except Exception as e:
            return f"ERROR: {e}".encode()
