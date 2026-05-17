"""Transport-agnostic client interface for the world model server.

The UI/agent code talks to a :class:`WorldModelClient`. Implementations:

  - :class:`InProcessClient` — wraps a :class:`WorldModelServer` directly,
    no serialization. Used when ``--model`` is a local path or ``hf:`` URI.
  - :class:`RemoteZMQClient` — speaks the binary ZMQ protocol against a
    server running elsewhere. Used for ``zmq://``/``tcp://`` URIs.

Use :func:`build_client` for URL-scheme dispatch — UI code should never
instantiate clients directly.
"""

from __future__ import annotations

import json
import struct
from dataclasses import asdict
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np

from nxwm.serve.server import WorldModelServer
from nxwm.serve.transports.zmq import (
    ACTION_SIZE,
    MODE_ACTION,
    MODE_INFO,
    MODE_INIT,
    MODE_RELOAD,
    MODE_RESEED,
)


class WorldModelClient(Protocol):
    """Interface every client implementation must provide."""

    def step(self, action: np.ndarray) -> bytes: ...

    def step_with_telemetry(
        self, action: np.ndarray
    ) -> tuple[bytes, dict[str, Any]]: ...

    def reseed(self, file: str, start_frame: int = 100) -> None: ...

    def reload(self, model_path: str) -> None: ...

    def info(self) -> dict[str, Any]: ...

    def init_from_frames(
        self, frames_jpeg: list[bytes], goal_jpeg: bytes | None = None
    ) -> None: ...

    # Detector control. Implementations that don't carry a detector raise
    # ``RuntimeError`` (no detector configured). Remote transports that
    # haven't been extended raise ``NotImplementedError``.
    def detector_config(self) -> dict[str, Any]: ...

    def apply_detector_params(
        self, params: dict[str, Any]
    ) -> dict[str, Any]: ...

    def reset_detector(self) -> None: ...

    def detector_debug_image(self) -> "np.ndarray | None": ...

    def close(self) -> None: ...


class InProcessClient:
    """Direct method calls on a :class:`WorldModelServer` — no serialization."""

    def __init__(self, server: WorldModelServer):
        self.server = server

    def step(self, action: np.ndarray) -> bytes:
        return self.server.step(action)

    def step_with_telemetry(
        self, action: np.ndarray
    ) -> tuple[bytes, dict[str, Any]]:
        return self.server.step_with_telemetry(action)

    def reseed(self, file: str, start_frame: int = 100) -> None:
        self.server.reseed(file, start_frame=start_frame)

    def reload(self, model_path: str) -> None:
        self.server.reload(model_path)

    def info(self) -> dict[str, Any]:
        return asdict(self.server.info())

    def init_from_frames(
        self, frames_jpeg: list[bytes], goal_jpeg: bytes | None = None
    ) -> None:
        self.server.init_from_frames(frames_jpeg, goal_jpeg=goal_jpeg)

    def detector_config(self) -> dict[str, Any]:
        return self.server.detector_config()

    def apply_detector_params(
        self, params: dict[str, Any]
    ) -> dict[str, Any]:
        return self.server.apply_detector_params(params)

    def reset_detector(self) -> None:
        self.server.reset_detector()

    def detector_debug_image(self) -> "np.ndarray | None":
        return self.server.detector_debug_image()

    def close(self) -> None:
        # Nothing to clean up — server is owned by the caller's process.
        pass


class RemoteZMQClient:
    """ZMQ REQ client speaking the binary protocol defined in
    :mod:`nxwm.serve.transports.zmq`.
    """

    def __init__(self, endpoint: str, *, recv_timeout_ms: int = 30_000):
        import zmq

        self._zmq = zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        self.socket.connect(endpoint)

    def _request(self, payload: bytes) -> bytes:
        self.socket.send(payload)
        return self.socket.recv()

    @staticmethod
    def _raise_if_error(resp: bytes) -> None:
        if resp.startswith(b"ERROR"):
            raise RuntimeError(resp.decode(errors="replace"))

    def step(self, action: np.ndarray) -> bytes:
        if action.dtype != np.float32:
            action = action.astype(np.float32, copy=False)
        if action.shape != (26,):
            raise ValueError(f"action must be (26,), got {action.shape}")
        payload = bytes([MODE_ACTION]) + action.tobytes()
        if len(payload) - 1 != ACTION_SIZE:
            raise ValueError(f"action size mismatch: expected {ACTION_SIZE} bytes")
        resp = self._request(payload)
        self._raise_if_error(resp)
        return resp

    def reseed(self, file: str, start_frame: int = 100) -> None:
        body = json.dumps({"file": file, "start_frame": int(start_frame)}).encode()
        resp = self._request(bytes([MODE_RESEED]) + body)
        self._raise_if_error(resp)
        if resp != b"OK":
            raise RuntimeError(f"unexpected reseed response: {resp!r}")

    def reload(self, model_path: str) -> None:
        body = json.dumps({"model_path": str(model_path)}).encode()
        resp = self._request(bytes([MODE_RELOAD]) + body)
        self._raise_if_error(resp)
        if resp != b"OK":
            raise RuntimeError(f"unexpected reload response: {resp!r}")

    def info(self) -> dict[str, Any]:
        resp = self._request(bytes([MODE_INFO]))
        self._raise_if_error(resp)
        return json.loads(resp.decode())

    def init_from_frames(
        self, frames_jpeg: list[bytes], goal_jpeg: bytes | None = None
    ) -> None:
        if not (1 <= len(frames_jpeg) <= 255):
            raise ValueError(f"num_frames must be in [1, 255], got {len(frames_jpeg)}")
        payload = bytes([len(frames_jpeg), 1 if goal_jpeg is not None else 0])
        for f in frames_jpeg:
            payload += struct.pack("<I", len(f)) + f
        if goal_jpeg is not None:
            payload += struct.pack("<I", len(goal_jpeg)) + goal_jpeg
        resp = self._request(bytes([MODE_INIT]) + payload)
        self._raise_if_error(resp)
        if resp != b"OK":
            raise RuntimeError(f"unexpected init response: {resp!r}")

    # Detector control — the ZMQ wire format doesn't carry detector
    # telemetry. Live tuning over a remote model would need a transport
    # extension; for now we surface a clear error rather than degrading.
    def step_with_telemetry(
        self, action: np.ndarray
    ) -> tuple[bytes, dict[str, Any]]:
        return self.step(action), {}

    def detector_config(self) -> dict[str, Any]:
        return {}

    def apply_detector_params(self, params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(
            "detector tuning over the ZMQ transport isn't supported; "
            "run nxwm ui with --model as a local path or hf: URI"
        )

    def reset_detector(self) -> None:
        raise NotImplementedError("see apply_detector_params")

    def detector_debug_image(self) -> "np.ndarray | None":
        return None

    def close(self) -> None:
        if not self.socket.closed:
            self.socket.close()
        if not self.context.closed:
            self.context.term()


def build_client(model_uri: str, **kwargs: Any) -> WorldModelClient:
    """URL-scheme dispatch.

      - ``zmq://host:port`` / ``tcp://host:port`` → :class:`RemoteZMQClient`
      - ``http://...`` / ``https://...``         → :class:`RemoteHTTPClient`
      - ``hf:owner/repo/file.pt``                 → :class:`InProcessClient` (resolved via ``huggingface_hub``)
      - ``./path``, ``/abs/path``, ``path``       → :class:`InProcessClient`

    ``kwargs`` flow into :class:`WorldModelServer` for in-process clients; for
    remote clients only ``recv_timeout_ms`` (plus ``api_prefix`` / ``verify_tls``
    for HTTP) are honored — the rest are silently dropped (a warning is the
    caller's responsibility, e.g. the CLI).
    """
    parsed = urlparse(model_uri)

    if parsed.scheme in ("zmq", "tcp"):
        endpoint = (
            model_uri.replace("zmq://", "tcp://", 1) if parsed.scheme == "zmq" else model_uri
        )
        recv_timeout_ms = int(kwargs.get("recv_timeout_ms", 30_000))
        return RemoteZMQClient(endpoint, recv_timeout_ms=recv_timeout_ms)

    if parsed.scheme in ("http", "https"):
        from nxwm.serve.remote_http import RemoteHTTPClient

        return RemoteHTTPClient(
            model_uri,
            api_prefix=str(kwargs.get("api_prefix", "/v1")),
            recv_timeout_ms=int(kwargs.get("recv_timeout_ms", 30_000)),
            verify_tls=bool(kwargs.get("verify_tls", True)),
        )

    # Default: local path or hf: URI → in-process. WorldModelServer's
    # constructor calls resolve_model_uri which handles both.
    _remote_only = {"recv_timeout_ms", "api_prefix", "verify_tls"}
    server_kwargs: dict[str, Any] = {
        k: v for k, v in kwargs.items() if k not in _remote_only
    }
    server = WorldModelServer(model_path=model_uri, **server_kwargs)
    return InProcessClient(server)
