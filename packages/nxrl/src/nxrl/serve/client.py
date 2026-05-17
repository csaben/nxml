"""Transport-agnostic policy client.

Implementations:
  - :class:`InProcessClient` — wraps a :class:`PolicyServer` directly.
  - :class:`RemoteZMQClient` — speaks the ZMQ binary protocol from
    :mod:`nxrl.serve.transports.zmq`.

Use :func:`build_client` for URL-scheme dispatch.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np

from nxrl.serve.server import PolicyServer
from nxrl.serve.transports.zmq import (
    ACTION_SIZE,
    MODE_INFO,
    MODE_RELOAD,
    WARMING_RESPONSE,
    encode_predict_frame_request,
    encode_predict_request,
)


class PolicyClient(Protocol):
    def predict(self, latents: np.ndarray) -> np.ndarray:
        ...

    def reload(self, model_path: str) -> None:
        ...

    def info(self) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...


class InProcessClient:
    def __init__(self, server: PolicyServer) -> None:
        self.server = server

    def predict(self, latents: np.ndarray) -> np.ndarray:
        return self.server.predict(latents)

    def reload(self, model_path: str) -> None:
        self.server.reload(model_path)

    def info(self) -> dict[str, Any]:
        return asdict(self.server.info())

    def close(self) -> None:
        pass


class RemoteZMQClient:
    def __init__(self, endpoint: str, *, recv_timeout_ms: int = 30_000) -> None:
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

    def predict(self, latents: np.ndarray) -> np.ndarray:
        payload = encode_predict_request(latents)
        resp = self._request(payload)
        self._raise_if_error(resp)
        if len(resp) != ACTION_SIZE:
            raise RuntimeError(f"unexpected predict response size: {len(resp)}")
        return np.frombuffer(resp, dtype=np.float32).copy()

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

    def close(self) -> None:
        if not self.socket.closed:
            self.socket.close()
        if not self.context.closed:
            self.context.term()


class RemoteZMQFrameClient:
    """Frame-mode counterpart to :class:`RemoteZMQClient`.

    Sends a JPEG per request; the server runs the VAE and keeps the
    sliding latent window. ``predict_frame`` returns ``None`` while the
    server is still warming up its window.

    Note: this client does NOT implement the :class:`PolicyClient`
    Protocol — its predict signature differs (bytes in, optional action
    out). Callers branch on instance type.
    """

    def __init__(self, endpoint: str, *, recv_timeout_ms: int = 30_000) -> None:
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

    def predict_frame(self, jpeg_bytes: bytes) -> np.ndarray | None:
        resp = self._request(encode_predict_frame_request(jpeg_bytes))
        self._raise_if_error(resp)
        if resp == WARMING_RESPONSE:
            return None
        if len(resp) != ACTION_SIZE:
            raise RuntimeError(f"unexpected predict_frame response size: {len(resp)}")
        return np.frombuffer(resp, dtype=np.float32).copy()

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

    def close(self) -> None:
        if not self.socket.closed:
            self.socket.close()
        if not self.context.closed:
            self.context.term()


def build_client(model_uri: str, **kwargs: Any) -> PolicyClient | RemoteZMQFrameClient:
    """URL-scheme dispatch.

      - ``zmq://host:port`` / ``tcp://host:port``            → :class:`RemoteZMQClient`
      - ``zmq+frames://host:port`` / ``tcp+frames://host:port`` → :class:`RemoteZMQFrameClient`
        (server-side VAE + sliding window — client just ships JPEGs)
      - ``http://`` / ``https://``                            → ``NotImplementedError``
      - ``hf:owner/repo/file.pt``                              → :class:`InProcessClient`
        (resolved via ``huggingface_hub`` inside ``PolicyServer``)
      - ``./path``, ``/abs/path``, ``path``                    → :class:`InProcessClient`

    For in-process clients, ``kwargs`` flow into :class:`PolicyServer`. For
    remote clients only ``recv_timeout_ms`` is honored — the rest are
    silently dropped.
    """
    parsed = urlparse(model_uri)

    if parsed.scheme in ("zmq+frames", "tcp+frames"):
        endpoint = "tcp://" + model_uri.split("://", 1)[1]
        recv_timeout_ms = int(kwargs.get("recv_timeout_ms", 30_000))
        return RemoteZMQFrameClient(endpoint, recv_timeout_ms=recv_timeout_ms)

    if parsed.scheme in ("zmq", "tcp"):
        endpoint = (
            model_uri.replace("zmq://", "tcp://", 1) if parsed.scheme == "zmq" else model_uri
        )
        recv_timeout_ms = int(kwargs.get("recv_timeout_ms", 30_000))
        return RemoteZMQClient(endpoint, recv_timeout_ms=recv_timeout_ms)

    if parsed.scheme in ("http", "https"):
        raise NotImplementedError("HTTP transport is Phase 2 work")

    server = PolicyServer(model_path=model_uri, **kwargs)
    return InProcessClient(server)
