"""HTTPS client speaking the protocol from :mod:`nxwm.serve.transports.http`.

Implements :class:`nxwm.serve.client.WorldModelClient`, so any caller that
already speaks to ``InProcessClient`` / ``RemoteZMQClient`` works against
a remote ``nxwm serve --transport http`` endpoint without code changes.

Per-frame work goes over a persistent WebSocket at ``/v1/session``; admin
ops (``info``, ``reload``) go over plain HTTP at ``/v1/*``. The WS is
opened lazily on the first ``step`` / ``init_from_frames`` / ``reseed``
call and reused for the lifetime of this client.
"""

from __future__ import annotations

import json
import struct
import urllib.parse
import urllib.request
from typing import Any
from urllib.error import HTTPError

import numpy as np

from nxwm.serve.transports.zmq import ACTION_SIZE  # 104 bytes — single source of truth

INIT_OPCODE = 0x01


def _join(base: str, path: str) -> str:
    """``base`` may include a path prefix (``https://host/v1``); ``path`` is
    appended without losing it. Trailing/leading slashes are normalized.
    """
    return base.rstrip("/") + "/" + path.lstrip("/")


def _http_to_ws(base: str) -> str:
    """Map an ``http(s)://`` base URL to its ``ws(s)://`` counterpart."""
    if base.startswith("https://"):
        return "wss://" + base[len("https://") :]
    if base.startswith("http://"):
        return "ws://" + base[len("http://") :]
    return base  # already ws(s):// — let the WS lib complain if it isn't


class RemoteHTTPClient:
    """Sync :class:`WorldModelClient` over HTTP + WebSocket.

    ``base_url`` is the prefix the server is mounted at — for ``nxwm serve``
    that's ``https://host:port`` (the app exposes ``/v1/*`` directly). If
    you front the server with a reverse proxy that mounts it at ``/wm``,
    pass ``https://host/wm``.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_prefix: str = "/v1",
        recv_timeout_ms: int = 30_000,
        verify_tls: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_prefix = api_prefix.strip("/")
        self.recv_timeout = recv_timeout_ms / 1000.0
        self._verify_tls = verify_tls

        self._ws = None  # websockets.sync.client.ClientConnection | None
        self._ws_url = _http_to_ws(_join(self.base_url, self.api_prefix + "/session"))
        self._http_root = _join(self.base_url, self.api_prefix)

    # ------------------------------------------------------------------
    # HTTP one-shots
    # ------------------------------------------------------------------

    def _http_get(self, path: str) -> Any:
        url = _join(self._http_root, path)
        req = urllib.request.Request(url, method="GET")
        return self._http_send(req)

    def _http_post_json(self, path: str, body: dict) -> Any:
        url = _join(self._http_root, path)
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        return self._http_send(req)

    def _http_send(self, req: urllib.request.Request) -> Any:
        ctx = None
        if not self._verify_tls:
            import ssl

            ctx = ssl._create_unverified_context()
        try:
            with urllib.request.urlopen(req, timeout=self.recv_timeout, context=ctx) as resp:
                payload = resp.read()
        except HTTPError as e:
            # Server returned an error JSON body — surface its message.
            body = e.read().decode(errors="replace") if e.fp is not None else ""
            raise RuntimeError(f"HTTP {e.code}: {body or e.reason}") from e
        if not payload:
            return None
        return json.loads(payload.decode("utf-8"))

    def info(self) -> dict[str, Any]:
        return self._http_get("info") or {}

    def reload(self, model_path: str) -> None:
        out = self._http_post_json("reload", {"model_path": str(model_path)})
        if not isinstance(out, dict) or not out.get("ok"):
            raise RuntimeError(f"unexpected reload response: {out!r}")
        # The server invalidated the session bound to our WS — drop our cached
        # connection so the next call opens a fresh one and gets a "ready".
        self._close_ws()

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    def _ws_conn(self):
        if self._ws is not None:
            return self._ws
        try:
            from websockets.sync.client import connect
        except ImportError as e:  # pragma: no cover - dependency surface
            raise RuntimeError(
                "websockets>=12 is required for the HTTPS transport "
                "(pip install websockets or uv sync the nxwm extras)"
            ) from e

        ssl_ctx = None
        if self._ws_url.startswith("wss://") and not self._verify_tls:
            import ssl

            ssl_ctx = ssl._create_unverified_context()

        ws = connect(
            self._ws_url,
            open_timeout=self.recv_timeout,
            close_timeout=self.recv_timeout,
            max_size=64 * 1024 * 1024,
            ssl=ssl_ctx,
        )
        # First frame is always a JSON "ready" — drain it. We don't need any
        # of its fields right now, but reading clears the socket so the first
        # real reply isn't mistaken for ready state.
        try:
            ready = ws.recv(timeout=self.recv_timeout)
        except Exception:
            ws.close()
            raise
        if isinstance(ready, bytes) or not ready:
            ws.close()
            raise RuntimeError(f"expected ready text frame, got {ready!r}")
        self._ws = ws
        return ws

    def _close_ws(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    @staticmethod
    def _maybe_raise(payload: Any) -> None:
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(payload["error"])

    def _ws_send_action(self, action: np.ndarray) -> tuple[bytes, dict[str, Any]]:
        if action.dtype != np.float32:
            action = action.astype(np.float32, copy=False)
        if action.shape != (26,):
            raise ValueError(f"action must be (26,), got {action.shape}")
        ws = self._ws_conn()
        ws.send(action.tobytes())
        text = ws.recv(timeout=self.recv_timeout)
        if isinstance(text, bytes):
            raise RuntimeError("expected telemetry text frame; got binary")
        payload = json.loads(text)
        self._maybe_raise(payload)
        jpeg = ws.recv(timeout=self.recv_timeout)
        if isinstance(jpeg, str):
            raise RuntimeError(f"expected JPEG binary frame; got text: {jpeg!r}")
        telemetry: dict[str, Any] = payload.get("telemetry") or {}
        return jpeg, telemetry

    def step(self, action: np.ndarray) -> bytes:
        jpeg, _ = self._ws_send_action(action)
        return jpeg

    def step_with_telemetry(self, action: np.ndarray) -> tuple[bytes, dict[str, Any]]:
        return self._ws_send_action(action)

    def init_from_frames(
        self, frames_jpeg: list[bytes], goal_jpeg: bytes | None = None
    ) -> None:
        if not (1 <= len(frames_jpeg) <= 255):
            raise ValueError(f"num_frames must be in [1, 255], got {len(frames_jpeg)}")
        payload = bytes([INIT_OPCODE, len(frames_jpeg), 1 if goal_jpeg is not None else 0])
        for f in frames_jpeg:
            payload += struct.pack("<I", len(f)) + f
        if goal_jpeg is not None:
            payload += struct.pack("<I", len(goal_jpeg)) + goal_jpeg
        ws = self._ws_conn()
        ws.send(payload)
        text = ws.recv(timeout=self.recv_timeout)
        if isinstance(text, bytes):
            raise RuntimeError("expected text reply to init")
        out = json.loads(text)
        self._maybe_raise(out)

    def reseed(self, file: str, start_frame: int = 100) -> None:
        self._ws_control({"op": "reseed", "file": file, "start_frame": int(start_frame)})

    def _ws_control(self, body: dict[str, Any]) -> dict[str, Any]:
        ws = self._ws_conn()
        ws.send(json.dumps(body))
        text = ws.recv(timeout=self.recv_timeout)
        if isinstance(text, bytes):
            raise RuntimeError("expected JSON text reply")
        out = json.loads(text)
        self._maybe_raise(out)
        return out

    # ------------------------------------------------------------------
    # Detector
    # ------------------------------------------------------------------

    def detector_config(self) -> dict[str, Any]:
        out = self._ws_control({"op": "detector_config"})
        return out.get("config") or {}

    def apply_detector_params(self, params: dict[str, Any]) -> dict[str, Any]:
        out = self._ws_control({"op": "detector_params", "params": params})
        return {k: v for k, v in out.items() if k != "type"}

    def reset_detector(self) -> None:
        self._ws_control({"op": "detector_reset"})

    def detector_debug_image(self) -> "np.ndarray | None":
        # The server doesn't fan the debug PNG out through the WS today —
        # the UI's `/api/detector/debug.png` route lives in nxwm.ui.routes,
        # not in the transport. A remote viewer that needs it can hit the
        # admin HTTP surface directly; for the WorldModelClient Protocol
        # we return None to keep the cv-less consumer path quiet.
        return None

    def close(self) -> None:
        self._close_ws()
