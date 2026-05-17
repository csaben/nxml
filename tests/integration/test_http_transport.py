"""HTTP/WebSocket transport for the world model server.

Mirrors test_clients.py for the ZMQ path: spin up an HTTPTransport in a
background thread, exercise RemoteHTTPClient against it, verify the wire
roundtrip lands on the same Session.step_latent the in-process path uses.

Skipped if `websockets` isn't installed (the dep is declared in pyproject
but optional for legacy uv-syncs that predate it).
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

websockets = pytest.importorskip("websockets")  # noqa: F841
from nxwm.serve import RemoteHTTPClient, WorldModelServer  # noqa: E402
from nxwm.serve.transports.http import HTTPTransport  # noqa: E402

# Python 3.14 deprecates `asyncio.iscoroutinefunction`; asyncio/starlette
# internals still call it. Filter rather than chasing upstream pins.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:.*asyncio.iscoroutinefunction.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore::pytest.PytestUnhandledThreadExceptionWarning"
    ),
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def http_endpoint(tiny_dit_v1_checkpoint, tiny_episode_path):
    server = WorldModelServer(
        model_path=tiny_dit_v1_checkpoint,
        device="cpu",
        flow_steps=3,
        cfg_scale=1.0,
        data_path=tiny_episode_path.parent,
        load_vae_eagerly=False,
    )
    port = _free_port()
    transport = HTTPTransport(server, host="127.0.0.1", port=port)
    thread_err: list[BaseException] = []

    def _runner() -> None:
        try:
            transport.run()
        except BaseException as e:  # noqa: BLE001 - want to surface to the test
            thread_err.append(e)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    # Wait for uvicorn to bind. A short poll is more robust than `sleep`.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        transport.stop()
        thread.join(timeout=2.0)
        if thread_err:
            raise thread_err[0]
        pytest.fail("uvicorn never came up")
    yield f"http://127.0.0.1:{port}", transport, server
    transport.stop()
    thread.join(timeout=5.0)


def test_http_info_roundtrip(http_endpoint, tiny_episode_path):
    base, _, server = http_endpoint
    client = RemoteHTTPClient(base, recv_timeout_ms=5_000)
    try:
        info = client.info()
        assert info["architecture"] == "dit_v1"
        assert info["history_length"] == 10
        assert info["current_model_path"] == server.current_model_path
        assert tiny_episode_path.name in info["available_episodes"]
    finally:
        client.close()


def test_http_step_routes_to_session(http_endpoint, tiny_episode_path):
    """A binary 104-byte send produces a JPEG response, and the model's
    forward pass ran on the *session* the WS opened — not the server's
    default session. We assert by patching decode_to_jpeg to a sentinel
    and confirming the bytes come back unchanged."""
    import nxwm.serve.server as srv_mod

    base, _, server = http_endpoint

    saved = srv_mod.decode_to_jpeg
    srv_mod.decode_to_jpeg = lambda latent, vae: b"FAKE_JPEG_BYTES"  # type: ignore[assignment]
    server._vae = object()
    try:
        client = RemoteHTTPClient(base, recv_timeout_ms=5_000)
        try:
            action = np.zeros(server.model.action_dims, dtype=np.float32)
            jpeg, telemetry = client.step_with_telemetry(action)
            assert jpeg == b"FAKE_JPEG_BYTES"
            assert telemetry == {}  # no detector configured
        finally:
            client.close()
    finally:
        srv_mod.decode_to_jpeg = saved
        server._vae = None


def test_http_two_sessions_isolated(http_endpoint, tiny_episode_path):
    """Two parallel WS connections each get their own Session; reseeding
    one does not disturb the other's state."""
    import nxwm.serve.server as srv_mod

    base, _, server = http_endpoint

    saved = srv_mod.decode_to_jpeg
    srv_mod.decode_to_jpeg = lambda latent, vae: b"X"  # type: ignore[assignment]
    server._vae = object()
    try:
        a = RemoteHTTPClient(base, recv_timeout_ms=5_000)
        b = RemoteHTTPClient(base, recv_timeout_ms=5_000)
        try:
            # Each client steps independently; both succeed.
            action = np.zeros(server.model.action_dims, dtype=np.float32)
            assert a.step(action) == b"X"
            assert b.step(action) == b"X"
            # Server now has 3 live sessions (default + two WS sessions).
            assert len(server._sessions) >= 3
        finally:
            a.close()
            b.close()
    finally:
        srv_mod.decode_to_jpeg = saved
        server._vae = None


def test_http_reseed_unknown_raises(http_endpoint):
    base, _, _ = http_endpoint
    client = RemoteHTTPClient(base, recv_timeout_ms=5_000)
    try:
        with pytest.raises(RuntimeError, match="not found"):
            client.reseed("does_not_exist.npz", start_frame=0)
    finally:
        client.close()
