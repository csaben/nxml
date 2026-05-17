"""Verify InProcessClient + RemoteZMQClient produce identical responses for the
same model + actions, and that build_client dispatches by URL scheme.
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest
from nxwm.serve import (
    InProcessClient,
    RemoteZMQClient,
    WorldModelServer,
    build_client,
)
from nxwm.serve.transports.zmq import ZMQTransport


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def in_process_server(tiny_dit_v1_checkpoint, tiny_episode_path):
    server = WorldModelServer(
        model_path=tiny_dit_v1_checkpoint,
        device="cpu",
        flow_steps=3,
        cfg_scale=1.0,
        data_path=tiny_episode_path.parent,
        load_vae_eagerly=False,
    )
    yield server


@pytest.fixture
def zmq_endpoint(in_process_server):
    """Start a ZMQTransport in a background thread, yield (endpoint, transport, server)."""
    port = _free_port()
    transport = ZMQTransport(in_process_server, port=port, host="127.0.0.1")
    thread = threading.Thread(
        target=transport.run, kwargs={"poll_timeout_ms": 50}, daemon=True
    )
    thread.start()
    time.sleep(0.1)
    yield f"tcp://127.0.0.1:{port}", transport, in_process_server
    transport.stop()
    thread.join(timeout=2.0)


def test_in_process_client_info_keys(in_process_server, tiny_episode_path):
    client = InProcessClient(in_process_server)
    info = client.info()
    # Modern keys (asdict of ServerInfo)
    assert info["architecture"] == "dit_v1"
    assert info["history_length"] == 10
    assert info["current_episode_file"] == tiny_episode_path.name
    assert tiny_episode_path.name in info["available_episodes"]
    client.close()


def test_remote_client_info_keys(zmq_endpoint):
    endpoint, _, server = zmq_endpoint
    client = RemoteZMQClient(endpoint, recv_timeout_ms=5000)
    try:
        info = client.info()
        # Wire-format keys (legacy compat) — different from in-process modern keys
        assert info["history_length"] == 10
        assert info["latent_scale"] == pytest.approx(0.18215)
        assert info["current_model"] == server.current_model_path
    finally:
        client.close()


def test_in_process_and_remote_step_latents_match(zmq_endpoint, tiny_episode_path):
    """Same model + same RNG + same action via either path → identical latent.

    JPEG encoding requires VAE which we skip in tests. Instead, both paths
    drive ``server.step_latent`` directly and we compare the latents stored
    on the server. Since the remote and in-process clients both ultimately
    call ``server.step``/``step_latent``, ``server.last_latent`` is the
    common comparison point.
    """
    endpoint, _, server = zmq_endpoint
    server.reseed(tiny_episode_path.name, start_frame=0)

    action = np.zeros(server.model.action_dims, dtype=np.float32)
    # Drive via step_latent directly (in-process, bypassing JPEG).
    import torch

    torch.manual_seed(7)
    in_proc_latent = server.step_latent(action).clone()

    # Reset state and drive via the remote client. The transport calls
    # server.step which calls step_latent internally + JPEG-encodes. We
    # don't have a VAE, so use a temporary monkeypatched decode_to_jpeg.
    server.reseed(tiny_episode_path.name, start_frame=0)
    import nxwm.serve.server as srv_mod

    saved = srv_mod.decode_to_jpeg
    srv_mod.decode_to_jpeg = lambda latent, vae: b"FAKE"  # type: ignore[assignment]
    server._vae = object()  # any non-None — patched decode_to_jpeg ignores it
    try:
        client = RemoteZMQClient(endpoint, recv_timeout_ms=5000)
        try:
            torch.manual_seed(7)
            resp = client.step(action)
            assert resp == b"FAKE"
        finally:
            client.close()
    finally:
        srv_mod.decode_to_jpeg = saved
        server._vae = None

    # Both paths drove server.step_latent with the same RNG → same last_latent.
    torch.testing.assert_close(server.last_latent, in_proc_latent, atol=0, rtol=0)


def test_remote_client_reseed_reload_roundtrip(zmq_endpoint, tiny_dit_v1_checkpoint, tiny_episode_path):
    endpoint, _, _ = zmq_endpoint
    client = RemoteZMQClient(endpoint, recv_timeout_ms=5000)
    try:
        client.reseed(tiny_episode_path.name, start_frame=0)
        client.reload(str(tiny_dit_v1_checkpoint))
    finally:
        client.close()


def test_remote_client_reseed_unknown_raises(zmq_endpoint):
    endpoint, _, _ = zmq_endpoint
    client = RemoteZMQClient(endpoint, recv_timeout_ms=5000)
    try:
        with pytest.raises(RuntimeError, match="ERROR"):
            client.reseed("does_not_exist.npz", start_frame=0)
    finally:
        client.close()


def test_build_client_dispatch_zmq():
    """zmq:// URI builds a RemoteZMQClient (without connecting to anything in particular)."""
    # We don't bind a server — just verify the dispatch yields the right type.
    # The client constructor connects but doesn't block until first request.
    client = build_client("zmq://127.0.0.1:1", recv_timeout_ms=100)
    try:
        assert isinstance(client, RemoteZMQClient)
    finally:
        client.close()


def test_build_client_dispatch_tcp():
    client = build_client("tcp://127.0.0.1:1", recv_timeout_ms=100)
    try:
        assert isinstance(client, RemoteZMQClient)
    finally:
        client.close()


def test_build_client_http_dispatch():
    """http(s):// URIs build a RemoteHTTPClient without connecting yet."""
    from nxwm.serve import RemoteHTTPClient

    client = build_client("https://example.com:8443", recv_timeout_ms=100)
    try:
        assert isinstance(client, RemoteHTTPClient)
        assert client.base_url == "https://example.com:8443"
    finally:
        client.close()


def test_build_client_local_path(tiny_dit_v1_checkpoint):
    client = build_client(
        str(tiny_dit_v1_checkpoint),
        device="cpu",
        load_vae_eagerly=False,
    )
    try:
        assert isinstance(client, InProcessClient)
        info = client.info()
        assert info["architecture"] == "dit_v1"
    finally:
        client.close()


def test_remote_client_init_from_frames_dispatches(zmq_endpoint):
    """The remote client correctly frames a MODE_INIT message.

    We don't have a VAE wired up, so init_from_frames will fail server-side
    when it tries to encode JPEGs — but we *should* get a clear ERROR back
    rather than a malformed response, proving the wire framing works.
    """
    endpoint, _, _ = zmq_endpoint
    client = RemoteZMQClient(endpoint, recv_timeout_ms=5000)
    try:
        with pytest.raises(RuntimeError):
            # Bytes won't decode as a valid JPEG, server will raise, client
            # converts the b"ERROR..." response to RuntimeError.
            client.init_from_frames([b"not-a-real-jpeg"])
    finally:
        client.close()
