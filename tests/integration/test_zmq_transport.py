"""Verbatim wire-protocol tests for ZMQTransport.

Three layers of test:

1. **Pure parser tests** — parse_message + _parse_init_payload on
   hand-crafted byte strings, no ZMQ socket involved.
2. **Info-shape test** — info_to_legacy_wire produces the exact JSON keys
   the legacy clients expect.
3. **End-to-end socket round-trip** — bind a real REP socket on a free
   port, drive a stub server through it, verify request/response framing.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time

import numpy as np
import pytest
import zmq
from nxwm.serve.server import ServerInfo
from nxwm.serve.transports.zmq import (
    ACTION_SIZE,
    MODE_ACTION,
    MODE_INFO,
    MODE_INIT,
    MODE_LEGACY,
    MODE_RELOAD,
    MODE_RESEED,
    ZMQTransport,
    _parse_init_payload,
    info_to_legacy_wire,
    parse_message,
)

# ---------------------------------------------------------------------------
# Pure parser tests
# ---------------------------------------------------------------------------


def test_parse_message_legacy_naked_action():
    msg = b"\x00" * ACTION_SIZE
    mode, data = parse_message(msg)
    assert mode == MODE_LEGACY
    assert data == msg


def test_parse_message_action_with_prefix():
    msg = bytes([MODE_ACTION]) + b"\x00" * ACTION_SIZE
    mode, data = parse_message(msg)
    assert mode == MODE_ACTION
    assert len(data) == ACTION_SIZE


def test_parse_message_init():
    payload = b"\x02\x00\x00\x00\x00\x01\x00\x00\x00\xff"
    mode, data = parse_message(bytes([MODE_INIT]) + payload)
    assert mode == MODE_INIT
    assert data == payload


def test_parse_message_reseed():
    payload = b'{"file":"a.npz","start_frame":7}'
    mode, data = parse_message(bytes([MODE_RESEED]) + payload)
    assert mode == MODE_RESEED
    assert data == payload


def test_parse_message_reload():
    payload = b'{"model_path":"/x/y.pt"}'
    mode, data = parse_message(bytes([MODE_RELOAD]) + payload)
    assert mode == MODE_RELOAD
    assert data == payload


def test_parse_message_info_singleton():
    mode, data = parse_message(bytes([MODE_INFO]))
    assert mode == MODE_INFO
    assert data == b""


def test_parse_message_unknown_falls_back_to_legacy():
    """Garbled input falls back to MODE_LEGACY (legacy behavior)."""
    mode, _ = parse_message(b"\x99\x88\x77")
    assert mode == MODE_LEGACY


def test_parse_init_payload_no_goal():
    frames = [b"jpeg-bytes-A", b"jpeg-bytes-B-longer"]
    payload = bytes([len(frames), 0])
    for f in frames:
        payload += struct.pack("<I", len(f)) + f
    parsed_frames, goal = _parse_init_payload(payload)
    assert parsed_frames == frames
    assert goal is None


def test_parse_init_payload_with_goal():
    frames = [b"f0", b"f1"]
    goal = b"goal-jpeg-data"
    payload = bytes([len(frames), 1])
    for f in frames:
        payload += struct.pack("<I", len(f)) + f
    payload += struct.pack("<I", len(goal)) + goal
    parsed_frames, parsed_goal = _parse_init_payload(payload)
    assert parsed_frames == frames
    assert parsed_goal == goal


def test_parse_init_payload_too_short_raises():
    with pytest.raises(ValueError, match="too short"):
        _parse_init_payload(b"\x00")


def test_parse_init_payload_size_mismatch_raises():
    payload = bytes([1, 0]) + struct.pack("<I", 100) + b"only-12-bytes"
    with pytest.raises(ValueError, match="size mismatch"):
        _parse_init_payload(payload)


def test_parse_init_payload_missing_goal_size_raises():
    frames = [b"f0"]
    payload = bytes([len(frames), 1])
    for f in frames:
        payload += struct.pack("<I", len(f)) + f
    # has_goal=1 but no trailing goal-size header
    with pytest.raises(ValueError, match="Missing goal size"):
        _parse_init_payload(payload)


# ---------------------------------------------------------------------------
# Info wire-shape test
# ---------------------------------------------------------------------------


def test_info_to_legacy_wire_keys():
    info = ServerInfo(
        current_model_path="/path/to/model.pt",
        architecture="dit_v1",
        config={"embed_dim": 64},
        history_length=10,
        goal_offset=30,
        flow_steps=5,
        cfg_scale=1.0,
        latent_scale=0.18215,
        current_episode_frame=42,
        current_episode_file="ep.npz",
        available_episodes=["a.npz", "b.npz"],
        available_checkpoints=["/c/best.pt"],
    )
    wire = info_to_legacy_wire(info)
    expected_keys = {
        "current_model",
        "npz_files",
        "checkpoints",
        "flow_steps",
        "cfg_scale",
        "history_length",
        "goal_offset",
        "latent_scale",
        "current_episode_frame",
    }
    assert set(wire.keys()) == expected_keys
    assert wire["current_model"] == "/path/to/model.pt"
    assert wire["npz_files"] == ["a.npz", "b.npz"]
    assert wire["current_episode_frame"] == 42


# ---------------------------------------------------------------------------
# End-to-end socket round-trip with stub server
# ---------------------------------------------------------------------------


class _StubServer:
    """Minimal stand-in for WorldModelServer to exercise the transport's dispatch."""

    def __init__(self):
        self.calls: list[tuple] = []
        self._info = ServerInfo(
            current_model_path="/stub.pt",
            architecture="dit_v1",
            config={"embed_dim": 64},
            history_length=10,
            goal_offset=30,
            flow_steps=5,
            cfg_scale=1.0,
            latent_scale=0.18215,
            current_episode_frame=0,
            current_episode_file=None,
            available_episodes=[],
            available_checkpoints=[],
        )

    def step(self, action: np.ndarray) -> bytes:
        self.calls.append(("step", action.copy()))
        return b"FAKE_JPEG"

    def info(self) -> ServerInfo:
        self.calls.append(("info",))
        return self._info

    def reseed(self, file: str, start_frame: int) -> None:
        self.calls.append(("reseed", file, start_frame))

    def reload(self, model_path: str) -> None:
        self.calls.append(("reload", model_path))

    def init_from_frames(self, frames_jpeg, goal_jpeg) -> None:
        self.calls.append(("init", len(frames_jpeg), goal_jpeg is not None))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def transport_pair():
    """Spin up a ZMQTransport on a free port, return (server, transport, req_socket)."""
    port = _free_port()
    stub = _StubServer()
    transport = ZMQTransport(stub, port=port, host="127.0.0.1")
    thread = threading.Thread(target=transport.run, kwargs={"poll_timeout_ms": 50}, daemon=True)
    thread.start()

    req_ctx = zmq.Context()
    req = req_ctx.socket(zmq.REQ)
    req.setsockopt(zmq.LINGER, 0)
    req.setsockopt(zmq.RCVTIMEO, 5000)
    req.connect(f"tcp://127.0.0.1:{port}")
    # Tiny grace period to ensure the REP socket has bound + the run loop is polling.
    time.sleep(0.1)

    yield stub, transport, req

    req.close()
    req_ctx.term()
    transport.stop()
    thread.join(timeout=2.0)


def test_roundtrip_info(transport_pair):
    stub, _, req = transport_pair
    req.send(bytes([MODE_INFO]))
    resp = req.recv()
    payload = json.loads(resp.decode())
    assert payload["current_model"] == "/stub.pt"
    assert payload["history_length"] == 10
    assert ("info",) in stub.calls


def test_roundtrip_reseed(transport_pair):
    stub, _, req = transport_pair
    req.send(bytes([MODE_RESEED]) + b'{"file":"x.npz","start_frame":99}')
    resp = req.recv()
    assert resp == b"OK"
    assert ("reseed", "x.npz", 99) in stub.calls


def test_roundtrip_reload(transport_pair):
    stub, _, req = transport_pair
    req.send(bytes([MODE_RELOAD]) + b'{"model_path":"/y.pt"}')
    resp = req.recv()
    assert resp == b"OK"
    assert ("reload", "/y.pt") in stub.calls


def test_roundtrip_action_with_prefix(transport_pair):
    stub, _, req = transport_pair
    action = np.arange(26, dtype=np.float32)
    req.send(bytes([MODE_ACTION]) + action.tobytes())
    resp = req.recv()
    assert resp == b"FAKE_JPEG"
    kind, sent = stub.calls[-1]
    assert kind == "step"
    np.testing.assert_array_equal(sent, action)


def test_roundtrip_action_legacy_naked(transport_pair):
    """Naked 104-byte action (no prefix) must still dispatch as MODE_LEGACY."""
    stub, _, req = transport_pair
    action = np.full(26, 0.5, dtype=np.float32)
    req.send(action.tobytes())
    resp = req.recv()
    assert resp == b"FAKE_JPEG"
    np.testing.assert_array_equal(stub.calls[-1][1], action)


def test_roundtrip_action_bad_size(transport_pair):
    """Action prefix with wrong-size payload returns ERROR, doesn't dispatch."""
    _, _, req = transport_pair
    # 1-byte mode prefix + 50 random bytes = neither full ACTION_SIZE+1 nor naked.
    # Fall-through path: returns MODE_LEGACY with 51 bytes; len != ACTION_SIZE → ERROR.
    req.send(bytes([MODE_ACTION]) + b"\x00" * 50)
    resp = req.recv()
    assert resp.startswith(b"ERROR")


def test_roundtrip_init(transport_pair):
    stub, _, req = transport_pair
    frames = [b"jpegA", b"jpegBB"]
    payload = bytes([len(frames), 0])
    for f in frames:
        payload += struct.pack("<I", len(f)) + f
    req.send(bytes([MODE_INIT]) + payload)
    resp = req.recv()
    assert resp == b"OK"
    assert ("init", len(frames), False) in stub.calls


def test_roundtrip_init_with_goal(transport_pair):
    stub, _, req = transport_pair
    frames = [b"jpegA"]
    goal = b"jpegGOAL"
    payload = bytes([len(frames), 1])
    for f in frames:
        payload += struct.pack("<I", len(f)) + f
    payload += struct.pack("<I", len(goal)) + goal
    req.send(bytes([MODE_INIT]) + payload)
    resp = req.recv()
    assert resp == b"OK"
    assert ("init", 1, True) in stub.calls


def test_roundtrip_init_malformed_returns_error(transport_pair):
    """Bad init payload returns ERROR but does not crash the loop."""
    _, _, req = transport_pair
    req.send(bytes([MODE_INIT]) + b"\x01")  # truncated header
    resp = req.recv()
    assert resp.startswith(b"ERROR")


def test_loop_survives_error_response(transport_pair):
    """After an error response, the loop must still answer subsequent requests."""
    _, _, req = transport_pair
    req.send(bytes([MODE_INIT]) + b"\x01")
    err = req.recv()
    assert err.startswith(b"ERROR")

    req.send(bytes([MODE_INFO]))
    payload = json.loads(req.recv().decode())
    assert payload["current_model"] == "/stub.pt"
