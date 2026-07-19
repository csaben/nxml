import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "deploy/cradle-ns"))
from dagger_control import Arbitrator, Mode
from dagger_inference_v2 import (
    MODE_FRAME_V2,
    MODE_INFO,
    MODE_RESET_V2,
    InferenceV2Client,
    InferenceV2Error,
    RemoteInferenceWorker,
    RemoteResult,
)


def revision(state="validated"):
    return {
        "revision_id": "revision-1",
        "state": state,
        "checkpoint_sha256": "a" * 64,
        "compatibility": {
            "architecture": "bc_transformer_v1",
            "action_spec_id": "switch_packets.v1",
            "action_dim": 26,
            "latent_shape": [4, 16, 32],
            "sequence_length": 32,
            "vae_profile": "sd-vae-ft-mse.rgb-bilinear-128x256.mode.scale-0.18215.v1",
        },
    }


def info(**updates):
    value = {
        "schema_id": "nxml.policy-inference-info.v2",
        "ready": True,
        "revision_id": "revision-1",
        "checkpoint_sha256": "a" * 64,
        "previous_revision_id": None,
        "action_spec_id": "switch_packets.v1",
        "action_dim": 26,
        "sequence_length": 32,
        "latent_shape": [4, 16, 32],
        "max_message_bytes": 2 * 1024 * 1024,
        "timeout_ms": 250,
    }
    value.update(updates)
    return value


def proposal(timestamp=10, state="proposal", **updates):
    value = {
        "schema_id": "nxml.policy-proposal.v2",
        "state": state,
        "action": [0.25] * 26 if state == "proposal" else [0.0] * 26,
        "frame_timestamp_ns": timestamp,
        "proposal_timestamp_ns": 1000,
        "processing_latency_ns": 2_000_000,
        "revision_id": "revision-1",
        "checkpoint_sha256": "a" * 64,
        "action_spec_id": "switch_packets.v1",
    }
    value.update(updates)
    return value


class Socket:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []
        self.closed = False

    def send(self, payload):
        self.sent.append(payload)

    def recv(self):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return json.dumps(response).encode()

    def close(self):
        self.closed = True


def test_info_reset_identity_and_timestamped_proposal():
    socket = Socket([info(), info(reset=True), proposal(10)])
    ticks = iter([100, 2_000_100])
    client = InferenceV2Client(
        "tcp://100.80.98.4:5557",
        revision(),
        timeout_ms=10,
        clock_ns=lambda: next(ticks),
        socket_factory=lambda: socket,
    )
    client.connect()
    result = client.predict_frame(10, b"jpeg")
    assert socket.sent[0] == bytes([MODE_INFO])
    assert socket.sent[1] == bytes([MODE_RESET_V2])
    assert socket.sent[2][0] == MODE_FRAME_V2
    assert result.transport_latency_ns == 2_000_000
    assert result.action is not None and result.action.shape == (26,)


def test_warming_is_neutral_and_bad_info_or_timeout_invalidates():
    warming_socket = Socket([info(), info(reset=True), proposal(10, state="warming")])
    ticks = iter([100, 200])
    client = InferenceV2Client(
        "tcp://host:5557",
        revision(),
        clock_ns=lambda: next(ticks),
        socket_factory=lambda: warming_socket,
    )
    assert client.predict_frame(10, b"jpeg").action is None

    bad = Socket([info(checkpoint_sha256="b" * 64)])
    client = InferenceV2Client("tcp://host:5557", revision(), socket_factory=lambda: bad)
    try:
        client.connect()
    except InferenceV2Error as error:
        assert "identity" in str(error)
    else:
        raise AssertionError("mismatched INFO was accepted")
    assert bad.closed

    failed = Socket([info(), info(reset=True), RuntimeError("timed out")])
    client = InferenceV2Client("tcp://host:5557", revision(), socket_factory=lambda: failed)
    try:
        client.predict_frame(10, b"jpeg")
    except InferenceV2Error as error:
        assert "transport failure" in str(error)
    else:
        raise AssertionError("transport timeout was accepted")
    assert failed.closed


@dataclass
class Frame:
    sequence: int
    monotonic_ns: int
    jpeg: bytes = b"jpeg"


class Source:
    def __init__(self, frames):
        self.frames = list(frames)
        self.calls = 0

    def latest_mjpeg(self, *, after_sequence, timeout):
        self.calls += 1
        if self.frames:
            return self.frames.pop(0)
        time.sleep(0.002)
        return None


class Client:
    def __init__(self, results):
        self.revision = revision()
        self.info = None
        self.results = list(results)
        self.connects = 0
        self.closed = 0

    def connect(self):
        self.info = info()
        self.connects += 1

    def predict_frame(self, timestamp, jpeg):
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def invalidate(self):
        self.info = None
        self.closed += 1

    close = invalidate


def wait_for(predicate):
    for _ in range(300):
        if predicate():
            return
        time.sleep(0.003)
    raise AssertionError("worker did not reach expected state")


def test_worker_is_disabled_by_default_and_feeds_arbitrator_only_after_enable():
    now = time.monotonic_ns()
    source = Source([Frame(0, now)])
    result = RemoteResult(
        np.full(26, 0.5, np.float32),
        "proposal",
        now,
        now + 2_000_000,
        2_000_000,
        1_000_000,
        999,
    )
    client = Client([result])
    worker = RemoteInferenceWorker(source=source, client=client, stale_ns=100_000_000)
    worker.start()
    time.sleep(0.02)
    assert source.calls == 0 and worker.latest_proposal() is None
    worker.enable()
    wait_for(lambda: worker.status()["proposals"] == 1)
    policy = worker.latest_proposal()
    assert policy is not None
    arb = Arbitrator(stale_ns=100_000_000)
    arb.transition(mode=Mode.PURE_AI)
    assert arb.apply(policy.monotonic_ns, None, policy).source == "policy"
    worker.disable()
    assert worker.latest_proposal() is None
    worker.stop()


def test_warming_and_error_clear_proposal_and_reconnect_with_no_stale_reuse():
    now = time.monotonic_ns()
    results = [
        RemoteResult(None, "warming", now, now + 1, 1, 1, 1),
        InferenceV2Error("endpoint lost"),
        RemoteResult(
            np.full(26, 0.75, np.float32),
            "proposal",
            now + 2,
            now + 3,
            1,
            1,
            2,
        ),
    ]
    client = Client(results)
    source = Source([Frame(0, now), Frame(1, now + 1), Frame(2, now + 2)])
    disarms = []
    worker = RemoteInferenceWorker(
        source=source, client=client, reconnect_seconds=0.01, on_disarm=disarms.append
    )
    worker.start()
    worker.enable()
    wait_for(lambda: bool(disarms))
    assert "endpoint lost" in disarms[-1]
    wait_for(lambda: worker.status()["proposals"] == 1)
    assert worker.latest_proposal() is not None
    assert client.closed >= 1
    assert client.connects >= 2
    worker.stop()


def test_unvalidated_revision_cannot_construct_transport():
    try:
        InferenceV2Client("tcp://host:5557", revision("candidate"))
    except ValueError as error:
        assert "validated" in str(error)
    else:
        raise AssertionError("candidate transport was constructed")
