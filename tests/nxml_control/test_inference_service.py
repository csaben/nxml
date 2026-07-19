import json
import struct
import time

import numpy as np
from nxml_control.inference import (
    ACTION_BYTES,
    MAX_MESSAGE_BYTES,
    MODE_FRAME_V2,
    MODE_HEALTH_V2,
    MODE_INFO,
    MODE_PREDICT_FRAME,
    MODE_RELOAD_PATH,
    MODE_RELOAD_REVISION_V2,
    MODE_RESET_V2,
    MODE_ROLLBACK_V2,
    ImmutableInferenceService,
    LoadedRevision,
)


class Server:
    def __init__(self, value):
        self.value = value
        self.frames = 0

    def predict_frame(self, _jpeg):
        self.frames += 1
        if self.frames == 1:
            return None
        return np.full(26, self.value, dtype=np.float32)

    def reset_frame_window(self):
        self.frames = 0


class Loader:
    def __init__(self):
        self.fail = set()

    def load(self, revision_id):
        if revision_id in self.fail:
            raise ValueError("checkpoint digest mismatch")
        value = {"r1": 1.0, "r2": 2.0}[revision_id]
        return LoadedRevision(
            revision_id,
            revision_id * 32,
            Server(value),
            {"sequence_length": 32, "latent_shape": [4, 16, 32]},
        )


def decode(service, message):
    return json.loads(service.handle(message))


def frame(timestamp, payload=b"jpeg"):
    return bytes([MODE_FRAME_V2]) + struct.pack("<Q", timestamp) + payload


def test_v2_warming_proposal_stale_reset_and_health():
    service = ImmutableInferenceService(Loader(), "r1", timeout_ms=1000)
    warming = decode(service, frame(10))
    assert warming["state"] == "warming" and warming["action"] == [0.0] * 26
    proposal = decode(service, frame(11))
    assert proposal["state"] == "proposal"
    assert proposal["action"] == [1.0] * 26
    assert proposal["revision_id"] == "r1"
    assert proposal["checkpoint_sha256"] == "r1" * 32
    assert proposal["proposal_timestamp_ns"] > 0
    assert proposal["processing_latency_ns"] >= 0
    stale = decode(service, frame(11))
    assert stale["state"] == "error" and stale["action"] == [0.0] * 26
    assert decode(service, bytes([MODE_RESET_V2]))["reset"] is True
    assert decode(service, bytes([MODE_HEALTH_V2]))["ready"] is True
    assert decode(service, bytes([MODE_INFO]))["revision_id"] == "r1"


def test_registry_reload_rollback_and_digest_failure_are_atomic():
    loader = Loader()
    service = ImmutableInferenceService(loader, "r1")
    stale = decode(
        service,
        bytes([MODE_RELOAD_REVISION_V2])
        + json.dumps({"revision_id": "r2", "expected_revision_id": "wrong"}).encode(),
    )
    assert stale["state"] == "error"
    assert service.info()["revision_id"] == "r1"
    loaded = decode(
        service,
        bytes([MODE_RELOAD_REVISION_V2])
        + json.dumps({"revision_id": "r2", "expected_revision_id": "r1"}).encode(),
    )
    assert loaded["revision_id"] == "r2" and loaded["previous_revision_id"] == "r1"
    rolled = decode(
        service,
        bytes([MODE_ROLLBACK_V2])
        + json.dumps({"expected_revision_id": "r2"}).encode(),
    )
    assert rolled["revision_id"] == "r1"
    loader.fail.add("r2")
    failed = decode(
        service,
        bytes([MODE_RELOAD_REVISION_V2])
        + json.dumps({"revision_id": "r2", "expected_revision_id": "r1"}).encode(),
    )
    assert failed["state"] == "error"
    assert service.info()["revision_id"] == "r1"
    path_reload = decode(service, bytes([MODE_RELOAD_PATH]) + b'{"model_path":"/tmp/x"}')
    assert path_reload["state"] == "error"


def test_v1_frame_wire_remains_compatible():
    service = ImmutableInferenceService(Loader(), "r1")
    assert service.handle(bytes([MODE_PREDICT_FRAME]) + b"jpeg") == b"\x00"
    action = service.handle(bytes([MODE_PREDICT_FRAME]) + b"jpeg")
    assert len(action) == ACTION_BYTES
    np.testing.assert_array_equal(
        np.frombuffer(action, dtype=np.float32), np.ones(26, dtype=np.float32)
    )


def test_message_limit_and_latency_budget_fail_closed():
    service = ImmutableInferenceService(Loader(), "r1", timeout_ms=1)
    oversized = decode(service, b"x" * (MAX_MESSAGE_BYTES + 1))
    assert oversized["state"] == "error"
    assert oversized["action"] == [0.0] * 26

    class SlowServer(Server):
        def predict_frame(self, jpeg):
            time.sleep(0.002)
            return super().predict_frame(jpeg)

    service._current.server = SlowServer(1.0)
    timed_out = decode(service, frame(10))
    assert timed_out["state"] == "error"
    assert timed_out["action"] == [0.0] * 26
    assert "latency budget" in timed_out["error"]
