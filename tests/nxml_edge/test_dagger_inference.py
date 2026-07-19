import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "deploy/cradle-ns"))
from dagger_control import Arbitrator, Mode
from dagger_inference import LocalInferenceWorker


@dataclass
class Frame:
    sequence: int
    monotonic_ns: int
    jpeg: bytes = b"jpeg"


class Source:
    def __init__(self, frames):
        self.frames = list(frames)

    def latest_mjpeg(self, *, after_sequence, timeout):
        if self.frames:
            return self.frames.pop(0)
        time.sleep(min(timeout, 0.005))
        return None


class Policy:
    sequence_length = 3

    def __init__(self, *, fail=False):
        self.count = 0
        self.fail = fail

    def reset_frame_window(self):
        self.count = 0

    def predict_frame(self, jpeg):
        if self.fail:
            raise RuntimeError("gpu lost")
        self.count += 1
        return None if self.count < self.sequence_length else np.arange(26, dtype=np.float32) / 25


class Runtime:
    def __init__(self, policy):
        self.policy = policy
        self.reasons = []

    def active_handle(self):
        return "immutable-r1", self.policy

    def neutral_disarm(self, reason):
        self.reasons.append(reason)


def wait_for(predicate):
    for _ in range(200):
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("worker did not reach expected state")


def test_history_freshness_timestamp_and_26d_proposal():
    now = time.monotonic_ns()
    runtime = Runtime(Policy())
    worker = LocalInferenceWorker(
        source=Source([Frame(i, now + i) for i in range(3)]), runtime=runtime
    )
    worker.start()
    wait_for(lambda: worker.status()["proposals"] == 1)
    proposal = worker.latest_proposal()
    status = worker.status()
    worker.stop()
    assert proposal is not None and proposal.action.shape == (26,)
    assert proposal.revision == "immutable-r1"
    assert proposal.observation_monotonic_ns == now + 2
    assert status["history_frames"] == status["sequence_length"] == 3
    assert status["health"] == "healthy" and status["armed"] is False
    assert status["inference_latency_ms"] is not None

    # The transport publishes the exact Proposal contract consumed by the
    # existing local arbitrator; no network or capture work enters apply().
    arb = Arbitrator(stale_ns=250_000_000)
    arb.transition(mode=Mode.PURE_AI)
    applied = arb.apply(proposal.monotonic_ns, None, proposal)
    assert applied.source == "policy"
    assert applied.revision == "immutable-r1"
    np.testing.assert_array_equal(applied.action, proposal.action)


def test_stale_observation_neutral_disarms_without_policy_call():
    policy = Policy()
    runtime = Runtime(policy)
    worker = LocalInferenceWorker(
        source=Source([Frame(0, time.monotonic_ns() - 1_000_000_000)]), runtime=runtime
    )
    worker.start()
    wait_for(lambda: bool(runtime.reasons))
    worker.stop()
    assert policy.count == 0
    assert worker.latest_proposal() is None
    assert runtime.reasons[-1] == "stale observation"


def test_model_error_neutral_disarms_and_does_not_escape_thread():
    runtime = Runtime(Policy(fail=True))
    worker = LocalInferenceWorker(source=Source([Frame(0, time.monotonic_ns())]), runtime=runtime)
    worker.start()
    wait_for(lambda: bool(runtime.reasons))
    worker.stop()
    assert worker.latest_proposal() is None
    assert "gpu lost" in runtime.reasons[-1]
    assert worker.status()["armed"] is False


def test_observation_stream_lag_disarms_after_last_fresh_frame():
    runtime = Runtime(Policy())
    worker = LocalInferenceWorker(
        source=Source([Frame(0, time.monotonic_ns())]),
        runtime=runtime,
        stale_ns=20_000_000,
    )
    worker.start()
    wait_for(lambda: "observation stream lag" in runtime.reasons)
    worker.stop()
    assert worker.latest_proposal() is None
    assert worker.status()["health"] == "stale"
