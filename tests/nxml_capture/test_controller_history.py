from __future__ import annotations

import numpy as np
from nxml_capture.controller_subscribe import ControllerSnapshot, ControllerSubscription


def sample(sequence: int, monotonic_ns: int) -> ControllerSnapshot:
    action = np.zeros(26, dtype=np.float32)
    action[0] = sequence
    return ControllerSnapshot(
        timestamp=monotonic_ns / 1e9,
        monotonic_ns=monotonic_ns,
        action=action,
        packet={"sequence": sequence},
        action_source="human",
    )


def subscription(*samples: ControllerSnapshot) -> ControllerSubscription:
    value = ControllerSubscription()
    with value._latest_lock:
        value._history.extend(samples)
        value._latest = samples[-1] if samples else None
    return value


def test_interleaved_frames_select_newest_prior_action() -> None:
    source = subscription(sample(1, 100), sample(2, 200), sample(3, 300))
    assert source.latest_at(timestamp=0, monotonic_ns=250).packet["sequence"] == 2
    assert source.latest_at(timestamp=0, monotonic_ns=350).packet["sequence"] == 3


def test_exact_equal_timestamp_selects_equal_sample() -> None:
    source = subscription(sample(1, 100), sample(2, 200))
    assert source.latest_at(timestamp=0, monotonic_ns=200).packet["sequence"] == 2


def test_no_prior_sample_returns_none() -> None:
    assert subscription(sample(1, 100)).latest_at(timestamp=0, monotonic_ns=99) is None


def test_scheduler_jitter_never_selects_a_future_latest_sample() -> None:
    source = subscription(sample(1, 100), sample(2, 200), sample(3, 300), sample(4, 400))
    # The subscription's latest value is far in the future relative to this frame.
    selected = source.latest_at(timestamp=0, monotonic_ns=205)
    assert selected is not None
    assert selected.packet["sequence"] == 2
    assert selected.monotonic_ns <= 205
