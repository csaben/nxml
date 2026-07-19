from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from nx_packets import BUTTON_INDEX, neutral_action
from nxml_capture import ControllerSnapshot, Synchronizer, VideoParquetEpisodeWriter
from nxml_capture.source import Frame


class _Source:
    is_open = True

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def latest(self) -> Frame | None:
        return None

    def frames(self):
        return iter(())


class _Controller:
    def __init__(self, snapshot: ControllerSnapshot | None, *, connected: bool = True) -> None:
        self.snapshot = snapshot
        self.is_connected = connected

    def latest(self) -> ControllerSnapshot | None:
        return self.snapshot

    def latest_at(self, *, timestamp: float, monotonic_ns: int | None = None):
        snapshot = self.snapshot
        if snapshot is None:
            return None
        if monotonic_ns is not None and snapshot.monotonic_ns is not None:
            return snapshot if snapshot.monotonic_ns <= monotonic_ns else None
        return snapshot if snapshot.timestamp <= timestamp else None

    def wait_for_first(self, timeout: float = 5.0) -> bool:
        return self.snapshot is not None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def _frame(timestamp: float = 10.1, monotonic_ns: int = 1_100_000_000) -> Frame:
    return Frame(
        timestamp=timestamp,
        monotonic_ns=monotonic_ns,
        image=np.zeros((16, 16, 3), dtype=np.uint8),
    )


def _snapshot(
    action: np.ndarray | None = None,
    *,
    timestamp: float = 10.0,
    monotonic_ns: int = 1_000_000_000,
    source: str | None = None,
) -> ControllerSnapshot:
    return ControllerSnapshot(
        timestamp=timestamp,
        monotonic_ns=monotonic_ns,
        action=neutral_action() if action is None else action,
        packet={},
        action_source=source,
    )


def test_explicit_human_mode_owns_full_fresh_neutral_packet() -> None:
    sync = Synchronizer(_Source(), _Controller(_snapshot()), driver="human", max_action_age=0.2)

    synced = sync._pair(_frame())

    assert synced is not None and synced.valid
    assert synced.active_driver == "human"
    assert synced.controller_id == "nxbt-orchestrator:switch_packets.v1"
    np.testing.assert_array_equal(synced.applied_action, neutral_action())
    np.testing.assert_array_equal(synced.human_action, neutral_action())
    assert synced.human_mask is not None and synced.human_mask.all()
    assert synced.ownership is not None and np.all(synced.ownership == 1)
    assert synced.action_timestamp == 10.0
    assert synced.action_monotonic_ns == 1_000_000_000
    assert synced.action_age == pytest.approx(0.1)


def test_stale_and_disconnected_packets_are_invalid_and_unowned() -> None:
    stale = Synchronizer(
        _Source(),
        _Controller(_snapshot(timestamp=9.0, monotonic_ns=100_000_000)),
        driver="human",
        max_action_age=0.2,
    )._pair(_frame())
    disconnected = Synchronizer(
        _Source(), _Controller(_snapshot(), connected=False), driver="human"
    )._pair(_frame())

    for synced, reason in (
        (stale, "stale_controller_sample"),
        (disconnected, "controller_disconnected"),
    ):
        assert synced is not None and not synced.valid
        assert synced.human_mask is not None and not synced.human_mask.any()
        assert synced.ownership is not None and not synced.ownership.any()
        assert synced.invalid_reasons == (reason,)


def test_unknown_or_inference_provenance_is_never_relabelled() -> None:
    unknown = Synchronizer(_Source(), _Controller(_snapshot()), driver="unknown")._pair(_frame())
    inference_action = neutral_action()
    inference_action[BUTTON_INDEX["A"]] = 1
    inference = Synchronizer(
        _Source(),
        _Controller(_snapshot(inference_action, source="inference")),
        driver="human",
    )._pair(_frame())

    assert unknown is not None and unknown.invalid_reasons == ("unknown_provenance",)
    assert inference is not None and inference.invalid_reasons == ("inference_provenance",)
    np.testing.assert_array_equal(inference.applied_action, neutral_action())
    assert inference.human_mask is not None and not inference.human_mask.any()
    assert inference.ownership is not None and not inference.ownership.any()


def test_wrong_action_spec_is_invalid_and_unowned() -> None:
    wrong = _snapshot(np.zeros(25, dtype=np.float32))
    synced = Synchronizer(_Source(), _Controller(wrong), driver="human")._pair(_frame())

    assert synced is not None and not synced.valid
    assert synced.invalid_reasons == ("wrong_action_spec",)
    assert synced.ownership is not None and not synced.ownership.any()


def test_human_capture_roundtrip_has_temporal_bounds_events_and_checksums(
    tmp_path: Path,
) -> None:
    action = neutral_action()
    action[BUTTON_INDEX["A"]] = 1
    controller = _Controller(_snapshot(action))
    sync = Synchronizer(_Source(), controller, driver="human", max_action_age=0.2)
    first = sync._pair(_frame())
    controller.snapshot = _snapshot(action, timestamp=10.1, monotonic_ns=1_100_000_000)
    second = sync._pair(_frame(10.2, 1_200_000_000))
    records = [first, second]
    writer = VideoParquetEpisodeWriter(tmp_path, episode_name="human", codec="h264", fps=30)
    episode_id = writer.episode_id
    for record in records:
        assert record is not None
        writer.append(record)
    writer.append_event(
        "human_session_started",
        timestamp=10.0,
        monotonic_ns=1_000_000_000,
        payload={"driver": "human"},
    )
    assert writer.close() is not None

    rows = pq.read_table(tmp_path / "human.parquet").to_pylist()
    assert [row["frame_monotonic_ns"] for row in rows] == [1_100_000_000, 1_200_000_000]
    assert [row["action_monotonic_ns"] for row in rows] == [1_000_000_000, 1_100_000_000]
    assert all(row["applied_action"] == row["human_action"] for row in rows)
    assert all(all(row["human_mask"]) for row in rows)
    assert all(set(row["ownership"]) == {1} for row in rows)
    assert all(row["invalid_reasons"] == [] for row in rows)
    assert all(row["action_monotonic_ns"] <= row["frame_monotonic_ns"] for row in rows)
    assert all(
        row["action_age"]
        == pytest.approx((row["frame_monotonic_ns"] - row["action_monotonic_ns"]) / 1e9)
        for row in rows
    )
    events = pq.read_table(tmp_path / "human.events.parquet").to_pylist()
    assert events[0]["kind"] == "human_session_started"

    manifest = json.loads((tmp_path / "human.manifest.json").read_text())
    assert manifest["episode_id"] == episode_id
    assert manifest["first_frame_timestamp_ns"] == 1_100_000_000
    assert manifest["last_frame_timestamp_ns"] == 1_200_000_000
    for name, metadata in manifest["files"].items():
        payload = (tmp_path / name).read_bytes()
        assert metadata["bytes"] == len(payload)
        assert metadata["sha256"] == hashlib.sha256(payload).hexdigest()


def test_no_prior_sample_is_neutral_invalid_and_unowned() -> None:
    future = _snapshot(timestamp=10.2, monotonic_ns=1_200_000_000)
    synced = Synchronizer(_Source(), _Controller(future), driver="human")._pair(_frame())
    assert synced is not None and not synced.valid
    assert synced.invalid_reasons == ("no_prior_controller_sample",)
    np.testing.assert_array_equal(synced.applied_action, neutral_action())
    assert synced.human_mask is not None and not synced.human_mask.any()
    assert synced.ownership is not None and not synced.ownership.any()


def test_exact_equal_sample_is_causally_valid() -> None:
    sample = _snapshot(timestamp=10.1, monotonic_ns=1_100_000_000)
    synced = Synchronizer(_Source(), _Controller(sample), driver="human")._pair(_frame())
    assert synced is not None and synced.valid and synced.action_age == 0


def test_future_sample_from_legacy_source_is_invariant_violation() -> None:
    class LegacyController:
        is_connected = True
        def latest(self):
            return _snapshot(timestamp=10.2, monotonic_ns=1_200_000_000)

    synced = Synchronizer(_Source(), LegacyController(), driver="human")._pair(_frame())
    assert synced is not None and not synced.valid
    assert synced.invalid_reasons == ("future_controller_sample",)
    assert synced.action_age < 0
    np.testing.assert_array_equal(synced.applied_action, neutral_action())
