from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).parents[2]
DEPLOY = ROOT / "deploy" / "cradle-ns"
sys.path.insert(0, str(DEPLOY))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


segments = load("dagger_segments_test", DEPLOY / "dagger_segments.py")
recording = load("dagger_recording_segments_test", DEPLOY / "dagger_recording.py")


def source(tmp_path: Path, *, index: int = 0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    episode = "ad9118b7-5f8b-40fe-b16f-3ae400fcfddb"
    base = tmp_path / f"{episode}.{index:06d}"
    paths = [Path(f"{base}.mkv"), Path(f"{base}.parquet"), Path(f"{base}.events.parquet")]
    for position, path in enumerate(paths):
        path.write_bytes(bytes([position]) * (position + 1))
    manifest = Path(f"{base}.manifest.json")
    manifest.write_text("{}")
    return segments.SegmentSource(episode, index, index * 100, (index + 1) * 100, *paths, manifest)


def test_prepare_is_exact_triplet_and_deterministic(tmp_path):
    item = source(tmp_path)
    first = segments.prepare_segment(item, tmp_path / "stage", "dataset")
    second = segments.prepare_segment(item, tmp_path / "stage2", "dataset")
    assert first.bundle == second.bundle
    assert first.bundle["schema_id"] == "nxml.segment-bundle.v1"
    assert first.bundle["timeline_start_ns"] == 0
    assert first.bundle["timeline_end_ns"] == 100
    with tarfile.open(first.tar_path) as archive:
        assert archive.getnames() == [item.video.name, item.actions.name, item.events.name]


class Client:
    def __init__(self):
        self.calls = 0
        self.closed = None

    def publish(self, prepared):
        self.calls += 1
        bundle = prepared.bundle
        return {
            "receipt_id": f"receipt-{bundle['sequence_index']}",
            "episode_id": bundle["episode_id"],
            "segment_id": bundle["segment_id"],
            "sequence_index": bundle["sequence_index"],
            "timeline_start_ns": bundle["timeline_start_ns"],
            "timeline_end_ns": bundle["timeline_end_ns"],
            "sha256": bundle["object_sha256"],
        }

    def close_episode(self, episode_id, receipts):
        self.closed = (episode_id, receipts)
        return {"state": "closed", "episode_id": episode_id}


def test_receipt_is_fsynced_before_sources_and_staging_deleted(tmp_path, monkeypatch):
    prepared = segments.prepare_segment(source(tmp_path), tmp_path / "stage", "dataset")
    journal = segments.SegmentJournal(tmp_path / "journal.json")
    client = Client()
    worker = segments.SegmentDeliveryWorker(client, journal)
    observed = []
    real_unlink = Path.unlink

    def checked_unlink(path, *args, **kwargs):
        state = json.loads(journal.path.read_text())
        key = f"{prepared.source.episode_id}:0"
        observed.append(state["receipts"].get(key, {}).get("receipt_id"))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", checked_unlink)
    worker.start()
    assert worker.submit(prepared)
    receipts = worker.wait_receipts(prepared.source.episode_id, 1, timeout=2)
    worker.close_episode(prepared.source.episode_id, 1)
    worker.stop()
    assert receipts[0]["receipt_id"] == "receipt-0"
    assert observed and set(observed) == {"receipt-0"}
    assert not prepared.tar_path.exists()
    assert client.closed is not None


def test_restart_reconciles_durable_pending_idempotently(tmp_path):
    prepared = segments.prepare_segment(source(tmp_path), tmp_path / "stage", "dataset")
    journal = segments.SegmentJournal(tmp_path / "journal.json")
    journal.store_pending(prepared)
    client = Client()
    worker = segments.SegmentDeliveryWorker(client, segments.SegmentJournal(journal.path))
    worker.start()
    worker.wait_receipts(prepared.source.episode_id, 1, timeout=2)
    worker.stop()
    assert client.calls == 1
    assert json.loads(journal.path.read_text())["pending"] == {}


def test_restart_reclaims_files_left_after_receipt_fsync(tmp_path):
    prepared = segments.prepare_segment(source(tmp_path), tmp_path / "stage", "dataset")
    journal = segments.SegmentJournal(tmp_path / "journal.json")
    journal.store_pending(prepared)
    journal.store_receipt(
        {
            "episode_id": prepared.source.episode_id,
            "sequence_index": 0,
            "receipt_id": "durable",
        }
    )
    assert prepared.source.video.exists() and prepared.tar_path.exists()
    worker = segments.SegmentDeliveryWorker(Client(), segments.SegmentJournal(journal.path))
    worker.start()
    worker.stop()
    assert not prepared.source.video.exists()
    assert not prepared.tar_path.exists()
    assert json.loads(journal.path.read_text())["cleanup"] == {}


def test_restart_recovers_close_intent_after_segment_receipt(tmp_path):
    prepared = segments.prepare_segment(source(tmp_path), tmp_path / "stage", "dataset")
    journal = segments.SegmentJournal(tmp_path / "journal.json")
    journal.store_pending(prepared)
    journal.store_close_request(prepared.source.episode_id, 1)
    client = Client()
    worker = segments.SegmentDeliveryWorker(client, segments.SegmentJournal(journal.path))
    worker.start()
    deadline = time.monotonic() + 2
    state = json.loads(journal.path.read_text())
    while state["close_requests"] and time.monotonic() < deadline:
        time.sleep(0.01)
        state = json.loads(journal.path.read_text())
    worker.stop()
    assert client.closed is not None
    assert state["close_requests"] == {}
    assert state["closes"][prepared.source.episode_id]["state"] == "closed"


def test_journal_keys_segment_zero_by_episode_uuid(tmp_path):
    journal = segments.SegmentJournal(tmp_path / "journal.json")
    first = source(tmp_path / "first")
    second = source(tmp_path / "second")
    second = segments.SegmentSource(
        "80895a91-9bf2-45d6-b352-276e329e510b",
        second.sequence_index,
        second.timeline_start_ns,
        second.timeline_end_ns,
        second.video,
        second.actions,
        second.events,
        second.manifest,
    )
    journal.reserve(first.episode_id, 0, 10)
    journal.store_source(first)
    journal.reserve(second.episode_id, 0, 10)
    journal.store_source(second)
    assert len(journal.value["pending"]) == 2
    assert all(key.endswith(":0") for key in journal.value["pending"])


def test_bounded_backpressure_never_deletes_unreceipted(tmp_path):
    first = segments.prepare_segment(source(tmp_path, index=0), tmp_path / "stage", "dataset")
    second = segments.prepare_segment(source(tmp_path, index=1), tmp_path / "stage", "dataset")
    worker = segments.SegmentDeliveryWorker(
        Client(),
        segments.SegmentJournal(tmp_path / "j.json"),
        max_pending_bytes=first.bundle["object_size_bytes"],
    )
    assert worker.submit(first)
    assert not worker.submit(second)
    assert first.source.video.exists()
    assert second.source.video.exists()
    assert worker.status()["backpressure"] is True


def test_startup_discovers_complete_unjournaled_triplet(tmp_path):
    item = source(tmp_path / "source")
    item.manifest.write_text(
        json.dumps(
            {
                "episode_id": item.episode_id,
                "first_frame_timestamp_ns": 100,
                "last_frame_timestamp_ns": 199,
                "fps_nominal": 30,
            }
        )
    )
    journal = segments.SegmentJournal(tmp_path / "journal.json")
    worker = segments.SegmentDeliveryWorker(
        Client(), journal, source_dir=tmp_path / "source", staging_dir=tmp_path / "stage"
    )
    worker._recover_unjournaled_sources()
    recovered = journal.pending_sources()
    assert len(recovered) == 1
    assert recovered[0].timeline_start_ns == 100
    assert recovered[0].timeline_end_ns == 33_333_532


def test_two_minute_faster_producer_hits_byte_gate_before_opening_next_segment(tmp_path):
    journal = segments.SegmentJournal(tmp_path / "journal.json")
    worker = segments.SegmentDeliveryWorker(Client(), journal, max_pending_bytes=1_000)
    episode = "3f37b81c-15d7-4118-aa3e-690295f162c5"
    # Simulate 120 seconds of 10-second rotations with a producer faster than delivery.
    admitted = 0
    for index in range(12):
        if not worker.reserve(episode, index, 300):
            break
        admitted += 1
    assert admitted == 3
    assert journal.buffered_bytes() == 900
    assert not worker.reserve(episode, admitted, 300)


def test_recorder_rotates_on_frame_boundaries_with_contiguous_timeline(tmp_path, monkeypatch):
    import numpy as np
    from nxml_capture import SyncedFrame

    frames = [
        SyncedFrame(
            timestamp=index / 30,
            frame=np.zeros((4, 4, 3), np.uint8),
            action=np.zeros(26, np.float32),
            action_age=0,
            frame_monotonic_ns=index * 40,
            action_monotonic_ns=index * 40,
        )
        for index in range(6)
    ]

    class Sync:
        invalid_samples = 0

        def __init__(self, *_args):
            pass

        def frames(self):
            yield from frames

    class History:
        def end_recording(self):
            pass

    class Worker:
        def __init__(self):
            self.items = []
            self.closed = threading.Event()

        def submit_source(self, item):
            self.items.append(item)
            return True

        def reserve(self, _episode_id, _index, _size):
            return True

        def close_episode(self, episode_id, count):
            assert count == 3
            assert all(item.episode_id == episode_id for item in self.items)
            self.closed.set()

    worker = Worker()
    monkeypatch.setattr(recording, "ArbitrationSynchronizer", Sync)
    session = recording.HumanRecordingSession(
        object(),
        output_dir=tmp_path / "source",
        history=History(),
        codec="ffv1",
        segment_worker=worker,
        segment_staging_dir=tmp_path / "stage",
        segment_duration_seconds=0.00000008,
    )
    writer = recording.VideoParquetEpisodeWriter(
        session.output_dir,
        episode_name="e1d588ba-a61e-47ea-b01e-b90a3b951245.000000",
        episode_id="e1d588ba-a61e-47ea-b01e-b90a3b951245",
        codec="ffv1",
        fps=30,
    )
    session._record(writer)
    assert worker.closed.wait(2)
    assert session.status()["frames"] == 6
    bounds = [(item.timeline_start_ns, item.timeline_end_ns) for item in worker.items]
    assert bounds == [(0, 80), (80, 160), (160, 33_333_533)]
    assert [item.sequence_index for item in worker.items] == [0, 1, 2]
    assert all(item.video.exists() for item in worker.items)
