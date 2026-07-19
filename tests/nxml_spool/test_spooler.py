"""Spooler end-to-end (no network): discover -> pack -> fake-upload -> verify -> delete."""

from __future__ import annotations

import json
import os
import tarfile
import time
from pathlib import Path

import pytest
from nxml_spool.episodes import discover_episodes
from nxml_spool.journal import Journal
from nxml_spool.shards import pack_shard, plan_shards
from nxml_spool.spooler import run_spooler
from nxml_spool.storage import (
    FilesystemStorageBackend,
    StorageCommit,
    StorageConflictError,
    StorageUnavailableError,
    StorageValidationError,
)


def _write_episode(root: Path, name: str, *, subdir: str | None = None, kb: int = 8) -> Path:
    d = root / subdir if subdir else root
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.mkv").write_bytes(os.urandom(kb * 1024))
    (d / f"{name}.parquet").write_bytes(os.urandom(1024))
    (d / f"{name}.manifest.json").write_text(json.dumps({"frame_count": 42}))
    return d / f"{name}.manifest.json"


def _age(paths_root: Path, seconds: float = 120.0) -> None:
    past = time.time() - seconds
    for p in paths_root.rglob("*"):
        if p.is_file():
            os.utime(p, (past, past))


class FakeUploader:
    def __init__(self) -> None:
        self.uploaded: dict[str, int] = {}

    def upload_and_verify(self, local_path: Path, path_in_repo: str) -> None:
        self.uploaded[path_in_repo] = local_path.stat().st_size


def test_discovery_settle_and_layouts(tmp_path: Path) -> None:
    _write_episode(tmp_path, "20260709_120000")  # flat (collect)
    _write_episode(tmp_path, "20260709_120500", subdir="20260709T120459")  # autopilot subdir
    fresh = _write_episode(tmp_path, "20260709_121000")  # not settled
    _age(tmp_path)
    os.utime(fresh, None)  # freshen the third episode's manifest

    eps = discover_episodes([tmp_path], settle_seconds=30)
    ids = {e.episode_id for e in eps}
    assert ids == {"20260709_120000", "20260709T120459__20260709_120500"}


def test_plan_shards_greedy(tmp_path: Path) -> None:
    _write_episode(tmp_path, "a", kb=8)
    _write_episode(tmp_path, "b", kb=8)
    _write_episode(tmp_path, "c", kb=8)
    _age(tmp_path)
    eps = discover_episodes([tmp_path], settle_seconds=30)
    groups = plan_shards(eps, shard_size_bytes=20 * 1024)  # fits two ~9.3KB episodes
    assert [len(g) for g in groups] == [2, 1]


def test_pack_shard_webdataset_layout(tmp_path: Path) -> None:
    _write_episode(tmp_path, "ep1")
    _age(tmp_path)
    eps = discover_episodes([tmp_path], settle_seconds=30)
    shard = pack_shard(eps, tmp_path / "staging", 7)
    assert shard.path.name == "shard-000007.tar"
    with tarfile.open(shard.path) as tar:
        names = sorted(tar.getnames())
    assert names == ["ep1.json", "ep1.mkv", "ep1.parquet"]
    sidecar = json.loads(shard.sidecar_path.read_text())
    assert sidecar["n_episodes"] == 1
    assert sidecar["sha256"] == shard.sha256


def test_run_spooler_end_to_end(tmp_path: Path) -> None:
    watch = tmp_path / "capture"
    _write_episode(watch, "20260709_130000")
    _write_episode(watch, "20260709_130500", subdir="sess1")
    _age(tmp_path)

    uploader = FakeUploader()
    state = tmp_path / "state"
    run_spooler(
        [watch],
        repo_id="fake/repo",
        state_dir=state,
        shard_size_mb=1,
        settle_seconds=30,
        uploader=uploader,
        once=True,
    )

    assert any(k.startswith("shards/shard-000000") for k in uploader.uploaded)
    assert any(k.startswith("meta/shard-000000") for k in uploader.uploaded)
    # Local episode files deleted; emptied autopilot subdir removed.
    assert not list(watch.rglob("*.mkv"))
    assert not (watch / "sess1").exists()
    journal = Journal(state / "journal.json")
    assert journal.stats()["episodes_shipped"] == 2
    status = json.loads((state / "status.json").read_text())
    assert status["pending_episodes"] == 0
    assert status["staging_shards"] == 0
    assert status["staging_bytes"] == 0
    assert status["admission_open"] is True
    assert status["disk_high_watermark"] == 0.85
    assert status["disk_low_watermark"] == 0.75

    # Idempotent second pass: nothing new to ship.
    run_spooler(
        [watch], repo_id="fake/repo", state_dir=state, shard_size_mb=1,
        settle_seconds=30, uploader=uploader, once=True,
    )
    assert Journal(state / "journal.json").stats()["shards_uploaded"] == 1


def test_no_delete_keeps_files(tmp_path: Path) -> None:
    watch = tmp_path / "capture"
    _write_episode(watch, "20260709_140000")
    _age(tmp_path)
    run_spooler(
        [watch], repo_id="fake/repo", state_dir=tmp_path / "state", shard_size_mb=1,
        settle_seconds=30, uploader=FakeUploader(), once=True, delete_after_upload=False,
    )
    assert list(watch.glob("*.mkv"))


class ExplodingUploader:
    def upload_and_verify(self, local_path: Path, path_in_repo: str) -> None:
        raise RuntimeError("simulated network failure")


def test_failed_upload_preserves_everything(tmp_path: Path) -> None:
    """A mid-upload failure must leave local episodes intact and the journal unchanged."""
    import pytest

    watch = tmp_path / "capture"
    _write_episode(watch, "20260710_030000")
    _age(tmp_path)
    state = tmp_path / "state"
    with pytest.raises(RuntimeError, match="simulated"):
        run_spooler(
            [watch], repo_id="fake/repo", state_dir=state, shard_size_mb=1,
            settle_seconds=30, uploader=ExplodingUploader(), once=True,
        )
    assert list(watch.glob("*.mkv")), "episode files must survive a failed upload"
    assert Journal(state / "journal.json").stats()["episodes_shipped"] == 0
    # Recovery: a working uploader ships the same episode on the next pass.
    ok = FakeUploader()
    run_spooler(
        [watch], repo_id="fake/repo", state_dir=state, shard_size_mb=1,
        settle_seconds=30, uploader=ok, once=True,
    )
    assert Journal(state / "journal.json").stats()["episodes_shipped"] == 1


class CommitThenCrashBackend:
    def __init__(self, delegate: FilesystemStorageBackend) -> None:
        self.delegate = delegate
        self.crashed = False

    def publish(self, shard):
        commit = self.delegate.publish(shard)
        if not self.crashed:
            self.crashed = True
            raise RuntimeError("crash after durable commit")
        return commit


def test_restart_after_remote_commit_is_exactly_once(tmp_path: Path) -> None:
    """Crash after marker publication but before journal flush is idempotent."""
    import pytest

    watch = tmp_path / "capture"
    _write_episode(watch, "episode")
    _age(tmp_path)
    state = tmp_path / "state"
    object_root = tmp_path / "objects"
    backend = CommitThenCrashBackend(FilesystemStorageBackend(object_root))

    with pytest.raises(RuntimeError, match="after durable commit"):
        run_spooler(
            [watch],
            repo_id="unused",
            state_dir=state,
            shard_size_mb=1,
            settle_seconds=30,
            storage=backend,
            once=True,
        )
    assert list(watch.glob("*.mkv")), "local source survives before journal commit"
    assert len(list(object_root.glob("commits/*.commit.json"))) == 1

    run_spooler(
        [watch],
        repo_id="unused",
        state_dir=state,
        shard_size_mb=1,
        settle_seconds=30,
        storage=backend,
        once=True,
    )
    assert Journal(state / "journal.json").stats()["episodes_shipped"] == 1
    assert len(list(object_root.glob("shards/*.tar"))) == 1
    assert len(list(object_root.glob("commits/*.commit.json"))) == 1
    assert not list(watch.glob("*.mkv"))


class ReceiptBackend:
    def publish(self, shard):
        receipt = {
            "commit_id": "receipt-1",
            "checksum": shard.sha256,
            "size_bytes": shard.size_bytes,
            "state": "committed",
        }
        return StorageCommit(
            "receipt-1",
            "uploads/shard.tar",
            shard.sha256,
            shard.size_bytes,
            receipt=receipt,
            authoritative_receipt=True,
        )

    def status(self):
        return {"backend": "control-plane", "cluster_connected": True}


def test_receipt_is_journaled_before_source_deletion(tmp_path: Path) -> None:
    watch = tmp_path / "capture"
    _write_episode(watch, "episode")
    _age(tmp_path)
    state = tmp_path / "state"
    run_spooler(
        [watch],
        repo_id="raw",
        state_dir=state,
        shard_size_mb=1,
        settle_seconds=30,
        storage=ReceiptBackend(),
        once=True,
    )
    journal = json.loads((state / "journal.json").read_text())
    assert journal["uploaded_shards"][0]["receipt"]["commit_id"] == "receipt-1"
    assert not list(watch.glob("*.mkv"))


def test_journal_failure_keeps_sources_after_remote_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    watch = tmp_path / "capture"
    _write_episode(watch, "episode")
    _age(tmp_path)

    def fail_journal(*_args, **_kwargs):
        raise OSError("journal fsync failed")

    monkeypatch.setattr(Journal, "record_uploaded_shard", fail_journal)
    with pytest.raises(OSError, match="journal fsync"):
        run_spooler(
            [watch],
            repo_id="raw",
            state_dir=tmp_path / "state",
            shard_size_mb=1,
            settle_seconds=30,
            storage=ReceiptBackend(),
            once=True,
        )
    assert list(watch.glob("*.mkv")), "deletion must follow durable receipt journal"


class RejectingBackend:
    def __init__(self, error):
        self.error = error

    def publish(self, _shard):
        raise self.error("cluster rejected shard")

    def status(self):
        return {"backend": "control-plane", "cluster_connected": True}


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (StorageConflictError, "identity_conflict"),
        (StorageValidationError, "validation_rejected"),
    ],
)
def test_semantic_rejection_is_durable_and_never_deletes(
    tmp_path: Path, error, kind: str
) -> None:
    watch = tmp_path / "capture"
    _write_episode(watch, "episode")
    _age(tmp_path)
    state = tmp_path / "state"
    run_spooler(
        [watch],
        repo_id="raw",
        state_dir=state,
        shard_size_mb=1,
        settle_seconds=30,
        storage=RejectingBackend(error),
        once=True,
    )
    assert list(watch.glob("*.mkv"))
    journal = Journal(state / "journal.json")
    stats = journal.stats()
    assert stats["episodes_blocked"] == 1
    assert next(iter(stats["blocked_episodes"].values()))["kind"] == kind
    assert json.loads((state / "status.json").read_text())["admission_open"] is False


class OutageThenRecoveryBackend(ReceiptBackend):
    def __init__(self) -> None:
        self.calls = 0

    def publish(self, shard):
        self.calls += 1
        if self.calls == 1:
            raise StorageUnavailableError("cluster offline")
        return super().publish(shard)


def test_cluster_outage_then_restart_recovers_without_deletion(tmp_path: Path) -> None:
    watch = tmp_path / "capture"
    _write_episode(watch, "episode")
    _age(tmp_path)
    state = tmp_path / "state"
    backend = OutageThenRecoveryBackend()
    with pytest.raises(StorageUnavailableError, match="offline"):
        run_spooler(
            [watch], repo_id="raw", state_dir=state, shard_size_mb=1,
            settle_seconds=30, storage=backend, once=True,
        )
    assert list(watch.glob("*.mkv"))
    run_spooler(
        [watch], repo_id="raw", state_dir=state, shard_size_mb=1,
        settle_seconds=30, storage=backend, once=True,
    )
    assert not list(watch.glob("*.mkv"))
    status = json.loads((state / "status.json").read_text())
    assert status["cluster_connected"] is True
