"""The spool loop: discover -> pack -> upload -> verify -> delete.

Deletion is gated twice: the shard upload must verify (size match via the Hub API), and
``--no-delete`` keeps everything local for dry runs. Status is mirrored to
``status.json`` in the state dir so the capture UI can render a live strip.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

from nxml_spool.episodes import discover_episodes
from nxml_spool.journal import Journal
from nxml_spool.shards import pack_shard, plan_shards
from nxml_spool.storage import HFDatasetStorageBackend, StorageBackend, StorageCommit

logger = logging.getLogger(__name__)


class Uploader:
    """HF Hub upload + size verification. Isolated for testing (FakeUploader in tests)."""

    def __init__(self, repo_id: str, *, private: bool = True) -> None:
        from huggingface_hub import HfApi

        self.api = HfApi()
        self.repo_id = repo_id
        self.api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    def upload_and_verify(self, local_path: Path, path_in_repo: str) -> None:
        self.api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type="dataset",
        )
        info = self.api.get_paths_info(self.repo_id, [path_in_repo], repo_type="dataset")
        remote_size = info[0].size if info else None
        local_size = local_path.stat().st_size
        if remote_size != local_size:
            raise RuntimeError(
                f"Upload verification failed for {path_in_repo}: "
                f"remote={remote_size} local={local_size}"
            )


class _LegacyUploaderBackend:
    """Preserve the pre-v2 uploader injection contract used by callers/tests."""

    def __init__(self, uploader: Uploader) -> None:
        self.uploader = uploader

    def publish(self, shard) -> StorageCommit:
        self.uploader.upload_and_verify(shard.path, f"shards/{shard.path.name}")
        self.uploader.upload_and_verify(
            shard.sidecar_path, f"meta/{shard.sidecar_path.name}"
        )
        return StorageCommit(
            shard.sha256,
            f"shards/{shard.path.name}",
            shard.sha256,
            shard.size_bytes,
        )


def _write_status(state_dir: Path, journal: Journal, extra: dict) -> None:
    status = {**journal.stats(), **extra, "updated_at": time.time()}
    tmp = state_dir / "status.json.tmp"
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(state_dir / "status.json")


def run_spooler(
    watch_dirs: list[Path],
    *,
    repo_id: str,
    state_dir: Path,
    shard_size_mb: int = 1024,
    settle_seconds: float = 30.0,
    poll_seconds: float = 15.0,
    delete_after_upload: bool = True,
    flush_partial_after_s: float = 900.0,
    uploader: Uploader | None = None,
    storage: StorageBackend | None = None,
    once: bool = False,
    disk_high_watermark: float = 0.85,
    disk_low_watermark: float = 0.75,
) -> None:
    """Run the spool loop (forever unless ``once``).

    A partial (below-target-size) shard is packed anyway when its oldest episode has
    waited more than ``flush_partial_after_s`` — bounds time-to-durability when capture
    stops for the day.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    journal = Journal(state_dir / "journal.json")
    if storage is not None and uploader is not None:
        raise ValueError("pass either storage or uploader, not both")
    backend: StorageBackend
    if storage is not None:
        backend = storage
    elif uploader is not None:
        backend = _LegacyUploaderBackend(uploader)
    else:
        backend = HFDatasetStorageBackend(repo_id)
    if not 0 < disk_low_watermark < disk_high_watermark < 1:
        raise ValueError("disk watermarks must satisfy 0 < low < high < 1")
    shard_size_bytes = shard_size_mb * 1024 * 1024
    staging_dir = state_dir / "staging"

    logger.info(
        f"Spooling {[str(d) for d in watch_dirs]} -> {repo_id} "
        f"(shards ~{shard_size_mb} MB, settle {settle_seconds}s)"
    )

    pressure_active = False
    while True:
        pending = [
            ep
            for ep in discover_episodes(watch_dirs, settle_seconds=settle_seconds)
            if not journal.is_shipped(ep.episode_id)
        ]
        disk = shutil.disk_usage(watch_dirs[0]) if watch_dirs else None
        disk_free_gb = disk.free / 1e9 if disk is not None else 0
        disk_used_fraction = (disk.used / disk.total) if disk is not None else 0
        if pressure_active:
            pressure_active = disk_used_fraction > disk_low_watermark
        else:
            pressure_active = disk_used_fraction >= disk_high_watermark
        groups = plan_shards(pending, shard_size_bytes=shard_size_bytes)

        for group in groups:
            group_bytes = sum(ep.size_bytes for ep in group)
            is_full = group_bytes >= shard_size_bytes * 0.9
            oldest_age = time.time() - min(ep.video_path.stat().st_mtime for ep in group)
            if (
                not is_full
                and oldest_age < flush_partial_after_s
                and not once
                and not pressure_active
            ):
                continue  # wait for more episodes before sealing a small shard

            shard = pack_shard(group, staging_dir, journal.next_shard_index)
            logger.info(
                f"Packed {shard.path.name}: {len(group)} episodes, "
                f"{shard.size_bytes / 1e6:.0f} MB — uploading"
            )
            _write_status(state_dir, journal, {"uploading": shard.path.name,
                                               "pending_episodes": len(pending),
                                               "disk_free_gb": round(disk_free_gb, 1),
                                               "disk_used_fraction": disk_used_fraction,
                                               "disk_pressure": pressure_active})
            commit = backend.publish(shard)
            if commit.checksum != shard.sha256:
                raise RuntimeError(f"backend committed wrong checksum for {shard.path.name}")
            journal.record_uploaded_shard(
                shard.path.name,
                shard.episode_ids,
                commit_id=commit.commit_id,
                checksum=commit.checksum,
            )
            logger.info(f"Verified {shard.path.name} on {repo_id}")

            shard.path.unlink()
            shard.sidecar_path.unlink()
            if delete_after_upload:
                for ep in group:
                    for f in ep.files:
                        f.unlink(missing_ok=True)
                    # Autopilot's per-episode subdirs: remove when emptied.
                    parent = ep.manifest_path.parent
                    if parent not in watch_dirs and not any(parent.iterdir()):
                        parent.rmdir()

        _write_status(
            state_dir,
            journal,
            {
                "uploading": None,
                "pending_episodes": len(
                    [
                        ep
                        for ep in discover_episodes(watch_dirs, settle_seconds=settle_seconds)
                        if not journal.is_shipped(ep.episode_id)
                    ]
                ),
                "disk_free_gb": round(disk_free_gb, 1),
                "disk_used_fraction": disk_used_fraction,
                "disk_pressure": pressure_active,
                "disk_high_watermark": disk_high_watermark,
                "disk_low_watermark": disk_low_watermark,
            },
        )
        if once:
            return
        time.sleep(poll_seconds)
