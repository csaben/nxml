"""Discovering completed episodes under the capture output directories.

An episode is COMPLETE when its ``{name}.manifest.json`` exists (the writers emit it last),
its video + parquet siblings exist, and none of the three has been modified for
``settle_seconds`` (the write sequence is not atomic; the settle window protects against
reading a mid-flush episode, including nxml-collect's post-close ``_stamp_metadata``
manifest rewrite).

Handles both on-disk layouts: nxml-collect writes flat files into the output dir;
nxml-autopilot's web mode writes one subdirectory per episode. Discovery just globs
``**/*.manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

VIDEO_SUFFIXES = (".mkv", ".mp4")


@dataclass(frozen=True)
class Episode:
    episode_id: str  # unique within the spool run (path-derived)
    manifest_path: Path
    video_path: Path
    parquet_path: Path
    events_path: Path | None = None

    @property
    def files(self) -> tuple[Path, ...]:
        files = [self.video_path, self.parquet_path]
        if self.events_path is not None:
            files.append(self.events_path)
        files.append(self.manifest_path)
        return tuple(files)

    @property
    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.files)


def _episode_id(watch_root: Path, manifest_path: Path) -> str:
    """Path-derived unique id: relative path with separators flattened.

    Flat layout: ``20260709_153000`` -> ``20260709_153000``.
    Autopilot subdirs: ``20260709T153000/20260709_193001`` -> ``20260709T153000__20260709_193001``.
    """
    rel = manifest_path.relative_to(watch_root)
    stem = rel.name.removesuffix(".manifest.json")
    parts = [*rel.parts[:-1], stem]
    return "__".join(parts)


def discover_episodes(
    watch_dirs: list[Path], *, settle_seconds: float = 30.0, now: float | None = None
) -> list[Episode]:
    """Completed, settled episodes across the watch dirs, oldest first."""
    now = time.time() if now is None else now
    episodes: list[Episode] = []
    for root in watch_dirs:
        if not root.is_dir():
            continue
        for manifest_path in root.rglob("*.manifest.json"):
            stem = manifest_path.name.removesuffix(".manifest.json")
            video_path = next(
                (
                    p
                    for suffix in VIDEO_SUFFIXES
                    if (p := manifest_path.with_name(stem + suffix)).is_file()
                ),
                None,
            )
            parquet_path = manifest_path.with_name(stem + ".parquet")
            if video_path is None or not parquet_path.is_file():
                continue  # npz debug episodes or partial writes: not spoolable
            newest_mtime = max(
                p.stat().st_mtime for p in (video_path, parquet_path, manifest_path)
            )
            if now - newest_mtime < settle_seconds:
                continue
            try:
                manifest = json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue  # mid-rewrite manifest; next scan gets it
            events_path = manifest_path.with_name(stem + ".events.parquet")
            if manifest.get("schema_version") == 2 and (
                not events_path.is_file()
                or not _checksums_match(manifest_path.parent, manifest)
            ):
                continue
            episodes.append(
                Episode(
                    episode_id=_episode_id(root, manifest_path),
                    manifest_path=manifest_path,
                    video_path=video_path,
                    parquet_path=parquet_path,
                    events_path=events_path if events_path.is_file() else None,
                )
            )
    return sorted(episodes, key=lambda e: e.video_path.stat().st_mtime)


def _checksums_match(root: Path, manifest: dict[str, object]) -> bool:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        return False
    for name, metadata in files.items():
        if not isinstance(name, str) or not isinstance(metadata, dict):
            return False
        path = root / name
        if not path.is_file() or path.stat().st_size != metadata.get("bytes"):
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != metadata.get("sha256"):
            return False
    return True
