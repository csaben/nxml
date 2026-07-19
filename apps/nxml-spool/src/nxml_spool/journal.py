"""Spool journal: which episodes are shipped, which shard index is next.

One JSON file, atomically rewritten. Restart-safe: an episode is only recorded as shipped
after its shard's upload is verified, so a crash between pack and verify re-packs those
episodes into a fresh shard (duplicate shard uploads are possible but duplicate *episodes
inside the dataset* are not, because shipped episodes are deleted from disk and skipped by
id — HF-side stale shards can be pruned manually if that ever happens).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


class Journal:
    def __init__(self, path: Path) -> None:
        self.path = path
        if path.is_file():
            data = json.loads(path.read_text())
        else:
            data = {
                "next_shard_index": 0,
                "shipped_episodes": {},
                "uploaded_shards": [],
                "blocked_episodes": {},
                "backend": {},
            }
        data.setdefault("blocked_episodes", {})
        data.setdefault("backend", {})
        self._data = data

    @property
    def next_shard_index(self) -> int:
        return self._data["next_shard_index"]

    def is_shipped(self, episode_id: str) -> bool:
        return episode_id in self._data["shipped_episodes"]

    def is_blocked(self, episode_id: str) -> bool:
        return episode_id in self._data["blocked_episodes"]

    def record_uploaded_shard(
        self,
        shard_name: str,
        episode_ids: list[str],
        *,
        commit_id: str | None = None,
        checksum: str | None = None,
        receipt: dict | None = None,
    ) -> None:
        now = time.time()
        self._data["uploaded_shards"].append(
            {
                "shard": shard_name,
                "time": now,
                "commit_id": commit_id,
                "checksum": checksum,
                "receipt": receipt,
            }
        )
        for episode_id in episode_ids:
            self._data["shipped_episodes"][episode_id] = {
                "shard": shard_name,
                "time": now,
                "commit_id": commit_id,
                "receipt": receipt,
            }
            self._data["blocked_episodes"].pop(episode_id, None)
        self._data["next_shard_index"] += 1
        self._data["backend"] = {
            "last_success_at": now,
            "last_error": None,
            "last_error_kind": None,
        }
        self._flush()

    def record_blocked(
        self,
        shard_name: str,
        episode_ids: list[str],
        *,
        checksum: str,
        kind: str,
        error: str,
    ) -> None:
        now = time.time()
        for episode_id in episode_ids:
            self._data["blocked_episodes"][episode_id] = {
                "shard": shard_name,
                "checksum": checksum,
                "kind": kind,
                "error": error,
                "time": now,
            }
        self._data["backend"] = {
            "last_error_at": now,
            "last_error": error,
            "last_error_kind": kind,
        }
        self._flush()

    def record_transient_error(self, error: str) -> None:
        self._data["backend"] = {
            **self._data.get("backend", {}),
            "last_error_at": time.time(),
            "last_error": error,
            "last_error_kind": "unavailable",
        }
        self._flush()

    def _flush(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w") as stream:
            stream.write(json.dumps(self._data, indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(self.path)
        directory_fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def stats(self) -> dict:
        return {
            "episodes_shipped": len(self._data["shipped_episodes"]),
            "shards_uploaded": len(self._data["uploaded_shards"]),
            "next_shard_index": self._data["next_shard_index"],
            "episodes_blocked": len(self._data["blocked_episodes"]),
            "blocked_episodes": self._data["blocked_episodes"],
            "backend_state": self._data["backend"],
        }
