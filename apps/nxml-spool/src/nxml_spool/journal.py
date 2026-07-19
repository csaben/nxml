"""Spool journal: which episodes are shipped, which shard index is next.

One JSON file, atomically rewritten. Restart-safe: an episode is only recorded as shipped
after its shard's upload is verified, so a crash between pack and verify re-packs those
episodes into a fresh shard (duplicate shard uploads are possible but duplicate *episodes
inside the dataset* are not, because shipped episodes are deleted from disk and skipped by
id — HF-side stale shards can be pruned manually if that ever happens).
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class Journal:
    def __init__(self, path: Path) -> None:
        self.path = path
        if path.is_file():
            data = json.loads(path.read_text())
        else:
            data = {"next_shard_index": 0, "shipped_episodes": {}, "uploaded_shards": []}
        self._data = data

    @property
    def next_shard_index(self) -> int:
        return self._data["next_shard_index"]

    def is_shipped(self, episode_id: str) -> bool:
        return episode_id in self._data["shipped_episodes"]

    def record_uploaded_shard(self, shard_name: str, episode_ids: list[str]) -> None:
        now = time.time()
        self._data["uploaded_shards"].append({"shard": shard_name, "time": now})
        for episode_id in episode_ids:
            self._data["shipped_episodes"][episode_id] = {"shard": shard_name, "time": now}
        self._data["next_shard_index"] += 1
        self._flush()

    def _flush(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=2))
        tmp.replace(self.path)

    def stats(self) -> dict:
        return {
            "episodes_shipped": len(self._data["shipped_episodes"]),
            "shards_uploaded": len(self._data["uploaded_shards"]),
            "next_shard_index": self._data["next_shard_index"],
        }
