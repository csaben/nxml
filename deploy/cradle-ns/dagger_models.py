"""Digest-verified model acquisition and atomic off-loop activation."""

from __future__ import annotations

import hashlib
import shutil
import threading
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED = {
    "architecture": "bc_transformer_v1",
    "action_spec_id": "switch_packets.v1",
    "action_dim": 26,
}


def check_compatibility(revision: dict) -> None:
    compatibility = revision.get("compatibility") or {}
    mismatches = {
        k: (compatibility.get(k), v) for k, v in REQUIRED.items() if compatibility.get(k) != v
    }
    if mismatches:
        raise ValueError(f"incompatible model revision: {mismatches}")


class RevisionCache:
    def __init__(self, root: Path):
        self.root = root

    def acquire(self, revision: dict) -> Path:
        check_compatibility(revision)
        digest = revision["checkpoint_sha256"]
        target = self.root / digest
        if target.is_file() and _sha(target) == digest:
            return target
        self.root.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".tmp")
        uri = revision["checkpoint_path"]
        if uri.startswith(("http://", "https://")):
            with urllib.request.urlopen(uri, timeout=60) as src, temp.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        else:
            source = Path(uri.removeprefix("file://"))
            shutil.copyfile(source, temp)
        if _sha(temp) != digest:
            temp.unlink(missing_ok=True)
            raise ValueError("checkpoint digest mismatch")
        temp.replace(target)
        return target


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ModelState:
    active: str | None = None
    previous: str | None = None
    loading: str | None = None
    error: str | None = None
    armed: bool = False


class AtomicModelRuntime:
    def __init__(self, cache: RevisionCache, loader: Callable[[Path], Any]):
        self.cache = cache
        self.loader = loader
        self._lock = threading.Lock()
        self._handle = None
        self._previous_handle = None
        self._state = ModelState()

    def state(self):
        with self._lock:
            return self._state

    def load_async(self, revision: dict) -> None:
        with self._lock:
            if self._state.loading:
                raise RuntimeError("model load already active")
            self._state = ModelState(
                self._state.active, self._state.previous, revision["revision_id"]
            )
        threading.Thread(
            target=self._load, args=(revision,), daemon=True, name="dagger-model-load"
        ).start()

    def _load(self, revision):
        try:
            handle = self.loader(self.cache.acquire(revision))
            with self._lock:
                old = self._state.active
                self._previous_handle = self._handle
                self._handle = handle
                self._state = ModelState(revision["revision_id"], old, armed=False)
        except Exception as error:
            with self._lock:
                self._handle = None
                self._state = ModelState(error=str(error), armed=False)

    def rollback(self) -> None:
        with self._lock:
            if self._state.previous is None or self._previous_handle is None:
                raise RuntimeError("no previous verified revision")
            active, previous = self._state.active, self._state.previous
            self._handle, self._previous_handle = self._previous_handle, self._handle
            self._state = ModelState(previous, active, armed=False)

    def neutral_disarm(self, reason: str):
        with self._lock:
            self._state = ModelState(
                self._state.active, self._state.previous, error=reason, armed=False
            )
