"""Storage-neutral immutable objects, with a local test backend."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol
from uuid import uuid4


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    size_bytes: int
    sha256: str


class ObjectStorage(Protocol):
    def put_if_absent(self, key: str, stream: BinaryIO) -> ObjectInfo: ...
    def inspect(self, key: str) -> ObjectInfo | None: ...
    def open(self, key: str) -> BinaryIO: ...


def _safe(key: str) -> Path:
    path = Path(key)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe object key: {key!r}")
    return path


class LocalObjectStorage:
    """Atomic local implementation. Existing keys are never overwritten."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / _safe(key)

    def put_if_absent(self, key: str, stream: BinaryIO) -> ObjectInfo:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.upload")
        digest, size = hashlib.sha256(), 0
        try:
            with temporary.open("xb") as output:
                while chunk := stream.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            os.link(temporary, target)
            temporary.unlink()
        except FileExistsError:
            temporary.unlink(missing_ok=True)
            existing = self.inspect(key)
            assert existing is not None
            return existing
        return ObjectInfo(key, size, digest.hexdigest())

    def inspect(self, key: str) -> ObjectInfo | None:
        path = self._path(key)
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return ObjectInfo(key, path.stat().st_size, digest.hexdigest())

    def open(self, key: str) -> BinaryIO:
        return self._path(key).open("rb")
