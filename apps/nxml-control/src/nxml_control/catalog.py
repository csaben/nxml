"""SQLite ingest catalog with transactional exactly-once registration."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class Upload:
    id: str
    idempotency_key: str
    object_key: str
    expected_size_bytes: int
    expected_sha256: str
    state: str
    actual_size_bytes: int | None
    actual_sha256: str | None
    dataset_id: str | None
    shard_id: str | None


class Catalog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _init(self) -> None:
        with self.connect() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS uploads(
              id TEXT PRIMARY KEY, idempotency_key TEXT NOT NULL UNIQUE,
              object_key TEXT NOT NULL UNIQUE, expected_size_bytes INTEGER NOT NULL,
              expected_sha256 TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN('created','uploaded','committed')),
              actual_size_bytes INTEGER, actual_sha256 TEXT, dataset_id TEXT, shard_id TEXT UNIQUE,
              manifest_json TEXT);
            """)

    def create_upload(
        self, *, idempotency_key: str, object_key: str, size_bytes: int, sha256: str
    ) -> Upload:
        with self.connect() as db:
            try:
                upload_id = str(uuid4())
                db.execute(
                    "INSERT INTO uploads VALUES(?,?,?,?,?,'created',NULL,NULL,NULL,NULL,NULL)",
                    (upload_id, idempotency_key, object_key, size_bytes, sha256),
                )
            except sqlite3.IntegrityError:
                row = db.execute(
                    "SELECT * FROM uploads WHERE idempotency_key=?", (idempotency_key,)
                ).fetchone()
                if row is None:
                    raise ValueError("object key is already reserved") from None
                upload = _upload(row)
                if (upload.object_key, upload.expected_size_bytes, upload.expected_sha256) != (
                    object_key,
                    size_bytes,
                    sha256,
                ):
                    raise ValueError("idempotency key reused with a different request") from None
                return upload
        return self.get(upload_id)

    def get(self, upload_id: str) -> Upload:
        with self.connect() as db:
            row = db.execute("SELECT * FROM uploads WHERE id=?", (upload_id,)).fetchone()
        if row is None:
            raise KeyError(upload_id)
        return _upload(row)

    def mark_uploaded(self, upload_id: str, *, size_bytes: int, sha256: str) -> Upload:
        with self.connect() as db:
            row = db.execute("SELECT * FROM uploads WHERE id=?", (upload_id,)).fetchone()
            if row is None:
                raise KeyError(upload_id)
            current = _upload(row)
            if current.state != "created":
                if (current.actual_size_bytes, current.actual_sha256) != (size_bytes, sha256):
                    raise ValueError("immutable object changed")
                return current
            db.execute(
                "UPDATE uploads SET state='uploaded',actual_size_bytes=?,actual_sha256=? WHERE id=?",
                (size_bytes, sha256, upload_id),
            )
        return self.get(upload_id)

    def commit(self, upload_id: str, *, dataset_id: str, shard_id: str, manifest: dict) -> Upload:
        with self.connect() as db:
            row = db.execute("SELECT * FROM uploads WHERE id=?", (upload_id,)).fetchone()
            if row is None:
                raise KeyError(upload_id)
            current = _upload(row)
            if current.state == "committed":
                if (current.dataset_id, current.shard_id) != (dataset_id, shard_id):
                    raise ValueError("committed upload cannot be reassigned")
                return current
            if current.state != "uploaded":
                raise ValueError("upload must be verified before commit")
            try:
                db.execute(
                    "UPDATE uploads SET state='committed',dataset_id=?,shard_id=?,manifest_json=? WHERE id=?",
                    (dataset_id, shard_id, json.dumps(manifest, sort_keys=True), upload_id),
                )
            except sqlite3.IntegrityError:
                raise ValueError("shard_id already registered") from None
        return self.get(upload_id)

    def health(self) -> dict[str, int | str]:
        with self.connect() as db:
            counts = dict(db.execute("SELECT state,count(*) FROM uploads GROUP BY state"))
        return {
            "status": "ok",
            **{s: int(counts.get(s, 0)) for s in ("created", "uploaded", "committed")},
        }


def _upload(row: sqlite3.Row) -> Upload:
    return Upload(**{name: row[name] for name in Upload.__dataclass_fields__})
