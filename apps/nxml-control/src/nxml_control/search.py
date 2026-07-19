"""Searchability metadata foundation; no embedding computation or vector store."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

INDEX_STATES = {"pending", "running", "complete", "failed"}
ARTIFACT_STATES = {"pending", "running", "complete", "failed"}


@dataclass(frozen=True)
class ClipReference:
    dataset_id: str
    episode_id: str
    window_start_ns: int | None = None
    window_end_ns: int | None = None

    def __post_init__(self):
        if not self.dataset_id or not self.episode_id:
            raise ValueError("dataset_id and episode_id are required")
        if (self.window_start_ns is None) != (self.window_end_ns is None):
            raise ValueError("clip window requires both start and end")
        if self.window_start_ns is not None and (
            self.window_start_ns < 0 or self.window_end_ns <= self.window_start_ns
        ):
            raise ValueError("clip window must be half-open [start_ns,end_ns) with end > start")


class SearchCatalog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._init()

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _init(self):
        with self.connect() as db:
            db.executescript("""
        CREATE TABLE IF NOT EXISTS index_versions(
          index_version TEXT PRIMARY KEY,idempotency_key TEXT NOT NULL UNIQUE,source_snapshot_id TEXT NOT NULL,
          embedding_model TEXT,embedding_version TEXT,state TEXT NOT NULL CHECK(state IN('pending','running','complete','failed')),
          error TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS derived_artifacts(
          artifact_id TEXT PRIMARY KEY,idempotency_key TEXT NOT NULL UNIQUE,dataset_id TEXT NOT NULL,episode_id TEXT NOT NULL,
          window_start_ns INTEGER,window_end_ns INTEGER,artifact_type TEXT NOT NULL,embedding_model TEXT,embedding_version TEXT,
          labels_json TEXT NOT NULL,derived_features_json TEXT NOT NULL,source_snapshot_id TEXT NOT NULL,index_version TEXT NOT NULL REFERENCES index_versions(index_version),
          status TEXT NOT NULL CHECK(status IN('pending','running','complete','failed')),uri TEXT,error TEXT,created_at TEXT NOT NULL,
          FOREIGN KEY(dataset_id,episode_id) REFERENCES episodes(dataset_id,id));
        CREATE TABLE IF NOT EXISTS annotations(
          annotation_id TEXT PRIMARY KEY,idempotency_key TEXT NOT NULL UNIQUE,dataset_id TEXT NOT NULL,episode_id TEXT NOT NULL,
          window_start_ns INTEGER NOT NULL,window_end_ns INTEGER NOT NULL,labels_json TEXT NOT NULL,source_snapshot_id TEXT,
          index_version TEXT,status TEXT NOT NULL CHECK(status IN('pending','running','complete','failed')),created_at TEXT NOT NULL,
          FOREIGN KEY(dataset_id,episode_id) REFERENCES episodes(dataset_id,id));
        CREATE INDEX IF NOT EXISTS artifacts_episode_idx ON derived_artifacts(dataset_id,episode_id,window_start_ns);
        CREATE INDEX IF NOT EXISTS annotations_episode_idx ON annotations(dataset_id,episode_id,window_start_ns);
        """)

    def create_index(
        self, *, idempotency_key, source_snapshot_id, embedding_model=None, embedding_version=None
    ):
        now = datetime.now(UTC).isoformat()
        version = "idx-" + uuid4().hex
        request = (source_snapshot_id, embedding_model, embedding_version)
        with self.connect() as db:
            if (
                db.execute(
                    "SELECT 1 FROM snapshots WHERE snapshot_id=?", (source_snapshot_id,)
                ).fetchone()
                is None
            ):
                raise KeyError(source_snapshot_id)
            try:
                db.execute(
                    "INSERT INTO index_versions VALUES(?,?,?,?,?,'pending',NULL,?,?)",
                    (version, idempotency_key, *request, now, now),
                )
            except sqlite3.IntegrityError:
                row = db.execute(
                    "SELECT * FROM index_versions WHERE idempotency_key=?", (idempotency_key,)
                ).fetchone()
                assert row is not None
                if (
                    row["source_snapshot_id"],
                    row["embedding_model"],
                    row["embedding_version"],
                ) != request:
                    raise ValueError("index idempotency key reused") from None
                return dict(row)
        return self.get_index(version)

    def get_index(self, version):
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM index_versions WHERE index_version=?", (version,)
            ).fetchone()
        if row is None:
            raise KeyError(version)
        return dict(row)

    def set_index_state(self, version, state, *, error=None):
        if state not in INDEX_STATES:
            raise ValueError("invalid index state")
        with self.connect() as db:
            current = db.execute(
                "SELECT state FROM index_versions WHERE index_version=?", (version,)
            ).fetchone()
            if current is None:
                raise KeyError(version)
            allowed = {
                "pending": {"running", "failed"},
                "running": {"complete", "failed"},
                "complete": set(),
                "failed": set(),
            }
            if state != current["state"] and state not in allowed[current["state"]]:
                raise ValueError("invalid index state transition")
            db.execute(
                "UPDATE index_versions SET state=?,error=?,updated_at=? WHERE index_version=?",
                (state, error, datetime.now(UTC).isoformat(), version),
            )
        return self.get_index(version)

    def create_artifact(
        self,
        *,
        idempotency_key,
        clip,
        artifact_type,
        source_snapshot_id,
        index_version,
        embedding_model=None,
        embedding_version=None,
        labels=None,
        derived_features=None,
        status="pending",
        uri=None,
    ):
        ref = ClipReference(**clip)
        if status not in ARTIFACT_STATES:
            raise ValueError("invalid artifact status")
        payload = (
            ref.dataset_id,
            ref.episode_id,
            ref.window_start_ns,
            ref.window_end_ns,
            artifact_type,
            embedding_model,
            embedding_version,
            json.dumps(labels or [], sort_keys=True),
            json.dumps(derived_features or {}, sort_keys=True),
            source_snapshot_id,
            index_version,
            status,
            uri,
        )
        with self.connect() as db:
            self._require_episode(db, ref)
            self._require_index(db, index_version, source_snapshot_id)
            artifact_id = str(uuid4())
            try:
                db.execute(
                    "INSERT INTO derived_artifacts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,?)",
                    (artifact_id, idempotency_key, *payload, datetime.now(UTC).isoformat()),
                )
            except sqlite3.IntegrityError:
                row = db.execute(
                    "SELECT * FROM derived_artifacts WHERE idempotency_key=?", (idempotency_key,)
                ).fetchone()
                if row is None:
                    raise
                if (
                    tuple(
                        row[name]
                        for name in (
                            "dataset_id",
                            "episode_id",
                            "window_start_ns",
                            "window_end_ns",
                            "artifact_type",
                            "embedding_model",
                            "embedding_version",
                            "labels_json",
                            "derived_features_json",
                            "source_snapshot_id",
                            "index_version",
                            "status",
                            "uri",
                        )
                    )
                    != payload
                ):
                    raise ValueError("artifact idempotency key reused") from None
                return _artifact(row)
        return self.get_artifact(artifact_id)

    def get_artifact(self, artifact_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM derived_artifacts WHERE artifact_id=?", (artifact_id,)
            ).fetchone()
        if row is None:
            raise KeyError(artifact_id)
        return _artifact(row)

    def set_artifact_state(self, artifact_id, status, *, error=None):
        if status not in ARTIFACT_STATES:
            raise ValueError("invalid artifact status")
        with self.connect() as db:
            row = db.execute(
                "SELECT status FROM derived_artifacts WHERE artifact_id=?", (artifact_id,)
            ).fetchone()
            if row is None:
                raise KeyError(artifact_id)
            allowed = {
                "pending": {"running", "failed"},
                "running": {"complete", "failed"},
                "complete": set(),
                "failed": set(),
            }
            if status != row["status"] and status not in allowed[row["status"]]:
                raise ValueError("invalid artifact state transition")
            db.execute(
                "UPDATE derived_artifacts SET status=?, error=? WHERE artifact_id=?",
                (status, error, artifact_id),
            )
        return self.get_artifact(artifact_id)

    def artifacts(self, dataset_id, episode_id):
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM derived_artifacts WHERE dataset_id=? AND episode_id=? ORDER BY window_start_ns,artifact_id",
                (dataset_id, episode_id),
            ).fetchall()
        return [_artifact(row) for row in rows]

    def create_annotation(
        self,
        *,
        idempotency_key,
        clip,
        labels,
        source_snapshot_id=None,
        index_version=None,
        status="complete",
    ):
        ref = ClipReference(**clip)
        if ref.window_start_ns is None:
            raise ValueError("annotations require a temporal clip window")
        if status not in ARTIFACT_STATES:
            raise ValueError("invalid annotation status")
        payload = (
            ref.dataset_id,
            ref.episode_id,
            ref.window_start_ns,
            ref.window_end_ns,
            json.dumps(labels, sort_keys=True),
            source_snapshot_id,
            index_version,
            status,
        )
        with self.connect() as db:
            self._require_episode(db, ref)
            annotation_id = str(uuid4())
            try:
                db.execute(
                    "INSERT INTO annotations VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (annotation_id, idempotency_key, *payload, datetime.now(UTC).isoformat()),
                )
            except sqlite3.IntegrityError:
                row = db.execute(
                    "SELECT * FROM annotations WHERE idempotency_key=?", (idempotency_key,)
                ).fetchone()
                assert row is not None
                if (
                    tuple(
                        row[name]
                        for name in (
                            "dataset_id",
                            "episode_id",
                            "window_start_ns",
                            "window_end_ns",
                            "labels_json",
                            "source_snapshot_id",
                            "index_version",
                            "status",
                        )
                    )
                    != payload
                ):
                    raise ValueError("annotation idempotency key reused") from None
                return _annotation(row)
        return self.annotations(ref.dataset_id, ref.episode_id)[-1]

    def annotations(self, dataset_id, episode_id):
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM annotations WHERE dataset_id=? AND episode_id=? ORDER BY window_start_ns,annotation_id",
                (dataset_id, episode_id),
            ).fetchall()
        return [_annotation(row) for row in rows]

    @staticmethod
    def _require_episode(db, ref):
        if (
            db.execute(
                "SELECT 1 FROM episodes WHERE dataset_id=? AND id=?",
                (ref.dataset_id, ref.episode_id),
            ).fetchone()
            is None
        ):
            raise KeyError(ref.episode_id)

    @staticmethod
    def _require_index(db, version, snapshot):
        row = db.execute(
            "SELECT source_snapshot_id FROM index_versions WHERE index_version=?", (version,)
        ).fetchone()
        if row is None:
            raise KeyError(version)
        if row[0] != snapshot:
            raise ValueError("index version source snapshot mismatch")


def _artifact(row):
    item = dict(row)
    item["labels"] = json.loads(item.pop("labels_json"))
    item["derived_features"] = json.loads(item.pop("derived_features_json"))
    return item


def _annotation(row):
    item = dict(row)
    item["labels"] = json.loads(item.pop("labels_json"))
    return item
