"""SQLite ingest catalog with transactional exactly-once registration."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from nxml_control.contracts import ShardManifestV2


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


@dataclass(frozen=True)
class Shard:
    id: str
    dataset_id: str
    upload_id: str
    object_key: str
    size_bytes: int
    sha256: str
    manifest: dict


@dataclass(frozen=True)
class Episode:
    id: str
    dataset_id: str
    shard_id: str
    ordinal: int
    manifest: dict


class InvalidManifestError(ValueError):
    pass


class IdentityConflictError(ValueError):
    pass


@dataclass(frozen=True)
class CommitReceipt:
    commit_id: str
    upload_id: str
    checksum: str
    size_bytes: int
    storage_key: str
    state: str
    committed_at: str
    dataset_id: str
    shard_id: str

    @property
    def id(self) -> str:
        return self.commit_id


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    dataset_id: str
    created_at: str
    control_source: str
    manifest: dict


class Catalog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
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

    def _init(self) -> None:
        with self.connect() as db:
            db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS uploads(
          id TEXT PRIMARY KEY,idempotency_key TEXT NOT NULL UNIQUE,object_key TEXT NOT NULL UNIQUE,
          expected_size_bytes INTEGER NOT NULL,expected_sha256 TEXT NOT NULL,
          state TEXT NOT NULL CHECK(state IN('created','uploaded','committed')),
          actual_size_bytes INTEGER,actual_sha256 TEXT,dataset_id TEXT,shard_id TEXT UNIQUE,manifest_json TEXT);
        CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS shards(
          id TEXT PRIMARY KEY,dataset_id TEXT NOT NULL REFERENCES datasets(id),upload_id TEXT NOT NULL UNIQUE REFERENCES uploads(id),
          object_key TEXT NOT NULL UNIQUE,size_bytes INTEGER NOT NULL,sha256 TEXT NOT NULL,manifest_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS episodes(
          id TEXT NOT NULL,dataset_id TEXT NOT NULL REFERENCES datasets(id),shard_id TEXT NOT NULL REFERENCES shards(id),
          ordinal INTEGER NOT NULL,manifest_json TEXT NOT NULL,PRIMARY KEY(dataset_id,id),UNIQUE(shard_id,ordinal));
        CREATE TABLE IF NOT EXISTS commit_receipts(
          commit_id TEXT PRIMARY KEY,upload_id TEXT NOT NULL UNIQUE REFERENCES uploads(id),checksum TEXT NOT NULL,
          size_bytes INTEGER NOT NULL,storage_key TEXT NOT NULL,state TEXT NOT NULL CHECK(state="committed"),
          committed_at TEXT NOT NULL,dataset_id TEXT NOT NULL,shard_id TEXT NOT NULL UNIQUE);
        CREATE TABLE IF NOT EXISTS episode_quality_dispositions(
          disposition_id TEXT PRIMARY KEY,idempotency_key TEXT NOT NULL UNIQUE,dataset_id TEXT NOT NULL,episode_id TEXT NOT NULL,
          schema_id TEXT NOT NULL,training_eligible INTEGER NOT NULL,reason TEXT NOT NULL,validator TEXT NOT NULL,validator_version TEXT NOT NULL,created_at TEXT NOT NULL,
          FOREIGN KEY(dataset_id,episode_id) REFERENCES episodes(dataset_id,id));
        CREATE INDEX IF NOT EXISTS episode_quality_latest_idx ON episode_quality_dispositions(dataset_id,episode_id,created_at,disposition_id);
        CREATE TABLE IF NOT EXISTS snapshots(
          snapshot_id TEXT PRIMARY KEY,dataset_id TEXT NOT NULL REFERENCES datasets(id),created_at TEXT NOT NULL,
          control_source TEXT NOT NULL,manifest_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS snapshot_shards(
          snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),shard_id TEXT NOT NULL REFERENCES shards(id),
          ordinal INTEGER NOT NULL,PRIMARY KEY(snapshot_id,shard_id),UNIQUE(snapshot_id,ordinal));
        CREATE INDEX IF NOT EXISTS shards_dataset_idx ON shards(dataset_id);
        CREATE INDEX IF NOT EXISTS episodes_shard_idx ON episodes(shard_id);
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
                item = _upload(row)
                if (item.object_key, item.expected_size_bytes, item.expected_sha256) != (
                    object_key,
                    size_bytes,
                    sha256,
                ):
                    raise ValueError("idempotency key reused with a different request") from None
                return item
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
            item = _upload(row)
            if item.state != "created":
                if (item.actual_size_bytes, item.actual_sha256) != (size_bytes, sha256):
                    raise ValueError("immutable object changed")
                return item
            db.execute(
                "UPDATE uploads SET state='uploaded',actual_size_bytes=?,actual_sha256=? WHERE id=?",
                (size_bytes, sha256, upload_id),
            )
        return self.get(upload_id)

    def commit(
        self, upload_id: str, *, dataset_id: str, shard_id: str, manifest: dict
    ) -> CommitReceipt:
        try:
            parsed = ShardManifestV2.model_validate(manifest)
        except ValidationError as error:
            raise InvalidManifestError(str(error)) from error
        normalized = parsed.model_dump(mode="json")
        episodes = _episode_entries(normalized)
        canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        with self.connect() as db:
            row = db.execute("SELECT * FROM uploads WHERE id=?", (upload_id,)).fetchone()
            if row is None:
                raise KeyError(upload_id)
            item = _upload(row)
            if item.state == "committed":
                if (item.dataset_id, item.shard_id) != (dataset_id, shard_id):
                    raise IdentityConflictError("committed upload cannot be reassigned")
                existing = db.execute(
                    "SELECT manifest_json FROM shards WHERE id=?", (shard_id,)
                ).fetchone()
                if existing is not None and existing[0] != canonical:
                    raise IdentityConflictError("committed manifest cannot be changed")
                receipt = db.execute(
                    "SELECT * FROM commit_receipts WHERE upload_id=?", (upload_id,)
                ).fetchone()
                assert receipt is not None
                return _receipt(receipt)
            if item.state != "uploaded":
                raise IdentityConflictError("upload must be verified before commit")
            commit_id = str(uuid4())
            committed_at = datetime.now(UTC).isoformat()
            try:
                db.execute("INSERT OR IGNORE INTO datasets VALUES(?)", (dataset_id,))
                db.execute(
                    "INSERT INTO shards VALUES(?,?,?,?,?,?,?)",
                    (
                        shard_id,
                        dataset_id,
                        upload_id,
                        item.object_key,
                        item.actual_size_bytes,
                        item.actual_sha256,
                        canonical,
                    ),
                )
                for ordinal, (episode_id, episode_manifest) in enumerate(episodes):
                    db.execute(
                        "INSERT INTO episodes VALUES(?,?,?,?,?)",
                        (
                            episode_id,
                            dataset_id,
                            shard_id,
                            ordinal,
                            json.dumps(episode_manifest, sort_keys=True, separators=(",", ":")),
                        ),
                    )
                db.execute(
                    "UPDATE uploads SET state='committed',dataset_id=?,shard_id=?,manifest_json=? WHERE id=?",
                    (dataset_id, shard_id, canonical, upload_id),
                )
                db.execute(
                    "INSERT INTO commit_receipts VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        commit_id,
                        upload_id,
                        item.actual_sha256,
                        item.actual_size_bytes,
                        item.object_key,
                        "committed",
                        committed_at,
                        dataset_id,
                        shard_id,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise IdentityConflictError("shard or episode already registered") from error
        return self.get_receipt(commit_id)

    def get_receipt(self, commit_id: str) -> CommitReceipt:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM commit_receipts WHERE commit_id=?", (commit_id,)
            ).fetchone()
        if row is None:
            raise KeyError(commit_id)
        return _receipt(row)

    def list_receipts(
        self, *, dataset_id: str | None = None, shard_id: str | None = None, limit=100, offset=0
    ) -> list[CommitReceipt]:
        clauses = []
        params = []
        if dataset_id is not None:
            clauses.append("dataset_id=?")
            params.append(dataset_id)
        if shard_id is not None:
            clauses.append("shard_id=?")
            params.append(shard_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connect() as db:
            rows = db.execute(
                f"SELECT * FROM commit_receipts{where} ORDER BY committed_at,commit_id LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return [_receipt(row) for row in rows]

    def set_episode_quality(
        self,
        dataset_id: str,
        episode_id: str,
        *,
        idempotency_key: str,
        training_eligible: bool,
        reason: str,
        validator: str,
        validator_version: str,
    ) -> dict:
        payload = (
            dataset_id,
            episode_id,
            "nxml.episode-quality.v1",
            int(training_eligible),
            reason,
            validator,
            validator_version,
        )
        with self.connect() as db:
            if (
                db.execute(
                    "SELECT 1 FROM episodes WHERE dataset_id=? AND id=?",
                    (dataset_id, episode_id),
                ).fetchone()
                is None
            ):
                raise KeyError(episode_id)
            try:
                disposition_id = str(uuid4())
                db.execute(
                    "INSERT INTO episode_quality_dispositions VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (disposition_id, idempotency_key, *payload, datetime.now(UTC).isoformat()),
                )
            except sqlite3.IntegrityError:
                row = db.execute(
                    "SELECT * FROM episode_quality_dispositions WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
                assert row is not None
                actual = tuple(
                    row[name]
                    for name in (
                        "dataset_id",
                        "episode_id",
                        "schema_id",
                        "training_eligible",
                        "reason",
                        "validator",
                        "validator_version",
                    )
                )
                if actual != payload:
                    raise ValueError("quality disposition idempotency key reused") from None
                return _quality(row)
            row = db.execute(
                "SELECT * FROM episode_quality_dispositions WHERE disposition_id=?",
                (disposition_id,),
            ).fetchone()
            return _quality(row)

    def episode_quality(self, dataset_id: str, episode_id: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM episode_quality_dispositions WHERE dataset_id=? AND episode_id=? ORDER BY created_at,disposition_id",
                (dataset_id, episode_id),
            ).fetchall()
        return [_quality(row) for row in rows]

    def create_snapshot(self, dataset_id: str, *, control_source: str = "all") -> Snapshot:
        if control_source not in {"all", "human", "policy"}:
            raise ValueError("control_source must be all, human, or policy")
        with self.connect() as db:
            shard_rows = db.execute(
                "SELECT id,sha256,size_bytes,object_key FROM shards WHERE dataset_id=? ORDER BY id",
                (dataset_id,),
            ).fetchall()
            if not shard_rows:
                raise KeyError(dataset_id)
            episode_rows = db.execute(
                "SELECT id,shard_id FROM episodes WHERE dataset_id=? ORDER BY shard_id,ordinal",
                (dataset_id,),
            ).fetchall()
            eligible_by_shard = {row["id"]: [] for row in shard_rows}
            exclusions = []
            for episode in episode_rows:
                quality = db.execute(
                    "SELECT * FROM episode_quality_dispositions WHERE dataset_id=? AND episode_id=? ORDER BY created_at DESC,disposition_id DESC LIMIT 1",
                    (dataset_id, episode["id"]),
                ).fetchone()
                if quality is not None and not bool(quality["training_eligible"]):
                    exclusions.append(
                        {
                            "episode_id": episode["id"],
                            "reason": quality["reason"],
                            "validator": quality["validator"],
                            "validator_version": quality["validator_version"],
                            "disposition_id": quality["disposition_id"],
                        }
                    )
                else:
                    eligible_by_shard[episode["shard_id"]].append(episode["id"])
            shards = [
                {**dict(row), "episode_ids": eligible_by_shard[row["id"]]}
                for row in shard_rows
                if eligible_by_shard[row["id"]]
            ]
            if not shards:
                raise ValueError("dataset has no training-eligible episodes")
            manifest = {
                "schema_id": "nxml.dataset-snapshot.v1",
                "dataset_id": dataset_id,
                "control_source": control_source,
                "shards": shards,
                "excluded_episodes": exclusions,
            }
            canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            snapshot_id = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
            created_at = datetime.now(UTC).isoformat()
            db.execute(
                "INSERT OR IGNORE INTO snapshots VALUES(?,?,?,?,?)",
                (snapshot_id, dataset_id, created_at, control_source, canonical),
            )
            for ordinal, row in enumerate(shards):
                db.execute(
                    "INSERT OR IGNORE INTO snapshot_shards VALUES(?,?,?)",
                    (snapshot_id, row["id"], ordinal),
                )
            stored = db.execute(
                "SELECT * FROM snapshots WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()
            assert stored is not None
            return _snapshot(stored)

    def list_snapshots(
        self,
        *,
        dataset_id: str | None = None,
        control_source: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Snapshot]:
        clauses, params = [], []
        if dataset_id is not None:
            clauses.append("dataset_id=?")
            params.append(dataset_id)
        if control_source is not None:
            clauses.append("control_source=?")
            params.append(control_source)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connect() as db:
            rows = db.execute(
                f"SELECT * FROM snapshots{where} ORDER BY created_at,snapshot_id LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return [_snapshot(row) for row in rows]

    def get_snapshot(self, snapshot_id: str) -> Snapshot:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM snapshots WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()
        if row is None:
            raise KeyError(snapshot_id)
        return _snapshot(row)

    def list_datasets(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                """SELECT d.id,count(DISTINCT s.id) shard_count,count(e.id) episode_count FROM datasets d LEFT JOIN shards s ON s.dataset_id=d.id LEFT JOIN episodes e ON e.shard_id=s.id GROUP BY d.id ORDER BY d.id"""
            ).fetchall()
        return [dict(row) for row in rows]

    def list_shards(self, dataset_id: str) -> list[Shard]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM shards WHERE dataset_id=? ORDER BY id", (dataset_id,)
            ).fetchall()
        return [_shard(row) for row in rows]

    def list_episodes(self, dataset_id: str) -> list[Episode]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM episodes WHERE dataset_id=? ORDER BY shard_id,ordinal", (dataset_id,)
            ).fetchall()
        return [_episode(row) for row in rows]

    def health(self) -> dict[str, int | str]:
        with self.connect() as db:
            counts = dict(db.execute("SELECT state,count(*) FROM uploads GROUP BY state"))
        return {
            "status": "ok",
            **{state: int(counts.get(state, 0)) for state in ("created", "uploaded", "committed")},
        }


def _episode_entries(manifest: dict) -> list[tuple[str, dict]]:
    raw = manifest.get("episodes", [])
    if not isinstance(raw, list):
        raise ValueError("manifest episodes must be a list")
    result = []
    for entry in raw:
        if isinstance(entry, str):
            episode_id, payload = entry, {"episode_id": entry}
        elif isinstance(entry, dict) and isinstance(entry.get("episode_id"), str):
            episode_id, payload = entry["episode_id"], entry
        else:
            raise ValueError("each manifest episode needs a string episode_id")
        if not episode_id:
            raise ValueError("episode_id cannot be empty")
        result.append((episode_id, payload))
    if len({item[0] for item in result}) != len(result):
        raise ValueError("manifest episode IDs must be unique")
    return result


def _upload(row: sqlite3.Row) -> Upload:
    return Upload(**{name: row[name] for name in Upload.__dataclass_fields__})


def _shard(row: sqlite3.Row) -> Shard:
    return Shard(
        row["id"],
        row["dataset_id"],
        row["upload_id"],
        row["object_key"],
        row["size_bytes"],
        row["sha256"],
        json.loads(row["manifest_json"]),
    )


def _episode(row: sqlite3.Row) -> Episode:
    return Episode(
        row["id"],
        row["dataset_id"],
        row["shard_id"],
        row["ordinal"],
        json.loads(row["manifest_json"]),
    )


def _receipt(row: sqlite3.Row) -> CommitReceipt:
    return CommitReceipt(**{name: row[name] for name in CommitReceipt.__dataclass_fields__})


def _snapshot(row: sqlite3.Row) -> Snapshot:
    return Snapshot(
        row["snapshot_id"],
        row["dataset_id"],
        row["created_at"],
        row["control_source"],
        json.loads(row["manifest_json"]),
    )


def _quality(row: sqlite3.Row) -> dict:
    item = dict(row)
    item["training_eligible"] = bool(item["training_eligible"])
    return item
