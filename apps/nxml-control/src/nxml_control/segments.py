"""Inactive v1 contracts for receipt-gated continuous recording segments."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nxml_control.catalog import Catalog
from nxml_control.service import IngestService
from nxml_control.storage import ObjectStorage

SHA256_PATTERN = r"^[0-9a-f]{64}$"
ID_PATTERN = r"^sha256:[0-9a-f]{64}$"


class SegmentMemberV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    role: Literal["video", "actions", "events"]
    path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_path(self):
        parts = self.path.split("/")
        if self.path.startswith("/") or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("segment member path must be a safe relative path")
        return self


class SegmentBundleV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["nxml.segment-bundle.v1"]
    dataset_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    segment_id: str = Field(pattern=ID_PATTERN)
    sequence_index: int = Field(ge=0)
    clock_id: str = Field(min_length=1)
    timeline_start_ns: int = Field(ge=0)
    timeline_end_ns: int = Field(gt=0)
    object_size_bytes: int = Field(gt=0)
    object_sha256: str = Field(pattern=SHA256_PATTERN)
    members: list[SegmentMemberV1] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_identity(self):
        if self.timeline_end_ns <= self.timeline_start_ns:
            raise ValueError("segment timeline must be non-empty and half-open")
        if self.segment_id != "sha256:" + self.object_sha256:
            raise ValueError("segment_id must be the object SHA-256 identity")
        roles = [member.role for member in self.members]
        if sorted(roles) != ["actions", "events", "video"]:
            raise ValueError("segment requires exactly one video/actions/events triplet")
        paths = [member.path for member in self.members]
        if len(paths) != len(set(paths)):
            raise ValueError("segment member paths must be unique")
        by_role = {member.role: member.path for member in self.members}
        if not by_role["video"].endswith((".mkv", ".mp4")):
            raise ValueError("video member must be MKV or MP4")
        if not by_role["actions"].endswith(".parquet"):
            raise ValueError("actions member must be parquet")
        if not by_role["events"].endswith(".events.parquet"):
            raise ValueError("events member must be events parquet")
        return self


class SegmentReferenceV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    segment_id: str = Field(pattern=ID_PATTERN)
    sequence_index: int = Field(ge=0)
    timeline_start_ns: int = Field(ge=0)
    timeline_end_ns: int = Field(gt=0)
    object_sha256: str = Field(pattern=SHA256_PATTERN)


class EpisodeCloseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["nxml.episode-close.v1"]
    dataset_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    clock_id: str = Field(min_length=1)
    timeline_start_ns: int = Field(ge=0)
    timeline_end_ns: int = Field(gt=0)
    segments: list[SegmentReferenceV1] = Field(min_length=1)


@dataclass(frozen=True)
class SegmentReceipt:
    receipt_id: str
    segment_id: str
    upload_id: str
    dataset_id: str
    episode_id: str
    sequence_index: int
    storage_key: str
    size_bytes: int
    sha256: str
    timeline_start_ns: int
    timeline_end_ns: int
    state: str
    committed_at: str


class SegmentCatalog:
    """Standalone/inactive catalog sharing the existing upload/object durability boundary."""

    def __init__(self, catalog: Catalog, ingest: IngestService, storage: ObjectStorage):
        self.catalog = catalog
        self.ingest = ingest
        self.storage = storage
        self._init()

    def _init(self):
        with self.catalog.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS segment_bundles(
              segment_id TEXT PRIMARY KEY,upload_id TEXT NOT NULL UNIQUE,dataset_id TEXT NOT NULL,
              episode_id TEXT NOT NULL,sequence_index INTEGER NOT NULL,clock_id TEXT NOT NULL,
              timeline_start_ns INTEGER NOT NULL,timeline_end_ns INTEGER NOT NULL,
              storage_key TEXT NOT NULL UNIQUE,size_bytes INTEGER NOT NULL,sha256 TEXT NOT NULL,
              manifest_json TEXT NOT NULL,committed_at TEXT NOT NULL,
              UNIQUE(dataset_id,episode_id,sequence_index));
            CREATE TABLE IF NOT EXISTS segment_receipts(
              receipt_id TEXT PRIMARY KEY,segment_id TEXT NOT NULL UNIQUE REFERENCES segment_bundles(segment_id),
              upload_id TEXT NOT NULL UNIQUE,committed_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS segment_quality_dispositions(
              disposition_id TEXT PRIMARY KEY,idempotency_key TEXT NOT NULL UNIQUE,segment_id TEXT NOT NULL,
              training_eligible INTEGER NOT NULL,reason TEXT NOT NULL,validator TEXT NOT NULL,
              validator_version TEXT NOT NULL,created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS episode_closes(
              close_id TEXT PRIMARY KEY,idempotency_key TEXT NOT NULL UNIQUE,dataset_id TEXT NOT NULL,
              episode_id TEXT NOT NULL UNIQUE,manifest_json TEXT NOT NULL,created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS episode_close_segments(
              close_id TEXT NOT NULL REFERENCES episode_closes(close_id),segment_id TEXT NOT NULL,
              ordinal INTEGER NOT NULL,PRIMARY KEY(close_id,segment_id),UNIQUE(close_id,ordinal));
            CREATE TABLE IF NOT EXISTS segment_snapshots(
              snapshot_id TEXT PRIMARY KEY,dataset_id TEXT NOT NULL,manifest_json TEXT NOT NULL,
              created_at TEXT NOT NULL);
            """)

    def commit_segment(self, upload_id: str, manifest: dict) -> SegmentReceipt:
        parsed = SegmentBundleV1.model_validate(manifest)
        upload = self.catalog.get(upload_id)
        canonical = json.dumps(
            parsed.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        if upload.state == "committed":
            with self.catalog.connect() as db:
                row = db.execute(
                    "SELECT r.receipt_id FROM segment_receipts r WHERE r.upload_id=?", (upload_id,)
                ).fetchone()
                if row is None:
                    raise ValueError("upload was committed by a different ingest contract")
                stored = db.execute(
                    "SELECT manifest_json FROM segment_bundles WHERE upload_id=?", (upload_id,)
                ).fetchone()
                if stored is None or stored[0] != canonical:
                    raise ValueError("immutable segment commit conflicts with original manifest")
            return self.get_receipt(row[0])
        if upload.state != "uploaded":
            raise ValueError("segment upload must be checksum-verified before commit")
        if (upload.actual_size_bytes, upload.actual_sha256) != (
            parsed.object_size_bytes,
            parsed.object_sha256,
        ):
            raise ValueError("segment object identity does not match verified upload")
        self.ingest.verify_members(upload_id, parsed.model_dump(mode="json"))
        now = datetime.now(UTC).isoformat()
        receipt_id = str(uuid4())
        with self.catalog.connect() as db:
            try:
                db.execute(
                    "INSERT INTO segment_bundles VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        parsed.segment_id,
                        upload_id,
                        parsed.dataset_id,
                        parsed.episode_id,
                        parsed.sequence_index,
                        parsed.clock_id,
                        parsed.timeline_start_ns,
                        parsed.timeline_end_ns,
                        upload.object_key,
                        parsed.object_size_bytes,
                        parsed.object_sha256,
                        canonical,
                        now,
                    ),
                )
                db.execute(
                    "INSERT INTO segment_receipts VALUES(?,?,?,?)",
                    (receipt_id, parsed.segment_id, upload_id, now),
                )
                db.execute(
                    "UPDATE uploads SET state='committed',dataset_id=?,shard_id=?,manifest_json=? WHERE id=? AND state='uploaded'",
                    (parsed.dataset_id, parsed.segment_id, canonical, upload_id),
                )
            except sqlite3.IntegrityError:
                row = db.execute(
                    "SELECT r.receipt_id,b.manifest_json FROM segment_receipts r "
                    "JOIN segment_bundles b ON b.segment_id=r.segment_id WHERE r.upload_id=?",
                    (upload_id,),
                ).fetchone()
                if row is None or row["manifest_json"] != canonical:
                    raise ValueError("segment identity, order, or upload already committed") from None
                return self.get_receipt(row["receipt_id"])
        return self.get_receipt(receipt_id)

    def get_receipt(self, receipt_id: str) -> SegmentReceipt:
        with self.catalog.connect() as db:
            row = db.execute(
                """SELECT r.receipt_id,b.segment_id,b.upload_id,b.dataset_id,b.episode_id,
                b.sequence_index,b.storage_key,b.size_bytes,b.sha256,b.timeline_start_ns,
                b.timeline_end_ns,r.committed_at FROM segment_receipts r
                JOIN segment_bundles b ON b.segment_id=r.segment_id WHERE r.receipt_id=?""",
                (receipt_id,),
            ).fetchone()
        if row is None:
            raise KeyError(receipt_id)
        return SegmentReceipt(**dict(row), state="committed")

    def set_quality(
        self,
        segment_id: str,
        *,
        idempotency_key: str,
        training_eligible: bool,
        reason: str,
        validator: str,
        validator_version: str,
    ) -> dict:
        payload = (
            segment_id,
            int(training_eligible),
            reason,
            validator,
            validator_version,
        )
        with self.catalog.connect() as db:
            if (
                db.execute(
                    "SELECT 1 FROM segment_bundles WHERE segment_id=?", (segment_id,)
                ).fetchone()
                is None
            ):
                raise KeyError(segment_id)
            try:
                disposition_id = str(uuid4())
                now = datetime.now(UTC).isoformat()
                db.execute(
                    "INSERT INTO segment_quality_dispositions VALUES(?,?,?,?,?,?,?,?)",
                    (disposition_id, idempotency_key, *payload, now),
                )
            except sqlite3.IntegrityError:
                row = db.execute(
                    "SELECT * FROM segment_quality_dispositions WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
                if (
                    row is None
                    or tuple(
                        row[key]
                        for key in (
                            "segment_id",
                            "training_eligible",
                            "reason",
                            "validator",
                            "validator_version",
                        )
                    )
                    != payload
                ):
                    raise ValueError("segment quality idempotency conflict") from None
                return dict(row)
        return self.quality(segment_id)[-1]

    def quality(self, segment_id: str) -> list[dict]:
        with self.catalog.connect() as db:
            rows = db.execute(
                "SELECT * FROM segment_quality_dispositions WHERE segment_id=? ORDER BY created_at,disposition_id",
                (segment_id,),
            ).fetchall()
        result = [dict(row) for row in rows]
        for item in result:
            item["training_eligible"] = bool(item["training_eligible"])
        return result

    def close_episode(self, close: dict, *, idempotency_key: str) -> dict:
        parsed = EpisodeCloseV1.model_validate(close)
        refs = sorted(parsed.segments, key=lambda item: item.sequence_index)
        if [item.sequence_index for item in refs] != list(range(len(refs))):
            raise ValueError("episode segment sequence must be contiguous from zero")
        if (
            refs[0].timeline_start_ns != parsed.timeline_start_ns
            or refs[-1].timeline_end_ns != parsed.timeline_end_ns
        ):
            raise ValueError("episode timeline bounds do not match segment bounds")
        for left, right in pairwise(refs):
            if left.timeline_end_ns != right.timeline_start_ns:
                relation = "overlap" if left.timeline_end_ns > right.timeline_start_ns else "gap"
                raise ValueError(f"episode segment timeline {relation}")
        normalized = parsed.model_dump(mode="json")
        normalized["segments"] = [item.model_dump(mode="json") for item in refs]
        canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        close_id = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
        with self.catalog.connect() as db:
            for ref in refs:
                row = db.execute(
                    "SELECT * FROM segment_bundles WHERE segment_id=?", (ref.segment_id,)
                ).fetchone()
                if row is None:
                    raise ValueError(f"episode close references missing segment {ref.segment_id}")
                expected = (
                    parsed.dataset_id,
                    parsed.episode_id,
                    ref.sequence_index,
                    parsed.clock_id,
                    ref.timeline_start_ns,
                    ref.timeline_end_ns,
                    ref.object_sha256,
                )
                actual = tuple(
                    row[key]
                    for key in (
                        "dataset_id",
                        "episode_id",
                        "sequence_index",
                        "clock_id",
                        "timeline_start_ns",
                        "timeline_end_ns",
                        "sha256",
                    )
                )
                if actual != expected:
                    raise ValueError(f"episode close segment metadata mismatch: {ref.segment_id}")
            try:
                now = datetime.now(UTC).isoformat()
                db.execute(
                    "INSERT INTO episode_closes VALUES(?,?,?,?,?,?)",
                    (
                        close_id,
                        idempotency_key,
                        parsed.dataset_id,
                        parsed.episode_id,
                        canonical,
                        now,
                    ),
                )
                db.executemany(
                    "INSERT INTO episode_close_segments VALUES(?,?,?)",
                    [(close_id, ref.segment_id, index) for index, ref in enumerate(refs)],
                )
            except sqlite3.IntegrityError:
                row = db.execute(
                    "SELECT * FROM episode_closes WHERE idempotency_key=?", (idempotency_key,)
                ).fetchone()
                if row is None or row["manifest_json"] != canonical:
                    raise ValueError("episode close idempotency conflict") from None
                return self.get_close(row["close_id"])
        return self.get_close(close_id)

    def get_close(self, close_id: str) -> dict:
        with self.catalog.connect() as db:
            row = db.execute(
                "SELECT * FROM episode_closes WHERE close_id=?", (close_id,)
            ).fetchone()
        if row is None:
            raise KeyError(close_id)
        item = dict(row)
        item["manifest"] = json.loads(item.pop("manifest_json"))
        return item

    def _snapshot_body(self, dataset_id: str) -> dict:
        included = []
        excluded = []
        with self.catalog.connect() as db:
            closes = db.execute(
                "SELECT * FROM episode_closes WHERE dataset_id=? ORDER BY episode_id", (dataset_id,)
            ).fetchall()
            for close in closes:
                segments = db.execute(
                    """SELECT b.*,q.training_eligible,q.reason,q.validator,q.validator_version,q.disposition_id
                    FROM episode_close_segments x JOIN segment_bundles b ON b.segment_id=x.segment_id
                    LEFT JOIN segment_quality_dispositions q ON q.disposition_id=(
                      SELECT disposition_id FROM segment_quality_dispositions
                      WHERE segment_id=b.segment_id ORDER BY created_at DESC,disposition_id DESC LIMIT 1)
                    WHERE x.close_id=? ORDER BY x.ordinal""",
                    (close["close_id"],),
                ).fetchall()
                failures = [row for row in segments if row["training_eligible"] == 0]
                if failures:
                    excluded.append(
                        {
                            "episode_id": close["episode_id"],
                            "segments": [
                                {
                                    "segment_id": row["segment_id"],
                                    "reason": row["reason"],
                                    "validator": row["validator"],
                                    "validator_version": row["validator_version"],
                                    "disposition_id": row["disposition_id"],
                                }
                                for row in failures
                            ],
                        }
                    )
                else:
                    included.append(
                        {
                            "episode_id": close["episode_id"],
                            "close_id": close["close_id"],
                            "segment_ids": [row["segment_id"] for row in segments],
                        }
                    )
        body = {
            "schema_id": "nxml.segment-snapshot.v1",
            "dataset_id": dataset_id,
            "episodes": included,
            "excluded_episodes": excluded,
        }
        return body

    def create_snapshot(self, dataset_id: str) -> dict:
        body = self._snapshot_body(dataset_id)
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        snapshot_id = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
        created_at = datetime.now(UTC).isoformat()
        with self.catalog.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO segment_snapshots VALUES(?,?,?,?)",
                (snapshot_id, dataset_id, canonical, created_at),
            )
        return self.get_snapshot(snapshot_id)

    def get_snapshot(self, snapshot_id: str) -> dict:
        with self.catalog.connect() as db:
            row = db.execute(
                "SELECT * FROM segment_snapshots WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()
        if row is None:
            raise KeyError(snapshot_id)
        return {
            **json.loads(row["manifest_json"]),
            "snapshot_id": row["snapshot_id"],
            "created_at": row["created_at"],
        }

    def iter_reconstruction(self, close_id: str) -> Iterator[tuple[dict, bytes]]:
        with self.catalog.connect() as db:
            rows = db.execute(
                """SELECT b.* FROM episode_close_segments x JOIN segment_bundles b
                ON b.segment_id=x.segment_id WHERE x.close_id=? ORDER BY x.ordinal""",
                (close_id,),
            ).fetchall()
        if not rows:
            raise KeyError(close_id)
        for row in rows:
            info = self.storage.inspect(row["storage_key"])
            if info is None or (info.size_bytes, info.sha256) != (row["size_bytes"], row["sha256"]):
                raise ValueError(f"durable segment object mismatch: {row['segment_id']}")
            with self.storage.open(row["storage_key"]) as source:
                yield json.loads(row["manifest_json"]), source.read()

    def status(self, dataset_id: str) -> dict:
        with self.catalog.connect() as db:
            segment = db.execute(
                "SELECT count(*) count,coalesce(sum(size_bytes),0) bytes,max(committed_at) latest FROM segment_bundles WHERE dataset_id=?",
                (dataset_id,),
            ).fetchone()
            closes = db.execute(
                "SELECT count(*) FROM episode_closes WHERE dataset_id=?", (dataset_id,)
            ).fetchone()[0]
        snapshot = self._snapshot_body(dataset_id)
        return {
            "schema_id": "nxml.segment-status.v1",
            "dataset_id": dataset_id,
            "committed_segments": segment["count"],
            "committed_bytes": segment["bytes"],
            "closed_episodes": closes,
            "eligible_episodes": len(snapshot["episodes"]),
            "excluded_episodes": len(snapshot["excluded_episodes"]),
            "latest_segment_committed_at": segment["latest"],
        }
