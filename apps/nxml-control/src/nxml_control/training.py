"""Persisted BC job control with pluggable execution."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4


@dataclass(frozen=True)
class TrainingSpec:
    snapshot_id: str
    config: dict
    idempotency_key: str


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: str
    checkpoint_sha256: str
    metrics: dict[str, float]
    logs: list[str]


class TrainingExecutor(Protocol):
    """Implement with a queue/Slurm/Kubernetes adapter for real GPU execution."""

    def execute(self, job_id: str, spec: TrainingSpec) -> TrainingResult: ...


class FakeTrainingExecutor:
    def execute(self, job_id: str, spec: TrainingSpec) -> TrainingResult:
        import hashlib

        payload = json.dumps(
            {"job_id": job_id, "snapshot_id": spec.snapshot_id, "config": spec.config},
            sort_keys=True,
        ).encode()
        return TrainingResult(
            f"fake://{job_id}.pt",
            hashlib.sha256(payload).hexdigest(),
            {"loss": 0.125, "examples": 4.0},
            ["executor=fake", "bc smoke complete"],
        )


class TrainingJobs:
    def __init__(self, db_path: str | Path, executor: TrainingExecutor) -> None:
        self.db_path = Path(db_path)
        self.executor = executor
        self._init()

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _init(self):
        with self._db() as db:
            db.executescript("""
        CREATE TABLE IF NOT EXISTS training_jobs(
          job_id TEXT PRIMARY KEY,idempotency_key TEXT NOT NULL UNIQUE,snapshot_id TEXT NOT NULL,
          config_json TEXT NOT NULL,state TEXT NOT NULL CHECK(state IN('queued','running','succeeded','failed')),
          created_at TEXT NOT NULL,updated_at TEXT NOT NULL,checkpoint_path TEXT,checkpoint_sha256 TEXT,error TEXT);
        CREATE TABLE IF NOT EXISTS training_logs(job_id TEXT NOT NULL,line_no INTEGER NOT NULL,message TEXT NOT NULL,PRIMARY KEY(job_id,line_no));
        CREATE TABLE IF NOT EXISTS training_metrics(job_id TEXT NOT NULL,name TEXT NOT NULL,value REAL NOT NULL,PRIMARY KEY(job_id,name));
        """)

    def submit(self, spec: TrainingSpec) -> dict:
        canonical = json.dumps(spec.config, sort_keys=True, separators=(",", ":"))
        now = datetime.now(UTC).isoformat()
        job_id = str(uuid4())
        with self._db() as db:
            try:
                db.execute(
                    "INSERT INTO training_jobs VALUES(?,?,?,?,\x27queued\x27,?,?,NULL,NULL,NULL)",
                    (job_id, spec.idempotency_key, spec.snapshot_id, canonical, now, now),
                )
            except sqlite3.IntegrityError:
                row = db.execute(
                    "SELECT * FROM training_jobs WHERE idempotency_key=?", (spec.idempotency_key,)
                ).fetchone()
                assert row is not None
                if (row["snapshot_id"], row["config_json"]) != (spec.snapshot_id, canonical):
                    raise ValueError(
                        "idempotency key reused with a different training request"
                    ) from None
                return self.get(row["job_id"])
        self.run(job_id)
        return self.get(job_id)

    def run(self, job_id: str) -> None:
        job = self.get(job_id)
        spec = TrainingSpec(job["snapshot_id"], job["config"], job["idempotency_key"])
        now = datetime.now(UTC).isoformat()
        with self._db() as db:
            db.execute(
                "UPDATE training_jobs SET state='running',updated_at=? WHERE job_id=? AND state='queued'",
                (now, job_id),
            )
        try:
            result = self.executor.execute(job_id, spec)
            now = datetime.now(UTC).isoformat()
            with self._db() as db:
                db.execute(
                    "UPDATE training_jobs SET state='succeeded',updated_at=?,checkpoint_path=?,checkpoint_sha256=? WHERE job_id=?",
                    (now, result.checkpoint_path, result.checkpoint_sha256, job_id),
                )
                db.executemany(
                    "INSERT INTO training_logs VALUES(?,?,?)",
                    [(job_id, i, line) for i, line in enumerate(result.logs)],
                )
                db.executemany(
                    "INSERT INTO training_metrics VALUES(?,?,?)",
                    [(job_id, name, value) for name, value in sorted(result.metrics.items())],
                )
        except Exception as error:
            with self._db() as db:
                db.execute(
                    "UPDATE training_jobs SET state='failed',updated_at=?,error=? WHERE job_id=?",
                    (datetime.now(UTC).isoformat(), str(error), job_id),
                )

    def get(self, job_id: str) -> dict:
        with self._db() as db:
            row = db.execute("SELECT * FROM training_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        item = dict(row)
        item["config"] = json.loads(item.pop("config_json"))
        return item

    def logs(self, job_id: str) -> list[str]:
        self.get(job_id)
        with self._db() as db:
            return [
                row[0]
                for row in db.execute(
                    "SELECT message FROM training_logs WHERE job_id=? ORDER BY line_no", (job_id,)
                )
            ]

    def metrics(self, job_id: str) -> dict[str, float]:
        self.get(job_id)
        with self._db() as db:
            return dict(
                db.execute(
                    "SELECT name,value FROM training_metrics WHERE job_id=? ORDER BY name",
                    (job_id,),
                )
            )
