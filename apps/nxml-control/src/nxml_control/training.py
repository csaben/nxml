"""Persisted asynchronous BC job control with pluggable execution."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shlex
import sqlite3
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
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
    available: bool

    def execute(self, job_id: str, spec: TrainingSpec) -> TrainingResult: ...
    def cancel(self, job_id: str) -> bool: ...


class FakeTrainingExecutor:
    """Deterministic test-only executor; production CLI never selects it."""

    available = True

    def execute(self, job_id, spec):
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

    def cancel(self, job_id):
        return False


class TrainingExecutionError(RuntimeError):
    def __init__(self, message, logs):
        super().__init__(message)
        self.logs = logs


class DisabledTrainingExecutor:
    """Production-safe executor state: job submission is unavailable, never faked."""

    available = False

    def execute(self, job_id, spec):
        raise RuntimeError("training executor is not configured")

    def cancel(self, job_id):
        return False


class SubprocessTrainingExecutor:
    """Real worker protocol: COMMAND --request PATH --result PATH.

    Request JSON (0440) is immutable ``nxml.bc-job.v1``. The worker atomically
    writes result JSON containing checkpoint_path, checkpoint_sha256, metrics,
    and optional logs. Checkpoint existence and SHA-256 are verified here.
    """

    available = True

    def __init__(self, command: str | list[str], work_root: str | Path):
        self.command = shlex.split(command) if isinstance(command, str) else list(command)
        if not self.command:
            raise ValueError("BC worker command cannot be empty")
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self._processes = {}
        self._lock = threading.Lock()

    def execute(self, job_id, spec):
        job_dir = self.work_root / job_id
        job_dir.mkdir(mode=0o750)
        request_path = job_dir / "request.json"
        result_path = job_dir / "result.json"
        request_path.write_text(
            json.dumps(
                {
                    "schema_id": "nxml.bc-job.v1",
                    "created_at": datetime.now(UTC).isoformat(),
                    "job_id": job_id,
                    "snapshot_id": spec.snapshot_id,
                    "config": spec.config,
                },
                sort_keys=True,
            )
        )
        request_path.chmod(0o440)
        command = [*self.command, "--request", str(request_path), "--result", str(result_path)]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=job_dir,
            env={**os.environ, "NXML_JOB_ID": job_id, "NXML_SNAPSHOT_ID": spec.snapshot_id},
        )
        with self._lock:
            self._processes[job_id] = process
        try:
            stdout, _ = process.communicate()
        finally:
            with self._lock:
                self._processes.pop(job_id, None)
        logs = stdout.splitlines()
        if process.returncode != 0:
            raise TrainingExecutionError(
                f"BC worker exited {process.returncode}: " + " | ".join(logs[-10:]), logs
            )
        if not result_path.is_file():
            raise TrainingExecutionError("BC worker did not write result.json", logs)
        result = json.loads(result_path.read_text())
        if result.get("schema_id") != "nxml.bc-result.v1":
            raise TrainingExecutionError("BC worker result schema must be nxml.bc-result.v1", logs)
        if result.get("job_id") != job_id or result.get("snapshot_id") != spec.snapshot_id:
            raise TrainingExecutionError("BC worker result lineage does not match request", logs)
        result_path.chmod(0o440)
        checkpoint = Path(result["checkpoint_path"])
        if not checkpoint.is_file():
            raise TrainingExecutionError(f"BC worker checkpoint missing: {checkpoint}", logs)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if not hmac.compare_digest(digest.encode(), str(result["checkpoint_sha256"]).encode()):
            raise TrainingExecutionError("BC worker checkpoint digest mismatch", logs)
        metrics = {str(key): float(value) for key, value in result.get("metrics", {}).items()}
        return TrainingResult(str(checkpoint), digest, metrics, [*logs, *result.get("logs", [])])

    def cancel(self, job_id):
        with self._lock:
            process = self._processes.get(job_id)
            if process is None or process.poll() is not None:
                return False
            process.terminate()
            return True


class TrainingJobs:
    def __init__(self, db_path: str | Path, executor: TrainingExecutor, *, run_async: bool = False):
        self.db_path = Path(db_path)
        self.executor = executor
        self.available = getattr(executor, "available", True)
        self.run_async = run_async
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nxml-bc")
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
              config_json TEXT NOT NULL,state TEXT NOT NULL CHECK(state IN('queued','running','succeeded','failed','cancelling','cancelled')),
              created_at TEXT NOT NULL,updated_at TEXT NOT NULL,checkpoint_path TEXT,checkpoint_sha256 TEXT,error TEXT);
            CREATE TABLE IF NOT EXISTS training_logs(job_id TEXT NOT NULL,line_no INTEGER NOT NULL,message TEXT NOT NULL,PRIMARY KEY(job_id,line_no));
            CREATE TABLE IF NOT EXISTS training_metrics(job_id TEXT NOT NULL,name TEXT NOT NULL,value REAL NOT NULL,PRIMARY KEY(job_id,name));
            """)
            schema = db.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='training_jobs'"
            ).fetchone()[0]
            if "cancelling" not in schema:
                db.executescript("""
                ALTER TABLE training_jobs RENAME TO training_jobs_old;
                CREATE TABLE training_jobs(
                  job_id TEXT PRIMARY KEY,idempotency_key TEXT NOT NULL UNIQUE,snapshot_id TEXT NOT NULL,
                  config_json TEXT NOT NULL,state TEXT NOT NULL CHECK(state IN('queued','running','succeeded','failed','cancelling','cancelled')),
                  created_at TEXT NOT NULL,updated_at TEXT NOT NULL,checkpoint_path TEXT,checkpoint_sha256 TEXT,error TEXT);
                INSERT INTO training_jobs SELECT * FROM training_jobs_old;
                DROP TABLE training_jobs_old;
                """)

    def submit(self, spec):
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
        if self.run_async:
            self._pool.submit(self.run, job_id)
        else:
            self.run(job_id)
        return self.get(job_id)

    def run(self, job_id):
        job = self.get(job_id)
        spec = TrainingSpec(job["snapshot_id"], job["config"], job["idempotency_key"])
        now = datetime.now(UTC).isoformat()
        with self._db() as db:
            changed = db.execute(
                "UPDATE training_jobs SET state='running',updated_at=? WHERE job_id=? AND state='queued'",
                (now, job_id),
            ).rowcount
        if changed != 1:
            return
        try:
            result = self.executor.execute(job_id, spec)
            now = datetime.now(UTC).isoformat()
            with self._db() as db:
                changed = db.execute(
                    "UPDATE training_jobs SET state='succeeded',updated_at=?,checkpoint_path=?,checkpoint_sha256=? WHERE job_id=? AND state='running'",
                    (now, result.checkpoint_path, result.checkpoint_sha256, job_id),
                ).rowcount
                if changed == 1:
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
                failure_logs = getattr(error, "logs", [])
                db.executemany(
                    "INSERT OR REPLACE INTO training_logs VALUES(?,?,?)",
                    [(job_id, i, line) for i, line in enumerate(failure_logs)],
                )
                db.execute(
                    "UPDATE training_jobs SET state='failed',updated_at=?,error=? WHERE job_id=? AND state='running'",
                    (datetime.now(UTC).isoformat(), str(error), job_id),
                )

    def get(self, job_id):
        with self._db() as db:
            row = db.execute("SELECT * FROM training_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        item = dict(row)
        item["config"] = json.loads(item.pop("config_json"))
        return item

    def list(self, *, state=None, snapshot_id=None, limit=100, offset=0):
        clauses = []
        params = []
        if state is not None:
            clauses.append("state=?")
            params.append(state)
        if snapshot_id is not None:
            clauses.append("snapshot_id=?")
            params.append(snapshot_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._db() as db:
            rows = db.execute(
                f"SELECT job_id FROM training_jobs{where} ORDER BY created_at,job_id LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return [self.get(row["job_id"]) for row in rows]

    def logs(self, job_id):
        self.get(job_id)
        with self._db() as db:
            return [
                row[0]
                for row in db.execute(
                    "SELECT message FROM training_logs WHERE job_id=? ORDER BY line_no", (job_id,)
                )
            ]

    def metrics(self, job_id):
        self.get(job_id)
        with self._db() as db:
            return dict(
                db.execute(
                    "SELECT name,value FROM training_metrics WHERE job_id=? ORDER BY name",
                    (job_id,),
                )
            )

    def artifacts(self, job_id):
        job = self.get(job_id)
        if not job["checkpoint_path"]:
            return []
        return [
            {
                "artifact_type": "policy_checkpoint",
                "uri": job["checkpoint_path"],
                "sha256": job["checkpoint_sha256"],
                "status": "complete" if job["state"] == "succeeded" else job["state"],
            }
        ]

    def cancel(self, job_id):
        job = self.get(job_id)
        if job["state"] not in {"queued", "running"}:
            raise ValueError(f"cannot cancel job in state {job['state']}")
        if job["state"] == "queued":
            with self._db() as db:
                changed = db.execute(
                    "UPDATE training_jobs SET state='cancelled',updated_at=?,error='cancelled by request' WHERE job_id=? AND state='queued'",
                    (datetime.now(UTC).isoformat(), job_id),
                ).rowcount
        else:
            with self._db() as db:
                changed = db.execute(
                    "UPDATE training_jobs SET state='cancelling',updated_at=? WHERE job_id=? AND state='running'",
                    (datetime.now(UTC).isoformat(), job_id),
                ).rowcount
            if changed == 1 and not self.executor.cancel(job_id):
                with self._db() as db:
                    db.execute(
                        "UPDATE training_jobs SET state='running',updated_at=? WHERE job_id=? AND state='cancelling'",
                        (datetime.now(UTC).isoformat(), job_id),
                    )
                raise ValueError("executor could not cancel running job")
            if changed == 1:
                with self._db() as db:
                    changed = db.execute(
                        "UPDATE training_jobs SET state='cancelled',updated_at=?,error='cancelled by request' WHERE job_id=? AND state='cancelling'",
                        (datetime.now(UTC).isoformat(), job_id),
                    ).rowcount
        if changed != 1:
            raise ValueError("training job state changed during cancellation")
        return self.get(job_id)
