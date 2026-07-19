"""Model revision registry and compare-and-swap deployment manager."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Protocol
from uuid import uuid4


class ConflictError(ValueError):
    pass


class CandidateError(ValueError):
    pass


@dataclass(frozen=True)
class PreparedCandidate:
    revision_id: str
    checkpoint_path: str
    info: dict[str, Any]
    handle: Any = None


class DeploymentRuntime(Protocol):
    def prepare(self, revision: dict) -> PreparedCandidate: ...
    def smoke(self, candidate: PreparedCandidate) -> None: ...
    def activate(self, candidate: PreparedCandidate) -> None: ...


class FakePolicyRuntime:
    def __init__(
        self, *, fail_prepare: set[str] | None = None, fail_activate: set[str] | None = None
    ):
        self.active_revision: str | None = None
        self.fail_prepare = fail_prepare or set()
        self.fail_activate = fail_activate or set()

    def prepare(self, revision):
        if revision["revision_id"] in self.fail_prepare:
            raise RuntimeError("candidate load failed")
        return PreparedCandidate(
            revision["revision_id"], revision["checkpoint_path"], revision["compatibility"]
        )

    def smoke(self, candidate):
        if candidate.info.get("smoke_fail"):
            raise RuntimeError("smoke inference failed")

    def activate(self, candidate):
        if candidate.revision_id in self.fail_activate:
            raise RuntimeError("runtime activation failed")
        self.active_revision = candidate.revision_id


class ModelRegistry:
    def __init__(
        self,
        path: str | Path,
        runtime: DeploymentRuntime,
        *,
        action_spec_id="switch_packets.v1",
        action_dim=26,
    ):
        self.path = Path(path)
        self.runtime = runtime
        self.action_spec_id = action_spec_id
        self.action_dim = action_dim
        self._lock = Lock()
        self._init()

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def _init(self):
        db = self._connect()
        try:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS model_revisions(
              revision_id TEXT PRIMARY KEY,model_id TEXT NOT NULL,checkpoint_path TEXT NOT NULL,checkpoint_sha256 TEXT NOT NULL,
              source_snapshot_id TEXT NOT NULL,source_config_json TEXT NOT NULL,source_commit_id TEXT NOT NULL,
              compatibility_json TEXT NOT NULL,evaluation_json TEXT NOT NULL,state TEXT NOT NULL CHECK(state IN('candidate','validated','active','rejected','retired')),created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS deployment_state(singleton INTEGER PRIMARY KEY CHECK(singleton=1),active_revision TEXT,previous_revision TEXT,generation INTEGER NOT NULL);
            INSERT OR IGNORE INTO deployment_state VALUES(1,NULL,NULL,0);
            CREATE TABLE IF NOT EXISTS activation_requests(idempotency_key TEXT PRIMARY KEY,operation TEXT NOT NULL,target_revision TEXT NOT NULL,expected_revision TEXT,expected_generation INTEGER,result_json TEXT NOT NULL);
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(activation_requests)")}
            if "expected_generation" not in columns:
                db.execute("ALTER TABLE activation_requests ADD COLUMN expected_generation INTEGER")
            db.commit()
        finally:
            db.close()

    def register(
        self,
        *,
        model_id,
        checkpoint_path,
        checkpoint_sha256,
        source_snapshot_id,
        source_config,
        source_commit_id,
        compatibility,
        evaluation,
    ) -> dict:
        revision_id = str(uuid4())
        now = datetime.now(UTC).isoformat()
        db = self._connect()
        try:
            db.execute(
                "INSERT INTO model_revisions VALUES(?,?,?,?,?,?,?,?,?,\x27candidate\x27,?)",
                (
                    revision_id,
                    model_id,
                    checkpoint_path,
                    checkpoint_sha256,
                    source_snapshot_id,
                    json.dumps(source_config, sort_keys=True),
                    source_commit_id,
                    json.dumps(compatibility, sort_keys=True),
                    json.dumps(evaluation, sort_keys=True),
                    now,
                ),
            )
            db.commit()
        finally:
            db.close()
        return self.get(revision_id)

    def get(self, revision_id) -> dict:
        db = self._connect()
        try:
            row = db.execute(
                "SELECT * FROM model_revisions WHERE revision_id=?", (revision_id,)
            ).fetchone()
        finally:
            db.close()
        if row is None:
            raise KeyError(revision_id)
        item = dict(row)
        for field in ("source_config", "compatibility", "evaluation"):
            item[field] = json.loads(item.pop(field + "_json"))
        return item

    def list_revisions(
        self,
        *,
        model_id: str | None = None,
        state: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        clauses, params = [], []
        if model_id is not None:
            clauses.append("model_id=?")
            params.append(model_id)
        if state is not None:
            clauses.append("state=?")
            params.append(state)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        db = self._connect()
        try:
            rows = db.execute(
                f"SELECT revision_id FROM model_revisions{where} ORDER BY created_at,revision_id LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        finally:
            db.close()
        return [self.get(row["revision_id"]) for row in rows]

    def deployment(self) -> dict:
        db = self._connect()
        try:
            row = db.execute(
                "SELECT active_revision,previous_revision,generation FROM deployment_state WHERE singleton=1"
            ).fetchone()
        finally:
            db.close()
        return dict(row)

    def validate(self, revision_id) -> dict:
        revision = self.get(revision_id)
        try:
            candidate = self.runtime.prepare(revision)
            if (
                candidate.info.get("action_spec_id") != self.action_spec_id
                or int(candidate.info.get("action_dim", -1)) != self.action_dim
            ):
                raise CandidateError("action-spec compatibility failed")
            self.runtime.smoke(candidate)
        except Exception as error:
            self._set_state(revision_id, "rejected")
            raise CandidateError(str(error)) from error
        self._set_state(revision_id, "validated")
        return self.get(revision_id)

    def promote(
        self, revision_id, *, expected_revision=None, expected_generation=None, idempotency_key
    ) -> dict:
        return self._activate(
            "promote", revision_id, expected_revision, expected_generation, idempotency_key
        )

    def rollback(
        self, *, expected_revision=None, expected_generation=None, idempotency_key
    ) -> dict:
        state = self.deployment()
        target = state["previous_revision"]
        if target is None:
            raise ConflictError("no previous-known-good revision")
        return self._activate(
            "rollback", target, expected_revision, expected_generation, idempotency_key
        )

    def _activate(self, operation, target, expected_revision, expected_generation, key) -> dict:
        db = self._connect()
        try:
            prior = db.execute(
                "SELECT * FROM activation_requests WHERE idempotency_key=?", (key,)
            ).fetchone()
            if prior:
                identity = (
                    prior["operation"],
                    prior["target_revision"],
                    prior["expected_revision"],
                    prior["expected_generation"],
                )
                if identity != (operation, target, expected_revision, expected_generation):
                    raise ConflictError("activation idempotency key reused")
                return json.loads(prior["result_json"])
        finally:
            db.close()
        revision = self.get(target)
        if revision["state"] not in {"validated", "active", "retired"}:
            raise ConflictError("target revision is not validated")
        try:
            candidate = self.runtime.prepare(revision)
            self.runtime.smoke(candidate)
            if (
                candidate.info.get("action_spec_id") != self.action_spec_id
                or int(candidate.info.get("action_dim", -1)) != self.action_dim
            ):
                raise CandidateError("action-spec compatibility failed")
        except Exception as error:
            raise CandidateError(str(error)) from error
        with self._lock:
            db = self._connect()
            try:
                db.execute("BEGIN IMMEDIATE")
                state = db.execute("SELECT * FROM deployment_state WHERE singleton=1").fetchone()
                active, generation = state["active_revision"], state["generation"]
                if expected_generation is not None and generation != expected_generation:
                    raise ConflictError(
                        f"deployment generation changed: expected {expected_generation}, found {generation}"
                    )
                if (
                    expected_generation is None or expected_revision is not None
                ) and active != expected_revision:
                    raise ConflictError(
                        f"active revision changed: expected {expected_revision}, found {active}"
                    )
                self.runtime.activate(candidate)
                if active and active != target:
                    db.execute(
                        "UPDATE model_revisions SET state='retired' WHERE revision_id=?", (active,)
                    )
                db.execute(
                    "UPDATE model_revisions SET state='active' WHERE revision_id=?", (target,)
                )
                previous = active if active != target else state["previous_revision"]
                new_generation = generation + 1
                db.execute(
                    "UPDATE deployment_state SET active_revision=?,previous_revision=?,generation=? WHERE singleton=1",
                    (target, previous, new_generation),
                )
                result = {
                    "operation": operation,
                    "active_revision": target,
                    "previous_revision": previous,
                    "generation": new_generation,
                }
                db.execute(
                    "INSERT INTO activation_requests(idempotency_key,operation,target_revision,expected_revision,expected_generation,result_json) VALUES(?,?,?,?,?,?)",
                    (
                        key,
                        operation,
                        target,
                        expected_revision,
                        expected_generation,
                        json.dumps(result, sort_keys=True),
                    ),
                )
                db.commit()
                return result
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()

    def _set_state(self, revision_id, state):
        db = self._connect()
        try:
            db.execute(
                "UPDATE model_revisions SET state=? WHERE revision_id=?", (state, revision_id)
            )
            db.commit()
        finally:
            db.close()


class PolicyServerRuntime:
    """Atomic holder around separately preloaded nxrl PolicyServer instances."""

    def __init__(self, *, initial_server=None, device="cuda", server_factory=None):
        self._server = initial_server
        self.device = device
        self._factory = server_factory
        self._runtime_lock = Lock()

    @property
    def server(self):
        with self._runtime_lock:
            return self._server

    def prepare(self, revision):
        if self._factory is None:
            from nxrl.serve.server import PolicyServer

            factory = PolicyServer
        else:
            factory = self._factory
        server = factory(model_path=revision["checkpoint_path"], device=self.device)
        raw = server.info()
        from dataclasses import asdict, is_dataclass

        info = asdict(raw) if is_dataclass(raw) else dict(raw)
        info.setdefault("action_spec_id", revision["compatibility"].get("action_spec_id"))
        return PreparedCandidate(revision["revision_id"], revision["checkpoint_path"], info, server)

    def smoke(self, candidate):
        import numpy as np

        info = candidate.info
        shape = (int(info["sequence_length"]), *(int(v) for v in info["latent_shape"]))
        action = candidate.handle.predict(np.zeros(shape, dtype=np.float32))
        if tuple(action.shape) != (int(info.get("action_dim", -1)),):
            raise RuntimeError("smoke inference returned incompatible action shape")
        if not np.isfinite(action).all():
            raise RuntimeError("smoke inference returned non-finite action")

    def activate(self, candidate):
        with self._runtime_lock:
            self._server = candidate.handle

    def predict(self, latents):
        server = self.server
        if server is None:
            raise RuntimeError("no active policy revision")
        return server.predict(latents)

    def info(self):
        server = self.server
        if server is None:
            raise RuntimeError("no active policy revision")
        return server.info()
