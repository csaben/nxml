"""Registry-bound, immutable-revision policy inference over ZMQ."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sqlite3
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import zmq

ACTION_DIM: Final = 26
ACTION_BYTES: Final = ACTION_DIM * 4
MODE_PREDICT: Final = 0x00
MODE_PREDICT_FRAME: Final = 0x01
MODE_RELOAD_PATH: Final = 0x03
MODE_INFO: Final = 0x04
MODE_FRAME_V2: Final = 0x11
MODE_RESET_V2: Final = 0x12
MODE_RELOAD_REVISION_V2: Final = 0x13
MODE_HEALTH_V2: Final = 0x14
MODE_ROLLBACK_V2: Final = 0x15
MAX_MESSAGE_BYTES: Final = 2 * 1024 * 1024


@dataclass
class LoadedRevision:
    revision_id: str
    digest: str
    server: Any
    compatibility: dict[str, Any]


class RevisionLoader:
    def __init__(
        self,
        catalog_path: str | Path,
        checkpoint_root: str | Path,
        *,
        device: str = "cuda:0",
        vae_path: str = "stabilityai/sd-vae-ft-mse",
        server_factory: Callable[..., Any] | None = None,
    ):
        self.catalog_path = Path(catalog_path)
        self.checkpoint_root = Path(checkpoint_root).resolve()
        self.device = device
        self.vae_path = vae_path
        self.server_factory = server_factory

    def load(self, revision_id: str) -> LoadedRevision:
        with sqlite3.connect(f"file:{self.catalog_path}?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT * FROM model_revisions WHERE revision_id=?", (revision_id,)
            ).fetchone()
        if row is None:
            raise KeyError("model revision not found")
        if row["state"] not in {"validated", "active", "retired"}:
            raise ValueError("model revision is not validated")
        compatibility = json.loads(row["compatibility_json"])
        path = Path(row["checkpoint_path"])
        resolved = path.resolve(strict=True)
        if path != resolved or not resolved.is_file():
            raise ValueError("checkpoint path must be canonical and regular")
        try:
            resolved.relative_to(self.checkpoint_root)
        except ValueError as error:
            raise ValueError("checkpoint is outside managed root") from error
        digest = hashlib.sha256()
        with resolved.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        actual = digest.hexdigest()
        if not hmac.compare_digest(actual, row["checkpoint_sha256"]):
            raise ValueError("checkpoint digest mismatch")
        if self.server_factory is None:
            from nxrl.serve.server import PolicyServer

            factory = PolicyServer
        else:
            factory = self.server_factory
        server = factory(
            model_path=str(resolved),
            device=self.device,
            enable_frame_mode=True,
            vae_path=self.vae_path,
        )
        info = asdict(server.info())
        actual_contract = {
            "architecture": info["architecture"],
            "action_dim": int(info["action_dim"]),
            "sequence_length": int(info["sequence_length"]),
            "latent_shape": list(info["latent_shape"]),
        }
        expected_contract = {
            key: compatibility[key]
            for key in ("architecture", "action_dim", "sequence_length", "latent_shape")
        }
        if actual_contract != expected_contract:
            raise ValueError("loaded checkpoint compatibility mismatch")
        if compatibility.get("action_spec_id") != "switch_packets.v1":
            raise ValueError("unsupported action specification")
        shape = (actual_contract["sequence_length"], *actual_contract["latent_shape"])
        proposal = server.predict(np.zeros(shape, dtype=np.float32))
        if proposal.shape != (ACTION_DIM,) or not np.isfinite(proposal).all():
            raise ValueError("checkpoint smoke inference failed")
        return LoadedRevision(revision_id, actual, server, compatibility)


class ImmutableInferenceService:
    def __init__(self, loader: RevisionLoader, initial_revision: str, *, timeout_ms: int = 250):
        self.loader = loader
        self.timeout_ns = timeout_ms * 1_000_000
        self._lock = threading.Lock()
        self._current = loader.load(initial_revision)
        self._previous: LoadedRevision | None = None
        self._last_frame_timestamp_ns: int | None = None

    @staticmethod
    def _neutral(**fields) -> dict:
        return {"action": [0.0] * ACTION_DIM, **fields}

    def info(self) -> dict:
        with self._lock:
            current = self._current
            previous = self._previous
            return {
                "schema_id": "nxml.policy-inference-info.v2",
                "ready": True,
                "revision_id": current.revision_id,
                "checkpoint_sha256": current.digest,
                "previous_revision_id": previous.revision_id if previous else None,
                "action_spec_id": "switch_packets.v1",
                "action_dim": ACTION_DIM,
                "sequence_length": current.compatibility["sequence_length"],
                "latent_shape": current.compatibility["latent_shape"],
                "max_message_bytes": MAX_MESSAGE_BYTES,
                "timeout_ms": self.timeout_ns // 1_000_000,
            }

    def reset(self) -> dict:
        with self._lock:
            self._current.server.reset_frame_window()
            self._last_frame_timestamp_ns = None
        return {**self.info(), "reset": True}

    def reload(self, revision_id: str, expected_revision_id: str | None) -> dict:
        with self._lock:
            if expected_revision_id != self._current.revision_id:
                raise ValueError("stale expected revision")
        candidate = self.loader.load(revision_id)
        with self._lock:
            if expected_revision_id != self._current.revision_id:
                raise ValueError("stale expected revision")
            self._previous = self._current
            self._current = candidate
            self._last_frame_timestamp_ns = None
        return self.info()

    def rollback(
        self, expected_revision_id: str | None, target_revision_id: str | None = None
    ) -> dict:
        with self._lock:
            if expected_revision_id != self._current.revision_id:
                raise ValueError("stale expected revision")
            if self._previous is None:
                if target_revision_id is None:
                    raise ValueError("no previous revision; explicit rollback target required")
                target_id = target_revision_id
            else:
                target_id = self._previous.revision_id
                if target_revision_id is not None and target_revision_id != target_id:
                    raise ValueError("rollback target does not match previous revision")
        return self.reload(target_id, expected_revision_id)

    def predict_frame(self, frame_timestamp_ns: int, jpeg: bytes) -> dict:
        started = time.monotonic_ns()
        with self._lock:
            if (
                self._last_frame_timestamp_ns is not None
                and frame_timestamp_ns <= self._last_frame_timestamp_ns
            ):
                raise ValueError("stale or out-of-order frame timestamp")
            self._last_frame_timestamp_ns = frame_timestamp_ns
            current = self._current
            action = current.server.predict_frame(jpeg)
        proposal_ns = time.monotonic_ns()
        latency_ns = proposal_ns - started
        base = {
            "schema_id": "nxml.policy-proposal.v2",
            "state": "warming" if action is None else "proposal",
            "frame_timestamp_ns": frame_timestamp_ns,
            "proposal_timestamp_ns": proposal_ns,
            "processing_latency_ns": latency_ns,
            "revision_id": current.revision_id,
            "checkpoint_sha256": current.digest,
            "action_spec_id": "switch_packets.v1",
        }
        if latency_ns > self.timeout_ns:
            raise TimeoutError("inference processing exceeded latency budget")
        if action is None:
            return self._neutral(**base)
        proposal = np.asarray(action, dtype=np.float32)
        if proposal.shape != (ACTION_DIM,) or not np.isfinite(proposal).all():
            raise ValueError("inference returned invalid action")
        return {**base, "action": proposal.tolist()}

    def handle(self, message: bytes) -> bytes:
        try:
            if not message or len(message) > MAX_MESSAGE_BYTES:
                raise ValueError("invalid message size")
            mode, payload = message[0], message[1:]
            if mode in {MODE_INFO, MODE_HEALTH_V2} and not payload:
                return json.dumps(self.info(), sort_keys=True).encode()
            if mode == MODE_RESET_V2 and not payload:
                return json.dumps(self.reset(), sort_keys=True).encode()
            if mode == MODE_RELOAD_PATH:
                raise ValueError("filesystem-path reload is disabled; use registry revision ID")
            if mode in {MODE_RELOAD_REVISION_V2, MODE_ROLLBACK_V2}:
                body = json.loads(payload)
                if mode == MODE_RELOAD_REVISION_V2:
                    result = self.reload(body["revision_id"], body.get("expected_revision_id"))
                else:
                    result = self.rollback(
                        body.get("expected_revision_id"), body.get("target_revision_id")
                    )
                return json.dumps(result, sort_keys=True).encode()
            if mode == MODE_FRAME_V2:
                if len(payload) < 8:
                    raise ValueError("v2 frame payload is missing monotonic timestamp")
                frame_ns = struct.unpack("<Q", payload[:8])[0]
                result = self.predict_frame(frame_ns, payload[8:])
                return json.dumps(result, sort_keys=True).encode()
            if mode == MODE_PREDICT_FRAME:
                with self._lock:
                    action = self._current.server.predict_frame(payload)
                return b"\x00" if action is None else np.asarray(action, dtype=np.float32).tobytes()
            if mode == MODE_PREDICT:
                from nxrl.serve.transports.zmq import parse_predict_payload

                with self._lock:
                    action = self._current.server.predict(parse_predict_payload(payload))
                return np.asarray(action, dtype=np.float32).tobytes()
            raise ValueError("unknown inference operation")
        except Exception as error:
            response = self._neutral(
                schema_id="nxml.policy-proposal.v2",
                state="error",
                error=str(error),
                proposal_timestamp_ns=time.monotonic_ns(),
            )
            return json.dumps(response, sort_keys=True).encode()


def serve(service: ImmutableInferenceService, host: str, port: int) -> None:
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.MAXMSGSIZE, MAX_MESSAGE_BYTES)
    socket.bind(f"tcp://{host}:{port}")
    try:
        while True:
            socket.send(service.handle(socket.recv()))
    finally:
        socket.close()
        context.term()


def main() -> None:
    parser = argparse.ArgumentParser(description="NXML immutable policy inference service")
    parser.add_argument("--state-dir", default="/var/lib/nxml-control")
    parser.add_argument("--checkpoint-dir", default="/var/lib/nxml-control/checkpoints")
    parser.add_argument("--revision-id", required=True)
    parser.add_argument("--host", default="100.80.98.4")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-ms", type=int, default=250)
    args = parser.parse_args()
    loader = RevisionLoader(
        Path(args.state_dir) / "catalog.sqlite3",
        args.checkpoint_dir,
        device=args.device,
    )
    serve(
        ImmutableInferenceService(loader, args.revision_id, timeout_ms=args.timeout_ms),
        args.host,
        args.port,
    )
