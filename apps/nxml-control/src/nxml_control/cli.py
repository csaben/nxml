from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from nxml_control.auth import load_service_token
from nxml_control.training import (
    DisabledTrainingExecutor,
    FakeTrainingExecutor,
    SubprocessTrainingExecutor,
)

CRADLE_TAILNET_IP = "100.80.98.4"
DEFAULT_PORT = 8787
DEFAULT_INGEST_RESERVED_BYTES = 10 * 1024 * 1024 * 1024


def _directory(path: str, *, mode: int = 0o750) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True, mode=mode)
    if not os.access(result, os.R_OK | os.W_OK | os.X_OK):
        raise PermissionError(f"service path is not readable/writable: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="NXML authenticated ML control plane")
    parser.add_argument("--state-dir", default="/var/lib/nxml-control")
    parser.add_argument("--checkpoint-dir", default="/var/lib/nxml-control/checkpoints")
    parser.add_argument("--log-dir", default="/var/log/nxml-control")
    parser.add_argument("--host", default=CRADLE_TAILNET_IP)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--token-file", help="0600 Bearer token file; defaults to NXML_CONTROL_TOKEN_FILE"
    )
    parser.add_argument(
        "--bc-worker-command",
        default=os.environ.get("NXML_BC_WORKER_COMMAND"),
        help="worker command implementing nxml.bc-job.v1; defaults to NXML_BC_WORKER_COMMAND",
    )
    parser.add_argument("--job-dir", default="/var/lib/nxml-control/jobs")
    parser.add_argument(
        "--ingest-reserved-bytes",
        type=int,
        default=int(os.environ.get("NXML_INGEST_RESERVED_BYTES", DEFAULT_INGEST_RESERVED_BYTES)),
        help="free-byte headroom reserved before accepting new uploads",
    )
    parser.add_argument(
        "--allow-fake-training",
        action="store_true",
        help="development only: use the synchronous fake BC executor",
    )
    args = parser.parse_args()

    token = load_service_token(token_file=args.token_file)
    state_dir = _directory(args.state_dir)
    checkpoint_dir = _directory(args.checkpoint_dir)
    log_dir = _directory(args.log_dir)
    job_dir = _directory(args.job_dir)
    if args.bc_worker_command:
        training_executor = SubprocessTrainingExecutor(args.bc_worker_command, job_dir)
        training_async = True
    elif args.allow_fake_training:
        training_executor = FakeTrainingExecutor()
        training_async = False
    else:
        training_executor = DisabledTrainingExecutor()
        training_async = False
    log_path = log_dir / "control.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    log_path.chmod(0o640)

    import uvicorn

    from nxml_control.api import create_app
    from nxml_control.models import PolicyServerRuntime

    app = create_app(
        state_dir=state_dir,
        auth_token=token,
        checkpoint_dir=checkpoint_dir,
        training_executor=training_executor,
        training_async=training_async,
        ingest_reserved_bytes=args.ingest_reserved_bytes,
        deployment_runtime=PolicyServerRuntime(device="cpu"),
        allow_fake_deployment_runtime=False,
    )
    app.state.checkpoint_dir = checkpoint_dir
    app.state.log_dir = log_dir
    uvicorn.run(app, host=args.host, port=args.port, proxy_headers=False, server_header=False)
