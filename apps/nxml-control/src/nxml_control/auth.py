"""Bearer service-token loading and constant-time verification."""

from __future__ import annotations

import hmac
import os
import stat
from pathlib import Path

ENV_TOKEN = "NXML_CONTROL_TOKEN"
ENV_TOKEN_FILE = "NXML_CONTROL_TOKEN_FILE"


def load_service_token(
    *, token_file: str | Path | None = None, environ: dict[str, str] | None = None
) -> str:
    env = os.environ if environ is None else environ
    path_value = str(token_file) if token_file is not None else env.get(ENV_TOKEN_FILE)
    if path_value:
        path = Path(path_value)
        file_stat = path.stat()
        mode = stat.S_IMODE(file_stat.st_mode)
        if not stat.S_ISREG(file_stat.st_mode):
            raise PermissionError(f"token path must be a regular file: {path}")
        if not mode & stat.S_IRUSR or mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise PermissionError(
                f"token file must be owner-readable and not group/world-writable, got {mode:04o}: {path}"
            )
        token = path.read_text().strip()
    else:
        token = env.get(ENV_TOKEN, "").strip()
    if len(token) < 32:
        raise ValueError("NXML control service token must be at least 32 characters")
    return token


def bearer_matches(authorization: str | None, expected_token: str) -> bool:
    if authorization is None:
        candidate = ""
    else:
        scheme, separator, candidate = authorization.partition(" ")
        if not separator or scheme.lower() != "bearer":
            candidate = ""
    return hmac.compare_digest(candidate.encode(), expected_token.encode())
