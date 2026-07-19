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
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode != 0o600:
            raise PermissionError(f"token file must have mode 0600, got {mode:04o}: {path}")
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
