from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class EdgeConfig(BaseModel):
    game: str = "pokemon-za"
    switch_mac: str | None = None
    policy_uri: str
    cluster_url: str = "http://cradle:8787"
    cluster_token_path: Path | None = Path("~/.config/nxml/cluster.token").expanduser()
    cluster_stale_after_seconds: float = Field(default=15.0, gt=0)
    capture_identity: str = "/dev/v4l/by-id/REPLACE_ME"
    capture_device_index: int = 0
    tailnet_host: str = "cradle-ns"
    bind_host: str = "0.0.0.0"
    edge_port: int = Field(default=8090, ge=1, le=65535)
    auth_mode: Literal["token", "tailscale"] = "token"
    tailscale_allowed_login: str | None = None
    tailscale_allowed_node_ids: list[str] = Field(default_factory=list)
    autopilot_port: int = Field(default=8080, ge=1, le=65535)
    orchestrator_port: int = Field(default=7777, ge=1, le=65535)
    capture_path: Path = Path("~/captures/pokemon-za").expanduser()
    spool_state_path: Path = Path("~/.local/state/nxml-spool").expanduser()
    initial_driver: Literal["human-priority", "human-takeover"] = "human-takeover"
    bt_unit: Literal["nxml-bt.service"] = "nxml-bt.service"
    autopilot_unit: Literal["nxml-autopilot.service"] = "nxml-autopilot.service"

    @field_validator("capture_identity")
    @classmethod
    def stable_capture_identity(cls, value: str) -> str:
        if not value.startswith("/dev/v4l/by-id/"):
            raise ValueError("capture_identity must use a stable /dev/v4l/by-id path")
        return value

    @model_validator(mode="after")
    def tailscale_auth_is_loopback_only(self) -> EdgeConfig:
        if self.auth_mode == "tailscale":
            if self.bind_host not in {"127.0.0.1", "::1"}:
                raise ValueError("Tailscale identity mode requires a loopback bind_host")
            if not self.tailscale_allowed_login or not self.tailscale_allowed_node_ids:
                raise ValueError("Tailscale identity mode requires an explicit user/device allowlist")
        return self

    @property
    def tailnet_url(self) -> str:
        if self.auth_mode == "tailscale":
            return f"https://{self.tailnet_host}"
        return f"http://{self.tailnet_host}:{self.edge_port}"

    @property
    def autopilot_url(self) -> str:
        return f"http://127.0.0.1:{self.autopilot_port}"


class ConfigStore:
    def __init__(self, config_path: Path, token_path: Path) -> None:
        self.config_path = config_path
        self.token_path = token_path

    def load(self) -> EdgeConfig:
        return EdgeConfig.model_validate_json(self.config_path.read_text())

    def save(self, config: EdgeConfig) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self.config_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(config.model_dump(mode="json"), indent=2) + "\n")
        temporary.chmod(0o600)
        temporary.replace(self.config_path)

    def ensure_web_token(self) -> str:
        if self.token_path.is_file():
            token = self.token_path.read_text().strip()
            if len(token) < 32:
                raise RuntimeError("stored web token is unexpectedly short")
            return token
        self.token_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        token = secrets.token_hex(32)
        temporary = self.token_path.with_suffix(".tmp")
        temporary.write_text(token + "\n")
        temporary.chmod(0o600)
        temporary.replace(self.token_path)
        return token
