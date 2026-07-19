from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Dependency(StrEnum):
    BLUETOOTH = "bluetooth"
    SWITCH = "switch"
    CAPTURE = "capture"
    POLICY = "policy"
    AUTOPILOT = "autopilot"


class SessionState(StrEnum):
    STOPPED = "stopped"
    BLOCKED = "blocked"
    STARTING = "starting"
    READY = "ready"
    DISCONNECTED = "disconnected"
    EJECTED = "ejected"
    ERROR = "error"


class DriverState(StrEnum):
    HUMAN = "human"
    AI = "policy"
    HYBRID = "human+policy"
    MACRO = "macro"
    DISCONNECTED = "disconnected"
    EJECTED = "ejected"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class Check(BaseModel):
    dependency: Dependency
    ready: bool
    summary: str
    detail: dict[str, object] = Field(default_factory=dict)
    operator_steps: list[str] = Field(default_factory=list)


class EdgeStatus(BaseModel):
    state: SessionState
    driver: DriverState
    ready: bool
    blocked_on: Dependency | None = None
    checks: list[Check]
    policy_id: str | None = None
    policy_revision: str | None = None
    previous_policy_revision: str | None = None
    ejected: bool = False
    tailnet_url: str
    logs: list[str] = Field(default_factory=list)
