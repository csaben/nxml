from __future__ import annotations

from pathlib import Path

import pytest
from nxml_edge.adapters import (
    FakeBluetoothAdapter,
    FakeDeviceAdapter,
    FakePolicyAdapter,
    FakeRuntimeAdapter,
    FakeServiceAdapter,
)
from nxml_edge.config import ConfigStore, EdgeConfig
from nxml_edge.supervisor import EdgeSupervisor


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def edge(tmp_path: Path):
    config = EdgeConfig(
        policy_uri="zmq+frames://cradle:5557",
        capture_identity="/dev/v4l/by-id/fake-hagibis",
        capture_path=tmp_path / "captures",
        spool_state_path=tmp_path / "spool",
    )
    store = ConfigStore(tmp_path / "edge.json", tmp_path / "token")
    store.save(config)
    services = FakeServiceAdapter()
    supervisor = EdgeSupervisor(
        config,
        store,
        devices=FakeDeviceAdapter(),
        bluetooth=FakeBluetoothAdapter(),
        policy=FakePolicyAdapter(),
        runtime=FakeRuntimeAdapter(),
        services=services,
    )
    return supervisor, services, store
