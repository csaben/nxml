from __future__ import annotations

from pathlib import Path

import click
import uvicorn

from nxml_edge.adapters import (
    AutopilotClient,
    BluetoothctlAdapter,
    FakeBluetoothAdapter,
    FakeDeviceAdapter,
    FakePolicyAdapter,
    FakeRuntimeAdapter,
    FakeServiceAdapter,
    HttpPolicyAdapter,
    HttpRuntimeAdapter,
    SystemctlServiceAdapter,
    UdevDeviceAdapter,
)
from nxml_edge.config import ConfigStore
from nxml_edge.supervisor import EdgeSupervisor
from nxml_edge.web import create_app


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=Path("~/.config/nxml/edge.json").expanduser(),
    show_default=True,
)
@click.option(
    "--token-file",
    type=click.Path(path_type=Path),
    default=Path("~/.config/nxml/autopilot.token").expanduser(),
    show_default=True,
)
@click.option(
    "--fixture-adapters", is_flag=True, help="Use fake non-hardware adapters for UI testing."
)
def main(config_path: Path, token_file: Path, fixture_adapters: bool) -> None:
    """Run the unprivileged NXML edge supervisor."""
    store = ConfigStore(config_path, token_file)
    config = store.load()
    token = store.ensure_web_token()
    if fixture_adapters:
        devices = FakeDeviceAdapter()
        bluetooth = FakeBluetoothAdapter()
        policy = FakePolicyAdapter()
        runtime = FakeRuntimeAdapter()
        services = FakeServiceAdapter()
    else:
        client = AutopilotClient(config.autopilot_url, token)
        devices = UdevDeviceAdapter()
        bluetooth = BluetoothctlAdapter()
        policy = HttpPolicyAdapter(client)
        runtime = HttpRuntimeAdapter(client)
        services = SystemctlServiceAdapter()
    supervisor = EdgeSupervisor(
        config,
        store,
        devices=devices,
        bluetooth=bluetooth,
        policy=policy,
        runtime=runtime,
        services=services,
    )
    uvicorn.run(create_app(supervisor, token=token), host=config.bind_host, port=config.edge_port)


if __name__ == "__main__":
    main()
