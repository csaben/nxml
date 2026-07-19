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
from nxml_edge.auth import TailscaleAuthenticator, TailscaleWhoIsResolver
from nxml_edge.cluster import ClusterClient, ClusterDashboard, HttpTransport
from nxml_edge.config import ConfigStore
from nxml_edge.preview import CapturePreview, NxbtStateClient
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
        bluetooth = BluetoothctlAdapter(f"http://127.0.0.1:{config.orchestrator_port}")
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
    cluster_token = None
    if config.cluster_token_path and config.cluster_token_path.is_file():
        cluster_token = config.cluster_token_path.read_text().strip()
    cluster = ClusterDashboard(
        ClusterClient(HttpTransport(config.cluster_url, cluster_token)),
        stale_after=config.cluster_stale_after_seconds,
    )
    auth = None
    if config.auth_mode == "tailscale":
        auth = TailscaleAuthenticator(
            tailnet_host=config.tailnet_host,
            allowed_login=config.tailscale_allowed_login or "",
            allowed_node_ids=config.tailscale_allowed_node_ids,
            resolver=TailscaleWhoIsResolver(),
        )
    uvicorn.run(
        create_app(
            supervisor,
            token=token if auth is None else None,
            auth=auth,
            cluster=cluster,
            preview=CapturePreview(config.capture_identity),
            controller=NxbtStateClient(f"http://127.0.0.1:{config.orchestrator_port}"),
        ),
        host=config.bind_host,
        port=config.edge_port,
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
