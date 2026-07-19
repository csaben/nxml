from __future__ import annotations

import stat

import pytest
from nxml_edge.config import EdgeConfig


def test_config_rejects_unstable_capture_path() -> None:
    with pytest.raises(ValueError, match="/dev/v4l/by-id"):
        EdgeConfig(policy_uri="fixture", capture_identity="/dev/video0")


def test_token_is_persistent_secret(edge) -> None:
    _, _, store = edge
    first = store.ensure_web_token()
    second = store.ensure_web_token()
    assert first == second
    assert len(first) == 64
    assert stat.S_IMODE(store.token_path.stat().st_mode) == 0o600


def test_tailscale_mode_preserves_private_cluster_token_wiring(tmp_path) -> None:
    token_path = tmp_path / "cluster.token"
    config = EdgeConfig(
        policy_uri="fixture",
        auth_mode="tailscale",
        tailnet_host="cradle-ns.tail.test",
        bind_host="127.0.0.1",
        tailscale_allowed_login="csaben@github",
        tailscale_allowed_node_ids=["node-inari"],
        cluster_token_path=token_path,
    )
    assert config.cluster_token_path == token_path
    assert config.tailnet_url == "https://cradle-ns.tail.test"


def test_tailscale_mode_rejects_direct_tailnet_backend_bind() -> None:
    with pytest.raises(ValueError, match="loopback"):
        EdgeConfig(
            policy_uri="fixture",
            auth_mode="tailscale",
            bind_host="100.73.109.68",
            tailscale_allowed_login="csaben@github",
            tailscale_allowed_node_ids=["node-inari"],
        )
