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
