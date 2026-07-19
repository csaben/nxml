from __future__ import annotations

import pytest
from nxml_edge.adapters import SystemctlServiceAdapter


def test_systemctl_adapter_has_literal_scope_aware_allowlist() -> None:
    adapter = SystemctlServiceAdapter()
    assert adapter._command("start", "nxml-bt.service") == [
        "systemctl",
        "start",
        "nxml-bt.service",
    ]
    assert adapter._command("restart", "nxml-autopilot.service") == [
        "systemctl",
        "--user",
        "restart",
        "nxml-autopilot.service",
    ]


@pytest.mark.parametrize(
    ("action", "unit"),
    [("enable", "nxml-bt.service"), ("start", "ssh.service"), ("", "")],
)
def test_systemctl_adapter_rejects_every_non_allowlisted_operation(action: str, unit: str) -> None:
    with pytest.raises(PermissionError):
        SystemctlServiceAdapter()._command(action, unit)
