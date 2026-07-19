from __future__ import annotations

from pathlib import Path

import pytest
from nxml_edge.adapters import SystemctlServiceAdapter


def test_systemctl_adapter_has_literal_scope_aware_allowlist() -> None:
    adapter = SystemctlServiceAdapter()
    assert adapter._command("start", "nxml-bt.service") == [
        "systemctl",
        "--no-ask-password",
        "start",
        "nxml-bt.service",
    ]
    assert adapter._command("restart", "nxml-autopilot.service") == [
        "systemctl",
        "--no-ask-password",
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


def test_systemctl_adapter_surfaces_noninteractive_authorization_failure(monkeypatch) -> None:
    class Result:
        returncode = 1
        stdout = ""
        stderr = "Interactive authentication required."

    monkeypatch.setattr("nxml_edge.adapters.subprocess.run", lambda *args, **kwargs: Result())
    with pytest.raises(PermissionError, match="PolicyKit did not authorize"):
        SystemctlServiceAdapter().start("nxml-bt.service")


def test_polkit_rule_is_exactly_scoped_and_logs_decision_fields() -> None:
    rule = Path("deploy/cradle-ns/polkit/49-nxml-edge.rules").read_text()
    assert 'action.id !== "org.freedesktop.systemd1.manage-units"' in rule
    assert 'subject.user !== "arelius"' in rule
    assert 'unit === "nxml-bt.service"' in rule
    for verb in ("start", "stop", "restart"):
        assert f'verb === "{verb}"' in rule
    assert "polkit.log" in rule
    assert "subject.local" in rule
    assert "subject.active" in rule
    assert "ssh.service" not in rule
