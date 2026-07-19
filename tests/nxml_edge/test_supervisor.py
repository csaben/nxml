from __future__ import annotations

from nxml_edge.adapters import BluetoothProbe, CaptureProbe, PolicyProbe, RuntimeProbe
from nxml_edge.models import Dependency, DriverState, SessionState


def _healthy_spool(supervisor) -> None:
    supervisor.config.spool_state_path.mkdir(parents=True, exist_ok=True)
    (supervisor.config.spool_state_path / "status.json").write_text(
        '{"admission_open": true, "cluster_connected": true}'
    )


def test_ready_status_has_unmistakable_driver(edge) -> None:
    supervisor, _, _ = edge
    status = supervisor.status()
    assert status.ready is True
    assert status.state is SessionState.READY
    assert status.driver is DriverState.HUMAN
    assert status.policy_revision == "rev-2"
    assert status.previous_policy_revision == "rev-1"


def test_human_capture_readiness_ignores_policy_and_autopilot(edge) -> None:
    supervisor, _, _ = edge
    _healthy_spool(supervisor)
    supervisor.policy.result = PolicyProbe(False, error="off")
    supervisor.runtime.result = RuntimeProbe(False, error="off")

    status = supervisor.status()

    assert status.human_capture_ready is True
    assert status.human_blocked_on is None
    assert "not required" in next(
        check.summary for check in status.checks if check.dependency is Dependency.POLICY
    ).lower()
    assert "not required" in next(
        check.summary for check in status.checks if check.dependency is Dependency.AUTOPILOT
    ).lower()


def test_human_capture_reports_spool_or_cluster_blocker(edge) -> None:
    supervisor, _, _ = edge
    supervisor.config.spool_state_path.mkdir(parents=True, exist_ok=True)
    (supervisor.config.spool_state_path / "status.json").write_text(
        '{"admission_open": false, "cluster_connected": false}'
    )

    status = supervisor.status()

    assert status.human_capture_ready is False
    assert status.human_blocked_on is Dependency.SPOOL


def test_start_stops_at_switch_and_guides_operator(edge) -> None:
    supervisor, services, _ = edge
    supervisor.bluetooth.result = BluetoothProbe(True, False, "00:11:22:33:44:55")
    status = supervisor.start_session()
    assert status.blocked_on is Dependency.SWITCH
    assert services.calls == [("start", "nxml-bt.service")]
    switch_check = next(check for check in status.checks if check.dependency is Dependency.SWITCH)
    assert "Change Grip / Order" in " ".join(switch_check.operator_steps)
    assert switch_check.detail["switch_mac"] == "00:11:22:33:44:55"


def test_retry_is_idempotent_while_preset_mac_service_is_active(edge) -> None:
    supervisor, services, _ = edge
    supervisor.bluetooth.result = BluetoothProbe(True, False, "58:2F:40:23:3C:CA")
    services.states["nxml-bt.service"] = "active"

    first = supervisor.retry()
    second = supervisor.retry()

    assert services.calls == []
    assert first.blocked_on is Dependency.SWITCH
    assert second.blocked_on is Dependency.SWITCH
    switch_check = next(c for c in second.checks if c.dependency is Dependency.SWITCH)
    assert switch_check.summary == "Connecting to configured Nintendo Switch"
    assert switch_check.detail["switch_mac"] == "58:2F:40:23:3C:CA"


def test_retry_starts_inactive_bluetooth_once_then_becomes_idempotent(edge) -> None:
    supervisor, services, _ = edge
    supervisor.bluetooth.result = BluetoothProbe(True, False, "58:2F:40:23:3C:CA")

    supervisor.retry()
    supervisor.retry()

    assert services.calls == [("start", "nxml-bt.service")]


def test_dependency_state_machine_reports_first_failed_stage(edge) -> None:
    supervisor, _, _ = edge
    supervisor.bluetooth.result = BluetoothProbe(False, False, error="adapter off")
    assert supervisor.status().blocked_on is Dependency.BLUETOOTH

    supervisor.bluetooth.result = BluetoothProbe(True, True)
    supervisor.devices.result = CaptureProbe(False, "/dev/v4l/by-id/missing")
    assert supervisor.status().blocked_on is Dependency.CAPTURE

    supervisor.devices.result = CaptureProbe(True, "/dev/v4l/by-id/fake", "/dev/video9")
    supervisor.policy.result = PolicyProbe(False, error="registry unavailable")
    assert supervisor.status().blocked_on is Dependency.POLICY


def test_start_only_allowlisted_services(edge) -> None:
    supervisor, services, _ = edge
    supervisor.runtime.result = RuntimeProbe(False, error="not started")
    supervisor.start_session()
    assert services.calls == [
        ("start", "nxml-bt.service"),
        ("start", "nxml-autopilot.service"),
    ]


def test_eject_is_persistent_until_explicit_rearm(edge) -> None:
    supervisor, _, _ = edge
    ejected = supervisor.eject()
    assert ejected.state is SessionState.EJECTED
    assert ejected.driver is DriverState.EJECTED
    assert ejected.ejected is True
    assert supervisor.status().ejected is True
    rearmed = supervisor.rearm()
    assert rearmed.ejected is False
    assert rearmed.driver is DriverState.NEUTRAL
