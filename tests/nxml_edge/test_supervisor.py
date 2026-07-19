from __future__ import annotations

from nxml_edge.adapters import BluetoothProbe, CaptureProbe, PolicyProbe, RuntimeProbe
from nxml_edge.models import Dependency, DriverState, SessionState


def test_ready_status_has_unmistakable_driver(edge) -> None:
    supervisor, _, _ = edge
    status = supervisor.status()
    assert status.ready is True
    assert status.state is SessionState.READY
    assert status.driver is DriverState.HUMAN
    assert status.policy_revision == "rev-2"
    assert status.previous_policy_revision == "rev-1"


def test_start_stops_at_switch_and_guides_operator(edge) -> None:
    supervisor, services, _ = edge
    supervisor.bluetooth.result = BluetoothProbe(True, False, "00:11:22:33:44:55")
    status = supervisor.start_session()
    assert status.blocked_on is Dependency.SWITCH
    assert services.calls == [("start", "nxml-bt.service")]
    switch_check = next(check for check in status.checks if check.dependency is Dependency.SWITCH)
    assert "Change Grip / Order" in " ".join(switch_check.operator_steps)


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
