from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Protocol


@dataclass(frozen=True, slots=True)
class CaptureProbe:
    present: bool
    stable_path: str
    device_path: str | None = None
    udev: dict[str, str] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BluetoothProbe:
    powered: bool
    switch_connected: bool
    switch_mac: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PolicyProbe:
    ready: bool
    policy_id: str | None = None
    revision: str | None = None
    previous_revision: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeProbe:
    ready: bool
    driver: str = "unknown"
    ejected: bool = False
    error: str | None = None


class DeviceAdapter(Protocol):
    def probe(self, stable_path: str) -> CaptureProbe: ...


class BluetoothAdapter(Protocol):
    def probe(self, switch_mac: str | None) -> BluetoothProbe: ...


class PolicyAdapter(Protocol):
    def probe(self, policy_uri: str) -> PolicyProbe: ...


class RuntimeAdapter(Protocol):
    def probe(self) -> RuntimeProbe: ...

    def eject(self) -> RuntimeProbe: ...

    def rearm(self) -> RuntimeProbe: ...


class ServiceAdapter(Protocol):
    def status(self, unit: str) -> str: ...

    def start(self, unit: str) -> None: ...

    def stop(self, unit: str) -> None: ...

    def restart(self, unit: str) -> None: ...


class FakeDeviceAdapter:
    def __init__(self, probe: CaptureProbe | None = None) -> None:
        self.result = probe or CaptureProbe(True, "/dev/v4l/by-id/fake", "/dev/video9")

    def probe(self, stable_path: str) -> CaptureProbe:
        return self.result


class FakeBluetoothAdapter:
    def __init__(self, probe: BluetoothProbe | None = None) -> None:
        self.result = probe or BluetoothProbe(True, True, "00:11:22:33:44:55")

    def probe(self, switch_mac: str | None) -> BluetoothProbe:
        return self.result


class FakePolicyAdapter:
    def __init__(self, probe: PolicyProbe | None = None) -> None:
        self.result = probe or PolicyProbe(True, "fixture-policy", "rev-2", "rev-1")

    def probe(self, policy_uri: str) -> PolicyProbe:
        return self.result


class FakeRuntimeAdapter:
    def __init__(self, probe: RuntimeProbe | None = None) -> None:
        self.result = probe or RuntimeProbe(True, "human", False)

    def probe(self) -> RuntimeProbe:
        return self.result

    def eject(self) -> RuntimeProbe:
        self.result = RuntimeProbe(True, "ejected", True)
        return self.result

    def rearm(self) -> RuntimeProbe:
        self.result = RuntimeProbe(True, "neutral", False)
        return self.result


class FakeServiceAdapter:
    ALLOWED = frozenset({"nxml-bt.service", "nxml-autopilot.service"})

    def __init__(self) -> None:
        self.states = {unit: "inactive" for unit in self.ALLOWED}
        self.calls: list[tuple[str, str]] = []

    def _allow(self, unit: str) -> None:
        if unit not in self.ALLOWED:
            raise PermissionError(f"unit is not allowlisted: {unit}")

    def status(self, unit: str) -> str:
        self._allow(unit)
        return self.states[unit]

    def start(self, unit: str) -> None:
        self._allow(unit)
        self.states[unit] = "active"
        self.calls.append(("start", unit))

    def stop(self, unit: str) -> None:
        self._allow(unit)
        self.states[unit] = "inactive"
        self.calls.append(("stop", unit))

    def restart(self, unit: str) -> None:
        self._allow(unit)
        self.states[unit] = "active"
        self.calls.append(("restart", unit))


class SymlinkDeviceAdapter:
    """Read-only stable-link adapter; udev enrichment is injected later."""

    def probe(self, stable_path: str) -> CaptureProbe:
        path = Path(stable_path)
        if not path.is_symlink():
            return CaptureProbe(False, stable_path, error="stable capture symlink is missing")
        try:
            device = str(path.resolve(strict=True))
        except OSError as error:
            return CaptureProbe(False, stable_path, error=str(error))
        return CaptureProbe(True, stable_path, device_path=device)


class UdevDeviceAdapter(SymlinkDeviceAdapter):
    def probe(self, stable_path: str) -> CaptureProbe:
        basic = super().probe(stable_path)
        if not basic.present or basic.device_path is None:
            return basic
        try:
            result = subprocess.run(
                ["udevadm", "info", "--query=property", "--name", basic.device_path],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            properties = dict(
                line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
            )
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            return CaptureProbe(
                True,
                stable_path,
                basic.device_path,
                error=f"udev metadata unavailable: {error}",
            )
        return CaptureProbe(True, stable_path, basic.device_path, properties)


class BluetoothctlAdapter:
    def probe(self, switch_mac: str | None) -> BluetoothProbe:
        try:
            show = subprocess.run(
                ["bluetoothctl", "show"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout
            powered = "Powered: yes" in show
            if not switch_mac:
                return BluetoothProbe(powered, False, error="Switch MAC is not configured")
            info = subprocess.run(
                ["bluetoothctl", "info", switch_mac],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            return BluetoothProbe(powered, "Connected: yes" in info.stdout, switch_mac)
        except (OSError, subprocess.SubprocessError) as error:
            return BluetoothProbe(False, False, switch_mac, str(error))


class SystemctlServiceAdapter:
    """Exact systemd allowlist; values never become executable command text."""

    ALLOWED: ClassVar[dict[str, tuple[str, str]]] = {
        "nxml-bt.service": ("system", "nxml-bt.service"),
        "nxml-autopilot.service": ("user", "nxml-autopilot.service"),
    }
    ACTIONS = frozenset({"start", "stop", "restart"})

    def _command(self, action: str, unit: str) -> list[str]:
        if action not in self.ACTIONS or unit not in self.ALLOWED:
            raise PermissionError(f"systemd operation is not allowlisted: {action} {unit}")
        scope, literal_unit = self.ALLOWED[unit]
        return ["systemctl", *(["--user"] if scope == "user" else []), action, literal_unit]

    def _run(self, action: str, unit: str) -> None:
        subprocess.run(self._command(action, unit), check=True, timeout=10)

    def status(self, unit: str) -> str:
        command = self._command("start", unit)
        command[-2] = "is-active"
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=3)
        return result.stdout.strip() or "unknown"

    def start(self, unit: str) -> None:
        self._run("start", unit)

    def stop(self, unit: str) -> None:
        self._run("stop", unit)

    def restart(self, unit: str) -> None:
        self._run("restart", unit)


class AutopilotClient:
    """Authenticated consumer of the UI-owned U4 runtime contract."""

    def __init__(self, base_url: str, token: str, *, timeout: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def status(self) -> dict[str, object]:
        return self._request("GET", "/runtime/status")

    def eject(self) -> dict[str, object]:
        return self._request("POST", "/runtime/eject")

    def rearm(self) -> dict[str, object]:
        return self._request("POST", "/runtime/rearm")

    def _request(self, method: str, path: str) -> dict[str, object]:
        request = urllib.request.Request(
            self.base_url + path,
            method=method,
            headers={"X-Autopilot-Token": self.token},
            data=b"" if method == "POST" else None,
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())


class HttpRuntimeAdapter:
    def __init__(self, client: AutopilotClient) -> None:
        self.client = client

    def probe(self) -> RuntimeProbe:
        try:
            payload = self.client.status()
        except (OSError, ValueError, urllib.error.URLError) as error:
            return RuntimeProbe(False, error=str(error))
        capture = payload.get("capture") if isinstance(payload.get("capture"), dict) else {}
        controller = (
            payload.get("controller") if isinstance(payload.get("controller"), dict) else {}
        )
        orchestrator = (
            payload.get("orchestrator") if isinstance(payload.get("orchestrator"), dict) else {}
        )
        ready = bool(
            payload.get("attached")
            and not payload.get("ejected")
            and capture.get("open")
            and not capture.get("stale")
            and controller.get("transport_fresh")
            and orchestrator.get("connected")
        )
        detail = str(payload.get("driver_detail") or payload.get("active_driver") or "unknown")
        return RuntimeProbe(ready, detail, bool(payload.get("ejected")))

    def eject(self) -> RuntimeProbe:
        self.client.eject()
        return self.probe()

    def rearm(self) -> RuntimeProbe:
        self.client.rearm()
        return self.probe()


class HttpPolicyAdapter:
    def __init__(self, client: AutopilotClient) -> None:
        self.client = client

    def probe(self, policy_uri: str) -> PolicyProbe:
        try:
            payload = self.client.status()
        except (OSError, ValueError, urllib.error.URLError) as error:
            return PolicyProbe(False, error=str(error))
        policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
        return PolicyProbe(
            bool(policy.get("ready")),
            str(policy["id"]) if policy.get("id") is not None else None,
            str(policy["revision"]) if policy.get("revision") is not None else None,
            (
                str(policy["previous_revision"])
                if policy.get("previous_revision") is not None
                else None
            ),
            str(policy["last_error"]) if policy.get("last_error") else None,
        )
