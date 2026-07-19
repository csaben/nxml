from __future__ import annotations

import threading
import time
from collections import deque

from nxml_edge.adapters import (
    BluetoothAdapter,
    DeviceAdapter,
    PolicyAdapter,
    RuntimeAdapter,
    ServiceAdapter,
)
from nxml_edge.config import ConfigStore, EdgeConfig
from nxml_edge.models import Check, Dependency, DriverState, EdgeStatus, SessionState


class EdgeSupervisor:
    def __init__(
        self,
        config: EdgeConfig,
        store: ConfigStore,
        *,
        devices: DeviceAdapter,
        bluetooth: BluetoothAdapter,
        policy: PolicyAdapter,
        runtime: RuntimeAdapter,
        services: ServiceAdapter,
    ) -> None:
        self.config = config
        self.store = store
        self.devices = devices
        self.bluetooth = bluetooth
        self.policy = policy
        self.runtime = runtime
        self.services = services
        self._lock = threading.RLock()
        self._logs: deque[str] = deque(maxlen=100)
        self._starting = False

    def _log(self, message: str) -> None:
        self._logs.append(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}")

    def status(self) -> EdgeStatus:
        with self._lock:
            bt = self.bluetooth.probe(self.config.switch_mac)
            capture = self.devices.probe(self.config.capture_identity)
            policy = self.policy.probe(self.config.policy_uri)
            runtime = self.runtime.probe()
            checks = [
                Check(
                    dependency=Dependency.BLUETOOTH,
                    ready=bt.powered,
                    summary="Bluetooth adapter powered" if bt.powered else "Bluetooth unavailable",
                    detail={"error": bt.error} if bt.error else {},
                    operator_steps=["Verify bluetooth.service and adapter power"]
                    if not bt.powered
                    else [],
                ),
                Check(
                    dependency=Dependency.SWITCH,
                    ready=bt.switch_connected,
                    summary="Nintendo Switch connected"
                    if bt.switch_connected
                    else "Switch awaiting connection",
                    detail={"switch_mac": bt.switch_mac or self.config.switch_mac},
                    operator_steps=(
                        [
                            "Wake the Switch with a real controller",
                            "Open Controllers → Change Grip / Order",
                            "Use Retry after the root nxml-bt service reports Switch connected",
                        ]
                        if not bt.switch_connected
                        else []
                    ),
                ),
                Check(
                    dependency=Dependency.CAPTURE,
                    ready=capture.present,
                    summary="Stable capture device present"
                    if capture.present
                    else "Capture card missing",
                    detail={
                        "stable_path": capture.stable_path,
                        "device_path": capture.device_path,
                        "udev": capture.udev,
                        "error": capture.error,
                    },
                    operator_steps=["Reconnect the configured capture card and retry"]
                    if not capture.present
                    else [],
                ),
                Check(
                    dependency=Dependency.POLICY,
                    ready=policy.ready,
                    summary="Policy endpoint ready" if policy.ready else "Policy unavailable",
                    detail={
                        "policy_id": policy.policy_id,
                        "revision": policy.revision,
                        "error": policy.error,
                    },
                ),
                Check(
                    dependency=Dependency.AUTOPILOT,
                    ready=runtime.ready,
                    summary="Autopilot ready" if runtime.ready else "Autopilot unavailable",
                    detail={"error": runtime.error, "spool": runtime.spool},
                ),
            ]
            blocked = next((check.dependency for check in checks if not check.ready), None)
            if runtime.ejected:
                state = SessionState.EJECTED
                driver = DriverState.EJECTED
            elif blocked is None:
                state = SessionState.READY
                driver = _driver(runtime.driver)
            elif self._starting:
                state = SessionState.STARTING
                driver = DriverState.DISCONNECTED
            else:
                state = SessionState.BLOCKED
                driver = DriverState.DISCONNECTED
            return EdgeStatus(
                state=state,
                driver=driver,
                ready=blocked is None and not runtime.ejected,
                blocked_on=blocked,
                checks=checks,
                policy_id=policy.policy_id,
                policy_revision=policy.revision,
                previous_policy_revision=policy.previous_revision,
                ejected=runtime.ejected,
                tailnet_url=self.config.tailnet_url,
                logs=list(self._logs),
            )

    def start_session(self) -> EdgeStatus:
        with self._lock:
            self._starting = True
            self.store.ensure_web_token()
            self.services.start(self.config.bt_unit)
            self._log(f"requested start for {self.config.bt_unit}")
            status = self.status()
            prerequisite_order = (
                Dependency.BLUETOOTH,
                Dependency.SWITCH,
                Dependency.CAPTURE,
            )
            if status.blocked_on not in prerequisite_order:
                self.services.start(self.config.autopilot_unit)
                self._log(f"requested start for {self.config.autopilot_unit}")
            self._starting = False
            return self.status()

    def retry(self) -> EdgeStatus:
        with self._lock:
            current = self.status()
            if current.blocked_on in (Dependency.BLUETOOTH, Dependency.SWITCH):
                self.services.restart(self.config.bt_unit)
                self._log(f"requested restart for {self.config.bt_unit}")
            elif current.blocked_on is Dependency.AUTOPILOT:
                self.services.restart(self.config.autopilot_unit)
                self._log(f"requested restart for {self.config.autopilot_unit}")
            return self.status()

    def stop_session(self) -> EdgeStatus:
        with self._lock:
            self.services.stop(self.config.autopilot_unit)
            self._log(f"requested stop for {self.config.autopilot_unit}")
            return self.status()

    def eject(self) -> EdgeStatus:
        with self._lock:
            self.runtime.eject()
            self._log("emergency eject requested through runtime API")
            return self.status()

    def rearm(self) -> EdgeStatus:
        with self._lock:
            self.runtime.rearm()
            self._log("explicit re-arm requested through runtime API")
            return self.status()


def _driver(value: str) -> DriverState:
    try:
        return DriverState(value)
    except ValueError:
        return DriverState.UNKNOWN
