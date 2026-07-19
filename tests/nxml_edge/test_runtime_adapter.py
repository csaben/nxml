from __future__ import annotations

from nxml_edge.adapters import HttpPolicyAdapter, HttpRuntimeAdapter


class FakeClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.eject_calls = 0
        self.rearm_calls = 0

    def status(self) -> dict[str, object]:
        return self.payload

    def eject(self) -> dict[str, object]:
        self.eject_calls += 1
        self.payload["ejected"] = True
        self.payload["driver_detail"] = "ejected"
        return self.payload

    def rearm(self) -> dict[str, object]:
        self.rearm_calls += 1
        self.payload["ejected"] = False
        self.payload["driver_detail"] = "neutral"
        return self.payload


def _green_payload() -> dict[str, object]:
    return {
        "attached": True,
        "ejected": False,
        "active_driver": "blended",
        "driver_detail": "human+policy",
        "capture": {"open": True, "stale": False, "age_ms": 12},
        "controller": {"transport_fresh": True, "transport_sample_age_ms": 20},
        "orchestrator": {"connected": True, "reachable": True},
        "policy": {
            "ready": True,
            "id": "za-ppo",
            "revision": "rev-7",
            "previous_revision": "rev-6",
        },
    }


def test_runtime_adapter_consumes_u4_freshness_and_driver_contract() -> None:
    runtime = HttpRuntimeAdapter(FakeClient(_green_payload()))
    probe = runtime.probe()
    assert probe.ready is True
    assert probe.driver == "human+policy"


def test_stale_capture_or_controller_degrades_runtime() -> None:
    payload = _green_payload()
    payload["capture"]["stale"] = True  # type: ignore[index]
    assert HttpRuntimeAdapter(FakeClient(payload)).probe().ready is False
    payload = _green_payload()
    payload["controller"]["transport_fresh"] = False  # type: ignore[index]
    assert HttpRuntimeAdapter(FakeClient(payload)).probe().ready is False
    payload = _green_payload()
    payload["orchestrator"]["connected"] = False  # type: ignore[index]
    assert HttpRuntimeAdapter(FakeClient(payload)).probe().ready is False


def test_policy_adapter_preserves_active_and_previous_revision() -> None:
    policy = HttpPolicyAdapter(FakeClient(_green_payload())).probe("ignored")
    assert policy.ready is True
    assert policy.policy_id == "za-ppo"
    assert policy.revision == "rev-7"
    assert policy.previous_revision == "rev-6"


def test_policy_activation_failure_remains_not_ready_with_error() -> None:
    payload = _green_payload()
    payload["policy"] = {
        "ready": False,
        "id": "candidate",
        "revision": "rev-bad",
        "previous_revision": "rev-7",
        "last_error": "candidate warmup failed",
    }
    policy = HttpPolicyAdapter(FakeClient(payload)).probe("ignored")
    assert policy.ready is False
    assert policy.previous_revision == "rev-7"
    assert policy.error == "candidate warmup failed"


def test_eject_and_rearm_delegate_to_u4_api() -> None:
    client = FakeClient(_green_payload())
    runtime = HttpRuntimeAdapter(client)
    assert runtime.eject().ejected is True
    assert client.eject_calls == 1
    assert runtime.rearm().ejected is False
    assert client.rearm_calls == 1
