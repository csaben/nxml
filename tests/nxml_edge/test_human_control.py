from __future__ import annotations

import json

import pytest
from nx_packets import ACTION_DIM, neutral_action
from nxml_edge.human_control import (
    HumanActionRequest,
    HumanControlBridge,
    HumanControlError,
    NxbtActionClient,
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class Transport:
    def __init__(self) -> None:
        self.actions: list[list[float]] = []
        self.fail = False

    def post_human(self, vector: list[float]) -> None:
        self.actions.append(vector)
        if self.fail:
            raise RuntimeError("offline")


def request(session_id: str, sequence: int, vector: list[float]) -> HumanActionRequest:
    return HumanActionRequest(
        session_id=session_id,
        sequence=sequence,
        client_timestamp_ms=1000 + sequence,
        vector=vector,
    )


def test_enable_action_disable_always_brackets_with_neutral() -> None:
    clock = Clock()
    transport = Transport()
    bridge = HumanControlBridge(transport, clock=clock, start_watchdog=False)
    enabled = bridge.enable()
    action = neutral_action().tolist()
    action[25] = 1

    bridge.apply(request(str(enabled["session_id"]), 0, action))
    bridge.disable("client_disabled")

    assert transport.actions == [neutral_action().tolist(), action, neutral_action().tolist()]
    assert bridge.status()["enabled"] is False
    assert bridge.status()["neutral_reason"] == "client_disabled"


def test_stale_heartbeat_neutralizes_and_latches_off() -> None:
    clock = Clock()
    transport = Transport()
    bridge = HumanControlBridge(
        transport, stale_after=0.25, clock=clock, start_watchdog=False
    )
    bridge.enable()
    clock.now = 0.3

    assert bridge.expire_if_stale() is True
    assert transport.actions[-1] == neutral_action().tolist()
    assert bridge.status()["enabled"] is False
    assert bridge.status()["neutral_reason"] == "stale_heartbeat"


def test_rate_and_sequence_are_bounded() -> None:
    clock = Clock()
    bridge = HumanControlBridge(Transport(), clock=clock, max_hz=40, start_watchdog=False)
    session = str(bridge.enable()["session_id"])
    bridge.apply(request(session, 0, neutral_action().tolist()))
    clock.now = 0.01
    with pytest.raises(HumanControlError) as rate:
        bridge.apply(request(session, 1, neutral_action().tolist()))
    assert rate.value.status_code == 429
    clock.now = 0.03
    with pytest.raises(HumanControlError) as duplicate:
        bridge.apply(request(session, 0, neutral_action().tolist()))
    assert duplicate.value.status_code == 409


def test_wrong_action_shape_or_range_is_rejected() -> None:
    with pytest.raises(ValueError):
        request("session", 0, [0.0] * (ACTION_DIM - 1))
    invalid = neutral_action().tolist()
    invalid[0] = 1.1
    with pytest.raises(ValueError):
        request("session", 0, invalid)


def test_server_error_disables_and_attempts_neutral() -> None:
    clock = Clock()
    transport = Transport()
    bridge = HumanControlBridge(transport, clock=clock, start_watchdog=False)
    session = str(bridge.enable()["session_id"])
    transport.fail = True

    with pytest.raises(HumanControlError) as error:
        bridge.apply(request(session, 0, neutral_action().tolist()))

    assert error.value.status_code == 502
    assert bridge.status()["enabled"] is False
    assert bridge.status()["neutral_reason"] == "server_error"
    assert transport.actions[-1] == neutral_action().tolist()


def test_nxbt_client_hard_codes_human_source(monkeypatch) -> None:
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def read(self) -> bytes:
            return b'{"applied": true}'

    def urlopen(req, timeout):
        captured["payload"] = json.loads(req.data)
        captured["url"] = req.full_url
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    NxbtActionClient("http://127.0.0.1:7777").post_human(neutral_action().tolist())

    assert captured["url"] == "http://127.0.0.1:7777/action"
    assert captured["payload"] == {"vector": neutral_action().tolist(), "source": "human"}
    assert "policy" not in captured["payload"]
