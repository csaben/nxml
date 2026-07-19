from __future__ import annotations

import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from nx_packets import neutral_action
from nxml_edge.human_control import HumanControlBridge
from nxml_edge.web import create_app
from starlette.requests import Request
from starlette.websockets import WebSocketDisconnect


class _Preview:
    stream_active = False

    def frames(self):
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfixture\r\n"

    def stream(self):
        yield b"--ffmpeg\r\nContent-type: image/jpeg\r\n\r\nfixture\r\n"


class _Actions:
    def post_human(self, vector: list[float]) -> dict[str, object]:
        return {"status": 200, "applied": True}


def _request(token: str | None = None, *, bearer: bool = False) -> Request:
    headers = []
    if token is not None:
        name = b"authorization" if bearer else b"x-nxml-edge-token"
        value = f"Bearer {token}".encode() if bearer else token.encode()
        headers = [(name, value)]
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "query_string": b"",
            "headers": headers,
        }
    )


def _endpoint(app, path: str):
    return next(route.endpoint for route in app.routes if getattr(route, "path", None) == path)


def _wait_until(condition, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not condition():
        time.sleep(0.01)


def test_status_and_session_controls(edge) -> None:
    supervisor, services, _ = edge
    app = create_app(supervisor, token="edge-secret")
    with pytest.raises(HTTPException) as error:
        _endpoint(app, "/api/status")(_request())
    assert error.value.status_code == 401
    status = _endpoint(app, "/api/status")(_request("edge-secret"))
    assert status.driver == "human"

    _endpoint(app, "/api/session/start")(_request("edge-secret"))
    assert ("start", "nxml-bt.service") in services.calls
    assert _endpoint(app, "/api/session/eject")(_request("edge-secret")).state == "ejected"
    assert _endpoint(app, "/api/session/rearm")(_request("edge-secret")).ejected is False


def test_bearer_auth_reaches_status_and_mocked_start_route(edge) -> None:
    supervisor, services, _ = edge
    app = create_app(supervisor, token="edge-secret")

    status = _endpoint(app, "/api/status")(_request("edge-secret", bearer=True))
    _endpoint(app, "/api/session/start")(_request("edge-secret", bearer=True))

    assert status.driver == "human"
    assert ("start", "nxml-bt.service") in services.calls


@pytest.mark.parametrize("token", [None, "", "wrong-secret"])
def test_missing_or_bad_bearer_is_rejected(edge, token) -> None:
    supervisor, _, _ = edge
    app = create_app(supervisor, token="edge-secret")
    with pytest.raises(HTTPException) as error:
        _endpoint(app, "/api/status")(_request(token, bearer=True))
    assert error.value.status_code == 401


def test_minimal_ui_renders(edge) -> None:
    supervisor, _, _ = edge
    app = create_app(supervisor, token="edge-secret")
    with pytest.raises(HTTPException) as error:
        _endpoint(app, "/")(_request())
    assert error.value.status_code == 401
    response = _endpoint(app, "/")(_request("edge-secret"))
    assert response.status_code == 200
    html = response.body.decode()
    assert "EMERGENCY EJECT" in html
    assert 'type="password"' not in html
    assert "sessionStorage" not in html
    assert "localStorage" not in html
    assert "URLSearchParams" not in html
    assert "Checking Tailnet identity" in html
    assert "/api/preview.mjpeg" in html
    assert "Human Capture" in html


def test_preview_requires_same_auth_as_status(edge) -> None:
    supervisor, _, _ = edge
    app = create_app(supervisor, token="edge-secret", preview=_Preview())
    endpoint = _endpoint(app, "/api/preview.mjpeg")

    with pytest.raises(HTTPException) as error:
        endpoint(_request())
    assert error.value.status_code == 401
    response = endpoint(_request("edge-secret"))
    assert response.media_type == "multipart/x-mixed-replace; boundary=frame"


def test_ui_has_explicit_human_enable_and_known_standard_mapping(edge) -> None:
    supervisor, _, _ = edge
    app = create_app(supervisor, token="edge-secret")
    html = _endpoint(app, "/")(_request("edge-secret")).body.decode()

    assert "Connect / Enable Human Control" in html
    assert "HUMAN CONTROL OFF" in html
    assert "STICK_DEADZONE=0.15" in html
    assert "0:24,1:25" in html  # standard south/east -> Switch B/A
    assert "2:22,3:23" in html  # standard west/north -> Switch Y/X
    assert "source: 'policy'" not in html
    assert "gamepad-telemetry" in html
    assert "proxy-telemetry" in html


def test_preview_stream_requires_auth_and_is_single_client(edge) -> None:
    supervisor, _, _ = edge
    preview = _Preview()
    app = create_app(supervisor, token="edge-secret", preview=preview)
    endpoint = _endpoint(app, "/api/preview/stream.mjpeg")

    with pytest.raises(HTTPException) as error:
        endpoint(_request())
    assert error.value.status_code == 401

    response = endpoint(_request("edge-secret"))
    assert response.media_type == "multipart/x-mixed-replace; boundary=ffmpeg"

    preview.stream_active = True
    with pytest.raises(HTTPException) as busy:
        endpoint(_request("edge-secret"))
    assert busy.value.status_code == 409


def test_human_ws_owns_a_session_and_neutralizes_on_close(edge) -> None:
    supervisor, _, _ = edge
    bridge = HumanControlBridge(_Actions(), start_watchdog=False)
    app = create_app(supervisor, token="edge-secret", human_control=bridge)

    with TestClient(app) as client:
        with client.websocket_connect("/api/human/ws?token=edge-secret") as ws:
            first = ws.receive_json()
            assert first["type"] == "enabled"
            session = first["status"]["session_id"]
            action = neutral_action().tolist()
            action[25] = 1.0
            ws.send_json(
                {
                    "session_id": session,
                    "sequence": 0,
                    "client_timestamp_ms": 1.0,
                    "vector": action,
                }
            )
            _wait_until(lambda: bridge.status()["actions_accepted"] == 1)

        # Assert before TestClient exits: lifespan shutdown would also
        # disable the bridge (reason "server_shutdown") and mask this.
        _wait_until(lambda: bridge.status()["enabled"] is False)
        status = bridge.status()
        assert status["enabled"] is False
        assert status["neutral_reason"] == "socket_closed"
        assert status["actions_accepted"] == 1


def test_human_ws_handshake_requires_edge_auth(edge) -> None:
    supervisor, _, _ = edge
    bridge = HumanControlBridge(_Actions(), start_watchdog=False)
    app = create_app(supervisor, token="edge-secret", human_control=bridge)

    with (
        TestClient(app) as client,
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/api/human/ws"),
    ):
        pass
    assert bridge.status()["enabled"] is False


def test_human_enable_route_is_not_reachable_without_edge_auth(edge) -> None:
    supervisor, _, _ = edge
    bridge = HumanControlBridge(_Actions(), start_watchdog=False)
    app = create_app(supervisor, token="edge-secret", human_control=bridge)

    with pytest.raises(HTTPException) as error:
        _endpoint(app, "/api/human/enable")(_request())
    assert error.value.status_code == 401
    assert bridge.status()["enabled"] is False
