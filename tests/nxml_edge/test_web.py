from __future__ import annotations

import pytest
from fastapi import HTTPException
from nxml_edge.web import create_app
from starlette.requests import Request


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
    response = _endpoint(app, "/")()
    assert response.status_code == 200
    html = response.body.decode()
    assert "EMERGENCY EJECT" in html
    assert 'type="password"' in html
    assert "sessionStorage" in html
    assert "localStorage" not in html
    assert "URLSearchParams" not in html
    assert "Authorization" in html
    assert "Log out" in html
