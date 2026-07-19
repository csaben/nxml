from __future__ import annotations

import pytest
from fastapi import HTTPException
from nxml_edge.web import create_app
from starlette.requests import Request


def _request(token: str | None = None) -> Request:
    headers = [] if token is None else [(b"x-nxml-edge-token", token.encode())]
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


def test_minimal_ui_renders(edge) -> None:
    supervisor, _, _ = edge
    app = create_app(supervisor, token="edge-secret")
    response = _endpoint(app, "/")()
    assert response.status_code == 200
    assert "EMERGENCY EJECT" in response.body.decode()
