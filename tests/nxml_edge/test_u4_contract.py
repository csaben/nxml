from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi import HTTPException
from nxml_autopilot.web import create_app
from nxml_capture.source import Frame
from nxml_mux.input_devices.readers import WebGamepadReader
from starlette.requests import Request


class FakeSource:
    is_open = True

    def latest(self) -> Frame:
        return Frame(
            timestamp=time.time(),
            monotonic_ns=time.monotonic_ns(),
            image=np.zeros((8, 8, 3), dtype=np.uint8),
        )


class FakeRuntime:
    def __init__(self) -> None:
        self.ejected = False
        self.calls = 0

    def emergency_eject(self) -> None:
        self.ejected = True
        self.calls += 1

    def rearm(self) -> None:
        self.ejected = False

    def runtime_status(self) -> dict[str, object]:
        return {
            "ejected": self.ejected,
            "active_driver": "safety" if self.ejected else "none",
        }


def _request(token: str | None = None) -> Request:
    headers = [] if token is None else [(b"x-autopilot-token", token.encode())]
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "query_string": b"",
            "headers": headers,
        }
    )


def _endpoint(app, path: str):
    return next(route.endpoint for route in app.routes if getattr(route, "path", None) == path)


@pytest.mark.anyio
async def test_published_u4_eject_contract_is_authenticated_idempotent_and_rearmable() -> None:
    app = create_app(WebGamepadReader(), FakeSource(), token="runtime-secret")
    runtime = FakeRuntime()
    app.state.runtime = runtime
    eject = _endpoint(app, "/runtime/eject")
    rearm = _endpoint(app, "/runtime/rearm")

    with pytest.raises(HTTPException) as error:
        await eject(_request())
    assert error.value.status_code == 401
    assert (await eject(_request("runtime-secret"))).body
    assert (await eject(_request("runtime-secret"))).body
    assert runtime.ejected is True
    assert runtime.calls == 2
    await rearm(_request("runtime-secret"))
    assert runtime.ejected is False
