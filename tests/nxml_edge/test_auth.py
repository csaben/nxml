from __future__ import annotations

import pytest
from fastapi import HTTPException
from nxml_edge.auth import TailscaleAuthenticator, TailscaleIdentity
from starlette.requests import Request


class Resolver:
    def __init__(self, identity: TailscaleIdentity) -> None:
        self.identity = identity
        self.addresses: list[str] = []

    def resolve(self, address: str) -> TailscaleIdentity:
        self.addresses.append(address)
        return self.identity


def request(
    *,
    method: str = "GET",
    peer: str = "127.0.0.1",
    login: str = "csaben@github",
    source: str = "100.101.164.118",
    origin: str | None = None,
) -> Request:
    headers = [
        (b"host", b"cradle-ns.tailb1b51d.ts.net"),
        (b"tailscale-user-login", login.encode()),
        (b"x-forwarded-for", source.encode()),
    ]
    if origin:
        headers.append((b"origin", origin.encode()))
    return Request(
        {
            "type": "http",
            "method": method,
            "path": "/api/status",
            "query_string": b"",
            "headers": headers,
            "client": (peer, 12345),
        }
    )


def authenticator(identity: TailscaleIdentity | None = None):
    resolver = Resolver(identity or TailscaleIdentity("csaben@github", "node-inari"))
    auth = TailscaleAuthenticator(
        tailnet_host="cradle-ns.tailb1b51d.ts.net",
        allowed_login="csaben@github",
        allowed_node_ids=["node-inari"],
        resolver=resolver,
    )
    return auth, resolver


def test_allowed_user_and_device_are_resolved_with_whois() -> None:
    auth, resolver = authenticator()
    auth(request())
    assert resolver.addresses == ["100.101.164.118"]


@pytest.mark.parametrize(
    "identity",
    [
        TailscaleIdentity("mallory@example.com", "node-inari"),
        TailscaleIdentity("csaben@github", "wrong-device"),
    ],
)
def test_wrong_user_or_device_is_rejected(identity) -> None:
    auth, _ = authenticator(identity)
    with pytest.raises(HTTPException) as error:
        auth(request(login=identity.login))
    assert error.value.status_code == 403


def test_spoofed_headers_on_direct_backend_connection_are_rejected() -> None:
    auth, resolver = authenticator()
    with pytest.raises(HTTPException, match="direct backend") as error:
        auth(request(peer="100.101.164.118"))
    assert error.value.status_code == 403
    assert resolver.addresses == []


def test_preview_from_wrong_tailnet_device_is_rejected() -> None:
    auth, _ = authenticator(TailscaleIdentity("csaben@github", "wrong-device"))
    preview_request = request()
    preview_request.scope["path"] = "/api/preview.mjpeg"
    with pytest.raises(HTTPException) as error:
        auth(preview_request)
    assert error.value.status_code == 403


@pytest.mark.parametrize("origin", [None, "http://cradle-ns.tailb1b51d.ts.net", "https://evil.test"])
def test_control_mutations_require_exact_https_origin(origin) -> None:
    auth, _ = authenticator()
    with pytest.raises(HTTPException, match="cross-origin") as error:
        auth(request(method="POST", origin=origin))
    assert error.value.status_code == 403


def test_same_origin_control_is_authorized() -> None:
    auth, _ = authenticator()
    auth(
        request(
            method="POST",
            origin="https://cradle-ns.tailb1b51d.ts.net",
        )
    )
