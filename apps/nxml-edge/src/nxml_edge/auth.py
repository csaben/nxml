from __future__ import annotations

import ipaddress
import json
import subprocess
import time
from dataclasses import dataclass
from typing import Protocol

from fastapi import HTTPException, Request
from starlette.requests import HTTPConnection


@dataclass(frozen=True, slots=True)
class TailscaleIdentity:
    login: str
    stable_node_id: str


class IdentityResolver(Protocol):
    def resolve(self, address: str) -> TailscaleIdentity: ...


class TailscaleWhoIsResolver:
    """Resolve cryptographic Tailnet identity through the local tailscaled API."""

    def __init__(self, *, ttl_seconds: float = 30.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[float, TailscaleIdentity]] = {}

    def resolve(self, address: str) -> TailscaleIdentity:
        cached = self._cache.get(address)
        if cached and time.monotonic() - cached[0] < self.ttl_seconds:
            return cached[1]
        try:
            result = subprocess.run(
                ["/usr/bin/tailscale", "whois", "--json", address],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            payload = json.loads(result.stdout)
            identity = TailscaleIdentity(
                login=payload["UserProfile"]["LoginName"],
                stable_node_id=payload["Node"]["StableID"],
            )
        except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
            raise HTTPException(503, f"Tailscale identity lookup failed: {error}") from error
        self._cache[address] = (time.monotonic(), identity)
        return identity


class TailscaleAuthenticator:
    """Trust Serve identity headers only across its loopback proxy boundary."""

    SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

    def __init__(
        self,
        *,
        tailnet_host: str,
        allowed_login: str,
        allowed_node_ids: list[str],
        resolver: IdentityResolver,
    ) -> None:
        if not allowed_login or not allowed_node_ids:
            raise ValueError("Tailscale auth requires an explicit login and device allowlist")
        self.tailnet_host = tailnet_host.rstrip(".").lower()
        self.allowed_login = allowed_login.lower()
        self.allowed_node_ids = frozenset(allowed_node_ids)
        self.resolver = resolver

    def __call__(self, request: Request) -> None:
        self.authorize_connection(request, state_changing=request.method not in self.SAFE_METHODS)

    def authorize_connection(self, request: HTTPConnection, *, state_changing: bool) -> None:
        """Shared check for HTTP requests and WebSocket handshakes.

        WebSocket callers pass ``state_changing=True``: the socket carries
        controller input, so the same-origin proof is mandatory even though
        the handshake itself is a GET.
        """
        peer = request.client.host if request.client else ""
        try:
            if not ipaddress.ip_address(peer).is_loopback:
                raise ValueError
        except ValueError as error:
            raise HTTPException(403, "direct backend access is forbidden") from error

        host = request.headers.get("host", "").split(":", 1)[0].rstrip(".").lower()
        if host != self.tailnet_host:
            raise HTTPException(403, "untrusted Host for Tailnet UI")

        login_header = request.headers.get("tailscale-user-login", "").lower()
        forwarded = request.headers.get("x-forwarded-for", "")
        source = forwarded.split(",")[-1].strip()
        try:
            source = str(ipaddress.ip_address(source))
        except ValueError as error:
            raise HTTPException(401, "missing trusted Tailscale source identity") from error
        identity = self.resolver.resolve(source)
        if (
            login_header != identity.login.lower()
            or identity.login.lower() != self.allowed_login
            or identity.stable_node_id not in self.allowed_node_ids
        ):
            raise HTTPException(403, "Tailscale user or device is not authorized")

        if state_changing:
            origin = request.headers.get("origin", "").rstrip("/").lower()
            if origin != f"https://{self.tailnet_host}":
                raise HTTPException(403, "cross-origin control request rejected")
