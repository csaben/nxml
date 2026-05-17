"""Implementation of ``nxwm ui``.

Builds a :class:`WorldModelClient` via URL-scheme dispatch and launches the
FastAPI app from :mod:`nxwm.ui.app`.
"""

from __future__ import annotations

import socket
from typing import Any


def _build_display_urls(host: str, port: int) -> list[str]:
    """Return a prioritized list of URLs the user can paste into a browser.

    For ``0.0.0.0`` (LAN exposure) we include the machine's hostname and any
    non-loopback IPv4 addresses we can resolve. For a specific bind address we
    just echo back what the user asked for.
    """
    if host != "0.0.0.0":
        display = "localhost" if host in ("127.0.0.1", "::1") else host
        return [f"http://{display}:{port}"]

    candidates: list[str] = []
    try:
        hostname = socket.gethostname()
        if hostname:
            candidates.append(f"http://{hostname}:{port}")
    except OSError:
        pass

    candidates.extend(f"http://{ip}:{port}" for ip in _enumerate_lan_ips())
    candidates.append(f"http://localhost:{port}")
    return candidates


def _enumerate_lan_ips() -> list[str]:
    """Best-effort: list non-loopback IPv4 addresses on this host.

    Tries ``hostname -I`` first (Linux) then falls back to a UDP-connect probe
    that reveals just the default-route source IP. Failure modes return empty.
    """
    import subprocess

    seen: set[str] = set()
    out: list[str] = []

    try:
        result = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=2, check=False
        )
        if result.returncode == 0:
            for ip in result.stdout.split():
                if ip and not ip.startswith("127.") and ":" not in ip and ip not in seen:
                    seen.add(ip)
                    out.append(ip)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    if not out:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                if ip and not ip.startswith("127."):
                    out.append(ip)
        except OSError:
            pass

    return out


def _parse_detector_args(pairs: tuple[str, ...]) -> dict[str, Any]:
    """Convert ``--detector-arg key=value`` flags into a kwargs dict.

    Cast attempts in order: int → float → str. Lets us pass numeric thresholds
    on the command line without repeating their type in the flag.
    """
    out: dict[str, Any] = {}
    for raw in pairs:
        if "=" not in raw:
            raise ValueError(f"--detector-arg expects key=value, got {raw!r}")
        key, _, val = raw.partition("=")
        key = key.strip()
        val = val.strip()
        for caster in (int, float):
            try:
                out[key] = caster(val)
                break
            except ValueError:
                continue
        else:
            out[key] = val
    return out


def _build_detector(name: str, args: dict[str, Any]):
    """Import the right registration module by detector ``game:`` prefix.

    nxwm itself doesn't depend on game packages — so the CLI is where we
    bridge them. Importing the adapter module triggers the
    ``register_detector`` side-effect.
    """
    game_prefix = name.split(":", 1)[0]
    if game_prefix == "pokemon_za":
        import nxml_games.pokemon_za.detector_adapter  # noqa: F401  (registers)
    else:
        raise ValueError(f"unknown detector namespace: {game_prefix!r}")

    from nxwm.env.detectors import detector_registry

    if name not in detector_registry:
        known = ", ".join(sorted(detector_registry)) or "(none)"
        raise ValueError(f"detector {name!r} not registered; known: {known}")
    return detector_registry[name](**args)


def run_ui(
    *,
    game: str,
    model: str,
    port: int,
    host: str,
    no_browser: bool,
    device: str | None,
    flow_steps: int,
    cfg_scale: float,
    data_path: str | None,
    vae_path: str | None,
    detector_name: str | None,
    detector_args: tuple[str, ...],
) -> None:
    import webbrowser

    import click

    try:
        from nxwm.ui.app import build_app
    except ImportError as e:
        raise click.UsageError(
            "UI app not available. Install with `uv sync --extra ui` and ensure"
            " `nxwm.ui` is implemented (Step 6.5)."
        ) from e

    from nxwm.serve import build_client

    is_remote = model.startswith(("zmq://", "tcp://", "http://", "https://"))
    in_process_only = {
        "device": device,
        "flow_steps": flow_steps,
        "cfg_scale": cfg_scale,
        "data_path": data_path,
        "vae_path": vae_path,
    }
    set_overrides = {k: v for k, v in in_process_only.items() if v is not None}
    if is_remote and set_overrides:
        click.echo(
            f"warning: in-process flags ignored for remote model: {sorted(set_overrides)}",
            err=True,
        )

    if detector_name is not None:
        if is_remote:
            raise click.UsageError(
                "--detector requires an in-process model; ZMQ transport doesn't carry telemetry"
            )
        try:
            detector = _build_detector(detector_name, _parse_detector_args(detector_args))
        except (ValueError, KeyError, FileNotFoundError) as e:
            raise click.UsageError(str(e)) from e
        set_overrides["detector"] = detector

    client_kwargs: dict[str, Any] = {} if is_remote else set_overrides
    client = build_client(model, **client_kwargs)
    app = build_app(client=client, game=game)

    # When bound to 0.0.0.0, list LAN-accessible URLs so a browser on another
    # machine on the same network knows where to point. localhost-only binds
    # show just the loopback URL.
    urls = _build_display_urls(host, port)
    primary_url = urls[0]
    click.echo(f"nxwm-ui serving for {game} (model: {model})")
    for url in urls:
        click.echo(f"  → {url}")

    if not no_browser:
        import contextlib

        with contextlib.suppress(Exception):
            webbrowser.open(primary_url)

    try:
        import uvicorn

        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        client.close()
