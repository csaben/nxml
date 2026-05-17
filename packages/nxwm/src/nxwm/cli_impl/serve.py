"""Implementation of ``nxwm serve``."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_serve(
    *,
    model: str,
    port: int,
    host: str,
    transport: str,
    device: str | None,
    data_path: str | None,
    checkpoint_dir: str | None,
    vae_path: str | None,
    flow_steps: int,
    cfg_scale: float,
    ssl_certfile: str | None = None,
    ssl_keyfile: str | None = None,
    root_path: str = "",
) -> None:
    import click

    from nxwm.serve import WorldModelServer

    server_kwargs: dict[str, Any] = {
        "flow_steps": flow_steps,
        "cfg_scale": cfg_scale,
    }
    if device is not None:
        server_kwargs["device"] = device
    if data_path is not None:
        server_kwargs["data_path"] = Path(data_path)
    if checkpoint_dir is not None:
        server_kwargs["checkpoint_dir"] = Path(checkpoint_dir)
    if vae_path is not None:
        server_kwargs["vae_path"] = vae_path

    server = WorldModelServer(model_path=model, **server_kwargs)

    if transport == "zmq":
        from nxwm.serve.transports.zmq import ZMQTransport

        zmq_transport = ZMQTransport(server, port=port, host=host)
        click.echo(f"nxwm serve: zmq REP on tcp://{host}:{port} (model: {model})")
        try:
            zmq_transport.run()
        finally:
            zmq_transport.shutdown()
        return

    if transport == "http":
        from nxwm.serve.transports.http import HTTPTransport

        # `host="*"` is the ZMQ default; for uvicorn the equivalent is `0.0.0.0`.
        bind_host = "0.0.0.0" if host == "*" else host
        scheme = "https" if ssl_certfile and ssl_keyfile else "http"
        http_transport = HTTPTransport(
            server,
            host=bind_host,
            port=port,
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
            root_path=root_path,
        )
        click.echo(
            f"nxwm serve: http on {scheme}://{bind_host}:{port}{root_path}/v1  "
            f"(model: {model})"
        )
        try:
            http_transport.run()
        finally:
            http_transport.shutdown()
        return

    raise click.UsageError(f"transport {transport!r} not supported")
