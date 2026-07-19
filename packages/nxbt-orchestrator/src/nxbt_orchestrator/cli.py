"""``nxbt-orchestrator`` CLI entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import socket
import sys

import uvicorn

from nxbt_orchestrator.server import ServerConfig, create_app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nxbt-orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the HTTP/WS controller server.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7777)
    serve.add_argument("--update-rate", type=int, default=120)
    serve.add_argument("--override-window", type=float, default=0.3)
    serve.add_argument("--reconnect-address", default=None)
    serve.add_argument("--recording-output-path", default="recorded_macro.json")
    serve.add_argument("--debug", action="store_true")
    serve.add_argument("--log-level", default="info")
    return parser


def _install_signal_handlers(server: uvicorn.Server) -> None:
    # First signal: graceful shutdown so nxbt can remove the controller and
    # restore its BlueZ override. A repeat signal SIGKILLs the process group
    # (nxbt workers hold the HCI socket and can wedge); under systemd the
    # unit's TimeoutStopSec provides the same escalation automatically.
    state = {"count": 0}

    def handler(_signum: int, _frame: object) -> None:
        state["count"] += 1
        if state["count"] >= 2:
            print("\n[orchestrator] force-killing process group", flush=True)
            os.killpg(os.getpgrp(), signal.SIGKILL)
        print("\n[orchestrator] shutdown requested (repeat signal to force-kill)", flush=True)
        server.should_exit = True

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "serve":
        # Refuse to start before touching BlueZ: a second orchestrator on the
        # same adapter causes Bluetooth contention, DBus timeouts, and runaway
        # CPU. Failing on the port check keeps the blast radius at zero.
        if not _port_is_free(args.host, args.port):
            print(
                f"[orchestrator] {args.host}:{args.port} is already in use — another "
                "orchestrator instance is likely running. Inspect it with "
                "`pgrep -af nxbt-orchestrator` and `ss -ltnp | grep "
                f"':{args.port}'` before starting a new one.",
                file=sys.stderr,
                flush=True,
            )
            return 2
        # Become our own process group leader so killpg only reaps our nxbt
        # workers, not the parent shell or sudo wrapper.
        with contextlib.suppress(OSError):
            os.setpgrp()

        config = ServerConfig(
            host=args.host,
            port=args.port,
            update_rate=args.update_rate,
            override_window=args.override_window,
            reconnect_address=args.reconnect_address,
            recording_output_path=args.recording_output_path,
            debug=args.debug,
        )
        app = create_app(config)
        u_config = uvicorn.Config(app, host=config.host, port=config.port, log_level=args.log_level)
        server = uvicorn.Server(u_config)
        server.install_signal_handlers = lambda: _install_signal_handlers(server)
        server.run()
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
