"""Client to ``nxbt-orchestrator``'s ``/ws/state`` WebSocket stream.

Each tick from the orchestrator is a JSON-encoded ``Packet`` (one frame of
controller state in the structured form defined by ``nx_packets``). The
subscription thread converts every packet to the canonical 26-dim float32
action vector and stores the latest snapshot for synchronizers/recorders.

A background thread runs an asyncio loop so synchronous consumers (the
recorder, the synchronizer) can poll :meth:`latest` without managing async
themselves. Reconnect-on-drop is handled transparently.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from dataclasses import dataclass

import numpy as np
import websockets
from nx_packets import Packet, packet_to_action


@dataclass(frozen=True, slots=True)
class ControllerSnapshot:
    timestamp: float
    action: np.ndarray  # (26,) float32
    packet: dict[str, object]  # raw orchestrator packet (dict)


class ControllerSubscription:
    """Maintains a live read of orchestrator controller state.

    Spawns a background asyncio loop on its own thread. ``latest()`` is safe
    to call from any thread.
    """

    def __init__(
        self,
        url: str = "ws://127.0.0.1:7777/ws/state",
        *,
        reconnect_backoff: float = 1.0,
    ) -> None:
        self.url = url
        self.reconnect_backoff = reconnect_backoff
        self._latest: ControllerSnapshot | None = None
        self._latest_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._connected = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._connected.clear()
        self._thread = threading.Thread(
            target=self._run_thread,
            daemon=True,
            name="controller-subscribe",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        loop = self._loop
        task = self._task
        if loop is not None and task is not None and not task.done():
            loop.call_soon_threadsafe(task.cancel)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
            self._loop = None
            self._task = None

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def latest(self) -> ControllerSnapshot | None:
        with self._latest_lock:
            return self._latest

    def wait_for_first(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.latest() is not None:
                return True
            time.sleep(0.05)
        return False

    def _run_thread(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            self._task = loop.create_task(self._consume())
            with contextlib.suppress(asyncio.CancelledError):
                loop.run_until_complete(self._task)
        finally:
            loop.close()

    async def _consume(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    async with websockets.connect(self.url) as ws:
                        self._connected.set()
                        async for message in ws:
                            if self._stop_event.is_set():
                                break
                            self._ingest(message)
                except asyncio.CancelledError:
                    return
                except Exception:
                    # Backoff and retry; orchestrator may be restarting.
                    self._connected.clear()
                    if self._stop_event.is_set():
                        return
                    try:
                        await asyncio.sleep(self.reconnect_backoff)
                    except asyncio.CancelledError:
                        return
                finally:
                    self._connected.clear()
        except asyncio.CancelledError:
            return

    def _ingest(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        try:
            raw = json.loads(message)
            packet = Packet.model_validate(raw)
        except (json.JSONDecodeError, ValueError):
            return
        snapshot = ControllerSnapshot(
            timestamp=time.time(),
            action=packet_to_action(packet),
            packet=raw,
        )
        with self._latest_lock:
            self._latest = snapshot
