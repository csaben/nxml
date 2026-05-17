"""Bridge :class:`NxbtController` thread-side state callbacks to asyncio.

The controller's update loop is a regular thread; FastAPI WebSockets run in
asyncio. :class:`StateStream` wraps an :class:`asyncio.Queue` per subscriber
and uses ``loop.call_soon_threadsafe`` to push from the update thread into
the asyncio loop. Bounded queues drop on overflow so a slow client cannot
back-pressure the 120 Hz update loop.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from nxbt_orchestrator.controller import NxbtController, _deep_copy_packet


class StateStream:
    def __init__(self, controller: NxbtController, *, queue_size: int = 4) -> None:
        self._controller = controller
        self._queue_size = queue_size

    def subscribe(self) -> _Subscription:
        loop = asyncio.get_running_loop()
        return _Subscription(self._controller, loop, self._queue_size)


class _Subscription:
    def __init__(
        self,
        controller: NxbtController,
        loop: asyncio.AbstractEventLoop,
        queue_size: int,
    ) -> None:
        self._controller = controller
        self._loop = loop
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=queue_size)
        self._controller.subscribe_state(self._on_state)

    def _on_state(self, packet: dict[str, Any]) -> None:
        # Runs on the controller's update thread. Copy out of the shared
        # packet before crossing into asyncio so the consumer sees a stable
        # snapshot.
        snapshot = _deep_copy_packet(packet)
        self._loop.call_soon_threadsafe(self._enqueue, snapshot)

    def _enqueue(self, snapshot: dict[str, Any]) -> None:
        if self._queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
        self._queue.put_nowait(snapshot)

    async def __aiter__(self):
        # Caller is responsible for calling ``close()`` on disconnect; both
        # ``server.ws_state`` and any other consumer wrap iteration in a
        # try/finally that does so.
        while True:
            yield await self._queue.get()

    def close(self) -> None:
        self._controller.unsubscribe_state(self._on_state)
