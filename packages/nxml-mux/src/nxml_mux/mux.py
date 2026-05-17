"""``ControllerMux`` — poll N sources, apply a strategy, expose a merged action.

Side-effect-free at output: the mux doesn't POST to ``nxbt-orchestrator``
itself. The caller drives the loop, calls ``tick()`` for the next action,
and decides where to send it.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from nxml_mux.source import ActionSnapshot, ActionSource


class Strategy(Protocol):
    def merge(self, snapshots: list[ActionSnapshot]) -> np.ndarray:
        ...


class ControllerMux:
    def __init__(self, sources: list[ActionSource], strategy: Strategy) -> None:
        self.sources = list(sources)
        self.strategy = strategy
        self._latest: np.ndarray | None = None

    def start(self) -> None:
        for s in self.sources:
            s.start()

    def stop(self) -> None:
        for s in self.sources:
            s.stop()

    def tick(self) -> np.ndarray:
        snapshots: list[ActionSnapshot] = []
        for s in self.sources:
            snap = s.latest()
            if snap is not None:
                snapshots.append(snap)
        action = self.strategy.merge(snapshots)
        self._latest = action
        return action

    def latest(self) -> np.ndarray | None:
        return self._latest

    def __enter__(self) -> ControllerMux:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()
