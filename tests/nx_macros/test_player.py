"""MacroPlayer behavior — calls poster correctly, respects loop / stop."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from nx_macros import MacroPlayer, macro_from_arrays
from nx_packets import ACTION_DIM


def _silent_macro(n: int, dt: float, name: str = "m"):
    return macro_from_arrays(
        name=name,
        tick_hz=1 / dt if dt > 0 else 30.0,
        actions=np.arange(n * ACTION_DIM, dtype=np.float32).reshape(n, ACTION_DIM),
        dts=[0.0] + [dt] * (n - 1) if n > 0 else [],
    )


def test_player_blocking_calls_poster_per_frame():
    macro = _silent_macro(5, dt=0.0)
    seen: list[np.ndarray] = []
    MacroPlayer(poster=lambda a: seen.append(a.copy())).play(macro)
    assert len(seen) == 5
    np.testing.assert_array_equal(seen[0], macro.actions()[0])
    np.testing.assert_array_equal(seen[-1], macro.actions()[-1])


def test_player_empty_macro_is_noop():
    macro = _silent_macro(0, dt=0.0)
    seen: list[np.ndarray] = []
    MacroPlayer(poster=lambda a: seen.append(a)).play(macro)
    assert seen == []


def test_player_async_runs_and_completes():
    macro = _silent_macro(3, dt=0.01)
    seen: list[np.ndarray] = []
    p = MacroPlayer(poster=lambda a: seen.append(a))
    p.play_async(macro)
    deadline = time.time() + 1.0
    while p.is_playing and time.time() < deadline:
        time.sleep(0.005)
    assert not p.is_playing
    assert len(seen) == 3


def test_player_async_loop_then_stop():
    macro = _silent_macro(3, dt=0.005)
    counter = {"n": 0}
    barrier = threading.Event()

    def poster(_a):
        counter["n"] += 1
        if counter["n"] >= 5:
            barrier.set()

    p = MacroPlayer(poster=poster)
    p.play_async(macro, loop=True)
    assert barrier.wait(2.0), "loop never accumulated enough frames"
    p.stop()
    assert not p.is_playing
    # Sanity: we got more than one full pass through the macro.
    assert counter["n"] >= 5


def test_player_async_double_start_raises():
    macro = _silent_macro(3, dt=0.05)
    p = MacroPlayer(poster=lambda _a: None)
    p.play_async(macro)
    try:
        with pytest.raises(RuntimeError):
            p.play_async(macro)
    finally:
        p.stop()


def test_player_poster_exception_does_not_crash_thread():
    macro = _silent_macro(3, dt=0.0)
    calls = {"n": 0}

    def poster(_a):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated")

    MacroPlayer(poster=poster).play(macro)
    assert calls["n"] == 3  # all frames attempted despite the middle failure
