# nx-macros

Record and replay 26-dim Switch action macros.

A **macro** is a list of `(action_vector, dt)` frames, where `action_vector`
is the canonical 26-dim float vector from `nx-packets`. Macros capture
exactly what the orchestrator sees on `POST /action`, so replaying through
the same endpoint reproduces the original input stream.

## Why a separate package

- The orchestrator stays dumb (it already accepts 26-dim vectors; no new
  endpoint needed for replay).
- Recording is a tee on the action loop in whichever app produces actions
  (`nxml-autopilot` today, others later) — not orchestrator state.
- Macros are JSON, in a directory, with no DB. `MacroStore(root)` is the
  whole index.

## Usage

```python
import time
import numpy as np
from nx_macros import Macro, MacroPlayer, MacroRecorder, MacroStore

# Record
rec = MacroRecorder(tick_hz=30.0)
rec.start("walk-and-jump")
for _ in range(60):
    rec.append(action=np.zeros(26, dtype=np.float32), t=time.time())
    time.sleep(1 / 30)
macro = rec.stop()

# Save / load
store = MacroStore(root="./data/macros/pokemon-za")
store.save(macro)
loaded = store.load("walk-and-jump")

# Replay through whatever poster you have (e.g. autopilot's `_post_action`).
def poster(action: np.ndarray) -> None:
    ...  # POST to /action, or call a mux source, or whatever

player = MacroPlayer(poster=poster)
player.play_async(loaded, loop=True)
# ... user does something else ...
player.stop()
```

## Format

`<name>.json` under the store root:

```json
{
  "name": "walk-and-jump",
  "tick_hz": 30.0,
  "metadata": {"game": "pokemon-za", "recorded_at": "2026-05-10T12:34:56"},
  "frames": [
    {"action": [0.0, 0.0, ..., 1.0], "dt": 0.0},
    {"action": [0.0, 0.0, ..., 1.0], "dt": 0.0334},
    ...
  ]
}
```

`dt` is wall-clock seconds since the previous frame. The first frame's `dt`
is `0.0`. `tick_hz` is informational — the player uses `dt` for timing.

## Bridging to `nx-packets`

`Macro.to_packet_data()` returns a `nx_packets.PacketData` for tools
that consume the structured `Packet` form (e.g. nxbt's own macro DSL).
