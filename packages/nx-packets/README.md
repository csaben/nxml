# nx-packets

Canonical Nintendo Switch input types and the **26-dim action contract** used
across the nxml ecosystem.

This package is the single source of truth for two things:

1. **Structured packet types** (`Packet`, `StickData`, `PacketData`) — the
   shape of one frame of controller state, used by emulators and Bluetooth
   senders.
2. **The 26-dim float32 action vector** — the contract between world models,
   policies, UIs, and the orchestrator. See `action_spec.py` for the exact
   layout (indices 0..25, sticks 0..3 continuous, buttons 4..25 binary).

## Why a dedicated package?

- It's the contract that crosses every boundary (nxwm, nxrl,
  nxbt-orchestrator, nxml-mux). Putting it in any one of those would force
  the others to depend on heavy transitive deps (torch, etc.) just to format
  an action.
- The layout is **stable**: existing checkpoints' `action_encoder` weights
  depend on this exact ordering. Any change is a v2 of the action spec and
  needs a migration story. Living in its own small package signals that.
- It's lightweight (`pydantic`, `numpy`) — fits on a Raspberry Pi just to
  format actions for a Bluetooth-only client.

## Usage

```python
import numpy as np
from nx_packets import (
    ACTION_DIM, BUTTON_INDEX,
    Packet, StickData,
    action_to_packet, packet_to_action, neutral_action,
)

# Build an action that presses A only
action = neutral_action()
action[BUTTON_INDEX["A"]] = 1.0

# Convert to a Packet for sending over Bluetooth
packet = action_to_packet(action)
assert packet.A is True

# And back
roundtrip = packet_to_action(packet)
np.testing.assert_array_equal(roundtrip, action)
```

## Auxiliary fields

`StickData` carries `LS_UP`/`LS_LEFT`/`LS_RIGHT`/`LS_DOWN` (and `RS_*`
counterparts). These are **auxiliary** binary signals consumed by downstream
nxbt/switch-emulation tools — they are NOT part of the 26-dim action vector.
Leave them defaulted to `False` when synthesizing actions from a model.
