# nxml-core

Workspace-internal core: protocols, registry, self-describing checkpoint
format, and model-URI resolver. Torch-free until you save or load a
checkpoint.

## Surface

- **`save_checkpoint` / `load_checkpoint`** — round-trip a model via a
  payload with `architecture`, `config`, `state_dict`, and `nxml_version`.
  Pair with a `Registry` so any consumer can reconstruct the model class
  from the checkpoint alone.
- **`Registry`** — generic name → class dispatcher with a paired config
  class. Used by `nxwm` for world-model architectures and by `nxrl` for
  policies and algorithms.
- **`WorldModel` Protocol** — architecture-agnostic interface:
  `init_rollout_state` / `step_rollout` / `update_goal` plus
  `latent_shape` / `context_length` / `action_dims`. Lets `nxrl`'s PPO
  rollouts drive any registered `nxwm` world model.
- **`resolve_model_uri`** — accepts `hf:owner/repo/path/to/file.pt`,
  `file:./local/path`, or a bare path. Returns a local `pathlib.Path`
  (downloads from HF if needed).

## Usage

```python
from pathlib import Path

from nxml_core import load_checkpoint, resolve_model_uri, save_checkpoint
from nxwm.core import architecture_registry

# Save (in a trainer).
save_checkpoint(
    architecture="dit_v1",
    config=my_config_dataclass,
    state_dict=model.state_dict(),
    path=Path("checkpoints/best.pt"),
)

# Load (in a server / eval / UI).
model, config, ckpt = load_checkpoint(
    resolve_model_uri("hf:arelius/nxwm-pokemon-za/run_021_best.pt"),
    registry=architecture_registry,
)
```
