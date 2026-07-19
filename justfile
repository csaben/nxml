# Justfile for the nxml workspace

# Sync all packages in the workspace (editable installs) + all optional extras.
sync:
    uv sync --all-packages --all-extras

# Run the full pytest suite across all packages.
test:
    uv run pytest

# Smoke train: 2 epochs of the tiny dit_v1 config (~60 s on a small GPU).
train-tiny:
    uv run python -m nxwm.training.launcher configs/wm/wm_dit_v1_tiny.yaml --world-size 1

# cradle-ns Switch stack (Bluetooth orchestrator + edge UI). Host-local only.
ns-up:
    deploy/cradle-ns/nsctl up

ns-down:
    deploy/cradle-ns/nsctl down

ns-status:
    deploy/cradle-ns/nsctl status

ns-logs:
    deploy/cradle-ns/nsctl logs
