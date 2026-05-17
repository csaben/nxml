"""Smoke tests for the nxwm CLI.

Two things being verified:
  1. ``nxwm --help`` is fast and torch-free (lazy imports).
  2. ``nxwm rollout --format npz`` runs end-to-end against the tiny migrated
     checkpoint and the npz fixture, writing a non-empty output file.

We use ``subprocess`` rather than calling click.testing.CliRunner so that the
``--help`` test sees the *real* import cost the user would see.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def _run_nxwm(*args: str, env: dict[str, str] | None = None, timeout: float = 15) -> subprocess.CompletedProcess:
    """Invoke the installed ``nxwm`` console script."""
    full_env = {**os.environ, **(env or {})}
    full_env.setdefault("CUDA_VISIBLE_DEVICES", "")
    return subprocess.run(
        ["uv", "run", "--no-sync", "nxwm", *args],
        capture_output=True,
        timeout=timeout,
        check=False,
        cwd=WORKSPACE_ROOT,
        env=full_env,
    )


def test_help_is_fast_and_torch_free():
    """``nxwm --help`` should not import torch (lazy imports keep startup snappy)."""
    t0 = time.perf_counter()
    result = _run_nxwm("--help", timeout=10)
    elapsed = time.perf_counter() - t0
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    out = result.stdout.decode()
    for sub in ("train", "rollout", "serve", "ui"):
        assert sub in out, f"missing subcommand in help: {sub}\n{out}"
    # No CUDA-init warnings: sign that torch wasn't imported during --help.
    err = result.stderr.decode().lower()
    assert "cuda" not in err, f"unexpected CUDA-related stderr:\n{result.stderr!r}"
    assert elapsed < 5.0, f"--help took {elapsed:.2f}s (expected < 5s)"


def test_help_does_not_import_torch():
    """Belt-and-suspenders: import nxwm.cli in a fresh interpreter, check sys.modules."""
    code = (
        "import sys; "
        "assert 'torch' not in sys.modules, list(sys.modules)[:5]; "
        "import nxwm.cli; "
        "assert 'torch' not in sys.modules, [m for m in sys.modules if 'torch' in m]; "
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        timeout=10,
        check=False,
        cwd=WORKSPACE_ROOT,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_subcommand_help_lists_options():
    result = _run_nxwm("rollout", "--help", timeout=10)
    assert result.returncode == 0
    out = result.stdout.decode()
    for opt in ("--model", "--seed-episode", "--steps", "--output", "--format"):
        assert opt in out


def test_rollout_npz_smoke(tiny_dit_v1_checkpoint: Path, tiny_episode_path: Path, tmp_path: Path):
    """End-to-end: nxwm rollout --format npz produces a non-empty output."""
    out = tmp_path / "rollout.npz"
    result = _run_nxwm(
        "rollout",
        "--model", str(tiny_dit_v1_checkpoint),
        "--seed-episode", str(tiny_episode_path),
        "--start-frame", "0",
        "--steps", "3",
        "--output", str(out),
        "--format", "npz",
        "--device", "cpu",
        "--flow-steps", "2",
        timeout=120,
    )
    assert result.returncode == 0, (
        f"rollout failed:\nstdout={result.stdout.decode(errors='replace')}\n"
        f"stderr={result.stderr.decode(errors='replace')}"
    )
    assert out.exists()

    data = np.load(out)
    assert "latents" in data and "actions" in data
    assert data["latents"].shape[0] == 3
    assert data["actions"].shape == (3, 26)


def test_rollout_npz_with_remote_uri_rejected(tmp_path: Path):
    """--format npz requires latent access; remote URI should be rejected with a clear error."""
    result = _run_nxwm(
        "rollout",
        "--model", "zmq://127.0.0.1:1",
        "--seed-episode", str(WORKSPACE_ROOT / "tests/fixtures/tiny_episode.npz"),
        "--start-frame", "0",
        "--steps", "2",
        "--output", str(tmp_path / "x.npz"),
        "--format", "npz",
        timeout=15,
    )
    assert result.returncode != 0
    err = result.stderr.decode().lower()
    assert "npz" in err and "latent" in err


@pytest.mark.parametrize("subcmd", ["train", "rollout", "serve", "ui"])
def test_each_subcommand_help_works(subcmd: str):
    result = _run_nxwm(subcmd, "--help", timeout=10)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
