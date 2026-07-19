"""Rank-0 file interface for the live web viewer: status.json + metrics.jsonl + recon GIFs.

The trainer writes; ``nxwm-mira watch`` (and anything else) reads. status.json is rewritten
atomically (tmp + replace) so readers never see a partial file; metrics.jsonl gets one JSON
line per log event.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class LiveStatusWriter:
    def __init__(self, output_dir: str | Path, total_steps: int, run_name: str) -> None:
        self.output_dir = Path(output_dir)
        self.total_steps = total_steps
        self.run_name = run_name
        self.wandb_url: str | None = None
        self.val_episodes: list[int] | None = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.output_dir / "metrics.jsonl"
        self._status_path = self.output_dir / "status.json"

    def log(self, step: int, stats: dict) -> None:
        """Append a metrics line and rewrite status.json."""
        record = {"step": step, "time": time.time(), **_jsonable(stats)}
        with self._metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self._write_status(step, stats)

    def _write_status(self, step: int, stats: dict) -> None:
        latest_recon = self.latest_recon()
        status = {
            "run_name": self.run_name,
            "step": step,
            "total_steps": self.total_steps,
            "wandb_url": self.wandb_url,
            "val_episodes": self.val_episodes,
            "latest_recon": (
                f"recons/{latest_recon.name}" if latest_recon is not None else None
            ),
            "updated_at": time.time(),
            "stats": _jsonable(stats),
        }
        tmp = self._status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(self._status_path)

    def latest_recon(self) -> Path | None:
        latest = self.output_dir / "recons" / "latest.gif"
        return latest if latest.is_file() else None


def _jsonable(stats: dict) -> dict:
    out = {}
    for k, v in stats.items():
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
    return out


def prune_recons(recons_dir: Path, *, keep_every: int, keep_recent: int = 20) -> None:
    """Retention for recon GIFs: keep every ``keep_every``-th viz plus the most recent N."""
    gifs = sorted(recons_dir.glob("step_*.gif"))
    if len(gifs) <= keep_recent:
        return
    for i, gif in enumerate(gifs[:-keep_recent]):
        if keep_every > 0 and i % keep_every == 0:
            continue
        gif.unlink(missing_ok=True)
