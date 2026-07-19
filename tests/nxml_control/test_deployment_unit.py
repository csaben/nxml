from __future__ import annotations

from pathlib import Path


def test_production_unit_keeps_python_worker_threads_available() -> None:
    unit = (Path(__file__).parents[2] / "apps/nxml-control/deploy/nxml-control.service").read_text()

    assert "MemoryDenyWriteExecute=true" not in unit
    assert "NoNewPrivileges=true" in unit
    assert "ProtectSystem=strict" in unit
    assert "ProtectHome=true" in unit
    assert "ReadWritePaths=/var/lib/nxml-control /var/log/nxml-control" in unit
