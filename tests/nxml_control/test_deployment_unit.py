from __future__ import annotations

from pathlib import Path


def test_production_unit_keeps_python_worker_threads_available() -> None:
    unit = (Path(__file__).parents[2] / "apps/nxml-control/deploy/nxml-control.service").read_text()

    assert "MemoryDenyWriteExecute=true" not in unit
    assert "NoNewPrivileges=true" in unit
    assert "ProtectSystem=strict" in unit
    assert "ProtectHome=true" in unit
    assert "ReadWritePaths=/var/lib/nxml-control /var/log/nxml-control" in unit


def test_inference_unit_is_tailnet_bound_and_gpu0_only() -> None:
    unit = (Path(__file__).parents[2] / "apps/nxml-control/deploy/nxml-inference.service").read_text()

    assert "Environment=CUDA_VISIBLE_DEVICES=0" in unit
    assert "--device cuda:0" in unit
    assert "--host 100.80.98.4" in unit
    assert "--port 5557" in unit
    assert "--revision-id ${NXML_INFERENCE_REVISION_ID}" in unit
    assert "--checkpoint-dir /var/lib/nxml-control/checkpoints" in unit
    assert "NoNewPrivileges=true" in unit
    assert "ProtectSystem=strict" in unit
