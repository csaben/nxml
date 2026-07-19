from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "apps/nxml-control/deploy/nxml-stack"


def test_inference_admission_uses_free_vram_floor_without_killing_processes():
    text = SCRIPT.read_text()
    assert "NXML_INFERENCE_MIN_FREE_MIB:-4096" in text
    assert "memory.free" in text
    assert "free >= GPU0_MIN_FREE_MIB" in text
    assert "none will be stopped" in text
    assert "docker stop" not in text
    assert "kill " not in text


def test_training_capacity_is_explicitly_separate():
    text = SCRIPT.read_text()
    assert "training GPU capacity was not assessed or reserved" in text
    assert "no job submitted" in text
    assert "GPU1" in text
    assert "wait_for_readiness" in text
    assert "warming: readiness attempt" in text
    assert "readiness timed out; left running for diagnosis" in text
    assert "trap - ERR INT TERM" in text
