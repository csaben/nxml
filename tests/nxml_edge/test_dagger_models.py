import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "deploy/cradle-ns"))
from dagger_models import AtomicModelRuntime, RevisionCache, check_compatibility


def revision(path, digest, **compat):
    return {
        "revision_id": "r1",
        "checkpoint_path": str(path),
        "checkpoint_sha256": digest,
        "compatibility": {
            "architecture": "bc_transformer_v1",
            "action_spec_id": "switch_packets.v1",
            "action_dim": 26,
            **compat,
        },
    }


def test_digest_cache_compatibility_and_atomic_load(tmp_path):
    src = tmp_path / "model.pt"
    src.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    r = revision(src, digest)
    runtime = AtomicModelRuntime(RevisionCache(tmp_path / "cache"), lambda p: p.read_bytes())
    runtime.load_async(r)
    for _ in range(100):
        if runtime.state().loading is None:
            break
        time.sleep(0.01)
    assert runtime.state().active == "r1" and runtime.state().armed is False
    assert (tmp_path / "cache" / digest).read_bytes() == b"weights"
    second = {**r, "revision_id": "r2"}
    runtime.load_async(second)
    for _ in range(100):
        if runtime.state().loading is None:
            break
        time.sleep(0.01)
    runtime.rollback()
    assert runtime.state().active == "r1" and runtime.state().previous == "r2"


def test_bad_digest_and_incompatible_revision_never_arm(tmp_path):
    src = tmp_path / "bad"
    src.write_bytes(b"bad")
    r = revision(src, "0" * 64)
    runtime = AtomicModelRuntime(RevisionCache(tmp_path / "cache"), lambda p: p)
    runtime.load_async(r)
    for _ in range(100):
        if runtime.state().loading is None:
            break
        time.sleep(0.01)
    assert runtime.state().active is None and runtime.state().error
    try:
        check_compatibility(revision(src, hashlib.sha256(b"bad").hexdigest(), action_dim=25))
    except ValueError:
        pass
    else:
        raise AssertionError("incompatible revision accepted")
