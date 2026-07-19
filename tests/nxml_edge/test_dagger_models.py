import hashlib
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "deploy/cradle-ns"))
from dagger_models import AtomicModelRuntime, RevisionCache, check_compatibility


def revision(path, digest, **compat):
    return {
        "revision_id": "r1",
        "state": "validated",
        "checkpoint_sha256": digest,
        "artifact_id": "sha256:" + digest,
        "artifact_uri": "/v1/models/revisions/r1/artifacts/sha256:" + digest,
        "compatibility": {
            "architecture": "bc_transformer_v1",
            "action_spec_id": "switch_packets.v1",
            "action_dim": 26,
            "latent_shape": [4, 16, 32],
            "sequence_length": 32,
            "vae_profile": "sd-vae-ft-mse.rgb-bilinear-128x256.mode.scale-0.18215.v1",
            **compat,
        },
    }


def test_digest_cache_compatibility_and_atomic_load(tmp_path, monkeypatch):
    src = tmp_path / "model.pt"
    src.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    r = revision(src, digest)
    requests = []

    def open_artifact(request, timeout):
        requests.append(request)
        return io.BytesIO(b"weights")

    monkeypatch.setattr("urllib.request.urlopen", open_artifact)
    runtime = AtomicModelRuntime(
        RevisionCache(tmp_path / "cache", control_url="http://control", token="secret"),
        lambda p: p.read_bytes(),
    )
    runtime.load_async(r)
    for _ in range(100):
        if runtime.state().loading is None:
            break
        time.sleep(0.01)
    assert runtime.state().active == "r1" and runtime.state().armed is False
    assert (tmp_path / "cache" / digest).read_bytes() == b"weights"
    assert requests[0].full_url.endswith(r["artifact_uri"])
    assert requests[0].get_header("Authorization") == "Bearer secret"
    second = {
        **r,
        "revision_id": "r2",
        "artifact_uri": f"/v1/models/revisions/r2/artifacts/sha256:{digest}",
    }
    runtime.load_async(second)
    for _ in range(100):
        if runtime.state().loading is None:
            break
        time.sleep(0.01)
    runtime.rollback()
    assert runtime.state().active == "r1" and runtime.state().previous == "r2"


def test_bad_digest_and_incompatible_revision_never_arm(tmp_path, monkeypatch):
    src = tmp_path / "bad"
    src.write_bytes(b"bad")
    r = revision(src, "0" * 64)
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: io.BytesIO(b"bad"))
    runtime = AtomicModelRuntime(
        RevisionCache(tmp_path / "cache", control_url="http://control"), lambda p: p
    )
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


def test_checkpoint_path_cannot_bypass_immutable_artifact_route(tmp_path):
    digest = hashlib.sha256(b"weights").hexdigest()
    item = revision(tmp_path / "model", digest)
    item.pop("artifact_uri")
    try:
        RevisionCache(tmp_path / "cache").acquire(item)
    except ValueError as error:
        assert "immutable artifact identity" in str(error)
    else:
        raise AssertionError("checkpoint_path bypass was accepted")


def test_unvalidated_revision_is_never_downloaded(tmp_path, monkeypatch):
    digest = hashlib.sha256(b"weights").hexdigest()
    item = {**revision(tmp_path / "model", digest), "state": "candidate"}
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError())
    )
    runtime = AtomicModelRuntime(
        RevisionCache(tmp_path / "cache", control_url="http://control"), lambda path: path
    )
    try:
        runtime.load_async(item)
    except ValueError as error:
        assert "cluster-validated" in str(error)
    else:
        raise AssertionError("candidate was accepted")
    assert runtime.state().active is None


def test_failed_candidate_preserves_previous_ready_handle(tmp_path, monkeypatch):
    good = b"good"
    digest = hashlib.sha256(good).hexdigest()
    first = revision(tmp_path / "first", digest)
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: io.BytesIO(good))
    runtime = AtomicModelRuntime(
        RevisionCache(tmp_path / "cache", control_url="http://control"), lambda path: path
    )
    runtime.load_async(first)
    for _ in range(100):
        if runtime.state().loading is None:
            break
        time.sleep(0.01)
    broken_digest = hashlib.sha256(b"broken").hexdigest()
    broken = revision(tmp_path / "broken", broken_digest)
    broken["revision_id"] = "r-broken"
    broken["artifact_uri"] = f"/v1/models/revisions/r-broken/artifacts/sha256:{broken_digest}"
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: io.BytesIO(b"wrong"))
    runtime.load_async(broken)
    for _ in range(100):
        if runtime.state().loading is None:
            break
        time.sleep(0.01)
    state = runtime.state()
    assert state.active == "r1" and state.ready is True and state.error
    assert runtime.active_handle()[1] is not None
