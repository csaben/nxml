import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "deploy/cradle-ns"))
from dagger_action_plane import ActionPlane
from dagger_control import Mode, MuteMask, Proposal


class Orchestrator:
    def __init__(self):
        self.actions = []

    def post_action(self, action):
        self.actions.append(np.asarray(action))


class Inference:
    def __init__(self):
        self.client = SimpleNamespace(
            revision={"revision_id": "rev-1", "checkpoint_sha256": "a" * 64}
        )
        self.proposal = None

    def latest_proposal(self):
        return self.proposal


def test_arm_is_neutral_and_modes_require_explicit_arm():
    output, inference = Orchestrator(), Inference()
    plane = ActionPlane(output, inference)
    try:
        plane.set_mode(Mode.PURE_AI)
        raise AssertionError("unarmed AI mode was accepted")
    except RuntimeError:
        pass
    plane.arm()
    assert plane.status()["armed"] and plane.status()["mode"] == "human"
    assert not output.actions[-1].any()
    plane.set_mode(Mode.PURE_AI)
    assert plane.status()["mode"] == "pure_ai"
    assert not output.actions[-1].any()  # atomic mode boundary


def test_mute_takeover_release_and_eject_records_complete_history():
    output, inference = Orchestrator(), Inference()
    plane = ActionPlane(output, inference)
    plane.submit_human([0.0] * 26)
    plane.arm()
    plane.set_mute(MuteMask((True,) + (False,) * 25))
    plane.set_mode(Mode.HYBRID)

    takeover = np.zeros(26, np.float32)
    takeover[[4, 5, 24]] = 1
    plane.submit_human(takeover.tolist())
    policy_action = np.zeros(26, np.float32)
    policy_action[[0, 25]] = 1
    now = __import__("time").monotonic_ns()
    inference.proposal = Proposal(policy_action, now, "rev-1", now - 1)
    applied = plane.arbitrator.apply(now + 1, plane._human, inference.proposal)
    plane._append(applied, plane._human, inference.proposal)
    assert applied.takeover and applied.source == "human"
    assert applied.policy[0] == 1 and applied.muted_policy[0] == 0

    released_human = Proposal(np.zeros(26, np.float32), now + 2)
    released = plane.arbitrator.apply(now + 3, released_human, inference.proposal)
    plane._append(released, released_human, inference.proposal)
    assert released.boundary == "takeover_released" and not released.action.any()
    plane.eject()
    assert not plane.status()["armed"] and plane.status()["mode"] == "human"
    assert plane.status()["last_disarm_reason"] == "emergency_eject"
    assert not output.actions[-1].any()


def test_unarmed_inference_failure_cannot_claim_or_emit_action_authority():
    output, inference = Orchestrator(), Inference()
    plane = ActionPlane(output, inference)
    plane.inference_failure("stale remote proposal")
    assert plane.status()["armed"] is False
    assert plane.status()["mode"] == "human"
    assert output.actions == []
