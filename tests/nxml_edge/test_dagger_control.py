import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "deploy/cradle-ns"))
from dagger_control import Arbitrator, Mode, MuteMask, Proposal


def p(*, at=100, revision=None, dims=()):
    a = np.zeros(26, np.float32)
    for i in dims:
        a[i] = 1
    return Proposal(a, at, revision)


def test_modes_mute_and_full_packet_takeover_release_boundary():
    a = Arbitrator()
    a.transition(mode=Mode.HYBRID, mute=MuteMask((True,) + (False,) * 25))
    policy = p(revision="immutable-r1", dims=(0, 25))
    out = a.apply(110, p(dims=()), policy)
    assert out.source == "policy" and out.action[0] == 0 and out.action[25] == 1
    out = a.apply(111, p(dims=(4, 5)), policy)
    assert out.source == "human" and set(out.ownership) == {1}
    out = a.apply(112, p(dims=()), policy)
    assert out.source == "none" and out.boundary == "takeover_released"
    assert a.apply(113, p(), policy).source == "policy"


def test_eject_and_stale_policy_neutral_disarm():
    a = Arbitrator(stale_ns=5)
    a.transition(mode=Mode.PURE_AI)
    assert a.apply(100, None, p(at=0), eject=False).disarmed
    e = a.apply(101, None, p(at=101), eject=True)
    assert e.disarmed and e.boundary == "emergency_eject" and not e.action.any()


def test_configuration_change_is_atomic_neutral_boundary():
    a = Arbitrator()
    out = a.transition(mode=Mode.PURE_AI)
    assert out.boundary == "configuration_changed" and not out.action.any()
