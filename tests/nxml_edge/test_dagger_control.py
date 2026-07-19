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
    a = Arbitrator(stale_ns=5, hard_stall_ns=20)
    a.transition(mode=Mode.PURE_AI)
    gap = a.apply(6, None, p(at=0), eject=False)
    assert not gap.disarmed and not gap.valid and gap.gap_state == "transient_gap"
    assert a.apply(25, None, p(at=0), eject=False).disarmed
    e = a.apply(101, None, p(at=101), eject=True)
    assert e.disarmed and e.boundary == "emergency_eject" and not e.action.any()


def test_configuration_change_is_atomic_neutral_boundary():
    a = Arbitrator()
    out = a.transition(mode=Mode.PURE_AI)
    assert out.boundary == "configuration_changed" and not out.action.any()


def test_policy_hold_neutralizes_one_miss_and_recovers_before_hard_stall():
    a = Arbitrator(stale_ns=55_000_000)
    a.transition(mode=Mode.PURE_AI)
    policy = p(at=1_000_000_000, revision="r1", dims=(25,))
    assert a.apply(1_055_000_000, None, policy).source == "policy"
    gap = a.apply(1_055_000_001, None, policy)
    assert not gap.disarmed and not gap.valid and not gap.action.any()
    recovered = a.apply(1_070_000_000, None, p(at=1_070_000_000, revision="r1", dims=(25,)))
    assert recovered.source == "policy" and recovered.gap_state == "recovered"
    assert recovered.gap_duration_ns == 15_000_000


def test_policy_hard_stall_identity_independent_and_repeated_gap_budget():
    a = Arbitrator(stale_ns=55, hard_stall_ns=250)
    a.transition(mode=Mode.PURE_AI)
    assert not a.apply(100, None, None).disarmed
    assert a.apply(350, None, None).disarmed
    b = Arbitrator(stale_ns=5, hard_stall_ns=1_000, max_gaps_per_window=4)
    b.transition(mode=Mode.PURE_AI)
    for base in (100, 200, 300):
        assert not b.apply(base, None, None).disarmed
        b.apply(base + 1, None, p(at=base + 1, revision="r1"))
    assert b.apply(400, None, None).disarmed


def test_hybrid_takeover_remains_immediate_during_policy_gap():
    a = Arbitrator(stale_ns=5, hard_stall_ns=250)
    a.transition(mode=Mode.HYBRID)
    out = a.apply(100, p(at=100, dims=(4, 5, 25)), None)
    assert out.source == "human" and out.takeover and out.action[25] == 1


def test_human_mode_observes_gap_and_hard_stall_without_policy_ownership():
    a = Arbitrator(stale_ns=55, hard_stall_ns=250)
    policy = p(at=100, revision="r1")
    assert a.observe_policy(155, policy) == ("none", None, 0, False)
    assert a.observe_policy(156, policy) == (
        "transient_gap",
        "policy_transient_gap",
        1,
        False,
    )
    assert a.observe_policy(405, policy) == ("disarmed", "policy_stall", 250, True)
