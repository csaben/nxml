import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "deploy/cradle-ns"))
from dagger_control import Arbitrator, Mode, MuteMask, Proposal
from dagger_history import ArbitrationRecord, ArbitratorHistory


def proposal(at, *dims, revision=None):
    action = np.zeros(26, np.float32)
    action[list(dims)] = 1
    return Proposal(action, at, revision, observation_monotonic_ns=at - 5)


def frame(at):
    return SimpleNamespace(timestamp=10.0, monotonic_ns=at, image=np.zeros((4, 4, 3), np.uint8))


def record(applied, human=None, policy=None, mode=Mode.HUMAN):
    return ArbitrationRecord(
        applied,
        human,
        policy,
        "sha256:d" if policy else None,
        mode,
        applied.takeover,
        policy is not None,
        policy is not None,
    )


def test_history_selects_exact_or_newest_prior_never_future():
    history = ArbitratorHistory(max_age_ns=100)
    arb = Arbitrator(stale_ns=100)
    human1, human2 = proposal(10, 25), proposal(20, 24)
    history.append(record(arb.apply(10, human1, None), human1))
    history.append(record(arb.apply(20, human2, None), human2))

    assert history.pair(frame(20)).action_monotonic_ns == 20
    selected = history.pair(frame(19))
    assert selected.action_monotonic_ns == 10
    assert selected.action_age == 9 / 1e9
    assert selected.human_mask.all()  # neutral values do not erase full human ownership.
    assert not history.pair(frame(9)).valid
    assert history.pair(frame(9)).invalid_reasons == ("no_prior_arbitration_record",)


def test_history_stale_is_invalid_neutral_and_mute_keeps_both_policy_packets():
    history = ArbitratorHistory(max_age_ns=5)
    arb = Arbitrator(stale_ns=100)
    mask = MuteMask((True,) + (False,) * 25)
    arb.transition(mode=Mode.PURE_AI, mute=mask)
    policy = proposal(100, 0, 25, revision="rev")
    applied = arb.apply(101, proposal(101), policy)
    history.append(record(applied, proposal(101), policy, Mode.PURE_AI))

    row = history.pair(frame(102))
    assert row.policy_action[0] == 1 and row.muted_policy_action[0] == 0
    assert row.action[25] == 1 and set(row.ownership) == {2}
    assert row.policy_monotonic_ns == 100
    assert row.policy_observation_monotonic_ns == 95
    stale = history.pair(frame(107))
    assert not stale.valid and not stale.action.any()
    assert stale.invalid_reasons == ("stale_arbitration_record",)


def test_hybrid_takeover_and_release_boundary_are_complete_records():
    history = ArbitratorHistory()
    arb = Arbitrator(stale_ns=100)
    arb.transition(mode=Mode.HYBRID)
    policy = proposal(10, 25, revision="rev")
    takeover = proposal(11, 4, 5)
    first = arb.apply(12, takeover, policy)
    history.append(record(first, takeover, policy, Mode.HYBRID))
    released = arb.apply(13, proposal(13), policy)
    history.append(record(released, proposal(13), policy, Mode.HYBRID))

    assert history.pair(frame(12)).takeover
    boundary = history.pair(frame(13))
    assert not boundary.action.any() and boundary.ownership_source == "none"
    assert boundary.human_mask.all()


def test_human_mode_records_observed_policy_identity_and_muted_proposal():
    history = ArbitratorHistory()
    arb = Arbitrator(stale_ns=100)
    arb.transition(mute=MuteMask((True,) + (False,) * 25))
    human = proposal(10, 24)
    policy = proposal(10, 0, 25, revision="immutable-r1")
    applied = arb.apply(11, human, policy)
    history.append(
        ArbitrationRecord(
            applied,
            human,
            policy,
            "sha256:checkpoint",
            Mode.HUMAN,
            False,
            True,
            True,
        )
    )
    row = history.pair(frame(12))
    assert row.ownership_source == "human" and set(row.ownership) == {1}
    np.testing.assert_array_equal(row.applied_action, human.action)
    np.testing.assert_array_equal(row.policy_action, policy.action)
    assert row.policy_revision == "immutable-r1"
    assert row.policy_digest == "sha256:checkpoint"
    assert row.muted_policy_action[0] == 0 and row.muted_policy_action[25] == 1


def test_neutral_boundary_keeps_observed_policy_provenance_without_ownership():
    history = ArbitratorHistory()
    arb = Arbitrator(stale_ns=100)
    human = proposal(10)
    policy = proposal(10, 25, revision="immutable-r1")
    boundary = arb.transition(mode=Mode.PURE_AI)
    boundary = replace(boundary, monotonic_ns=11)
    history.append(
        ArbitrationRecord(
            boundary,
            human,
            policy,
            "sha256:checkpoint",
            Mode.PURE_AI,
            False,
            True,
            True,
        )
    )
    row = history.pair(frame(12))
    assert row.ownership_source == "none" and not row.applied_action.any()
    assert row.policy_revision == "immutable-r1"
    np.testing.assert_array_equal(row.muted_policy_action, policy.action)


def test_stale_history_row_is_invalid_unowned_and_has_no_fabricated_identity():
    history = ArbitratorHistory(max_age_ns=1)
    arb = Arbitrator(stale_ns=100)
    human = proposal(10)
    policy = proposal(10, 25, revision="immutable-r1")
    history.append(record(arb.apply(10, human, None), human, policy))
    row = history.pair(frame(12))
    assert not row.valid and row.invalid_reasons == ("stale_arbitration_record",)
    assert row.policy_revision is None and row.policy_digest is None
    assert not row.ownership.any() and not row.applied_action.any()
