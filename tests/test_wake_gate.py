"""WakeGate: cross-room arbitration and segment admission (pure, explicit time)."""

from __future__ import annotations

import pytest

from glados.core.wake import (
    Admission,
    Drop,
    SegmentFacts,
    WakeGate,
    WakeHit,
    WakeTimings,
    strip_wake_phrase,
)


def _gate(modes: dict[str, str] | None = None, gains: dict[str, float] | None = None) -> WakeGate:
    modes = modes or {}
    gains = gains or {}
    return WakeGate(
        timings=WakeTimings(armed_window_s=8.0, followup_window_s=6.0, dedup_window_s=0.5),
        mode_for_room=lambda room: modes.get(room, "required"),
        gain_for_room=lambda room: gains.get(room, 0.0),
    )


def _hit(client: str, room: str, at: float, energy: float = -30.0, score: float = 0.8) -> WakeHit:
    return WakeHit(client_id=client, room_id=room, score=score, energy_dbfs=energy, at=at)


def _seg(client: str, room: str, start: float, end: float, *, in_flight: bool = False,
         confirm: bool = False) -> SegmentFacts:
    return SegmentFacts(client_id=client, room_id=room, start=start, end=end,
                        turn_in_flight=in_flight, confirm_pending=confirm)


def _decide(gate: WakeGate, facts: SegmentFacts, now: float | None = None):
    return gate.decide(facts, facts.end + 1.0 if now is None else now)


def test_unwoken_segment_is_dropped() -> None:
    assert _decide(_gate(), _seg("k", "kitchen", 1.0, 2.0)) == Drop("not_woken")


def test_room_with_wake_off_admits_everything() -> None:
    gate = _gate({"kitchen": "off"})
    assert _decide(gate, _seg("k", "kitchen", 1.0, 2.0)) == Admission("wake_off")


def test_off_room_hits_do_not_enter_arbitration() -> None:
    gate = _gate({"living": "off"})
    assert gate.offer(_hit("l", "living", 1.0, energy=0.0), now=1.0) is None
    gate.offer(_hit("k", "kitchen", 1.1, energy=-40.0), now=1.1)
    assert gate.settle(2.0).room_id == "kitchen"


def test_first_hit_opens_window_and_returns_deadline() -> None:
    gate = _gate()
    assert gate.offer(_hit("k", "kitchen", 10.0), now=10.0) == pytest.approx(10.5)
    assert gate.offer(_hit("l", "living", 10.2), now=10.2) is None


def test_hit_inside_segment_admits_one_breath() -> None:
    gate = _gate()
    gate.offer(_hit("k", "kitchen", 1.5), now=1.5)
    assert _decide(gate, _seg("k", "kitchen", 1.0, 3.0)) == Admission("one_breath")


def test_hit_just_after_vad_end_still_counts_inside() -> None:
    gate = _gate()
    gate.offer(_hit("k", "kitchen", 3.2), now=3.2)
    assert _decide(gate, _seg("k", "kitchen", 1.0, 3.0)) == Admission("one_breath")


def test_segment_starting_inside_armed_window_is_admitted() -> None:
    gate = _gate()
    gate.offer(_hit("k", "kitchen", 1.0), now=1.0)
    assert _decide(gate, _seg("k", "kitchen", 8.5, 10.0)) == Admission("armed")


def test_segment_starting_after_armed_window_is_dropped() -> None:
    gate = _gate()
    gate.offer(_hit("k", "kitchen", 1.0), now=1.0)
    assert _decide(gate, _seg("k", "kitchen", 9.5, 10.0)) == Drop("not_woken")


def test_segment_that_started_before_the_wake_without_containing_it_is_dropped() -> None:
    gate = _gate()
    gate.offer(_hit("k", "kitchen", 5.0), now=5.0)
    assert _decide(gate, _seg("k", "kitchen", 1.0, 2.0), now=6.0) == Drop("not_woken")


def test_loudest_hit_wins_and_loser_room_drops_its_copy() -> None:
    gate = _gate()
    gate.offer(_hit("k", "kitchen", 1.0, energy=-20.0), now=1.0)
    gate.offer(_hit("l", "living", 1.2, energy=-35.0), now=1.2)
    assert gate.settle(1.5).room_id == "kitchen"
    assert _decide(gate, _seg("l", "living", 0.5, 2.0)) == Drop("lost_arbitration")
    assert _decide(gate, _seg("k", "kitchen", 0.5, 2.0)) == Admission("one_breath")


def test_room_gain_offset_shifts_the_winner() -> None:
    gate = _gate(gains={"living": 20.0})
    gate.offer(_hit("k", "kitchen", 1.0, energy=-20.0), now=1.0)
    gate.offer(_hit("l", "living", 1.1, energy=-35.0), now=1.1)
    assert gate.settle(2.0).room_id == "living"


def test_score_breaks_energy_tie() -> None:
    gate = _gate()
    gate.offer(_hit("k", "kitchen", 1.0, score=0.6), now=1.0)
    gate.offer(_hit("l", "living", 1.1, score=0.9), now=1.1)
    assert gate.settle(2.0).room_id == "living"


def test_loser_room_follow_up_window_cannot_rescue_its_copy() -> None:
    gate = _gate()
    gate.turn_ended("living", at=0.0)
    gate.offer(_hit("k", "kitchen", 1.0, energy=-20.0), now=1.0)
    gate.offer(_hit("l", "living", 1.1, energy=-40.0), now=1.1)
    assert _decide(gate, _seg("l", "living", 0.8, 2.0)) == Drop("lost_arbitration")


def test_hit_exactly_at_deadline_starts_a_new_window() -> None:
    gate = _gate()
    gate.offer(_hit("k", "kitchen", 1.0), now=1.0)
    assert gate.offer(_hit("l", "living", 1.5), now=1.5) == pytest.approx(2.0)
    assert _decide(gate, _seg("k", "kitchen", 0.8, 1.2), now=1.6) == Admission("one_breath")


def test_repeated_hits_from_one_client_keep_the_best() -> None:
    gate = _gate()
    gate.offer(_hit("k", "kitchen", 1.0, score=0.6), now=1.0)
    gate.offer(_hit("k", "kitchen", 1.1, score=0.9), now=1.1)
    winner = gate.settle(2.0)
    assert winner.score == 0.9


def test_winning_mic_owns_the_room_against_a_second_mic() -> None:
    gate = _gate()
    gate.offer(_hit("k1", "kitchen", 1.0, energy=-20.0), now=1.0)
    gate.offer(_hit("k2", "kitchen", 1.1, energy=-30.0), now=1.1)
    gate.settle(2.0)
    assert _decide(gate, _seg("k2", "kitchen", 3.0, 4.0)) == Drop("not_owner")
    assert _decide(gate, _seg("k1", "kitchen", 3.0, 4.0)) == Admission("armed")


def test_follow_up_after_turn_end_admits_without_wake() -> None:
    gate = _gate()
    gate.turn_ended("kitchen", at=20.0)
    assert _decide(gate, _seg("k", "kitchen", 25.0, 26.0)) == Admission("follow_up")
    assert _decide(gate, _seg("k", "kitchen", 26.5, 27.0)) == Drop("not_woken")


def test_turn_end_extends_ownership_through_follow_up() -> None:
    gate = _gate()
    gate.offer(_hit("k1", "kitchen", 1.0, energy=-20.0), now=1.0)
    gate.offer(_hit("k2", "kitchen", 1.1, energy=-30.0), now=1.1)
    gate.settle(2.0)
    gate.turn_ended("kitchen", at=15.0)
    assert _decide(gate, _seg("k2", "kitchen", 18.0, 19.0)) == Drop("not_owner")
    assert _decide(gate, _seg("k1", "kitchen", 18.0, 19.0)) == Admission("follow_up")


def test_confirm_pending_admits_without_wake() -> None:
    facts = _seg("k", "kitchen", 1.0, 2.0, confirm=True)
    assert _decide(_gate(), facts) == Admission("confirm")


def test_turn_in_flight_admits_for_barge_in_only() -> None:
    facts = _seg("k", "kitchen", 1.0, 2.0, in_flight=True)
    assert _decide(_gate(), facts) == Admission("barge_in_only")


def test_faulted_detector_fails_closed_but_keeps_barge_in() -> None:
    gate = _gate()
    gate.turn_ended("kitchen", at=0.5)
    gate.fault("k", "kitchen")
    assert _decide(gate, _seg("k", "kitchen", 1.0, 2.0)) == Drop("detector_fault")
    in_flight = _seg("k", "kitchen", 1.0, 2.0, in_flight=True)
    assert _decide(gate, in_flight) == Admission("barge_in_only")


def test_pending_deadline_reports_an_open_window_covering_the_segment() -> None:
    gate = _gate()
    gate.offer(_hit("k", "kitchen", 2.0), now=2.0)
    assert gate.pending_deadline(1.0, 2.1) == pytest.approx(2.5)
    assert gate.pending_deadline(0.0, 1.0) is None


def test_forget_client_drops_its_pending_hit() -> None:
    gate = _gate()
    gate.offer(_hit("k", "kitchen", 1.0), now=1.0)
    gate.forget_client("k")
    assert gate.settle(2.0) is None


def test_old_wins_are_pruned() -> None:
    gate = _gate()
    gate.offer(_hit("k", "kitchen", 1.0), now=1.0)
    gate.settle(2.0)
    assert _decide(gate, _seg("k", "kitchen", 0.9, 1.2), now=100.0) == Drop("not_woken")


@pytest.mark.parametrize(
    "heard, expected",
    [
        ("Hey GLaDOS, what time is it?", "what time is it?"),
        ("hey gladys what time is it", "what time is it"),
        ("Hey, Glad OS. Turn it up", "Turn it up"),
        ("HeyGLaDOS stop", "stop"),
        ("Hey GLaDOS.", ""),
        ("hey jarvis", ""),
        ("what time is it", "what time is it"),
        ("Hey Gladiator, hi", "Hey Gladiator, hi"),
    ],
)
def test_strip_wake_phrase(heard: str, expected: str) -> None:
    aliases = ["hey glados", "hey gladys", "hey jarvis"]
    assert strip_wake_phrase(heard, aliases) == expected
