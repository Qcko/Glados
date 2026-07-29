"""Unit tests for the handshake admission gate (caps + per-IP lockout).

Pure-sync tests against an injectable clock -- no sockets, no sleeps. The
wire-level behaviour (error codes, close, recovery through a real WS) is in
test_server_handshake.py.
"""

from __future__ import annotations

from glados.core.config import HandshakeConfig
from glados.core.handshake_gate import FailureOutcome, HandshakeGate, Verdict


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_gate(**overrides) -> tuple[HandshakeGate, FakeClock]:
    cfg = HandshakeConfig(
        max_pending=3,
        max_pending_per_ip=2,
        fail_threshold=3,
        fail_window_s=60.0,
        lockout_s=30.0,
        max_tracked_ips=4,
        **overrides,
    )
    clock = FakeClock()
    return HandshakeGate(cfg, clock=clock), clock


def lock_out(gate: HandshakeGate, ip: str, threshold: int = 3) -> None:
    for _ in range(threshold):
        gate.record_failure(ip)


# ---- pending caps -------------------------------------------------------


def test_admit_and_release_roundtrip() -> None:
    gate, _ = make_gate()
    assert gate.admit("10.0.0.1") is Verdict.OK
    assert gate.pending_count == 1
    gate.release("10.0.0.1")
    assert gate.pending_count == 0


def test_global_cap_rejects_overflow() -> None:
    gate, _ = make_gate()
    for ip in ("10.0.0.1", "10.0.0.2", "10.0.0.3"):
        assert gate.admit(ip) is Verdict.OK
    assert gate.admit("10.0.0.4") is Verdict.BUSY


def test_per_ip_cap_rejects_third_slot_from_one_peer() -> None:
    gate, _ = make_gate()
    assert gate.admit("10.0.0.1") is Verdict.OK
    assert gate.admit("10.0.0.1") is Verdict.OK
    assert gate.admit("10.0.0.1") is Verdict.BUSY
    # A different peer still gets the remaining global slot.
    assert gate.admit("10.0.0.2") is Verdict.OK


def test_record_failure_reports_each_transition() -> None:
    # The gate never logs; the server narrates from these outcomes.
    gate, _ = make_gate()  # fail_threshold=3
    assert gate.record_failure("10.0.0.5") is FailureOutcome.COUNTED
    assert gate.record_failure("10.0.0.5") is FailureOutcome.COUNTED
    assert gate.record_failure("10.0.0.5") is FailureOutcome.LOCKOUT_ENGAGED
    assert gate.record_failure("10.0.0.5") is FailureOutcome.ALREADY_LOCKED


def test_peek_mirrors_admit_without_taking_a_slot() -> None:
    gate, _ = make_gate()
    assert gate.peek("10.0.0.1") is Verdict.OK
    assert gate.pending_count == 0  # peek never takes a slot
    for ip in ("10.0.0.1", "10.0.0.2", "10.0.0.3"):
        gate.admit(ip)
    assert gate.peek("10.0.0.4") is Verdict.BUSY
    lock_out(gate, "10.0.0.9")
    # Lockout outranks capacity (mirrors admit), so a locked-out peer is
    # routed to the explanatory post-accept reject even when caps are full.
    assert gate.peek("10.0.0.9") is Verdict.LOCKED_OUT
    assert gate.pending_count == 3


def test_release_frees_both_caps() -> None:
    gate, _ = make_gate()
    gate.admit("10.0.0.1")
    gate.admit("10.0.0.1")
    gate.release("10.0.0.1")
    assert gate.admit("10.0.0.1") is Verdict.OK


# ---- failure lockout ----------------------------------------------------


def test_failures_below_threshold_do_not_lock() -> None:
    gate, _ = make_gate()
    gate.record_failure("10.0.0.1")
    gate.record_failure("10.0.0.1")
    assert gate.admit("10.0.0.1") is Verdict.OK


def test_threshold_failures_in_window_lock_the_ip_only() -> None:
    gate, _ = make_gate()
    lock_out(gate, "10.0.0.1")
    assert gate.admit("10.0.0.1") is Verdict.LOCKED_OUT
    assert gate.admit("10.0.0.2") is Verdict.OK


def test_failures_outside_window_do_not_accumulate() -> None:
    gate, clock = make_gate()
    gate.record_failure("10.0.0.1")
    gate.record_failure("10.0.0.1")
    clock.now += 61.0
    gate.record_failure("10.0.0.1")
    assert gate.admit("10.0.0.1") is Verdict.OK


def test_lockout_expires_after_lockout_s() -> None:
    gate, clock = make_gate()
    lock_out(gate, "10.0.0.1")
    clock.now += 29.0
    assert gate.admit("10.0.0.1") is Verdict.LOCKED_OUT
    clock.now += 1.0
    assert gate.admit("10.0.0.1") is Verdict.OK


def test_failures_during_lockout_do_not_extend_it() -> None:
    gate, clock = make_gate()
    lock_out(gate, "10.0.0.1")
    clock.now += 15.0
    gate.record_failure("10.0.0.1")
    clock.now += 15.0
    assert gate.admit("10.0.0.1") is Verdict.OK


def test_success_clears_failure_history() -> None:
    gate, _ = make_gate()
    gate.record_failure("10.0.0.1")
    gate.record_failure("10.0.0.1")
    gate.record_success("10.0.0.1")
    gate.record_failure("10.0.0.1")
    gate.record_failure("10.0.0.1")
    assert gate.admit("10.0.0.1") is Verdict.OK


def test_failure_straddling_lockout_expiry_starts_fresh_count() -> None:
    # A failure arriving just after the lockout expires must start a fresh
    # count (the failures that engaged the lockout were consumed by it),
    # not instantly re-lock.
    gate, clock = make_gate()
    lock_out(gate, "10.0.0.1")
    clock.now += 31.0
    gate.record_failure("10.0.0.1")
    assert gate.admit("10.0.0.1") is Verdict.OK


def test_release_without_admit_cannot_widen_the_cap() -> None:
    gate, _ = make_gate()
    gate.release("10.0.0.1")
    assert gate.pending_count == 0
    for ip in ("10.0.0.1", "10.0.0.2", "10.0.0.3"):
        assert gate.admit(ip) is Verdict.OK
    assert gate.admit("10.0.0.4") is Verdict.BUSY


# ---- tracking-table eviction --------------------------------------------


def test_eviction_prefers_expired_entries() -> None:
    gate, clock = make_gate()
    lock_out(gate, "10.0.0.1")  # locked; must survive eviction
    for ip in ("10.0.0.2", "10.0.0.3", "10.0.0.4"):
        gate.record_failure(ip)
    clock.now += 61.0  # .2-.4's failures expire; .1's lockout has too
    lock_out(gate, "10.0.0.1")  # re-lock with fresh state
    # Five distinct IPs would overflow max_tracked_ips=4; the expired
    # entries must go before any live one.
    gate.record_failure("10.0.0.5")
    gate.record_failure("10.0.0.6")
    assert gate.admit("10.0.0.1") is Verdict.LOCKED_OUT


def test_table_stays_bounded_under_ip_flood() -> None:
    gate, _ = make_gate()
    for i in range(50):
        gate.record_failure(f"10.1.0.{i}")
    assert len(gate._ips) <= 4 + 1  # cap plus the entry being touched
