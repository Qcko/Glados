"""Deterministic turn-outcome classification (core/turn_outcome.py)."""

from __future__ import annotations

from glados.core.turn_outcome import TurnRecord, classify


def _turn(*, final_text: str = "", loop_exhausted: bool = False) -> TurnRecord:
    return TurnRecord(final_text=final_text, loop_exhausted=loop_exhausted)


def test_no_tools_plain_answer_is_done() -> None:
    turn = _turn(final_text="It is Friday afternoon.")
    assert classify(turn) == "done"


def test_all_tools_ok_is_done() -> None:
    turn = _turn(final_text="Added milk to your cart.")
    turn.record_tool("dunnes.add_to_cart_by_name", ok=True, mutating=True)
    assert classify(turn) == "done"


def test_unrecovered_error_is_failed() -> None:
    turn = _turn(final_text="All set!")
    turn.record_tool("dunnes.view_cart", ok=False)
    assert classify(turn) == "failed"


def test_cheerful_question_after_errors_is_failed_not_needs_user() -> None:
    # Bake-off T6: two tool errors then an upbeat clarifying question.
    # Error priority must win over the question heuristic.
    turn = _turn(final_text="Anything else I can help with?")
    turn.record_tool("dunnes.view_cart", ok=False)
    turn.record_tool("dunnes.remove_from_cart", ok=False)
    assert classify(turn) == "failed"


def test_error_then_later_success_same_tool_is_recovered() -> None:
    turn = _turn(final_text="Done.")
    turn.record_tool("dunnes.view_cart", ok=False)
    turn.record_tool("dunnes.view_cart", ok=True)
    assert classify(turn) == "done"


def test_success_before_error_does_not_recover() -> None:
    turn = _turn(final_text="Done.")
    turn.record_tool("dunnes.view_cart", ok=True)
    turn.record_tool("dunnes.view_cart", ok=False)
    assert classify(turn) == "failed"


def test_recovery_is_per_tool() -> None:
    # search recovers; remove never succeeds -> still failed.
    turn = _turn(final_text="Done.")
    turn.record_tool("dunnes.search_products", ok=False)
    turn.record_tool("dunnes.search_products", ok=True)
    turn.record_tool("dunnes.remove_from_cart", ok=False)
    assert classify(turn) == "failed"


def test_question_with_no_tools_is_needs_user() -> None:
    turn = _turn(final_text="Which store did you mean — Cornelscourt or Bishopstown?")
    assert classify(turn) == "needs-user"


def test_question_after_successful_call_is_done() -> None:
    # Acted, then asked a genuine follow-up — not a hand-back.
    turn = _turn(final_text="I added milk. Did you also want bread?")
    turn.record_tool("dunnes.add_to_cart_by_name", ok=True, mutating=True)
    assert classify(turn) == "done"


def test_trailing_whitespace_question_still_needs_user() -> None:
    turn = _turn(final_text="What size?  \n")
    assert classify(turn) == "needs-user"


def test_loop_exhausted_is_failed() -> None:
    turn = _turn(final_text="I got stuck in a tool loop and stopped.", loop_exhausted=True)
    assert classify(turn) == "failed"
