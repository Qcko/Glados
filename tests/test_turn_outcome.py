"""Deterministic turn-outcome classification (core/turn_outcome.py)."""

from __future__ import annotations

from glados.core.turn_outcome import (
    TurnRecord,
    classify,
    is_action_request,
    is_time_request,
)


def _turn(
    *,
    final_text: str = "",
    loop_exhausted: bool = False,
    action_intent: bool = False,
) -> TurnRecord:
    return TurnRecord(
        final_text=final_text,
        loop_exhausted=loop_exhausted,
        action_intent=action_intent,
    )


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
    turn = _turn(final_text="Which store did you mean -- Cornelscourt or Bishopstown?")
    assert classify(turn) == "needs-user"


def test_question_after_successful_call_is_done() -> None:
    # Acted, then asked a genuine follow-up -- not a hand-back.
    turn = _turn(final_text="I added milk. Did you also want bread?")
    turn.record_tool("dunnes.add_to_cart_by_name", ok=True, mutating=True)
    assert classify(turn) == "done"


def test_trailing_whitespace_question_still_needs_user() -> None:
    turn = _turn(final_text="What size?  \n")
    assert classify(turn) == "needs-user"


def test_loop_exhausted_is_failed() -> None:
    turn = _turn(final_text="I got stuck in a tool loop and stopped.", loop_exhausted=True)
    assert classify(turn) == "failed"


# ---- Semantic goal-check (action intent vs. search-and-narrate drift) ----


def test_action_request_search_then_narrate_is_failed() -> None:
    # Bake-off T2: "Add milk" -> search_products 200 OK -> described JSON,
    # never added. Pure error-derivation calls this `done`; the goal-check
    # catches it because no mutating call landed.
    turn = _turn(
        final_text="I found 30 milk products: whole, skimmed, oat...",
        action_intent=True,
    )
    turn.record_tool("dunnes.search_products", ok=True, mutating=False)
    assert classify(turn) == "failed"


def test_action_request_with_successful_mutation_is_done() -> None:
    turn = _turn(final_text="Added milk to your cart.", action_intent=True)
    turn.record_tool("dunnes.search_products", ok=True, mutating=False)
    turn.record_tool("dunnes.add_to_cart_by_name", ok=True, mutating=True)
    assert classify(turn) == "done"


def test_action_request_punted_with_question_is_needs_user() -> None:
    # Asked to act, searched, then handed back with a question instead of
    # acting. A clarification request, not silent drift.
    turn = _turn(
        final_text="I found a few milks -- which one did you mean?",
        action_intent=True,
    )
    turn.record_tool("dunnes.search_products", ok=True, mutating=False)
    assert classify(turn) == "needs-user"


def test_read_request_search_only_is_done() -> None:
    # "What's in my cart?" is a read -- a successful search satisfies it, so
    # the goal-check must not demand a mutating call.
    turn = _turn(final_text="You have milk and bread.", action_intent=False)
    turn.record_tool("dunnes.view_cart", ok=True, mutating=False)
    assert classify(turn) == "done"


def test_action_request_no_tools_declarative_is_confabulated() -> None:
    # Asked to act, dispatched nothing, yet declares it done. Fabricated
    # completion -- the poisoned-history signature (SESSION 2026-06-15).
    turn = _turn(final_text="Sure, adding that now.", action_intent=True)
    assert classify(turn) == "confabulated"


def test_action_request_no_tools_question_is_needs_user() -> None:
    # Same zero-tool turn, but ends on a question -- an honest hand-back
    # (clarification), not a fabricated completion.
    turn = _turn(final_text="Which milk did you mean?", action_intent=True)
    assert classify(turn) == "needs-user"


def test_action_request_no_tools_empty_reply_is_done() -> None:
    # Nothing said, nothing claimed -- no fabrication to flag. (A no-op turn
    # like this is not committed to history anyway.)
    turn = _turn(final_text="", action_intent=True)
    assert classify(turn) == "done"


def test_read_request_no_tools_declarative_is_not_confabulated() -> None:
    # A read/question, not an action request, never trips confabulation even
    # with zero tools -- only side-effecting claims are flagged.
    turn = _turn(final_text="It is Friday afternoon.", action_intent=False)
    assert classify(turn) == "done"


def test_is_action_request_detects_imperatives() -> None:
    assert is_action_request("Add milk to the cart.")
    assert is_action_request("Please remove the eggs")
    assert is_action_request("Can you book a table for two")
    assert is_action_request("Set the quantity to 3")


def test_is_action_request_rejects_reads_and_questions() -> None:
    assert not is_action_request("What's in my cart?")
    assert not is_action_request("Which of my favorites are on sale?")
    assert not is_action_request("Show me the milk options")
    assert not is_action_request("Tell me what to add")


def test_is_time_request_detects_time_questions() -> None:
    assert is_time_request("What time is it?")
    assert is_time_request("what time is it right now")
    assert is_time_request("What's the time?")
    assert is_time_request("What is the time")
    assert is_time_request("Time is it.")  # ASR drops the lead-in
    assert is_time_request("Do you have the time?")
    assert is_time_request("do you know the time")
    assert is_time_request("Tell me the time")
    assert is_time_request("give me the time")
    assert is_time_request("What's the current time?")
    assert is_time_request("the time right now please")


def test_is_time_request_rejects_non_time_questions() -> None:
    # Bare "what time does X" is a planning question, not "what's the clock".
    assert not is_time_request("What time does the shop open?")
    assert not is_time_request("What time should I leave?")
    assert not is_time_request("Set a timer for five minutes")
    assert not is_time_request("Add a bottle of thyme to the cart")
    assert not is_time_request("Is it a good time to buy milk?")
    # A trailing clause means it's a planning question, not "what's the clock".
    assert not is_time_request("What's the time the train leaves?")
    assert not is_time_request("What is the time of the next bus")
