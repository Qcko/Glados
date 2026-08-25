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


def test_action_request_no_tools_empty_reply_is_failed() -> None:
    # Nothing claimed, so there is no fabrication to flag -- but nothing was
    # SAID either, and an unanswered turn is not a completed one. This used to
    # classify `done` by fall-through, which is how a silent turn reported
    # success (see `test_empty_reply_is_failed_even_when_tools_succeeded`).
    turn = _turn(final_text="", action_intent=True)
    assert classify(turn) == "failed"


def test_empty_reply_is_failed_even_when_tools_succeeded() -> None:
    # The regression this branch exists for: qwen3:4b burned its whole
    # num_predict budget on reasoning tokens and was cut before emitting any
    # content. The tool call had succeeded and nothing errored, so every other
    # guard passed and the turn was reported `done` while the user heard
    # silence (2026-08-25).
    turn = _turn(final_text="", action_intent=False)
    turn.record_tool("dunnes.scan_favorites_for_sales", ok=True)
    assert classify(turn) == "failed"


def test_whitespace_only_reply_is_failed() -> None:
    # Whitespace is not an answer; TTS would speak nothing.
    turn = _turn(final_text="   \n  ", action_intent=False)
    assert classify(turn) == "failed"


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


def test_empty_reply_after_a_successful_mutation_is_failed_but_not_retryable() -> None:
    # The safety interlock behind the silent-turn branch: such a turn IS a
    # failure, but it already changed external state, so both recovery paths
    # (scope fallback and specialist escalation) must refuse to replay it --
    # they gate on made_successful_mutation(). Without that, a cart-add whose
    # reply was swallowed would fire twice.
    turn = _turn(final_text="", action_intent=True)
    turn.record_tool("dunnes.add_to_cart_by_name", ok=True, mutating=True)
    assert classify(turn) == "failed"
    assert turn.made_successful_mutation() is True


# ---- Claimed a change it did not make -----------------------------------
#
# Every case below is a real turn from the 25-08-2026 bake-off. The two
# confabulations were classified `done` before this guard existed, because
# `_confabulated` only fires on a turn with ZERO tool calls and these called a
# DIFFERENT tool. The `done` cases are the false-positive net: a wrong
# accusation replaces a correct spoken reply with a canned failure line.


def _claim(final_text: str, calls) -> TurnRecord:
    turn = _turn(final_text=final_text)
    for tool, ok, mutating, args in calls:
        turn.record_tool(tool, ok, mutating=mutating, args=args)
    return turn


def test_claimed_add_with_only_a_remove_dispatched_is_confabulated() -> None:
    # Bake-off T4 on qwen3:8b with reasoning suppressed: it removed the
    # bananas, never called add(eggs), and announced the eggs anyway.
    turn = _claim(
        "Eggs added to cart.",
        [("dunnes.remove_from_cart_by_name", True, True, {"name": "bananas"})],
    )
    assert classify(turn) == "confabulated"


def test_claimed_removal_with_only_a_read_dispatched_is_confabulated() -> None:
    # Bake-off T6: view_cart ran, nothing was removed, and the reply said it
    # had been. Fails on both 4b arms and on 8b without reasoning.
    turn = _claim(
        "Milk removed from cart.",
        [("dunnes.view_cart", True, False, {})],
    )
    assert classify(turn) == "confabulated"


def test_claim_backed_by_a_matching_dispatch_is_done() -> None:
    turn = _claim(
        "Removed bananas, added 18 Irish Eggs Medium to your cart.",
        [
            ("dunnes.remove_from_cart_by_name", True, True, {"name": "bananas"}),
            ("dunnes.add_to_cart_by_name", True, True, {"quantity": 1, "query": "eggs"}),
        ],
    )
    assert classify(turn) == "done"


def test_id_only_dispatch_cannot_contradict_a_claim() -> None:
    # A productId never appears in a spoken reply, so it is no evidence either
    # way. Accusing here would fail every correct id-based removal.
    turn = _claim(
        "Removed Irish Low Fat Milk 3L from your cart. Your cart is now empty.",
        [
            ("dunnes.view_cart", True, False, {}),
            ("dunnes.remove_from_cart", True, True, {"productId": "100806893"}),
        ],
    )
    assert classify(turn) == "done"


def test_a_relative_adjust_may_be_narrated_as_a_removal() -> None:
    # The verb and the tool name disagree and that is FINE -- delta -1 from 1
    # really is a removal. Matching claim verbs against tool names would fail
    # this passing turn, which is why the check compares subjects instead.
    turn = _claim(
        "Removed 1 carton of Irish Low Fat Milk 3L. Your cart now has 1 milk.",
        [("dunnes.adjust_cart_quantity_by_name", True, True, {"name": "milk", "delta": -1})],
    )
    assert classify(turn) == "done"


def test_paraphrased_product_still_counts_as_backed() -> None:
    # The reply names the product, the argument named the query; they only have
    # to share a distinctive word.
    turn = _claim(
        "Added Kerrygold Pure Irish Butter 227g to your cart.",
        [("dunnes.add_to_cart_by_name", True, True, {"query": "irish butter"})],
    )
    assert classify(turn) == "done"


def test_a_read_turn_making_no_claim_is_untouched() -> None:
    turn = _claim(
        "Coca-Cola Original Taste 24 x 330ml and four more are on sale.",
        [("dunnes.scan_favorites_for_sales", True, False, {})],
    )
    assert classify(turn) == "done"


def test_an_offer_to_act_is_not_a_claim_of_having_acted() -> None:
    # Future tense is an intention. Only a completed claim can be a false one.
    turn = _claim("Shall I add milk to your cart?", [])
    assert classify(turn) == "needs-user"


def test_claim_check_does_not_need_action_intent() -> None:
    # The point of the guard: it never consults action_intent, because four of
    # five real mutation requests do not match is_action_request at all
    # ("Actually, add eggs instead", "...and then remove the milk").
    turn = _claim(
        "Milk removed from cart.",
        [("dunnes.view_cart", True, False, {})],
    )
    assert turn.action_intent is False
    assert classify(turn) == "confabulated"


# ---- Not this turn's doing: the false-positive net -----------------------
#
# Each of these was a REAL false positive before the guard learned to decline.
# They all silence a correct spoken reply and rewrite history, so they matter
# more than the catches do.


def test_a_denial_is_not_a_claim() -> None:
    turn = _claim("I have not added it to your cart yet.",
                  [("dunnes.view_cart", True, False, {})])
    assert classify(turn) == "done"


def test_reporting_that_nothing_changed_is_not_a_claim() -> None:
    turn = _claim("Nothing was removed -- your cart still has both milks.",
                  [("dunnes.view_cart", True, False, {})])
    assert classify(turn) == "done"


def test_a_change_someone_else_made_is_not_a_claim() -> None:
    turn = _claim("The sale price was updated by the store last night.",
                  [("dunnes.view_cart", True, False, {})])
    assert classify(turn) == "done"


def test_a_reference_to_an_earlier_turn_is_not_a_claim() -> None:
    turn = _claim("I already removed the bananas earlier; your cart has milk and eggs.",
                  [("dunnes.view_cart", True, False, {})])
    assert classify(turn) == "done"


def test_set_to_needs_a_number_to_be_a_quantity_claim() -> None:
    # "set to expire in 30 minutes" is not a quantity change.
    turn = _claim("Your cart is set to expire in 30 minutes.",
                  [("dunnes.view_cart", True, False, {})])
    assert classify(turn) == "done"


def test_an_offer_ending_in_a_question_is_never_confabulated() -> None:
    turn = _claim("Shall I have the milk removed?",
                  [("dunnes.view_cart", True, False, {})])
    assert classify(turn) != "confabulated"


def test_an_enum_only_argument_gives_nothing_to_compare() -> None:
    # Arguments exist but yield no speakable word, so there is no evidence
    # either way -- the guard must decline rather than accuse.
    turn = _claim("Kitchen light set to on.",
                  [("home.set_power", True, True, {"state": "on"})])
    assert classify(turn) == "done"


def test_a_non_digit_id_is_still_an_id() -> None:
    # Only Dunnes happens to use bare digits; a SKU or UUID is just as absent
    # from a spoken reply.
    turn = _claim("Removed it.",
                  [("dunnes.remove_from_cart", True, True, {"productId": "A100806893"})])
    assert classify(turn) == "done"


def test_a_batch_call_is_not_judged_by_its_recipe_name() -> None:
    turn = _claim(
        "Added 12 of 13 items.",
        [("dunnes.add_recipe_ingredients", True, True,
          {"recipe": "lasagne", "items": ["mince", "tomatoes"]})],
    )
    assert classify(turn) == "done"


def test_a_false_claim_beats_drift_so_the_reply_is_scrubbed() -> None:
    # Both drifted AND falsely claiming. `failed` would leave the false line
    # spoken and committed to history; only `confabulated` replaces it.
    turn = _turn(final_text="Milk removed from cart.", action_intent=True)
    turn.record_tool("dunnes.view_cart", ok=True, mutating=False, args={})
    assert classify(turn) == "confabulated"
