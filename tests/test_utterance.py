"""Reading intent off the user's utterance.

These are heuristics over a closed vocabulary, and both of them arm machinery
downstream: `is_action_request` decides whether a turn must land a mutating
call to count as done, and `is_time_request` forces a `time.now` dispatch. A
false positive in either is expensive, so the cases below are the measured
evidence, not illustrations -- most of them are phrasings that were observed
being got wrong.
"""

from __future__ import annotations

from glados.core.utterance import is_action_request, is_time_request


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


def test_discourse_markers_no_longer_hide_a_mutation_request() -> None:
    """Measured 25-08-2026 over 1372 logged utterances: these four phrasings
    account for 81 turns whose action guard was asleep, because the utterance
    opened with a discourse marker or a verb the list lacked."""
    assert is_action_request("Actually, add eggs instead.")
    assert is_action_request("Actually, just make it one milk.")
    assert is_action_request("Take one of the milks off.")
    assert is_action_request("Now add it back.")
    assert is_action_request("uh remove the milk")
    assert is_action_request("i mean add the eggs")


def test_polysemous_verbs_do_not_turn_reads_into_actions() -> None:
    """`take` and `make` were the two verbs worth adding and the two that can
    lead an ordinary read. Each of these was a verified false positive before
    the idiom guard."""
    assert not is_action_request("take a look at my cart")
    assert not is_action_request("make sure the milk is in there")
    assert not is_action_request("take your time")
    assert not is_action_request("make a note of that")


def test_a_leading_marker_does_not_make_a_read_an_action() -> None:
    """The lead-in group is skipped, not treated as an action itself -- the
    `^` anchor still has to meet a verb, which is what keeps the widening
    safe."""
    assert not is_action_request("so what is in my cart")
    assert not is_action_request("then what did I order")
    assert not is_action_request("um what time is it")


def test_a_marker_does_not_smuggle_an_idiom_past_the_guard() -> None:
    """The two widenings in this change interact: neither the extra verbs nor
    the extra lead-ins produces these alone. The idiom guard has to skip the
    same lead-ins the action regex does, or "actually, take a look at my cart"
    reads as an imperative to act and arms two guards on a plain read."""
    assert not is_action_request("actually, take a look at my cart")
    assert not is_action_request("ok, take a look at my cart")
    assert not is_action_request("now take a look")
    assert not is_action_request("just make sure the milk is there")
    assert not is_action_request("um make sure it is there")
