"""What the USER said -- intent read off the utterance, before anything runs.

Separate from `turn_outcome` on purpose. That module judges a turn from the
record of what happened: which tools were dispatched, whether they succeeded,
what the reply claimed. This one judges only the request, and it is consumed by
callers that have no interest in outcomes at all -- the tool router asks
`is_action_request` to pick a scope, and the organizer asks `is_time_request`
to force a `time.now` dispatch rather than let the model answer a clock
question from its prior.

Both are deliberate heuristics over a closed vocabulary, and both are anchored
so they stay tight. They run on ASR output: no reliable punctuation, dropped
lead-ins ("Time is it."), filler mid-sentence. Both widenings so far were
measured against real logged utterances before they shipped; do the same,
because widening either one widens what the guards downstream will demand.
Nothing enforces that -- and note the corpus available today is a scripted
bake-off suite rather than conversation, so it can show a phrasing is handled
but not that an unseen one is safe (see `DESIGN-turn-outcome-guards.md`).
"""

from __future__ import annotations

import re

# Imperative verbs that imply the user wants the assistant to *change* external
# state this turn, not just look something up. Matched at the start of the
# request (after an optional politeness/filler lead-in) so a mid-sentence
# mention ("tell me what to add") doesn't trip the action heuristic. Read verbs
# (show, list, find, what/which/is) are deliberately absent -- a read request is
# satisfied by a successful search, so it must not demand a mutating call.
_ACTION_VERBS = (
    "add",
    "remove",
    "delete",
    "buy",
    "order",
    "book",
    "set",
    "change",
    "update",
    "cancel",
    "clear",
    "empty",
    "put",
    "schedule",
    "send",
    "play",
    "turn",
    # Measured 25-08-2026 against 1372 logged utterances: `take` and `make`
    # each recover a real mutation request the guard slept through ("Take one
    # of the milks off", "Actually, just make it one milk" -- 44 occurrences
    # between them). Both are polysemous, so they carry the idiom guard below.
    # `swap`/`replace` gained nothing on that corpus but have no read sense at
    # all, so they cost nothing. `drop` is deliberately absent: zero measured
    # gain, and "drop me a hint" is a read wearing an imperative.
    "take",
    "make",
    "swap",
    "replace",
)
# Politeness forms, discourse markers and ASR filler all sit in front of the
# verb without changing the request. Shared by the action regex and the idiom
# guard below: matching only one of them against a marker-prefixed utterance
# opens a hole neither widening opens alone ("actually, take a look at my
# cart" read as an imperative to act).
_LEAD_IN = (
    r"(?:(?:please|hey|ok(?:ay)?|glados|can you|could you|would you"
    r"|actually|just|now|then|so|um|uh|er|yeah|i mean"
    r"|i(?:'d| would) like (?:you )?to|i want (?:you )?to)[\s,]+)*"
)

_ACTION_INTENT_RE = re.compile(
    # Discourse markers and ASR filler lead an utterance far more often than
    # the politeness forms did: "Actually, add eggs instead", "Now add it
    # back", "uh remove the milk" all reached the guard as non-actions. They
    # are safe to skip because they add no new string-start -- the `^` anchor
    # still has to meet an action verb, and that is what keeps this regex off
    # reads.
    rf"^{_LEAD_IN}(?:{'|'.join(_ACTION_VERBS)})\b",
    re.IGNORECASE,
)


# `take` and `make` carry a common non-mutating reading in exactly the position
# the anchor inspects, and the anchor cannot tell them apart -- "take a look at
# my cart" is a read, "take one of the milks off" is a removal. Every phrase
# below was a verified false positive when those two verbs were added. Kept as
# a short closed list rather than a general rule: these are idioms, and an
# idiom is enumerable where a part-of-speech judgement is not.
_IDIOMATIC_NON_ACTION = re.compile(
    rf"^{_LEAD_IN}"
    r"(?:take\s+(?:a\s+look|your\s+time|care|it\s+easy)"
    r"|make\s+(?:sure|sense|a\s+note|note))\b",
    re.IGNORECASE,
)


def is_action_request(text: str) -> bool:
    """True if the user's utterance reads as an imperative to mutate external
    state ("add milk to the cart") rather than a question or a read ("what's
    in my cart?"). Used by the goal-check to demand a successful mutating tool
    call before calling such a turn ``done``.

    It stays anchored at the START of the utterance on purpose. Splitting a
    compound instruction on its coordinators and testing each clause was
    measured and REJECTED 25-08-2026: splitting manufactures new string-starts,
    and 11 of these verbs have an ordinary non-imperative reading in that
    position ("...and change is due on Friday", "...and buy one get one free
    deals"), so it trades a bounded miss for an unbounded false positive. The
    anchor is the whole safety argument -- do not remove it."""
    stripped = text.strip() if text else ""
    if not stripped or _IDIOMATIC_NON_ACTION.match(stripped):
        return False
    return _ACTION_INTENT_RE.match(stripped) is not None


# Asking the current time reads as a *question*, not an imperative, so the local
# model answers from its prior and fabricates a plausible-but-wrong time instead
# of calling time.now (SESSION 2026-06-15 Finding 2: "What time is it?" -> 0 calls,
# wrong; "run time.now" -> real call). A sibling to is_action_request: detect the
# intent deterministically so the organizer can force the dispatch rather than
# trusting the model to. Anchored to phrasings that clearly ask for the current
# time -- bare "what time does the shop close" / "set a timer" must NOT match.
_TIME_REQUEST_RE = re.compile(
    r"(?:"
    r"what(?:'s| is)\s+(?:the\s+)?time\b"  # what's the time / what is the time
    r"|what\s+time\s+is\s+it\b"  # what time is it (now)
    r"|\btime\s+is\s+it\b"  # ASR drops the lead-in: "Time is it."
    r"|do\s+you\s+(?:have|know)\s+(?:the\s+)?time\b"  # do you have/know the time
    r"|(?:tell|give)\s+me\s+the\s+time\b"  # tell/give me the time
    r"|(?:got|have)\s+the\s+time\b"  # got/have the time
    r"|\bcurrent\s+time\b"  # the current time
    r"|\bthe\s+time\s+(?:right\s+)?now\b"  # the time (right) now
    r")"
    # The time-phrase must sit at the END of the clause (modulo trailing fillers
    # and punctuation). Without this, "what's the time the train leaves" matches
    # the loose first branch -- a planning question, not "what's the clock".
    r"(?=(?:\s+(?:now|right|currently|please|exactly|today|then))*[\s?.!,]*$)",
    re.IGNORECASE,
)


def is_time_request(text: str) -> bool:
    """True if the utterance asks for the current time ("what time is it?",
    "do you have the time?"). Drives the organizer's deterministic time
    intercept (force time.now), so it errs tight: it must not fire on
    timer-setting or "what time does X open" planning questions."""
    return bool(text) and _TIME_REQUEST_RE.search(text.strip()) is not None
