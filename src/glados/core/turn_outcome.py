"""Deterministic per-turn outcome classification.

A turn can end three ways from the harness/router's point of view:

- ``done`` -- the model did what was asked (or answered conversationally).
- ``needs-user`` -- the model handed back with a question without acting.
- ``failed`` -- a tool error went unrecovered, or the tool loop blew its budget.

The classification is derived from *observable* tool results, never from the
model's self-report. A drifting local model narrates failure cheerfully (see
the 2026-05-30 bake-off T6: two tool errors followed by an upbeat clarifying
question), so its narration is worthless as a completion signal. The wire
already carries the truth -- each dispatch records ``ok`` -- this module just
aggregates it to the turn level so the v2.6 router can use it as an
escalation input.

Pure error-derivation misses *semantic* drift, where a tool returns 200 OK but
the model never acts on the result (bake-off T2: ``search_products`` succeeded,
the model just never added to the cart). The lightweight goal-check below
closes the common case of that gap without leaving the deterministic lane: when
the user issued an *action* request ("add milk", "remove the eggs") and the
turn ran tools but never landed a successful *mutating* call, the action did
not happen -- that is drift, not success. It stays a heuristic on observable
signals (the user's verb + the per-tool ``mutating`` flag), not a model
self-report or a semantic judgement of the reply text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

TurnOutcomeKind = Literal["done", "needs-user", "failed", "confabulated"]

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
)
_ACTION_INTENT_RE = re.compile(
    r"^(?:(?:please|hey|ok(?:ay)?|glados|can you|could you|would you|i(?:'d| would) like (?:you )?to|i want (?:you )?to)[\s,]+)*"
    rf"(?:{'|'.join(_ACTION_VERBS)})\b",
    re.IGNORECASE,
)


def is_action_request(text: str) -> bool:
    """True if the user's utterance reads as an imperative to mutate external
    state ("add milk to the cart") rather than a question or a read ("what's
    in my cart?"). Used by the goal-check to demand a successful mutating tool
    call before calling such a turn ``done``."""
    return bool(text) and _ACTION_INTENT_RE.match(text.strip()) is not None


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


@dataclass(frozen=True)
class ToolRecord:
    """One tool dispatch within a turn, as the registry observed it."""

    tool: str  # qualified name, e.g. "dunnes.add_to_cart_by_name"
    ok: bool
    # Side-effecting calls (cart writes, checkout, login, money). Mirrors
    # ToolSpec.requires_confirmation -- the existing per-tool flag for
    # "this mutates external state".
    mutating: bool = False
    # Lowercased free-text argument values -- what the call was ABOUT
    # ("milk", "bananas"). Ids and numbers are excluded: they never appear in a
    # spoken reply, so they cannot corroborate or contradict one. Used only by
    # the claim check below, which asks whether the thing the reply says was
    # changed is the thing any successful call actually touched.
    subjects: tuple[str, ...] = ()


@dataclass
class TurnRecord:
    """Mutable accumulator a turn fills in as it runs, then classifies once."""

    tools: list[ToolRecord] = field(default_factory=list)
    final_text: str = ""
    loop_exhausted: bool = False
    # True when the user's request was an imperative to change external state
    # (set by the organizer via is_action_request on the user text). Drives the
    # goal-check: such a turn is only `done` if a mutating call actually landed.
    action_intent: bool = False

    def record_tool(
        self,
        tool: str,
        ok: bool,
        *,
        mutating: bool = False,
        args: dict | None = None,
    ) -> None:
        self.tools.append(
            ToolRecord(tool=tool, ok=ok, mutating=mutating, subjects=_subjects(args))
        )

    def made_successful_mutation(self) -> bool:
        """True if a side-effecting tool call landed this turn. The specialist
        escalation path checks this before re-running a `failed` turn: a turn
        that already mutated external state must not be replayed cold, or the
        side effect (cart write, checkout, send) fires twice."""
        return _has_successful_mutation(self.tools)


def classify(turn: TurnRecord) -> TurnOutcomeKind:
    """Reduce a finished turn to a single typed outcome.

    Priority order matters: an unrecovered tool error is ``failed`` even when
    the turn ends on a question (T6 ended on a cheerful question *after* two
    errors -- that is a failure, not a clarification request)."""
    if turn.loop_exhausted:
        return "failed"
    if _has_unrecovered_error(turn.tools):
        return "failed"
    if said_nothing(turn):
        return "failed"
    if _confabulated(turn) or _claimed_a_change_it_did_not_make(turn):
        # Ahead of the drift check on purpose. A turn can be both drifted AND
        # making a false claim, and only `confabulated` gets the reply replaced
        # and kept out of history (see Organizer._handle_confabulation);
        # `failed` would leave the false "Milk removed from cart." both spoken
        # and committed, which is the poisoning this whole module exists to
        # stop. The two are otherwise disjoint -- `_confabulated` needs a
        # zero-tool turn, `_action_drifted` needs tools -- so nothing else
        # changes order here.
        return "confabulated"
    if _action_drifted(turn):
        # Asked to act, ran tools, but never landed the mutation. Ending on a
        # question is a hand-back (the model punted instead of acting);
        # ending on a statement is silent drift (bake-off T2: searched,
        # narrated the JSON, never added). Both mean the action didn't happen.
        return "needs-user" if _ends_on_question(turn.final_text) else "failed"
    if _ends_on_question(turn.final_text) and not _has_successful_call(turn.tools):
        return "needs-user"
    return "done"


def said_nothing(turn: TurnRecord) -> bool:
    """The turn ended with no reply text at all. Every other guard here asks
    whether the model did the RIGHT thing; this one asks whether it answered,
    and an unanswered turn cannot be `done` however clean its tool calls were.

    Observed 2026-08-25 on qwen3:4b: a reasoning model spent its entire
    num_predict budget on `thinking` tokens and was cut at `done_reason=length`
    before emitting a single content token. Tools had all succeeded and nothing
    errored, so the turn fell through to `done` -- the harness reported success
    for a turn the user heard nothing from. Silence is the one failure the user
    cannot tell from a crash, so it must not classify as success."""
    return not turn.final_text.strip()


def _confabulated(turn: TurnRecord) -> bool:
    """The user asked to mutate state, the turn dispatched *no tools at all*,
    yet the reply declares the action done ("Sure, adding that now.", "Added
    milk to your cart."). That is a fabricated completion -- the signature of a
    poisoned history steering the model away from tool-calls (SESSION 2026-06-15:
    six consecutive zero-dispatch turns all narrating success). It is the
    zero-tool case `_action_drifted` deliberately leaves alone: there, a
    *declarative* ending is the discriminator from an honest hand-back. Ending on
    a question is a clarification ("which milk?") and stays `needs-user`."""
    return (
        turn.action_intent
        and not turn.tools
        and bool(turn.final_text.strip())
        and not _ends_on_question(turn.final_text)
    )


# Past-tense assertions that a change HAPPENED. Present and future forms are
# absent on purpose ("I'll add", "to add", "shall I remove") -- those are
# intentions, and only a completed claim can be a false one.
_CLAIM_RE = re.compile(
    r"\b(added|removed|deleted|cleared|emptied|updated|changed)\b"
    # "set ... to 2" is a quantity claim; "set to expire in 30 minutes" is not,
    # so the number is required rather than optional.
    r"|\bset\b[^.!?]{0,20}?\bto\s+\d",
    re.IGNORECASE,
)

# A claim inside one of these is not this turn reporting its own work:
# a denial, a report of what someone else did, or a reference back to an
# earlier turn. Each was a real false positive before it was excluded.
_NOT_THIS_TURNS_DOING = re.compile(
    r"\b(not|n't|nothing|never|no)\b"  # "I have not added it", "Nothing was removed"
    r"|\balready\b|\bearlier\b|\bpreviously\b|\blast night\b"  # a prior turn
    r"|\bby\s+the\s+\w+",  # "was updated by the store"
    re.IGNORECASE,
)

# Words too common to prove a reply is talking about a given tool argument.
_UNDISTINCTIVE = frozenset(
    {
        "a", "an", "and", "the", "to", "of", "for", "in", "on", "at", "it",
        "my", "your", "cart", "item", "items", "one", "two", "some", "all",
        "please", "thing", "things", "product", "products",
    }
)


def _claimed_a_change_it_did_not_make(turn: TurnRecord) -> bool:
    """The reply says something was added/removed/set, and the dispatch record
    does not support it.

    This is the sibling of `_confabulated` for turns that DID call tools -- and
    it is the one that fires in practice, because it never consults
    `action_intent`. Measured 25-08-2026: four of five real mutation requests
    ("Actually, add eggs instead", "...and then remove the milk", "Take one of
    the milks off") do not match `is_action_request` at all, so every guard
    keyed on it stays asleep. The reply is where the claim lives, and the
    dispatch record is the ground truth; neither depends on how the user
    phrased the request.

    Two real failures from the same day's bake-off, both previously `done`:
    the model removed the bananas, never called add(eggs), and said "Eggs added
    to cart"; and it called view_cart, removed nothing, and said "Milk removed
    from cart".

    It FAILS OPEN wherever it cannot judge, because a false positive replaces a
    correct spoken reply with a canned "no record of that" line -- telling a
    user their shopping did not happen when it did is its own kind of lie."""
    if _ends_on_question(turn.final_text):
        # "Shall I have the milk removed?" is an offer, not a report.
        return False
    if not any(_asserts_a_change(s) for s in _sentences(turn.final_text)):
        return False

    landed = [t for t in turn.tools if t.ok and t.mutating]
    if not landed:
        # Claimed a change with nothing side-effecting behind it at all.
        return True

    subject_words = {w for tool in landed for s in tool.subjects for w in _words(s)}
    if not subject_words:
        # Every successful call identified its target by an id, a number or an
        # enum -- none of which a spoken reply repeats. Nothing comparable, so
        # do not accuse. Note this tests for usable WORDS, not merely for the
        # presence of arguments: a call with only `state="on"` has arguments and
        # still gives us nothing to check against.
        return False
    return not (subject_words & _words(turn.final_text))


def _asserts_a_change(sentence: str) -> bool:
    return bool(
        _CLAIM_RE.search(sentence) and not _NOT_THIS_TURNS_DOING.search(sentence)
    )


def _sentences(text: str) -> list[str]:
    # Judged per sentence so one honest clause cannot excuse a false one, and
    # a negation in a neighbouring sentence cannot excuse a claim in this one.
    return [s for s in re.split(r"[.!?]+", text) if s.strip()]


def _words(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if w not in _UNDISTINCTIVE and len(w) > 2
    }


def _subjects(args: dict | None) -> tuple[str, ...]:
    """The free-text values a call was aimed at.

    Identifiers are excluded even though they arrive as strings: a spoken reply
    names the product, never its id, so keeping them made every id-based call
    look like a contradicted claim. The test is "contains a run of letters",
    not "is not all digits" -- `prod-123` and a UUID are just as absent from
    speech as `100806893`, and only Dunnes happens to use bare digits.

    A call carrying a list or dict argument yields nothing at all. Those are
    the batch flows (a recipe's worth of items), where the top-level strings
    describe the request rather than any one item, and judging a twelve-item
    add by the word "lasagne" would be worse than not judging it."""
    if not args:
        return ()
    if any(isinstance(v, (list, dict, tuple)) for v in args.values()):
        return ()
    return tuple(
        value
        for v in args.values()
        if isinstance(v, str)
        and (value := v.strip().lower())
        and re.search(r"[a-z]{3}", value)
    )


def _action_drifted(turn: TurnRecord) -> bool:
    """The user asked to mutate state, the turn ran at least one tool, yet no
    successful mutating call landed. A pure read/search can't satisfy an action
    request. Turns with no tool calls are out of scope here -- the search-and-
    narrate drift this targets always shows tool activity; the zero-tool
    fabricated-completion case is handled by `_confabulated`."""
    return (
        turn.action_intent
        and bool(turn.tools)
        and not _has_successful_mutation(turn.tools)
    )


def _has_successful_mutation(tools: list[ToolRecord]) -> bool:
    return any(t.ok and t.mutating for t in tools)


def _has_unrecovered_error(tools: list[ToolRecord]) -> bool:
    """An error on a tool is recovered only by a *later* successful call to
    the same tool. A success that precedes the error does not count."""
    for name in {t.tool for t in tools}:
        calls = [t for t in tools if t.tool == name]
        last_error = _last_index(calls, ok=False)
        if last_error is None:
            continue
        recovered_after = any(c.ok for c in calls[last_error + 1 :])
        if not recovered_after:
            return True
    return False


def _last_index(calls: list[ToolRecord], *, ok: bool) -> int | None:
    for i in range(len(calls) - 1, -1, -1):
        if calls[i].ok is ok:
            return i
    return None


def _has_successful_call(tools: list[ToolRecord]) -> bool:
    return any(t.ok for t in tools)


def _ends_on_question(text: str) -> bool:
    return text.rstrip().endswith("?")
